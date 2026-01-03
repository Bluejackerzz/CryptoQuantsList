import os
import datetime
import numpy as np
import pandas as pd
import networkx as nx
import ccxt
import yfinance as yf
from scipy.linalg import inv
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Cache
GLOBAL_PRICE_CACHE = pd.DataFrame()

DEFAULT_COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOT-USD', 'TRX-USD', 'TON-USD',
    'LINK-USD', 'MATIC-USD', 'NEAR-USD', 'ATOM-USD', 'LTC-USD',
    'BCH-USD', 'APT-USD', 'SUI-USD', 'ALGO-USD', 'ICP-USD',
    'HBAR-USD', 'FTM-USD', 'KAS-USD', 'SEI-USD', 'INJ-USD'
]

ALPHA_PARAM = 0.5

# --- CONNECTIONS ---
exchange_private = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_API_SECRET'),
    'enableRateLimit': True,
    'options': { 'defaultType': 'linear', 'adjustForTimeDifference': True },
    'urls': {
        'api': {
            'public': 'https://api-demo.bybit.com',
            'private': 'https://api-demo.bybit.com',
        }
    }
})

exchange_public = ccxt.bybit({
    'enableRateLimit': True,
    'options': { 'defaultType': 'linear' },
    'urls': {
        'api': {
            'public': 'https://api-demo.bybit.com', 
            'private': 'https://api-demo.bybit.com',
        }
    }
})

# --- DATA ENGINE ---

def fetch_crypto_data(tickers):
    global GLOBAL_PRICE_CACHE
    price_data = {}
    use_yahoo_fallback = False
    
    print(f"--- Attempting Data Fetch for {len(tickers)} assets ---")
    
    for ticker in tickers:
        try:
            symbol_root = ticker.split('-')[0]
            bybit_symbol = f"{symbol_root}/USDT"
            ohlcv = exchange_public.fetch_ohlcv(bybit_symbol, timeframe='D', limit=60)
            
            if not ohlcv: raise Exception("Empty")
            closes = [x[4] for x in ohlcv]
            
            if len(closes) > 30:
                price_data[ticker] = closes
            else: raise Exception("Short data")
            
        except Exception:
            if not price_data: 
                use_yahoo_fallback = True
                break

    if use_yahoo_fallback or len(price_data) < 2:
        print(">> SWITCHING TO YAHOO FINANCE FALLBACK <<")
        try:
            df = yf.download(tickers, period="60d", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df['Close']
                except KeyError:
                    if 'Close' in df.columns.get_level_values(0):
                        df = df.xs('Close', axis=1, level=0)
            
            df = df.dropna(axis=1, how='all')
            df = df.dropna()
            GLOBAL_PRICE_CACHE = df 
            return df
        except Exception as e:
            print(f"Yahoo Fallback Failed: {e}")
            return pd.DataFrame()

    df = pd.DataFrame(price_data)
    GLOBAL_PRICE_CACHE = df
    return df

def compute_quant_metrics(df_prices):
    returns = df_prices.pct_change().dropna()
    
    if returns.empty:
        return {}

    latest_returns = returns.iloc[-1].values 
    tickers = df_prices.columns.tolist()
    
    # 1. Laplacian
    correlation_matrix = returns.corr().values
    correlation_matrix = np.nan_to_num(correlation_matrix)
    
    adjacency = np.abs(correlation_matrix)
    np.fill_diagonal(adjacency, 0)
    
    degrees = np.sum(adjacency, axis=1)
    D = np.diag(degrees)
    L = D - adjacency
    
    I = np.eye(len(L))
    try:
        filter_matrix = inv(I - ALPHA_PARAM * L)
        h = np.dot(filter_matrix, latest_returns)
    except np.linalg.LinAlgError:
        h = latest_returns

    residuals = latest_returns - h
    
    # 2. Topology
    clipped_corr = np.clip(correlation_matrix, -1.0, 0.99999)
    distance_matrix = np.sqrt(2 * (1 - clipped_corr))
    
    betti_0_curve = []
    epsilons = np.linspace(0, 2, 20)
    n_assets = len(tickers)
    
    for eps in epsilons:
        G = nx.Graph()
        G.add_nodes_from(range(n_assets))
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if distance_matrix[i, j] < eps:
                    G.add_edge(i, j)
        b_0 = nx.number_connected_components(G)
        betti_0_curve.append(int(b_0))

    # 3. Nodes
    graph_nodes = []
    for i, ticker in enumerate(tickers):
        graph_nodes.append({
            "id": ticker,
            "residual": float(residuals[i])
        })

    # 4. Edges
    graph_edges = []
    edge_threshold = 0.3 if n_assets > 15 else 0.25 
    
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            weight = adjacency[i, j]
            if weight > edge_threshold: 
                graph_edges.append({
                    "source": tickers[i],
                    "target": tickers[j],
                    "weight": float(weight)
                })

    # 5. Signals
    results = {}
    for i, ticker in enumerate(tickers):
        resid = residuals[i]
        price = float(df_prices.iloc[-1, i])
        
        signal = "NEUTRAL"
        sl_price = 0.0
        tp_price = 0.0
        threshold = 0.015
        
        if resid > threshold:
            signal = "SHORT"
            reward_pct = abs(resid)
            risk_pct = reward_pct / 2.0
            tp_price = price * (1.0 - reward_pct)
            sl_price = price * (1.0 + risk_pct)

        elif resid < -threshold:
            signal = "LONG"
            reward_pct = abs(resid)
            risk_pct = reward_pct / 2.0
            tp_price = price * (1.0 + reward_pct)
            sl_price = price * (1.0 - risk_pct)
            
        results[ticker] = {
            "price": price,
            "residual_e": float(resid),
            "signal": signal,
            "rr_setup": {
                "entry": price,
                "sl": float(sl_price),
                "tp": float(tp_price)
            }
        }

    return {
        "market_data": results,
        "topology": {
            "epsilons": epsilons.tolist(),
            "betti_0": betti_0_curve
        },
        "network_graph": {
            "nodes": graph_nodes, 
            "edges": graph_edges
        },
        "tickers": tickers
    }

# --- ACCOUNT HELPER ---
def get_account_summary():
    balance_info = {"USDT": 0.0}
    orders_info = []
    try:
        bal = exchange_private.fetch_balance()
        if 'USDT' in bal:
            balance_info['USDT'] = bal['USDT']['free']
        try:
            orders = exchange_private.fetch_open_orders(params={'category': 'linear'})
            for o in orders:
                orders_info.append({
                    'symbol': o['symbol'],
                    'side': o['side'],
                    'price': o['price'],
                    'amount': o['amount']
                })
        except: pass
    except Exception as e:
        return {"error": str(e)}
        
    return {"balance": balance_info, "orders": orders_info}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    selected_coins = data.get('coins', DEFAULT_COINS)
    df = fetch_crypto_data(selected_coins)
    
    if df.empty or len(df.columns) < 2:
        return jsonify({"error": "Not enough valid data found."}), 400
        
    return jsonify(compute_quant_metrics(df))

@app.route('/account', methods=['GET'])
def account():
    return jsonify(get_account_summary())

@app.route('/history', methods=['POST'])
def history():
    global GLOBAL_PRICE_CACHE
    data = request.json
    ticker = data.get('ticker')
    # If true, we return BTC data alongside for overlay
    overlay_btc = data.get('overlay_btc', False)
    
    if GLOBAL_PRICE_CACHE.empty or ticker not in GLOBAL_PRICE_CACHE.columns:
        return jsonify({"error": "Data not in cache. Run Engine first."}), 400
        
    try:
        prices = GLOBAL_PRICE_CACHE[ticker].values.tolist()
        btc_prices = []
        
        if overlay_btc:
            # Find BTC ticker in columns (e.g., BTC-USD)
            btc_col = next((c for c in GLOBAL_PRICE_CACHE.columns if 'BTC' in c), None)
            if btc_col:
                btc_prices = GLOBAL_PRICE_CACHE[btc_col].values.tolist()
        
        labels = list(range(len(prices)))
        
        return jsonify({
            "ticker": ticker,
            "prices": prices,
            "btc_prices": btc_prices,
            "labels": labels
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Quant Engine X (Dual-Axis Mode)...")
    app.run(debug=True)