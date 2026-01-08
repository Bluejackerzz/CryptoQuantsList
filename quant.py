import os
import datetime
import time
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
    timestamps = []
    use_yahoo_fallback = False
    
    print(f"--- Attempting Data Fetch for {len(tickers)} assets ---")
    
    for ticker in tickers:
        try:
            time.sleep(0.2) # Anti-Ban
            symbol_root = ticker.split('-')[0]
            bybit_symbol = f"{symbol_root}/USDT"
            ohlcv = exchange_public.fetch_ohlcv(bybit_symbol, timeframe='D', limit=100)
            
            if not ohlcv: raise Exception("Empty")
            closes = [x[4] for x in ohlcv]
            
            if not timestamps and len(closes) > 30:
                raw_ts = [x[0] for x in ohlcv]
                timestamps = [datetime.datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d') for ts in raw_ts]

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
            df = yf.download(tickers, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df['Close']
                except KeyError:
                    if 'Close' in df.columns.get_level_values(0): df = df.xs('Close', axis=1, level=0)
            df = df.dropna(axis=1, how='all')
            df = df.dropna()
            df = df.tail(100)
            df.index = df.index.strftime('%Y-%m-%d')
            GLOBAL_PRICE_CACHE = df 
            return df
        except Exception as e:
            return pd.DataFrame()

    if timestamps:
        min_len = min([len(v) for v in price_data.values()] + [len(timestamps)])
        for k in price_data: price_data[k] = price_data[k][-min_len:]
        timestamps = timestamps[-min_len:]
        df = pd.DataFrame(price_data, index=timestamps)
    else:
        df = pd.DataFrame(price_data)

    GLOBAL_PRICE_CACHE = df
    return df

# --- MATH ALGORITHMS ---

def compute_hurst(time_series):
    try:
        ts = np.array(time_series)
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = m[0] * 2.0 
        return float(hurst)
    except: return 0.5

def compute_forecast_gnn(returns, adjacency):
    X_self = returns.shift(1).fillna(0).values 
    X_neighbors = np.dot(X_self, adjacency)    
    Y = returns.values                         
    forecasts = {}
    tickers = returns.columns.tolist()
    for i, ticker in enumerate(tickers):
        try:
            features = np.column_stack((X_self[:, i], X_neighbors[:, i]))
            target = Y[:, i]
            XtX = np.dot(features.T, features) + np.eye(2) * 1e-5
            weights = np.dot(np.linalg.inv(XtX), np.dot(features.T, target))
            current_self = returns.iloc[-1, i]
            current_neighbor = np.dot(returns.iloc[-1].values, adjacency[:, i])
            pred = weights[0] * current_self + weights[1] * current_neighbor
            forecasts[ticker] = {"pred": float(pred)}
        except: forecasts[ticker] = {"pred": 0.0}
    return forecasts

def compute_market_regime(laplacian, returns):
    try:
        eigenvals = np.linalg.eigvalsh(laplacian)
        eigenvals = np.sort(eigenvals)
        fiedler_value = float(eigenvals[1]) if len(eigenvals) > 1 else 0.0
        avg_momentum = np.mean(returns.iloc[-1].values)
        if fiedler_value < 0.5:
            phase = "DIVERGENT (CHOP)"
            bias = "NEUTRAL"
            desc = "Low connectivity. Assets moving independently."
        else:
            if avg_momentum > 0.005:
                phase = "BULL ACCUMULATION"
                bias = "LONG"
                desc = "High connectivity + Positive Momentum."
            elif avg_momentum < -0.005:
                phase = "BEAR DISTRIBUTION"
                bias = "SHORT"
                desc = "High connectivity + Negative Momentum."
            else:
                phase = "CONSOLIDATION"
                bias = "NEUTRAL"
                desc = "High connectivity, flat momentum."
        return { "phase": phase, "bias": bias, "desc": desc }
    except Exception as e: return {"phase": "ERROR", "bias": "NEUTRAL", "desc": str(e)}

def compute_quant_metrics(df_prices):
    returns = df_prices.pct_change().dropna()
    if returns.empty: return {}

    latest_returns = returns.iloc[-1].values 
    tickers = df_prices.columns.tolist()
    
    correlation_matrix = returns.corr().values
    correlation_matrix = np.nan_to_num(correlation_matrix)
    adjacency = np.maximum(correlation_matrix, 0)
    np.fill_diagonal(adjacency, 0)
    degrees = np.sum(adjacency, axis=1)
    D = np.diag(degrees)
    L = D - adjacency
    
    I = np.eye(len(L))
    try:
        filter_matrix = inv(I - ALPHA_PARAM * L)
        h = np.dot(filter_matrix, latest_returns)
    except np.linalg.LinAlgError: h = latest_returns

    residuals = latest_returns - h
    
    clipped_corr = np.clip(correlation_matrix, -1.0, 0.99999)
    distance_matrix = np.sqrt(2 * (1 - clipped_corr))
    betti_0_curve = []
    persistence_barcodes = []
    epsilons = np.linspace(0, 2, 40)
    n_assets = len(tickers)
    prev_components = n_assets
    
    for eps in epsilons:
        G = nx.Graph()
        G.add_nodes_from(range(n_assets))
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if distance_matrix[i, j] < eps: G.add_edge(i, j)
        n_curr = nx.number_connected_components(G)
        betti_0_curve.append(int(n_curr))
        if n_curr < prev_components:
            for _ in range(prev_components - n_curr): persistence_barcodes.append(float(eps))
        prev_components = n_curr
    for _ in range(prev_components): persistence_barcodes.append(2.0)

    forecasts = compute_forecast_gnn(returns, adjacency)
    regime_data = compute_market_regime(L, returns)

    graph_nodes = []
    graph_edges = []
    edge_threshold = 0.3 if n_assets > 15 else 0.25 
    for i in range(n_assets):
        graph_nodes.append({ "id": tickers[i], "residual": float(residuals[i]) })
        for j in range(i + 1, n_assets):
            weight = adjacency[i, j]
            if weight > edge_threshold: 
                graph_edges.append({"source": tickers[i], "target": tickers[j], "weight": float(weight)})

    results = {}
    for i, ticker in enumerate(tickers):
        resid = residuals[i]
        price = float(df_prices.iloc[-1, i])
        pred = forecasts[ticker]['pred']
        hurst_val = compute_hurst(df_prices[ticker].values)
        
        signal = "NEUTRAL"
        if resid < -0.015: signal = "LONG"
        elif resid > 0.015: signal = "SHORT"
        
        ai_conf = "WEAK"
        if (signal == "LONG" and pred > 0) or (signal == "SHORT" and pred < 0): ai_conf = "STRONG"
        
        hurst_label = "RANDOM"
        if hurst_val > 0.6: hurst_label = "TRENDING"
        if hurst_val < 0.4: hurst_label = "MEAN-REV"
        
        # Hurst Adaptive Default (UI Display)
        reward_mult = 0.04
        if hurst_val > 0.6: reward_mult = 0.06 
        elif hurst_val < 0.4: reward_mult = 0.03
        
        if signal == "LONG":
            rr_entry = price
            rr_sl = price * 0.98
            rr_tp = price * (1 + reward_mult)
        elif signal == "SHORT":
            rr_entry = price
            rr_sl = price * 1.02
            rr_tp = price * (1 - reward_mult)
        else:
            rr_entry, rr_sl, rr_tp = 0,0,0
        
        results[ticker] = {
            "price": price,
            "residual_e": float(resid),
            "signal": signal,
            "ai_forecast": { "price": price * (1+pred), "pct": pred, "conf": ai_conf },
            "hurst": { "value": hurst_val, "label": hurst_label },
            "rr_setup": { 
                "entry": rr_entry, 
                "sl": rr_sl, 
                "tp": rr_tp 
            }
        }

    return {
        "regime": regime_data,
        "market_data": results,
        "topology": { "epsilons": epsilons.tolist(), "betti_0": betti_0_curve, "persistence": persistence_barcodes },
        "network_graph": { "nodes": graph_nodes, "edges": graph_edges },
        "correlation_matrix": correlation_matrix.tolist(),
        "tickers": tickers
    }

def compute_rolling_surface(df_prices):
    if df_prices.empty: return {}
    returns = df_prices.pct_change().dropna()
    btc_col = next((c for c in returns.columns if 'BTC' in c), None)
    if not btc_col: return {}
    window_size = 30
    steps = len(returns) - window_size
    if steps < 1: return {}
    z_data = []
    dates = []
    tickers = [c for c in returns.columns if c != btc_col]
    for i in range(steps):
        window = returns.iloc[i : i + window_size]
        date_label = window.index[-1]
        corr_mat = window.corr()
        row = []
        for t in tickers:
            val = corr_mat.loc[btc_col, t]
            row.append(float(val) if not np.isnan(val) else 0.0)
        z_data.append(row)
        dates.append(str(date_label))
    return { "x": dates, "y": tickers, "z": z_data }

# --- BACKTESTING (Logic Fixed + Custom Position Size) ---
def run_backtest_logic(df_prices, simulation_days=30, start_capital=1000.0, tp_mode='hurst', trade_pct=0.05):
    required_len = 30 + simulation_days
    if df_prices.empty: return {"error": "No price data loaded."}
    if len(df_prices) < required_len:
        simulation_days = len(df_prices) - 31
        if simulation_days < 1: return {"error": f"Data too short ({len(df_prices)} days)."}
    
    relevant_df = df_prices.tail(30 + simulation_days)
    returns = relevant_df.pct_change().dropna()
    tickers = relevant_df.columns.tolist()
    prices_raw = relevant_df.values
    
    equity_curve = [float(start_capital)]
    trades_log = []
    window_size = 30
    test_days = len(returns) - window_size
    
    for i in range(test_days):
        current_idx = window_size + i - 1
        if current_idx >= len(returns): break
        
        window_returns = returns.iloc[i : current_idx + 1] 
        current_prices_window = relevant_df.iloc[i : current_idx + 1]

        corr_mat = window_returns.corr().values
        corr_mat = np.nan_to_num(corr_mat)
        adj = np.maximum(corr_mat, 0)
        np.fill_diagonal(adj, 0)
        D = np.diag(np.sum(adj, axis=1))
        L = D - adj
        
        try:
            I = np.eye(len(L))
            filt = inv(I - ALPHA_PARAM * L)
            latest_ret = window_returns.iloc[-1].values
            h = np.dot(filt, latest_ret)
            residuals = latest_ret - h
        except:
            equity_curve.append(equity_curve[-1])
            continue
            
        ai_forecasts = {}
        if tp_mode == 'ai':
            ai_forecasts = compute_forecast_gnn(window_returns, adj)

        if current_idx + 1 >= len(returns): break
        next_day_close_row = prices_raw[i + window_size + 1] if (i + window_size + 1) < len(prices_raw) else None
        if next_day_close_row is None: break
        
        date_label = returns.index[current_idx + 1]
        entry_prices_idx = i + window_size
        entry_prices = prices_raw[entry_prices_idx] 
        
        daily_pnl = 0.0
        for t_idx, ticker in enumerate(tickers):
            resid = residuals[t_idx]
            price = entry_prices[t_idx]
            next_close = next_day_close_row[t_idx]
            
            position = 0 
            if resid < -0.015: position = 1 
            elif resid > 0.015: position = -1
            
            if position != 0:
                # Dynamic Position Sizing based on current equity
                # trade_pct comes from user input (e.g. 0.05 for 5%)
                trade_size = equity_curve[-1] * trade_pct
                
                sl_price = price * 0.98 if position == 1 else price * 1.02
                tp_price = 0.0
                
                if tp_mode == 'ai':
                    pred_ret = ai_forecasts.get(ticker, {'pred':0})['pred']
                    target_pct = max(abs(pred_ret), 0.01)
                    if position == 1: tp_price = price * (1 + target_pct)
                    else: tp_price = price * (1 - target_pct)
                else:
                    h_val = compute_hurst(current_prices_window[ticker].values)
                    reward_mult = 0.04
                    if h_val > 0.6: reward_mult = 0.06
                    elif h_val < 0.4: reward_mult = 0.03
                    if position == 1: tp_price = price * (1 + reward_mult)
                    else: tp_price = price * (1 - reward_mult)
                
                exit_price = next_close
                if position == 1:
                    if next_close <= sl_price: exit_price = sl_price
                    elif next_close >= tp_price: exit_price = tp_price
                    trade_pnl = trade_size * ((exit_price - price) / price)
                else:
                    if next_close >= sl_price: exit_price = sl_price
                    elif next_close <= tp_price: exit_price = tp_price
                    trade_pnl = trade_size * ((price - exit_price) / price)

                daily_pnl += trade_pnl
                
                # Only log trades that actually happened (moved PnL)
                if abs(trade_pnl) > 0:
                    trades_log.append({
                        "date": str(date_label),
                        "ticker": ticker,
                        "side": "LONG" if position == 1 else "SHORT",
                        "entry": round(price, 4),
                        "tp": round(tp_price, 4),
                        "sl": round(sl_price, 4),
                        "pnl": round(trade_pnl, 2)
                    })

        new_equity = equity_curve[-1] + daily_pnl
        equity_curve.append(new_equity)

    total_return = ((equity_curve[-1] - start_capital) / start_capital) * 100
    win_trades = len([t for t in trades_log if t['pnl'] > 0])
    total_trades = len(trades_log)
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        "equity_curve": equity_curve,
        "labels": list(range(len(equity_curve))),
        "final_equity": equity_curve[-1],
        "total_return": total_return,
        "win_rate": win_rate,
        "trades": trades_log[::-1]
    }

def get_account_summary():
    balance_info = {"USDT": 0.0}
    orders_info = []
    try:
        bal = exchange_private.fetch_balance()
        if 'USDT' in bal: balance_info['USDT'] = bal['USDT']['free']
        try:
            orders = exchange_private.fetch_open_orders(params={'category': 'linear'})
            for o in orders:
                orders_info.append({ 'symbol': o['symbol'], 'side': o['side'], 'price': o['price'], 'amount': o['amount'] })
        except: pass
    except Exception as e: return {"error": str(e)}
    return {"balance": balance_info, "orders": orders_info}

# --- ROUTES ---
@app.route('/')
def index(): return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    selected_coins = data.get('coins', DEFAULT_COINS)
    df = fetch_crypto_data(selected_coins)
    if df.empty or len(df.columns) < 2: return jsonify({"error": "Data fetch failed."}), 400
    return jsonify(compute_quant_metrics(df))

@app.route('/surface', methods=['GET'])
def surface():
    global GLOBAL_PRICE_CACHE
    if GLOBAL_PRICE_CACHE.empty: return jsonify({"error": "No data."}), 400
    data = compute_rolling_surface(GLOBAL_PRICE_CACHE)
    if not data: return jsonify({"error": "Not enough history."}), 400
    return jsonify(data)

@app.route('/backtest', methods=['POST'])
def backtest():
    global GLOBAL_PRICE_CACHE
    if GLOBAL_PRICE_CACHE.empty: return jsonify({"error": "No data available. Run Engine First."}), 400
    data = request.json
    sim_days = int(data.get('days', 30))
    equity = float(data.get('capital', 1000.0))
    mode = data.get('mode', 'hurst')
    
    # NEW: Get trade size pct, default to 5% if missing
    trade_pct = float(data.get('size', 5.0)) / 100.0
    
    results = run_backtest_logic(GLOBAL_PRICE_CACHE, simulation_days=sim_days, start_capital=equity, tp_mode=mode, trade_pct=trade_pct)
    if "error" in results: return jsonify(results), 400
    return jsonify(results)

@app.route('/account', methods=['GET'])
def account(): return jsonify(get_account_summary())

@app.route('/history', methods=['POST'])
def history():
    global GLOBAL_PRICE_CACHE
    data = request.json
    ticker = data.get('ticker')
    overlay_btc = data.get('overlay_btc', False)
    if GLOBAL_PRICE_CACHE.empty or ticker not in GLOBAL_PRICE_CACHE.columns: return jsonify({"error": "No data."}), 400
    try:
        prices = GLOBAL_PRICE_CACHE[ticker].values.tolist()
        btc_prices = []
        if overlay_btc:
            btc_col = next((c for c in GLOBAL_PRICE_CACHE.columns if 'BTC' in c), None)
            if btc_col: btc_prices = GLOBAL_PRICE_CACHE[btc_col].values.tolist()
        labels = GLOBAL_PRICE_CACHE.index.tolist()
        return jsonify({ "ticker": ticker, "prices": prices, "btc_prices": btc_prices, "labels": labels })
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting CryptoQuantsList (Custom Trade Size)...")
    app.run(debug=True)