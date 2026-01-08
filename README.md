

![image1](https://media.discordapp.net/attachments/1216717532577140941/1458816133967777823/image.png?ex=696103e5&is=695fb265&hm=928e560d68fbea39a486b3779e07dd9bb89521e01105aeb54d2666ef172f9c7b&=&format=webp&quality=lossless&width=1699&height=864)
![image2](https://media.discordapp.net/attachments/1216717532577140941/1458815755905798207/image.png?ex=6961038b&is=695fb20b&hm=f1577a6062fd001d70898ca42feae86e32f6d77b162c2e94fc2e675f01e26569&=&format=webp&quality=lossless&width=1710&height=864)
![image3](https://cdn.discordapp.com/attachments/1216717532577140941/1458816481965113489/image.png?ex=69610438&is=695fb2b8&hm=598e1dadb842c7b51958038d4e1d7a9fc4c88493bbb1412d928cc730a8ce0d28)



# CryptoQuantsList

A simple python vibe coding project that i do, based on Laplacian Equation to determine the Average Weight of the Cryptocoins, this engine can determine when to Long/Short
therese a AI Bias based on GNN model, to add an conviction bias for your Trading Bias,this engine also have a Hurst Mathematical Equation to calculate the current/choosen coins is undwervalue/overvalue based on the Weighted average prices of the Cryptocoins

using YahooFinance API to extract all of availables Layer-1 Cryptocurrencies asset

## Current Version
the Current Stable Version, still has the Mesh Graph bugged.
![image](https://cdn.discordapp.com/attachments/1216717532577140941/1457070858794303622/image.png?ex=6960993b&is=695f47bb&hm=f91cfbf434b7f87d19704b48ddbd79caa3bd99d705e33f1c667968f7374dab60)
*how the Mesh supposed to works, current mesh is broken.
```
Current Version Update
-Adding backtesting calculator to check the engine performance
-Adding Hurst Model to calculate the trading Position(Entry,SL/TP) to minimize drawdown
-The Demo Equity(BYBIT API USAGE) hasnt been tested, expect errors
```
## How To run
Run this Command :
```
pip install -r requirements.txt
python quant.py
```
Go to your browser and paste 
```
http://127.0.0.1:5000/
```
## The ideas behind it.
special thanks to @sd_jeong on IG, as an inspiration for this project
