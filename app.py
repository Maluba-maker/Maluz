import streamlit as st
import yfinance as yf
import ta
import pandas as pd
import time

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Malagna", layout="wide")

# ================= PASSWORD =================
APP_PASSWORD = "malagna2026"

def check_password():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if not st.session_state.auth:
        st.markdown("<h2 style='text-align:center'>🔐 Secure Access</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password")
        if pwd == APP_PASSWORD:
            st.session_state.auth = True
            st.rerun()
        elif pwd:
            st.error("Incorrect password")
        st.stop()

check_password()

# ================= STYLES =================
st.markdown("""
<style>
body { background:#0b0f14; color:white; }
.block { background:#121722; padding:22px; border-radius:16px; margin-bottom:18px; }
.center { text-align:center; }
.signal-buy { color:#22c55e; font-size:60px; font-weight:800; }
.signal-sell { color:#ef4444; font-size:60px; font-weight:800; }
.signal-wait { color:#9ca3af; font-size:48px; font-weight:700; }
.metric { color:#9ca3af; margin-top:6px; }
.small { font-size:13px; color:#9ca3af; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="block">
<h1>Malagna</h1>
<div class="metric">20-Rule Dominant Engine • All Markets • True M5</div>
</div>
""", unsafe_allow_html=True)

# ================= MARKETS =================
CURRENCIES = {
    "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","AUD/USD":"AUDUSD=X","NZD/USD":"NZDUSD=X",
    "USD/JPY":"JPY=X","USD/CHF":"CHF=X","USD/CAD":"CAD=X",
    "EUR/GBP":"EURGBP=X","EUR/JPY":"EURJPY=X","GBP/JPY":"GBPJPY=X",
    "EUR/AUD":"EURAUD=X","EUR/CAD":"EURCAD=X","AUD/JPY":"AUDJPY=X",
    "GBP/CAD":"GBPCAD=X","CHF/JPY":"CHFJPY=X","NZD/JPY":"NZDJPY=X"
}

CRYPTO = {
    "BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","BNB/USD":"BNB-USD",
    "SOL/USD":"SOL-USD","XRP/USD":"XRP-USD","ADA/USD":"ADA-USD",
    "DOGE/USD":"DOGE-USD","AVAX/USD":"AVAX-USD","DOT/USD":"DOT-USD",
    "LINK/USD":"LINK-USD","MATIC/USD":"MATIC-USD"
}

COMMODITIES = {
    "Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F",
    "Brent Oil":"BZ=F","Natural Gas":"NG=F",
    "Copper":"HG=F","Corn":"ZC=F","Wheat":"ZW=F"
}

market = st.radio("Market", ["Currencies","Crypto","Commodities","Stocks"], horizontal=True)

if market == "Currencies":
    asset = st.selectbox("Pair", list(CURRENCIES.keys()))
    symbol = CURRENCIES[asset]

elif market == "Crypto":
    asset = st.selectbox("Crypto", list(CRYPTO.keys()))
    symbol = CRYPTO[asset]

elif market == "Commodities":
    asset = st.selectbox("Commodity", list(COMMODITIES.keys()))
    symbol = COMMODITIES[asset]

else:
    asset = st.text_input("Stock ticker (e.g. AAPL, TSLA, MSFT)").upper()
    symbol = asset

# ================= DATA =================
@st.cache_data(ttl=60)
def fetch(symbol, interval, period):
    return yf.download(symbol, interval=interval, period=period, progress=False)

data_5m  = fetch(symbol, "5m", "5d")
data_15m = fetch(symbol, "15m", "10d")

def indicators(df):
    if df.empty or "Close" not in df:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]
    close = close.astype(float)
    return {
        "close": close,
        "ema50": ta.trend.ema_indicator(close, 50),
        "ema200": ta.trend.ema_indicator(close, 200),
        "rsi": ta.momentum.rsi(close, 14),
        "macd": ta.trend.macd_diff(close)
    }

i5  = indicators(data_5m)
i15 = indicators(data_15m)

# ================= SUPPORT / RESISTANCE (SIMPLE & SAFE) =================
sr = {
    "support": False,
    "resistance": False
}

if i5:
    recent_low  = i5["close"].rolling(20).min().iloc[-1]
    recent_high = i5["close"].rolling(20).max().iloc[-1]
    price = i5["close"].iloc[-1]

    if abs(price - recent_low) / price < 0.002:
        sr["support"] = True
    if abs(price - recent_high) / price < 0.002:
        sr["resistance"] = True

# ================= CANDLE TYPE (M5) =================
def candle_type(df):
    o = float(df["Open"].iloc[-1])
    c = float(df["Close"].iloc[-1])
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])

    body = abs(c - o)
    full = h - l

    if full == 0:
        return "NEUTRAL"

    ratio = body / full
    if ratio >= 0.6:
        return "IMPULSE"
    elif ratio <= 0.3:
        return "NEUTRAL"
    else:
        return "REJECTION"

candle = candle_type(data_5m)

# ================= STRUCTURE & TREND =================
structure = "RANGE"
trend = "FLAT"

if i15:
    if i15["ema50"].iloc[-1] > i15["ema200"].iloc[-1]:
        structure = "BULLISH"
        trend = "UPTREND"
    elif i15["ema50"].iloc[-1] < i15["ema200"].iloc[-1]:
        structure = "BEARISH"
        trend = "DOWNTREND"

# ================= 20-RULE ENGINE =================
def evaluate_pairs(structure, sr, candle, trend):
    fired = []

    # ---- CATEGORY A (TREND) ----
    if structure == "BULLISH" and candle == "IMPULSE":
        fired.append(("BUY", 88, "Pair 1: Bullish trend acceleration"))
    if structure == "BULLISH" and trend == "UPTREND" and candle == "REJECTION":
        fired.append(("BUY", 85, "Pair 2: Pullback in uptrend"))
    if structure == "BULLISH" and trend == "UPTREND" and candle == "IMPULSE":
        fired.append(("BUY", 90, "Pair 3: Breakout continuation"))
    if structure == "BEARISH" and candle == "IMPULSE":
        fired.append(("SELL", 88, "Pair 4: Bearish trend acceleration"))
    if structure == "BEARISH" and trend == "DOWNTREND" and candle == "REJECTION":
        fired.append(("SELL", 85, "Pair 5: Pullback in downtrend"))

    # ---- CATEGORY B (SR) ----
    if sr["support"] and candle == "REJECTION":
        fired.append(("BUY", 87, "Pair 6: Support rejection"))
    if sr["resistance"] and candle == "REJECTION":
        fired.append(("SELL", 87, "Pair 7: Resistance rejection"))
    if sr["support"] and candle == "NEUTRAL" and structure == "BEARISH":
        fired.append(("BUY", 90, "Pair 8: Sell exhaustion"))
    if sr["resistance"] and candle == "NEUTRAL" and structure == "BULLISH":
        fired.append(("SELL", 90, "Pair 9: Buy exhaustion"))
    if sr["support"] and candle == "IMPULSE":
        fired.append(("BUY", 84, "Pair 10: Support impulse"))

    # ---- CATEGORY C (MEAN REVERSION) ----
    if sr["support"] and candle == "NEUTRAL" and trend == "DOWNTREND":
        fired.append(("BUY", 86, "Pair 11: Mean reversion low"))
    if sr["resistance"] and candle == "NEUTRAL" and trend == "UPTREND":
        fired.append(("SELL", 86, "Pair 12: Mean reversion high"))
    if sr["support"] and candle == "REJECTION" and structure == "RANGE":
        fired.append(("BUY", 88, "Pair 13: Oversold snapback"))
    if sr["resistance"] and candle == "REJECTION" and structure == "RANGE":
        fired.append(("SELL", 88, "Pair 14: Overbought snapback"))
    if candle == "IMPULSE" and structure == "RANGE":
        fired.append(("BUY", 83, "Pair 15: Volatility release"))

    # ---- CATEGORY D (MOMENTUM) ----
    if candle == "IMPULSE" and structure == "BULLISH" and trend == "UPTREND":
        fired.append(("BUY", 84, "Pair 16: Momentum alignment up"))
    if candle == "IMPULSE" and structure == "BEARISH" and trend == "DOWNTREND":
        fired.append(("SELL", 84, "Pair 17: Momentum alignment down"))
    if sr["support"] and structure == "BULLISH" and candle == "NEUTRAL":
        fired.append(("BUY", 89, "Pair 18: Hidden accumulation"))
    if sr["resistance"] and structure == "BEARISH" and candle == "NEUTRAL":
        fired.append(("SELL", 89, "Pair 19: Distribution"))
    if candle == "REJECTION" and trend in ["UPTREND","DOWNTREND"]:
        fired.append(("BUY" if trend=="UPTREND" else "SELL", 83, "Pair 20: Second-leg entry"))

    if not fired:
        return "WAIT", "No valid rule alignment", 0

    fired.sort(key=lambda x: x[1], reverse=True)
    top = fired[0]
    agree = [f for f in fired if f[0]==top[0]]
    oppose = [f for f in fired if f[0]!=top[0]]

    confidence = top[1] + (len(agree)-1)*3
    if oppose:
        confidence -= min(12, abs(top[1]-oppose[0][1]))

    confidence = max(50, min(99, confidence))

    return top[0], top[2], confidence

signal, reason, confidence = evaluate_pairs(structure, sr, candle, trend)

# ================= DISPLAY =================
signal_class = {
    "BUY":"signal-buy",
    "SELL":"signal-sell",
    "WAIT":"signal-wait"
}[signal]

st.markdown(f"""
<div class="block center">
<div class="{signal_class}">{signal}</div>
<div class="metric">{asset}</div>
<div class="metric">Confidence: {confidence}%</div>
<div class="small">{reason}</div>
<div class="small">Structure: {structure} • Trend: {trend} • Candle: {candle}</div>
</div>
""", unsafe_allow_html=True)

time.sleep(1)
st.rerun()
