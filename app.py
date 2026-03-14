import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd
import ta

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Maluz Signal Engine", layout="centered")

st.markdown("## 🔹 Maluz Signal Engine")
st.caption("Screenshot-based • STRICT Malagna Mirror")

# =============================
# PASSWORD
# =============================
PASSWORD = "maluz123"
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

def check_password():
    def entered():
        if hashlib.sha256(st.session_state["pw"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state.auth = True
        else:
            st.session_state.auth = False

    if "auth" not in st.session_state or not st.session_state.auth:
        st.text_input("🔐 Password", type="password", key="pw", on_change=entered)
        if "auth" in st.session_state and not st.session_state.auth:
            st.error("Incorrect password")
        st.stop()

check_password()

# =============================
# INPUT
# =============================
mode = st.radio("Input Mode", ["Upload Screenshot", "Camera"])
image = None

if mode == "Upload Screenshot":
    f = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg"])
    if f:
        image = np.array(Image.open(f))
        st.image(image, use_column_width=True)

if mode == "Camera":
    cam = st.camera_input("Capture chart")
    if cam:
        image = np.array(Image.open(cam))
        st.image(image, use_column_width=True)

# =============================
# IMAGE HELPERS
# =============================

def candle_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    roi = hsv[int(h * 0.55):int(h * 0.75), int(w * 0.7):]

    green = np.sum((roi[:, :, 0] > 35) & (roi[:, :, 0] < 85))
    red = np.sum((roi[:, :, 0] < 10) | (roi[:, :, 0] > 160))

    if green > red * 1.2:
        return "GREEN"
    if red > green * 1.2:
        return "RED"
    return "MIXED"


def candle_strength(gray):
    h, w = gray.shape
    roi = gray[int(h * 0.55):int(h * 0.75), int(w * 0.7):]
    v = np.std(roi)

    if v > 35:
        return "IMPULSE"
    if v < 18:
        return "REJECTION"
    return "NEUTRAL"


def extract_price_path(gray):
    """
    Extracts a visual price path using edge density.
    Works on light & dark chart themes.
    """
    h, w = gray.shape
    path = []

    edges = cv2.Canny(gray, 50, 150)

    for x in range(0, w, 4):
        column_edges = edges[:, x]
        ys = np.where(column_edges > 0)[0]

        if len(ys) > 10:
            path.append(np.mean(ys))

    return np.array(path)

def detect_structure_from_path(path):
    """
    Structure authority:
    - Higher lows → BULLISH
    - Lower highs → BEARISH
    """
    if len(path) < 30:
        return "RANGE"

    segments = np.array_split(path, 4)

    highs = [np.min(seg) for seg in segments]  # higher price = smaller Y
    lows  = [np.max(seg) for seg in segments]  # lower price = larger Y

    if highs[0] > highs[1] > highs[2] > highs[3] and \
       lows[0]  > lows[1]  > lows[2]  > lows[3]:
        return "BULLISH"

    if highs[0] < highs[1] < highs[2] < highs[3] and \
       lows[0]  < lows[1]  < lows[2]  < lows[3]:
        return "BEARISH"

    return "RANGE"
def detect_bias_from_path(path):
    """
    Fallback directional bias.
    """
    if len(path) < 30:
        return "NEUTRAL"

    left = np.mean(path[:len(path)//2])
    right = np.mean(path[len(path)//2:])

    # Screen coordinates: higher Y = lower price
    if right < left:
        return "BULLISH"
    if right > left:
        return "BEARISH"

    return "NEUTRAL"

def movement_strength(path):

    if len(path) < 40:
        return "WEAK"

    recent = np.mean(path[-10:])
    mid = np.mean(path[-25:-15])
    older = np.mean(path[-40:-30])

    move1 = abs(recent - mid)
    move2 = abs(mid - older)

    total_move = move1 + move2

    if total_move > 4:
        return "STRONG"

    if total_move > 2:
        return "MODERATE"

    return "WEAK"

def path_to_dataframe(path):

    if len(path) < 50:
        return None

    prices = -path  # invert screen coordinates

    data = []

    for i in range(1, len(prices)-1):

        open_p = prices[i-1]
        close_p = prices[i]

        high_p = max(open_p, close_p) + abs(prices[i]-prices[i-1])*0.2
        low_p  = min(open_p, close_p) - abs(prices[i]-prices[i-1])*0.2

        data.append({
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close_p
        })

    df = pd.DataFrame(data)

    return df

def indicators(df):

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    return {
        "close": close,
        "ema20": ta.trend.ema_indicator(close, 20),
        "ema50": ta.trend.ema_indicator(close, 50),
        "rsi": ta.momentum.rsi(close, 14),
        "macd": ta.trend.macd_diff(close),
        "atr": ta.volatility.average_true_range(high, low, close, 14),
        "adx": ta.trend.adx(high, low, close, 14)
    }

# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    candle = candle_strength(gray)
    color = candle_color(image)

    path = extract_price_path(gray)
    strength = movement_strength(path)
    df = path_to_dataframe(path)
    
    if df is None:
        st.warning("Chart not clear enough")
        st.stop()
    
    i = indicators(df)
    
    ema20 = i["ema20"].iloc[-1]
    ema50 = i["ema50"].iloc[-1]
    price = i["close"].iloc[-1]
    
    adx = i["adx"].iloc[-1]
    
    signal = "WAIT"
    reason = "No structure"
    conf = 0
    
    if ema20 > ema50 and price > ema20 and adx > 20:
        signal = "BUY"
        reason = "Bullish trend structure"
        conf = 80
    
    elif ema20 < ema50 and price < ema20 and adx > 20:
        signal = "SELL"
        reason = "Bearish trend structure"
        conf = 80
    
    # Adjust confidence based on strength
    if strength == "STRONG":
        conf += 5
    elif strength == "WEAK":
        conf -= 10

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=5)
    expiry = entry + timedelta(minutes=5)

    if signal == "BUY":
        st.success(f"🟢 BUY ({conf}%)")
    elif signal == "SELL":
        st.error(f"🔴 SELL ({conf}%)")
    else:
        st.info("⚪ WAIT")

st.code(f"""
SIGNAL: {signal}
CONFIDENCE: {conf}%
REASON: {reason}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
STRENGTH: {strength}
CANDLE: {candle}
COLOR: {color}
""")




