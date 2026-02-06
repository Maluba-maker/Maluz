import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

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

# =============================
# STRUCTURE–PHASE DECISION ENGINE
# =============================

def classify_market_state(structure, path):
    """
    Returns one of:
    UP_CONTINUATION
    DOWN_CONTINUATION
    UP_PULLBACK
    DOWN_PULLBACK
    """

    if len(path) < 30:
        return "NO_TRADE"

    recent = np.mean(path[-10:])
    prior  = np.mean(path[-30:-20])

    # Screen coordinates:
    # higher Y = lower price
    price_moving_down = recent > prior
    price_moving_up   = recent < prior

    if structure == "BULLISH":
        if price_moving_up:
            return "UP_CONTINUATION"
        else:
            return "UP_PULLBACK"

    if structure == "BEARISH":
        if price_moving_down:
            return "DOWN_CONTINUATION"
        else:
            return "DOWN_PULLBACK"

    return "NO_TRADE"


def evaluate_pairs(market_state):

    if market_state == "UP_CONTINUATION":
        return "BUY", "Uptrend continuation", 80

    if market_state == "DOWN_CONTINUATION":
        return "SELL", "Downtrend continuation", 80

    if market_state == "UP_PULLBACK":
        return "SELL", "Pullback in uptrend", 70

    if market_state == "DOWN_PULLBACK":
        return "BUY", "Pullback in downtrend", 70

    return "WAIT", "No clear structure", 0


# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    candle = candle_strength(gray)
    color = candle_color(image)

    path = extract_price_path(gray)
    structure = detect_structure_from_path(path)

    market_state = classify_market_state(structure, path)
    signal, reason, conf = evaluate_pairs(market_state)

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
STRUCTURE: {structure}
MARKET STATE: {market_state}
CANDLE: {candle}
COLOR: {color}
""")
