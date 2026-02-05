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


def detect_sr(gray):
    h, _ = gray.shape
    zone = gray[int(h * 0.45):int(h * 0.75), :]
    proj = np.sum(zone, axis=1)
    m = proj.mean()

    return {
        "support": np.sum(proj < m * 0.92) > 8,
        "resistance": np.sum(proj > m * 1.08) > 8
    }


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
    Extracts price path using edge density.
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

def detect_trend_from_price_path(path):
    if len(path) < 30:
        return "RANGE"

    segments = np.array_split(path, 4)

    highs = [np.min(seg) for seg in segments]  # higher price = smaller Y
    lows  = [np.max(seg) for seg in segments]  # lower price = larger Y

    higher_highs = highs[0] > highs[1] > highs[2] > highs[3]
    higher_lows  = lows[0]  > lows[1]  > lows[2]  > lows[3]

    lower_highs = highs[0] < highs[1] < highs[2] < highs[3]
    lower_lows  = lows[0]  < lows[1]  < lows[2]  < lows[3]

    if higher_highs and higher_lows:
        return "UPTREND"

    if lower_highs and lower_lows:
        return "DOWNTREND"

    return "RANGE"

def detect_phase_from_path(path):
    if len(path) < 30:
        return "RANGE"

    first = np.mean(path[:len(path)//3])
    middle = np.mean(path[len(path)//3:2*len(path)//3])
    last = np.mean(path[-len(path)//3:])

    slope1 = middle - first
    slope2 = last - middle

    if abs(slope2) > abs(slope1) * 0.8:
        return "CONTINUATION"

    return "PULLBACK"

def detect_structure_from_path(path):
    if len(path) < 30:
        return "RANGE"

    segments = np.array_split(path, 4)

    highs = [np.min(seg) for seg in segments]
    lows  = [np.max(seg) for seg in segments]

    if highs[0] < highs[1] < highs[2] < highs[3] and \
       lows[0]  < lows[1]  < lows[2]  < lows[3]:
        return "BEARISH"

    if highs[0] > highs[1] > highs[2] > highs[3] and \
       lows[0]  > lows[1]  > lows[2]  > lows[3]:
        return "BULLISH"

    return "RANGE"


def detect_bias_from_path(path):
    if len(path) < 30:
        return "NEUTRAL"

    left = np.mean(path[:len(path)//2])
    right = np.mean(path[len(path)//2:])

    # Screen coordinates: higher Y = lower price
    if right > left:
        return "BEARISH"

    if right < left:
        return "BULLISH"

    return "NEUTRAL"


def detect_pullback_state(path):
    if len(path) < 40:
        return None

    recent = np.mean(path[-10:])
    prior = np.mean(path[-25:-10])

    if abs(recent - prior) < 3:
        return "SLOWING"

    return "TURNING"

def detect_overextension(path):
    if len(path) < 30:
        return None

    mean = np.mean(path)
    recent = np.mean(path[-10:])

    deviation = (recent - mean) / (np.max(path) - np.min(path))

    if deviation > 0.18:
        return "OVERBOUGHT"

    if deviation < -0.18:
        return "OVERSOLD"

    return "NORMAL"

def gatekeeper(structure, trend, sr, candle):
    return 0, "OK"

# =============================
# STRUCTURE–PHASE DECISION ENGINE
# =============================
def evaluate_pairs(structure, sr, candle, trend, market_phase, pullback_state, bias):

    penalty, gate_note = gatekeeper(structure, trend, sr, candle)
    fired = []
    momentum_bonus = 0

    # ---- CATEGORY A (TREND & PULLBACK) ----
    if market_phase == "CONTINUATION":

        # Bullish continuation
        if structure == "BULLISH" and (
            candle == "IMPULSE" or (candle == "NEUTRAL" and color == "GREEN")
        ):
            fired.append(("BUY", 85, "Bullish continuation"))

        # Bearish continuation
        if structure == "BEARISH" and (
            candle == "IMPULSE" or (candle == "NEUTRAL" and color == "RED")
        ):
            fired.append(("SELL", 85, "Bearish continuation"))

    # === PULLBACK TRADES ===
    elif market_phase == "PULLBACK" and pullback_state == "TURNING":

        # Bullish pullback continuation
        if (
            structure == "BULLISH"
            and candle in ["REJECTION", "NEUTRAL"]
            and color == "GREEN"
        ):
        fired.append(("BUY", 80, "Bullish pullback continuation"))

        # Bearish pullback continuation
        if (
            structure == "BEARISH"
            and candle in ["REJECTION", "NEUTRAL"]
            and color == "RED"
        ):
            fired.append(("SELL", 80, "Bearish pullback continuation"))
    
    # ---- CATEGORY B (SR) ----
    if market_phase == "CONTINUATION":

        if trend == "UPTREND" and sr["support"] and candle == "REJECTION":
            fired.append(("BUY", 86, "Support hold in uptrend"))

        if trend == "DOWNTREND" and sr["resistance"] and candle == "REJECTION":
            fired.append(("SELL", 86, "Resistance hold in downtrend"))

    elif market_phase == "PULLBACK":

        if trend == "UPTREND" and sr["resistance"] and candle in ["REJECTION", "NEUTRAL"]:
            fired.append(("SELL", 72, "Pullback rejection at resistance"))

        if trend == "DOWNTREND" and sr["support"] and candle in ["REJECTION", "NEUTRAL"]:
            fired.append(("BUY", 72, "Pullback rejection at support"))

    # ---- CATEGORY C (RANGE) ----
    if market_phase == "RANGE":

        if sr["support"] and candle in ["NEUTRAL", "REJECTION"]:
            fired.append(("BUY", 75, "Range mean reversion (support)"))

        if sr["resistance"] and candle in ["NEUTRAL", "REJECTION"]:
            fired.append(("SELL", 75, "Range mean reversion (resistance)"))

    # ---- CATEGORY D (MOMENTUM) ----
    if market_phase == "CONTINUATION" and candle == "IMPULSE":
        momentum_bonus += 6

    if market_phase == "PULLBACK" and candle == "REJECTION":
        momentum_bonus += 3

    buys = [r for r in fired if r[0] == "BUY"]
    sells = [r for r in fired if r[0] == "SELL"]

    buy_score = sum(r[1] for r in buys)
    sell_score = sum(r[1] for r in sells)

    if buy_score == sell_score or not fired:
        return "WAIT", "No dominant side", 0

    dominant = buys if buy_score > sell_score else sells
    dominant.sort(key=lambda x: x[1], reverse=True)

    top = dominant[0]
    confidence = min(95, max(0, top[1] + momentum_bonus - penalty))

    if confidence < 65:
        return "WAIT", "Weak setup", confidence

    return top[0], top[2], confidence

# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sr = detect_sr(gray)
    candle = candle_strength(gray)
    color = candle_color(image)

    path = extract_price_path(gray)
    bias = detect_bias_from_path(path)
    
    trend = detect_trend_from_price_path(path)
    phase = detect_phase_from_path(path)
    pullback_state = detect_pullback_state(path)
    
    if trend == "UPTREND":
        structure = "BULLISH"
    elif trend == "DOWNTREND":
        structure = "BEARISH"
    else:
        structure = "RANGE"

    signal, reason, conf = evaluate_pairs(
        structure, sr, candle, trend, phase, pullback_state, bias
    )

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
TREND: {trend}
PHASE: {phase}
PULLBACK STATE: {pullback_state}
BIAS: {bias}
CANDLE: {candle}
COLOR: {color}
""")


















