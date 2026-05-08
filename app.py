import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd

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

def detect_breakout(path):

    if len(path) < 50:
        return "NONE"

    recent = path[-15:]
    prev = path[-50:-15]

    prev_high = np.min(prev)
    prev_low = np.max(prev)

    current = recent[-1]

    range_size = abs(prev_low - prev_high)

    buffer = range_size * 0.05

    # IMPORTANT:
    # smaller Y = higher price on charts

    if current < prev_high - buffer:
        return "UP_BREAK"

    if current > prev_low + buffer:
        return "DOWN_BREAK"

    return "NONE"

def quality_check(structure, momentum, breakout):

    if structure == "RANGE":
        return False, "Market has no structure"

    # ONLY block weak momentum (not moderate)
    if momentum == "WEAK":
        return False, "Weak momentum"

    if breakout == "NONE":
        return False, "No breakout"

    return True, "Valid setup"


def preprocess_chart(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ===== GREEN CANDLES =====
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    # ===== RED CANDLES =====
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = red_mask1 + red_mask2

    # ===== COMBINE =====
    mask = green_mask + red_mask
    kernel = np.ones((2,2), np.uint8)

    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # ===== CLEAN NOISE =====
    kernel = np.ones((2,2), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def extract_candles(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candles = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # REMOVE TINY NOISE
        if area < 15:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = h / max(w, 1)

        if aspect_ratio < 1.5:
            continue
        # FILTER BAD SHAPES
        if h < 8:
            continue
        
        if w > 25:
            continue

        candle = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "top": y,
            "bottom": y + h,
            "center": y + h//2
        }

        candles.append(candle)

    # SORT LEFT → RIGHT
    candles = sorted(candles, key=lambda c: c["x"])

    return candles

def candles_to_path(candles):

    if len(candles) < 20:
        return None

    path = [c["top"] for c in candles]

    return np.array(path)

def smooth_path(path):
    if len(path) < 20:
        return None

    return pd.Series(path).rolling(5).mean().dropna().values

def detect_structure(path):
    if len(path) < 40:
        return "RANGE"

    segments = np.array_split(path, 4)

    highs = [np.min(seg) for seg in segments]
    lows  = [np.max(seg) for seg in segments]

    if all(highs[i] > highs[i+1] for i in range(3)) and \
       all(lows[i] > lows[i+1] for i in range(3)):
        return "UPTREND"

    if all(highs[i] < highs[i+1] for i in range(3)) and \
       all(lows[i] < lows[i+1] for i in range(3)):
        return "DOWNTREND"

    return "RANGE"

def is_consolidating(path):
    recent = path[-30:]

    volatility = np.std(recent)
    avg_move = np.mean(np.abs(np.diff(recent)))

    return volatility < (avg_move * 1.2)

def momentum_strength(path):

    y = path[-30:]
    x = np.arange(len(y))

    slope = abs(np.polyfit(x, y, 1)[0])

    if slope > 0.5:
        return "STRONG"
    elif slope > 0.2:
        return "MODERATE"
    else:
        return "WEAK"

def generate_signal(structure, consolidating, momentum, breakout):

    if consolidating:
        return "WAIT", "Market is consolidating"

    # 🔥 PRIMARY: Breakout trades
    if breakout == "UP_BREAK" and momentum != "WEAK":
        return "BUY", "Breakout trade"

    if breakout == "DOWN_BREAK" and momentum != "WEAK":
        return "SELL", "Breakout trade"

    # 🔥 FALLBACK: Trend continuation
    if structure == "UPTREND" and momentum == "STRONG":
        return "BUY", "Trend continuation"

    if structure == "DOWNTREND" and momentum == "STRONG":
        return "SELL", "Trend continuation"

    return "WAIT", "No clear edge"

# =============================
# EXECUTION
# =============================

if image is not None and st.button("🔍 Analyse Market"):

    # ===== CROP MAIN CHART =====

    h, w, _ = image.shape
    
    image = image[
        int(h*0.12):int(h*0.78),
        int(w*0.02):int(w*0.98)
    ]
    mask = preprocess_chart(image)

    candles = extract_candles(mask)
    
    debug = image.copy()

    for c in candles:
        cv2.rectangle(
            debug,
            (c["x"], c["top"]),
            (c["x"] + c["w"], c["bottom"]),
            (0,255,0),
            1
        )
    
    st.image(debug, caption="Detected Candles")
    st.write(f"Detected Candles: {len(candles)}")
    
    path = candles_to_path(candles)

    if path is None:
        st.warning("Not enough candles detected")
        st.stop()
    
    path = smooth_path(path)
    
    if path is None:
        st.warning("Path smoothing failed")
        st.stop()

    structure = detect_structure(path)
    consolidating = is_consolidating(path)
    momentum = momentum_strength(path)
    breakout = detect_breakout(path)

    signal, reason = generate_signal(structure, consolidating, momentum, breakout)

    if signal == "BUY":
        st.success("🟢 BUY")
    elif signal == "SELL":
        st.error("🔴 SELL")
    else:
        st.info("⚪ WAIT")
    
    # ✅ NOW IT SHOWS AFTER SIGNAL
    st.write({
        "structure": structure,
        "momentum": momentum,
        "breakout": breakout,
        "consolidating": consolidating
    })
    
    st.code(f"""
    SIGNAL: {signal}
    STRUCTURE: {structure}
    MOMENTUM: {momentum}
    CONSOLIDATION: {consolidating}
    BREAKOUT: {breakout}
    REASON: {reason}
    """)




