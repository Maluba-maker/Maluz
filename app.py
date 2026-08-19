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
        st.image(image, width="stretch")

if mode == "Camera":
    cam = st.camera_input("Capture chart")

    if cam:
        image = np.array(Image.open(cam))
        st.image(image, use_column_width=True)

# =============================
# IMAGE HELPERS
# =============================

def detect_breakout(path):

    recent = path[-15:]
    prev = path[-50:-15]

    prev_high = np.max(prev)
    prev_low = np.min(prev)

    current = recent[-1]

    range_size = prev_high - prev_low

    # 🔥 MUCH SMALLER BUFFER
    buffer = range_size * 0.05

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

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    return thresh

def extract_price_path_v2(binary):

    h, w = binary.shape

    path = []

    for x in range(0, w, 3):

        column = binary[:, x]
        ys = np.where(column > 0)[0]

        if len(ys) > 10:
            mid = int(np.mean(ys))
            path.append(mid)

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
    lows = [np.max(seg) for seg in segments]

    if all(highs[i] > highs[i + 1] for i in range(3)) and \
       all(lows[i] > lows[i + 1] for i in range(3)):
        return "UPTREND"

    if all(highs[i] < highs[i + 1] for i in range(3)) and \
       all(lows[i] < lows[i + 1] for i in range(3)):
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

    binary = preprocess_chart(image)

    path = extract_price_path_v2(binary)
    path = smooth_path(path)

    if path is None:
        st.warning("Chart not clear enough")
        st.stop()

    structure = detect_structure(path)
    consolidating = is_consolidating(path)
    momentum = momentum_strength(path)
    breakout = detect_breakout(path)

    signal, reason = generate_signal(
        structure,
        consolidating,
        momentum,
        breakout
    )

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
