import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

# =============================
# PASSWORD PROTECTION
# =============================

PASSWORD = "maluz123"
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

def check_password():
    def password_entered():
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("🔐 Enter password", type="password",
                      key="password", on_change=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("🔐 Enter password", type="password",
                      key="password", on_change=password_entered)
        st.error("❌ Incorrect password")
        return False
    return True


if not check_password():
    st.stop()

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="Maluz", layout="centered")
st.title("📊 Maluz")
st.caption("OTC Screenshot-Based Market Analysis")

# =============================
# INPUT
# =============================

input_mode = st.radio("Select Input Mode",
                      ["Upload / Drag Screenshot", "Take Photo (Camera)"])

image = None

if input_mode == "Upload / Drag Screenshot":
    uploaded = st.file_uploader("Upload OTC chart screenshot",
                                type=["png", "jpg", "jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded))
        st.image(image, use_column_width=True)

if input_mode == "Take Photo (Camera)":
    camera_image = st.camera_input("Capture chart photo")
    if camera_image:
        image = np.array(Image.open(camera_image))
        st.image(image, use_column_width=True)

# =============================
# HELPER FUNCTIONS
# =============================

def extract_gray_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def region(gray, h, top, bottom):
    return gray[int(h * top):int(h * bottom), :]

def compute_edge_strength(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.mean(edges)

def detect_trend_from_red(hsv):
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask = (
        cv2.inRange(hsv, lower_red1, upper_red1) |
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    points = np.column_stack(np.where(mask > 0))
    if len(points) < 50:
        return "UNCLEAR"

    slope = np.polyfit(points[:, 1], points[:, 0], 1)[0]
    return "DOWN" if slope > 0 else "UP"

def detect_momentum_from_blue(hsv):
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([120, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    points = np.column_stack(np.where(mask > 0))

    if len(points) < 30:
        return "UNKNOWN"

    slope = np.polyfit(points[:, 1], points[:, 0], 1)[0]
    return "DOWN" if slope > 0 else "UP"

def detect_manipulation(gray, h):
    flags = []

    candle_zone = region(gray, h, 0.4, 0.65)
    recent_zone = region(gray, h, 0.35, 0.5)

    candle_energy = np.std(candle_zone)
    recent_energy = np.std(recent_zone)

    edge_strength = compute_edge_strength(gray)

    if edge_strength > 45:
        flags.append("Abnormal wick spikes")

    if abs(candle_energy - recent_energy) > 22:
        flags.append("Sudden volatility injection")

    if candle_energy < 15 and recent_energy > 30:
        flags.append("Artificial volatility release")

    return flags

# =============================
# CORE ANALYSIS
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Invalid image.")
        st.stop()

    gray, hsv = extract_gray_hsv(image)
    height, width = gray.shape

    # --- Market behaviour (isolated) ---
    manipulation_flags = detect_manipulation(gray, height)
    market_manipulated = len(manipulation_flags) > 0

    # --- Structure ---
    trend = detect_trend_from_red(hsv)

    # --- Momentum ---
    momentum = detect_momentum_from_blue(hsv)

    # =============================
    # DECISION LOGIC (UNCHANGED)
    # =============================

    if trend == "UP":
        signal = "BUY"
    elif trend == "DOWN":
        signal = "SELL"
    else:
        signal = "NO TRADE"

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    # =============================
    # OUTPUT
    # =============================

    if signal == "BUY":
        st.markdown(
            "<div style='background:#dcfce7;color:#166534;"
            "padding:14px;border-radius:8px;font-weight:700;'>"
            "🟢 BUY SIGNAL</div>",
            unsafe_allow_html=True
        )
    elif signal == "SELL":
        st.markdown(
            "<div style='background:#fee2e2;color:#991b1b;"
            "padding:14px;border-radius:8px;font-weight:700;'>"
            "🔴 SELL SIGNAL</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#e5e7eb;color:#374151;"
            "padding:14px;border-radius:8px;font-weight:700;'>"
            "⚪ NO TRADE</div>",
            unsafe_allow_html=True
        )

    st.code(f"""
SIGNAL: {signal}
TREND: {trend}
MOMENTUM: {momentum}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())

    # =============================
    # MARKET BEHAVIOUR (ADVISORY ONLY)
    # =============================

    if market_manipulated:
        st.error("🚨 Market Behaviour Alert: Possible Manipulation Detected")
        for f in manipulation_flags:
            st.write("•", f)
    else:
        st.success("✅ Market behaviour appears normal")

# ======================================================
# GPT TRADE OPINION (OPINION FIRST, EXPLANATION SECOND)
# ======================================================

st.markdown("### 🧠 GPT Trade Opinion")

try:
    from openai import OpenAI
    client = OpenAI()

    prompt = f"""
You are a professional OTC trading analyst.

You MUST follow this exact structure in your response.

First, give a short TRADE OPINION in one line, choosing ONLY one:
- "GOOD SIGNAL – CAN ENTER"
- "RISKY – BETTER TO WAIT"
- "AVOID – NO TRADE"

Second, give a clear explanation.

Rules:
- Do NOT change the signal
- Do NOT generate a new signal
- Do NOT suggest trade sizes

Trade details:
Signal: {final_signal}
Confidence: {confidence}%
Entry Time: {entry.strftime('%H:%M')}
Expiry Time: {expiry.strftime('%H:%M')}

Indicators used:
- Market structure direction
- Wick rejection
- Trend environment
- Momentum expansion
- 100-period Moving Average
- Bollinger Bands
- Stochastic Oscillator

Your response format MUST be:

TRADE OPINION:
<one line verdict>

EXPLANATION:
<short explanation>
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    gpt_text = response.output_text

    # ---- Split opinion & explanation ----
    if "EXPLANATION:" in gpt_text:
        opinion, explanation = gpt_text.split("EXPLANATION:", 1)
    else:
        opinion = gpt_text
        explanation = ""

    st.success(opinion.strip())
    st.info(explanation.strip())

except Exception as e:
    st.warning("GPT opinion unavailable.")







































