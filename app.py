import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

# =============================
# PASSWORD PROTECTION
# =============================

def check_password():
    def password_entered():
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("🔐 Enter password to access Maluz", type="password",
                      key="password", on_change=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("🔐 Enter password to access Maluz", type="password",
                      key="password", on_change=password_entered)
        st.error("❌ Incorrect password")
        return False
    return True


PASSWORD = "maluz123"
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

if not check_password():
    st.stop()

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="Maluz", layout="centered")
st.title("📊 Maluz")
st.caption("OTC Screenshot-Based Market Analysis")

# =============================
# INPUT MODE
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
# ANALYSIS
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Please upload or capture a valid screenshot.")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape

    # ======================================================
    # 1. MARKET BEHAVIOUR (UNSAFE CHECK – NON NEGOTIABLE)
    # ======================================================

    behaviour_flags = []

    candle_zone = gray[int(height * 0.4):int(height * 0.65), :]
    candle_energy = np.std(candle_zone)

    edges = cv2.Canny(gray, 50, 150)
    edge_strength = np.mean(edges)

    recent_slice = gray[int(height * 0.35):int(height * 0.5), :]
    recent_energy = np.std(recent_slice)

    if candle_energy < 18:
        behaviour_flags.append("Low volatility")

    if edge_strength > 45:
        behaviour_flags.append("Excessive wick spikes")

    if abs(candle_energy - recent_energy) > 20:
        behaviour_flags.append("Sudden behaviour change")

    if candle_energy < 18 or edge_strength > 45:
        st.warning("⚪ WAIT – Market unsafe / unstable")
        st.stop()

    # ======================================================
    # 2. MARKET STATE (TREND VS RANGE)
    # ======================================================

    market_state = "TREND"
    center_zone = gray[int(height * 0.45):int(height * 0.55), :]
    center_std = np.std(center_zone)

    if candle_energy < 22 and edge_strength < 25:
        market_state = "RANGE"

    if center_std < 12:
        market_state = "TRANSITION"

    if market_state != "TREND":
        st.info(f"🟡 WAIT – Market in {market_state.lower()} state")
        st.stop()

    # ======================================================
    # 3. STRUCTURE (WHO HAS CONTROL – HUMAN LOGIC)
    # ======================================================

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = (
        cv2.inRange(hsv, lower_red1, upper_red1) |
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    red_points = np.column_stack(np.where(red_mask > 0))

    if len(red_points) < 50:
        st.info("🟡 WAIT – No clear candle authority")
        st.stop()

    slope = np.polyfit(red_points[:, 1], red_points[:, 0], 1)[0]
    trend = "DOWN" if slope > 0 else "UP"

    # ======================================================
    # 4. MOMENTUM (SUPPORTING, NOT VETO)
    # ======================================================

    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_points = np.column_stack(np.where(blue_mask > 0))

    momentum = None
    if len(blue_points) > 30:
        slope_fast = np.polyfit(blue_points[:, 1], blue_points[:, 0], 1)[0]
        momentum = "DOWN" if slope_fast > 0 else "UP"

    # ======================================================
    # 5. REACTION / REJECTION (MANDATORY)
    # ======================================================

    if edge_strength < 25:
        st.info("🟡 WAIT – No reaction at pullback")
        st.stop()

    rejection = edge_strength >= 30
    st.session_state["recent_rejection"] = rejection

    if not rejection:
        st.info("🟡 WAIT – Pullback not finished")
        st.stop()

    # ======================================================
    # 6. FINAL HUMAN DECISION
    # ======================================================

    signal = "BUY" if trend == "UP" else "SELL"
    reason = "Dominant control with visible reaction (human logic)"

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    # =============================
    # SIGNAL DISPLAY
    # =============================

    if signal == "BUY":
        st.markdown(
            "<div style='background-color:#dcfce7; color:#166534; "
            "padding:14px; border-radius:8px; font-weight:700;'>"
            "🟢 BUY SIGNAL</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#fee2e2; color:#991b1b; "
            "padding:14px; border-radius:8px; font-weight:700;'>"
            "🔴 SELL SIGNAL</div>",
            unsafe_allow_html=True
        )

    st.code(f"""
SIGNAL: {signal}
REASON: {reason}
TREND: {trend}
MOMENTUM: {momentum}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())

    if behaviour_flags:
        st.warning("⚠️ Market Behaviour Advisory")
        for flag in behaviour_flags:
            st.write("•", flag)

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






































