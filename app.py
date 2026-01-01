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
        st.text_input(
            "🔐 Enter password to access Maluz",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False
    elif not st.session_state["authenticated"]:
        st.text_input(
            "🔐 Enter password to access Maluz",
            type="password",
            key="password",
            on_change=password_entered
        )
        st.error("❌ Incorrect password")
        return False
    else:
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

input_mode = st.radio(
    "Select Input Mode",
    ["Upload / Drag Screenshot", "Take Photo (Camera)"]
)

image = None

if input_mode == "Upload / Drag Screenshot":
    uploaded = st.file_uploader(
        "Upload OTC chart screenshot",
        type=["png", "jpg", "jpeg"]
    )
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
    # 0. TIME & CONTEXT STRICTNESS
    # ======================================================

    now = datetime.now()
    strict_mode = False

    if now.minute < 10 or now.minute > 50:
        strict_mode = True  # dead / rollover minutes

    # ======================================================
    # 1. MARKET BEHAVIOUR (RATE + CONSISTENCY)
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
    # 2. MARKET STATE (TREND vs RANGE vs TRANSITION)
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
    # 3. STRUCTURE (DOMINANT + AUTHORITY)
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

    if len(red_points) < 70:
        st.info("🟡 WAIT – Structure unclear")
        st.stop()

    slope = np.polyfit(red_points[:, 1], red_points[:, 0], 1)[0]

    if abs(slope) < 0.006:
        st.info("🟡 WAIT – Weak structure")
        st.stop()

    trend = "DOWN" if slope > 0 else "UP"

    # ======================================================
    # 4. MOMENTUM (RELATIVE + PULLBACK QUALITY)
    # ======================================================

    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_points = np.column_stack(np.where(blue_mask > 0))

    if len(blue_points) < 40:
        st.info("🟡 WAIT – No pullback")
        st.stop()

    slope_fast = np.polyfit(blue_points[:, 1], blue_points[:, 0], 1)[0]
    momentum = "DOWN" if slope_fast > 0 else "UP"

    # HUMAN RULE: only pullbacks, never late continuation
    if momentum == trend:
        st.info("🟡 WAIT – Late continuation")
        st.stop()

    # ======================================================
    # 5. LOCATION (ROOM + LOGIC)
    # ======================================================

    if edge_strength < 20:
        st.info("🟡 WAIT – No reaction at key zone")
        st.stop()

    # ======================================================
    # 6. REJECTION (TIMING + MEMORY)
    # ======================================================

    rejection = False

    if edge_strength >= 30:
        rejection = True

    if st.session_state.get("recent_rejection", False):
        rejection = True

    st.session_state["recent_rejection"] = rejection

    if not rejection:
        st.info("🟡 WAIT – Pullback not finished")
        st.stop()

    # ======================================================
    # 7. STOCHASTIC (CONFIRMATION, NOT PREDICTION)
    # ======================================================

    stoch_zone = gray[int(height * 0.78):height, :]
    stoch_avg = np.mean(stoch_zone)

    if trend == "UP" and stoch_avg < 105:
        st.info("🟡 WAIT – Momentum not turned up yet")
        st.stop()

    if trend == "DOWN" and stoch_avg > 145:
        st.info("🟡 WAIT – Momentum not turned down yet")
        st.stop()

    # ======================================================
    # 8. FINAL HUMAN SANITY FILTER
    # ======================================================

    confidence_score = 0

    if candle_energy > 25:
        confidence_score += 1
    if edge_strength > 30:
        confidence_score += 1
    if not strict_mode:
        confidence_score += 1
    if rejection:
        confidence_score += 1
    if momentum != trend:
        confidence_score += 1

    if confidence_score < 4:
        st.info("🟡 WAIT – Setup not clean enough")
        st.stop()

    # ======================================================
    # 9. FINAL DECISION (MATCHES HUMAN LOGIC)
    # ======================================================

    signal = "BUY" if trend == "UP" else "SELL"
    reason = "Clean pullback + rejection in dominant trend"

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
CONFIDENCE SCORE: {confidence_score}/5
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





































