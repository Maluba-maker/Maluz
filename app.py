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

    # =============================
    # 1️⃣ DOMINANT TREND (LONG MA)
    # =============================

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

    red_points = np.column_stack(np.where(red_mask > 0))

    if len(red_points) < 50:
        trend = "FLAT"
    else:
        slope = np.polyfit(red_points[:, 1], red_points[:, 0], 1)[0]

        # 🔴 ONLY CALIBRATION CHANGE
        if abs(slope) < 0.004:
            trend = "FLAT"
        elif slope > 0:
            trend = "DOWN"
        else:
            trend = "UP"

    if trend == "FLAT":
        st.warning("⚪ NO TRADE – Dominant trend flat")
        st.stop()

    # =============================
    # 2️⃣ MOMENTUM (FAST MA)
    # =============================

    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_points = np.column_stack(np.where(blue_mask > 0))

    if len(blue_points) < 30:
        momentum = "FLAT"
    else:
        slope_fast = np.polyfit(blue_points[:, 1], blue_points[:, 0], 1)[0]
        momentum = "DOWN" if slope_fast > 0 else "UP"

    # =============================
    # 3️⃣ STOCHASTIC ZONE
    # =============================

    stoch_zone = gray[int(height * 0.78):height, :]
    stoch_avg = np.mean(stoch_zone)

    if stoch_avg < 105:
        stochastic = "LOW"
    elif stoch_avg > 155:
        stochastic = "HIGH"
    else:
        stochastic = "MID"

    # =============================
    # 4️⃣ MANIPULATION CHECK (WARNING ONLY)
    # =============================

    manipulation_flags = []

    # Choppy candle energy
    recent = gray[int(height * 0.4):int(height * 0.65),
                  int(width * 0.6):width]
    if np.std(recent) < 18:
        manipulation_flags.append("Low volatility chop")

    # Bollinger squeeze
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    bb_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    bb_points = np.column_stack(np.where(bb_mask > 0))
    if len(bb_points) > 0:
        band_width = np.std(bb_points[:, 0])
        if band_width < height * 0.02:
            manipulation_flags.append("Bollinger squeeze")

    # Opposing momentum vs trend
    if trend != momentum:
        manipulation_flags.append("Counter-momentum pressure")

    if manipulation_flags:
        st.warning("⚠️ Possible manipulation / unstable conditions detected:")
        for f in manipulation_flags:
            st.write("•", f)

    # =============================
    # 5️⃣ FINAL DECISION (UNCHANGED)
    # =============================

    signal = "NO TRADE"
    reason = "Context not aligned"

    if trend == momentum:
        if trend == "UP":
            signal = "BUY"
            reason = "Trend continuation BUY"
        elif trend == "DOWN":
            signal = "SELL"
            reason = "Trend continuation SELL"

    if trend == "UP" and momentum == "DOWN" and stochastic == "LOW":
        signal = "BUY"
        reason = "Pullback BUY in uptrend"

    if trend == "DOWN" and momentum == "UP" and stochastic == "HIGH":
        signal = "SELL"
        reason = "Pullback SELL in downtrend"

    # =============================
    # OUTPUT
    # =============================

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    if signal == "NO TRADE":
        st.warning("⚪ NO TRADE")
    else:
        st.success(f"✅ {signal} SIGNAL")

    st.code(f"""
SIGNAL: {signal}
REASON: {reason}
TREND: {trend}
MOMENTUM: {momentum}
STOCHASTIC: {stochastic}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())

    # =============================
    # ⚠️ MANIPULATION WARNING (DISPLAY)
    # =============================

    st.markdown("### ⚠️ Market Behaviour Warning")

    if manipulation_score >= 3:
        st.error("🚨 HIGH MANIPULATION RISK\n\n" + ", ".join(manipulation_flags))
    elif manipulation_score == 2:
        st.warning("⚠️ POSSIBLE MANIPULATION\n\n" + ", ".join(manipulation_flags))
    else:
        st.success("✅ No abnormal manipulation detected")

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




















