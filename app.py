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
# ANALYSE
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Please upload or capture a valid screenshot.")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape

    # =============================
    # 1️⃣ TREND (LONG RED MA)
    # =============================

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

    red_points = np.column_stack(np.where(red_mask > 0))

    if len(red_points) < 50:
        st.warning("⚪ No clear trend")
        st.stop()

    ma_y = np.mean(red_points[:, 0])
    price_y = int(height * 0.45)

    trend = "DOWN" if price_y > ma_y else "UP"
    st.info(f"📈 Trend detected: {trend}")

    # =============================
    # 2️⃣ BOLLINGER REJECTION
    # =============================

    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    bb_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    bb_points = np.column_stack(np.where(bb_mask > 0))
    if len(bb_points) < 50:
        st.warning("⚪ No Bollinger reaction")
        st.stop()

    bb_y = np.mean(bb_points[:, 0])
    bb_touch = abs(price_y - bb_y) < height * 0.05

    if not bb_touch:
        st.warning("⚪ No Bollinger rejection")
        st.stop()

    # =============================
    # 3️⃣ FAST MA SLOPE (BLUE)
    # =============================

    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_points = np.column_stack(np.where(blue_mask > 0))
    if len(blue_points) < 30:
        st.warning("⚪ No momentum confirmation")
        st.stop()

    slope = np.polyfit(blue_points[:, 1], blue_points[:, 0], 1)[0]
    ma_direction = "DOWN" if slope > 0 else "UP"

    # =============================
    # 4️⃣ STOCHASTIC ZONE
    # =============================

    stoch_zone = gray[int(height * 0.78):height, :]
    stoch_avg = np.mean(stoch_zone)

    if stoch_avg < 110:
        stoch = "OVERSOLD"
    elif stoch_avg > 150:
        stoch = "OVERBOUGHT"
    else:
        st.warning("⚪ Stochastic not ready")
        st.stop()

    # =============================
    # 5️⃣ FINAL DECISION
    # =============================

    final_signal = "NO TRADE"
    confidence = 0

    if trend == "UP" and ma_direction == "UP" and stoch == "OVERSOLD":
        final_signal = "BUY"
        confidence = 83

    if trend == "DOWN" and ma_direction == "DOWN" and stoch == "OVERBOUGHT":
        final_signal = "SELL"
        confidence = 82

    # =============================
    # ⏱ TIMING
    # =============================

    now = datetime.now()
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    arrow = "⬆️" if final_signal == "BUY" else "⬇️" if final_signal == "SELL" else ""

    # =============================
    # 📤 OUTPUT
    # =============================

    st.markdown("---")

    if final_signal == "NO TRADE":
        st.warning("⚪ Signal generated: NO TRADE")
    else:
        st.success("✅ Signal generated")

    st.code(f"""
SIGNAL: {final_signal} {arrow}
CONFIDENCE: {confidence}%
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())


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








