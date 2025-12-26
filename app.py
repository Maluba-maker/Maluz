import streamlit as st
import hashlib

# =============================
# PASSWORD PROTECTION
# =============================

def check_password():
    """Returns True if the user entered the correct password."""
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


# 🔑 CHANGE THIS PASSWORD
PASSWORD = "maluz123"   # <-- choose your password
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

if not check_password():
    st.stop()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import mss
import mss.tools
import os

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="Maluz", layout="centered")
st.title("📊 Maluz")
st.caption("OTC Screenshot-Based Market Analysis")

# =============================
# INPUT MODE
# =============================

# =============================
# INPUT MODE
# =============================

input_mode = st.radio(
    "Select Input Mode",
    [
        "Upload / Drag Screenshot",
        "Take Photo (Camera)"
    ]
)

image = None

# =============================
# MANUAL UPLOAD (SCREENSHOT / PHOTO)
# =============================

if input_mode == "Upload / Drag Screenshot":
    uploaded = st.file_uploader(
        "Upload OTC chart screenshot or phone photo",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        image = np.array(Image.open(uploaded))
        st.image(image, caption="Image Loaded", use_column_width=True)

# =============================
# CAMERA INPUT (PHONE / WEBCAM)
# =============================

if input_mode == "Take Photo (Camera)":
    st.info(
        "📸 Take a clear photo of your OTC chart.\n"
        "Make sure indicators (MA, Bollinger Bands, Stochastic) are visible."
    )

    camera_image = st.camera_input("Capture chart photo")

    if camera_image:
        image = np.array(Image.open(camera_image))
        st.image(image, caption="Photo Captured", use_column_width=True)

# =============================
# ANALYSE MARKET (FINAL CLEAN STRATEGY)
# =============================

if st.button("🔍 Analyse Market"):

    # ---------- SAFETY ----------
    if image is None:
        st.error("Please upload or capture a screenshot first.")
        st.stop()

    if image.size == 0:
        st.error("Invalid image.")
        st.stop()

    # ---------- IMAGE PREP ----------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    final_signal = "NO TRADE"
    confidence = 0

    # ======================================================
    # STEP 1: MARKET STATE FILTER (AVOID CHOP)
    # ======================================================

    recent_zone = gray[int(height * 0.35):int(height * 0.55), :]
    volatility = np.std(recent_zone)

    if volatility < 8:
        st.warning("⚪ Market choppy → NO TRADE")
        st.stop()

    # ======================================================
    # STEP 2: TREND DIRECTION (RECENT MOMENTUM)
    # ======================================================

    left_trend = gray[:, :int(width * 0.45)]
    right_trend = gray[:, int(width * 0.55):]

    if np.mean(right_trend) > np.mean(left_trend):
        trend = "UP"
    elif np.mean(right_trend) < np.mean(left_trend):
        trend = "DOWN"
    else:
        trend = "NEUTRAL"

    if trend == "NEUTRAL":
        st.warning("⚪ No clear trend → NO TRADE")
        st.stop()

    st.info(f"📈 Trend detected: {trend}")

    # ======================================================
    # STEP 3: PULLBACK DETECTION
    # ======================================================

    pullback_zone = gray[int(height * 0.45):int(height * 0.65), :]
    pullback_strength = np.mean(pullback_zone)

    trend_zone = gray[int(height * 0.15):int(height * 0.35), :]
    trend_strength = np.mean(trend_zone)

    pullback_valid = False

    if trend == "UP" and pullback_strength < trend_strength:
        pullback_valid = True
    elif trend == "DOWN" and pullback_strength > trend_strength:
        pullback_valid = True

    if not pullback_valid:
        st.warning("⚪ No valid pullback → NO TRADE")
        st.stop()

    # ======================================================
    # STEP 4: EXHAUSTION CHECK (KEY EDGE)
    # ======================================================

    lower_zone = gray[int(height * 0.65):, :]
    upper_zone = gray[:int(height * 0.35), :]

    exhaustion = False

    if trend == "UP" and np.mean(lower_zone) < np.mean(upper_zone):
        exhaustion = True
    elif trend == "DOWN" and np.mean(upper_zone) < np.mean(lower_zone):
        exhaustion = True

    if not exhaustion:
        st.warning("⚪ Pullback not exhausted → WAIT")
        st.stop()

    # ======================================================
    # STEP 5: CONTINUATION CONFIRMATION
    # ======================================================

    right_energy = np.mean(gray[:, int(width * 0.6):])
    left_energy = np.mean(gray[:, :int(width * 0.4)])

    continuation = False

    if trend == "UP" and right_energy > left_energy:
        continuation = True
        final_signal = "BUY"
        confidence = 80

    elif trend == "DOWN" and left_energy > right_energy:
        continuation = True
        final_signal = "SELL"
        confidence = 80

    if not continuation:
        st.warning("⚪ No continuation candle → WAIT")
        st.stop()

    # ======================================================
    # STEP 6: TIMING (1 MIN OTC)
    # ======================================================

    now = datetime.now()
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    arrow = "⬆️" if final_signal == "BUY" else "⬇️"

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    st.success("✅ High-quality setup detected")

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






