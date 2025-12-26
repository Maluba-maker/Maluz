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
# ANALYSE BUTTON (FINAL LOGIC)
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Please upload or capture a screenshot first.")
        st.stop()

    # ---------- BASE IMAGE ----------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape

    decision = "WAIT"
    reason = ""
    confidence = 0

    # ======================================================
    # STEP 1: TREND DETECTION (100 MA – RED)
    # ======================================================

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    red_zone = red_mask[int(height * 0.15):int(height * 0.65), :]
    red_pixels = np.where(red_zone > 0)[0]

    if len(red_pixels) < 300:
        st.warning("WAIT — 100 MA not clearly detected")
        st.stop()

    avg_red_y = np.mean(red_pixels)
    mid_price_y = (height * 0.5)

    if mid_price_y < avg_red_y:
        trend = "UPTREND"
        confidence += 20
    elif mid_price_y > avg_red_y:
        trend = "DOWNTREND"
        confidence += 20
    else:
        st.warning("WAIT — No clear trend")
        st.stop()

    # ======================================================
    # STEP 2: TEST OF 100 MA
    # ======================================================

    test_zone = gray[int(height * 0.45):int(height * 0.55), :]
    if np.std(test_zone) < 15:
        st.warning("WAIT — No valid pullback (test)")
        st.stop()

    confidence += 20

    # ======================================================
    # STEP 3A: STOCHASTIC MOMENTUM (BLUE / ORANGE)
    # ======================================================

    stoch_zone = hsv[int(height * 0.75):height, :]

    blue_mask = cv2.inRange(
        stoch_zone, np.array([90, 80, 50]), np.array([130, 255, 255])
    )
    orange_mask = cv2.inRange(
        stoch_zone, np.array([10, 100, 100]), np.array([25, 255, 255])
    )

    blue_strength = np.count_nonzero(blue_mask)
    orange_strength = np.count_nonzero(orange_mask)

    if trend == "UPTREND" and blue_strength <= orange_strength:
        st.warning("WAIT — No bullish stochastic momentum")
        st.stop()

    if trend == "DOWNTREND" and orange_strength <= blue_strength:
        st.warning("WAIT — No bearish stochastic momentum")
        st.stop()

    confidence += 20

    # ======================================================
    # STEP 3B: FAST MA MOMENTUM (2 MA vs 5 MA)
    # ======================================================

    left_strength = np.mean(gray[:, :int(width * 0.45)])
    right_strength = np.mean(gray[:, int(width * 0.55):])

    if trend == "UPTREND" and right_strength <= left_strength:
        st.warning("WAIT — No bullish MA momentum")
        st.stop()

    if trend == "DOWNTREND" and left_strength <= right_strength:
        st.warning("WAIT — No bearish MA momentum")
        st.stop()

    confidence += 20

    # ======================================================
    # STEP 4: ENTRY QUALITY (YOUR ORIGINAL LOGIC)
    # ======================================================

    upper = gray[:int(height * 0.33), :]
    lower = gray[int(height * 0.66):, :]

    if trend == "UPTREND" and np.mean(lower) >= np.mean(upper):
        st.warning("WAIT — Weak bullish structure")
        st.stop()

    if trend == "DOWNTREND" and np.mean(upper) >= np.mean(lower):
        st.warning("WAIT — Weak bearish structure")
        st.stop()

    confidence += 20

    # ======================================================
    # FINAL DECISION
    # ======================================================

    decision = "BUY" if trend == "UPTREND" else "SELL"

    now = datetime.now()
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    arrow = "⬆️" if decision == "BUY" else "⬇️"

    st.success("✅ Signal generated")

    st.code(f"""
SIGNAL: {decision} {arrow}
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





