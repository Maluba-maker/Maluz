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
# ANALYSE BUTTON (FINAL)
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape

    votes = []
    confidence = 0

    # ======================================================
    # 🔵 TREND DETECTION (NOT DEPENDENT ON 100 MA POSITION)
    # ======================================================

    trend_votes = []

    # --- RED MA SLOPE (100 MA DIRECTION) ---
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red_1, upper_red_1) + \
               cv2.inRange(hsv, lower_red_2, upper_red_2)

    left_red = red_mask[:, :int(width * 0.4)]
    right_red = red_mask[:, int(width * 0.6):]

    if np.count_nonzero(right_red) > np.count_nonzero(left_red):
        trend_votes.append("UP")
    elif np.count_nonzero(left_red) > np.count_nonzero(right_red):
        trend_votes.append("DOWN")

    # --- MARKET STRUCTURE ---
    mid_zone = gray[int(height * 0.45):int(height * 0.55), :]
    left_mid = mid_zone[:, :int(width * 0.5)]
    right_mid = mid_zone[:, int(width * 0.5):]

    if np.mean(right_mid) > np.mean(left_mid):
        trend_votes.append("UP")
    elif np.mean(right_mid) < np.mean(left_mid):
        trend_votes.append("DOWN")

    # --- MOMENTUM FLOW ---
    right_energy = np.mean(gray[:, int(width * 0.6):])
    left_energy = np.mean(gray[:, :int(width * 0.4)])

    if right_energy > left_energy:
        trend_votes.append("UP")
    elif right_energy < left_energy:
        trend_votes.append("DOWN")

    # --- FINAL TREND ---
    if trend_votes.count("UP") >= 2:
        trend = "UP"
    elif trend_votes.count("DOWN") >= 2:
        trend = "DOWN"
    else:
        trend = "NEUTRAL"

    st.info(f"📈 Trend detected: {trend}")

    # ======================================================
    # 🧠 ENTRY CONFIRMATIONS (BLENDED STRATEGY)
    # ======================================================

    # --- WICK REJECTION ---
    upper = gray[:int(height * 0.33), :]
    lower = gray[int(height * 0.66):, :]

    if np.mean(lower) < np.mean(upper):
        votes.append("SELL")
        confidence += 20
    elif np.mean(upper) < np.mean(lower):
        votes.append("BUY")
        confidence += 20

    # --- BOLLINGER BANDS (PURPLE) ---
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    bb_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    if np.count_nonzero(bb_mask[:int(height * 0.35), :]) > 300:
        votes.append("SELL")
        confidence += 20

    if np.count_nonzero(bb_mask[int(height * 0.65):, :]) > 300:
        votes.append("BUY")
        confidence += 20

    # --- STOCHASTIC (BLUE vs ORANGE) ---
    stoch_zone = hsv[int(height * 0.75):height, :]

    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    blue_strength = np.count_nonzero(cv2.inRange(stoch_zone, lower_blue, upper_blue))
    orange_strength = np.count_nonzero(cv2.inRange(stoch_zone, lower_orange, upper_orange))

    if blue_strength > orange_strength and blue_strength > 300:
        votes.append("BUY")
        confidence += 20
    elif orange_strength > blue_strength and orange_strength > 300:
        votes.append("SELL")
        confidence += 20

    # ======================================================
    # 🔒 FINAL SIGNAL (TREND MUST MATCH)
    # ======================================================

    buy_votes = votes.count("BUY")
    sell_votes = votes.count("SELL")

    confidence = min(confidence, 100)

    if trend == "UP" and buy_votes >= 3 and confidence >= 80:
        final_signal = "BUY"
    elif trend == "DOWN" and sell_votes >= 3 and confidence >= 80:
        final_signal = "SELL"
    else:
        final_signal = "NO TRADE"
        confidence = 0

    # ======================================================
    # ⏱ TIMING
    # ======================================================

    now = datetime.now()
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    arrow = "⬆️" if final_signal == "BUY" else "⬇️" if final_signal == "SELL" else ""

    # ======================================================
    # 📤 OUTPUT
    # ======================================================

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







