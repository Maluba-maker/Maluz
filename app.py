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
st.caption("OTC Screenshot-Based Market Analysis (Human Logic)")

# =============================
# INPUT MODE (UNCHANGED)
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
# ANALYSE MARKET
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Please upload or capture a valid screenshot.")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape

    # =============================
    # 1️⃣ DOMINANT TREND (AUTHORITY)
    # =============================

    red_mask = cv2.inRange(hsv, (0,70,50), (10,255,255)) | \
               cv2.inRange(hsv, (170,70,50), (180,255,255))
    red_pts = np.column_stack(np.where(red_mask > 0))

    if len(red_pts) < 80:
        st.warning("⚪ NO TRADE – No dominant trend")
        st.stop()

    ma_y = np.mean(red_pts[:, 0])
    price_y = int(height * 0.45)

    trend = "UP" if price_y < ma_y else "DOWN"

    slope = np.polyfit(red_pts[:,1], red_pts[:,0], 1)[0]
    if abs(slope) < 0.015:
        st.warning("⚪ NO TRADE – Flat dominant trend")
        st.stop()

    # =============================
    # 2️⃣ BOLLINGER CONTEXT
    # =============================

    bb_mask = cv2.inRange(hsv, (125,50,50), (155,255,255))
    bb_pts = np.column_stack(np.where(bb_mask > 0))

    bb_y = np.mean(bb_pts[:,0])
    near_upper_bb = bb_y < height * 0.35
    near_lower_bb = bb_y > height * 0.65

    # =============================
    # 3️⃣ MOMENTUM STATE (FAST MA)
    # =============================

    blue_mask = cv2.inRange(hsv, (90,80,80), (120,255,255))
    blue_pts = np.column_stack(np.where(blue_mask > 0))

    if len(blue_pts) < 30:
        st.warning("⚪ NO TRADE – No momentum clarity")
        st.stop()

    fast_slope = np.polyfit(blue_pts[:,1], blue_pts[:,0], 1)[0]

    if fast_slope < -0.02:
        momentum = "UP"
    elif fast_slope > 0.02:
        momentum = "DOWN"
    else:
        momentum = "FLAT"

    if momentum == "FLAT":
        st.warning("⚪ NO TRADE – Momentum flat")
        st.stop()

    # =============================
    # 4️⃣ STOCHASTIC STATE (ZONE)
    # =============================

    stoch_zone = gray[int(height * 0.78):height, :]
    stoch_avg = np.mean(stoch_zone)

    if stoch_avg < 105:
        stoch = "LOW"
    elif stoch_avg > 155:
        stoch = "HIGH"
    else:
        stoch = "MID"

    # =============================
    # 5️⃣ FINAL HUMAN DECISION
    # =============================

    signal = "NO TRADE"
    reason = "Context not aligned"

    if trend == "UP":

        # Pullback BUY
        if momentum == "UP" and stoch in ["LOW", "MID"] and not near_upper_bb:
            signal = "BUY"
            reason = "Pullback BUY in uptrend"

        # Continuation BUY
        elif momentum == "UP" and stoch == "HIGH" and near_upper_bb:
            signal = "BUY"
            reason = "Continuation BUY (strong trend)"

    if trend == "DOWN":

        # Pullback SELL
        if momentum == "DOWN" and stoch in ["MID", "HIGH"] and not near_lower_bb:
            signal = "SELL"
            reason = "Pullback SELL in downtrend"

        # Continuation SELL
        elif momentum == "DOWN" and stoch == "LOW" and near_lower_bb:
            signal = "SELL"
            reason = "Continuation SELL (strong trend)"

    # =============================
    # OUTPUT
    # =============================

    st.markdown("---")

    if signal == "NO TRADE":
        st.warning("⚪ NO TRADE")
    else:
        st.success(f"✅ {signal} SIGNAL")

    now = datetime.now()
    entry = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    st.code(f"""
SIGNAL: {signal}
REASON: {reason}
TREND: {trend}
MOMENTUM: {momentum}
STOCHASTIC: {stoch}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""")


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















