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
        st.text_input("🔐 Enter password", type="password",
                      key="password", on_change=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("🔐 Enter password", type="password",
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
st.caption("OTC Screenshot-Based Market Analysis — Final Human Logic")

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
    # 1️⃣ DOMINANT TREND (LONG MA SLOPE)
    # =============================

    red_mask = cv2.inRange(hsv, (0,70,50), (10,255,255)) | \
               cv2.inRange(hsv, (170,70,50), (180,255,255))
    red_pts = np.column_stack(np.where(red_mask > 0))

    if len(red_pts) < 100:
        st.warning("⚪ NO TRADE – No dominant trend")
        st.stop()

    long_slope = np.polyfit(red_pts[:,1], red_pts[:,0], 1)[0]

    # FINAL calibrated threshold
    if long_slope < -0.01:
        trend = "UP"
    elif long_slope > 0.01:
        trend = "DOWN"
    else:
        st.warning("⚪ NO TRADE – Dominant trend flat")
        st.stop()

    ma_y = np.mean(red_pts[:,0])
    price_y = int(height * 0.45)

    price_above_ma = price_y < ma_y
    price_below_ma = price_y > ma_y

    # =============================
    # 2️⃣ FAST MA MOMENTUM
    # =============================

    blue_mask = cv2.inRange(hsv, (90,80,80), (120,255,255))
    blue_pts = np.column_stack(np.where(blue_mask > 0))

    if len(blue_pts) < 50:
        st.warning("⚪ NO TRADE – No momentum clarity")
        st.stop()

    fast_slope = np.polyfit(blue_pts[:,1], blue_pts[:,0], 1)[0]

    if fast_slope < -0.02:
        momentum = "UP"
    elif fast_slope > 0.02:
        momentum = "DOWN"
    else:
        st.warning("⚪ NO TRADE – Momentum flat")
        st.stop()

    # =============================
    # 3️⃣ STOCHASTIC (ZONE ONLY)
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
    # 4️⃣ LOCATION (BOLLINGER CONTEXT)
    # =============================

    bb_mask = cv2.inRange(hsv, (125,50,50), (155,255,255))
    bb_pts = np.column_stack(np.where(bb_mask > 0))
    bb_y = np.mean(bb_pts[:,0])

    near_resistance = bb_y < height * 0.35
    near_support = bb_y > height * 0.65

    # =============================
    # 5️⃣ FINAL TRADE DECISION
    # =============================

    signal = "NO TRADE"
    reason = "Context not aligned"

    if trend == "UP":
        if price_below_ma and stoch in ["LOW", "MID"] and not near_resistance:
            signal = "BUY"
            reason = "Pullback BUY in uptrend"
        elif price_above_ma and momentum == "UP" and stoch == "HIGH" and not near_resistance:
            signal = "BUY"
            reason = "Continuation BUY in uptrend"

    if trend == "DOWN":
        if price_above_ma and stoch in ["MID", "HIGH"] and not near_support:
            signal = "SELL"
            reason = "Pullback SELL in downtrend"
        elif price_below_ma and momentum == "DOWN" and stoch == "LOW" and not near_support:
            signal = "SELL"
            reason = "Continuation SELL in downtrend"

    # =============================
    # ⚠️ MANIPULATION DETECTION (ADVISORY ONLY)
    # =============================

    manipulation_score = 0
    manipulation_flags = []

    recent_zone = gray[int(height*0.35):int(height*0.6), int(width*0.55):width]
    if np.std(recent_zone) > 22:
        manipulation_score += 1
        manipulation_flags.append("Excessive wick noise")

    if abs(fast_slope) < 0.025:
        manipulation_score += 1
        manipulation_flags.append("Unstable momentum")

    if (stoch == "HIGH" and momentum != "DOWN") or (stoch == "LOW" and momentum != "UP"):
        manipulation_score += 1
        manipulation_flags.append("Stochastic not respected")

    if np.std(bb_pts[:,0]) < height * 0.08:
        manipulation_score += 1
        manipulation_flags.append("Bollinger trap zone")

    recent_failures = st.checkbox("⚠️ Recent trades failing on this pair")
    if recent_failures:
        manipulation_score += 1
        manipulation_flags.append("Recent setup failures")

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



















