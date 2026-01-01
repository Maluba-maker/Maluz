import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Maluz Signal Engine", layout="centered")

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
    else:
        return True

if not check_password():
    st.stop()

# =============================
# IMAGE VALIDATION
# =============================
def validate_image(image):
    if image is None:
        return False, "No image uploaded"
    if image.size == 0:
        return False, "Invalid image data"
    if len(image.shape) != 3:
        return False, "Image must be color"
    return True, "OK"

# =============================
# INPUT
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
# ANALYSIS FUNCTIONS
# =============================
def market_quality_ok(gray):
    return np.std(gray) >= 12

def detect_market_structure(gray):
    h, _ = gray.shape
    roi = gray[int(h*0.3):int(h*0.75), :]
    edges = cv2.Canny(roi, 50, 150)
    proj = np.sum(edges, axis=1)

    highs = np.where(proj > np.mean(proj) * 1.2)[0]
    lows  = np.where(proj < np.mean(proj) * 0.8)[0]

    if len(highs) < 2 or len(lows) < 2:
        return "RANGE"
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "BULLISH"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "BEARISH"
    return "RANGE"

def detect_support_resistance(gray):
    h, _ = gray.shape
    zone = gray[int(h*0.45):int(h*0.75), :]
    projection = np.sum(zone, axis=1)
    mean = np.mean(projection)

    has_support = len(np.where(projection < mean * 0.92)[0]) > 8
    has_resistance = len(np.where(projection > mean * 1.08)[0]) > 8

    volatility = np.std(zone)
    exhaustion = volatility < np.std(gray) * 0.75

    return {
        "support": has_support or exhaustion,
        "resistance": has_resistance or exhaustion
    }

def analyse_candle_behaviour(gray):
    h, w = gray.shape
    recent = gray[int(h*0.55):int(h*0.75), int(w*0.7):]
    std = np.std(recent)

    if std > 35:
        return "STRONG_MOMENTUM"
    if std < 18:
        return "WEAK_REJECTION"
    return "NEUTRAL"

def confirm_trend(gray):
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    left = np.mean(blur[:, :blur.shape[1]//3])
    right = np.mean(blur[:, blur.shape[1]//3:])

    if right > left + 3:
        return "UPTREND"
    if right < left - 3:
        return "DOWNTREND"
    return "FLAT"

def market_behaviour_warning(gray):
    h, _ = gray.shape
    vol = np.std(gray[int(h*0.4):int(h*0.7), :])
    edges = np.mean(cv2.Canny(gray, 50, 150))

    flags = []
    if vol < 18:
        flags.append("Low volatility / choppy market")
    if edges > 45:
        flags.append("Possible manipulation / spikes")
    return flags

# =============================
# FINAL DECISION ENGINE (PERMANENT FIX)
# =============================
def generate_signal(structure, sr, candle, trend):

    buy_score = 0
    sell_score = 0

    # BUY (reaction)
    if sr["support"]:
        buy_score += 1
    if candle == "WEAK_REJECTION":
        buy_score += 1
    if structure == "BULLISH":
        buy_score += 1

    # SELL (continuation / exhaustion)
    if sr["resistance"]:
        sell_score += 1
    if structure == "BEARISH":
        sell_score += 2
    if trend == "DOWNTREND":
        sell_score += 1
    if candle == "STRONG_MOMENTUM":
        sell_score += 1

    # 🔑 CRITICAL FIX:
    # Late impulse into resistance = SELL dominance
    if sr["resistance"] and candle != "STRONG_MOMENTUM":
        sell_score += 2

    # DOMINANT DECISION
    if sell_score > buy_score:
        return "SELL", "Sell dominant (exhaustion at resistance)"
    if buy_score > sell_score:
        return "BUY", "Buy dominant (reaction from support)"

    return "NO TRADE", "No clear dominance"

# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):

    valid, msg = validate_image(image)
    if not valid:
        st.error(msg)
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not market_quality_ok(gray):
        raw_signal, reason = "NO TRADE", "Market quality poor"
    else:
        structure = detect_market_structure(gray)
        sr = detect_support_resistance(gray)
        candle = analyse_candle_behaviour(gray)
        trend = confirm_trend(gray)

        raw_signal, reason = generate_signal(structure, sr, candle, trend)

    signal = raw_signal if raw_signal in ["BUY", "SELL"] else "WAIT"

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    warnings = market_behaviour_warning(gray)

    # =============================
    # OUTPUT
    # =============================
    if signal == "BUY":
        st.markdown(
            "<div style='background:#dcfce7;color:#166534;padding:14px;"
            "border-radius:8px;font-weight:700;'>🟢 BUY SIGNAL</div>",
            unsafe_allow_html=True
        )
    elif signal == "SELL":
        st.markdown(
            "<div style='background:#fee2e2;color:#991b1b;padding:14px;"
            "border-radius:8px;font-weight:700;'>🔴 SELL SIGNAL</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#e5e7eb;color:#374151;padding:14px;"
            "border-radius:8px;font-weight:700;'>⚪ WAIT</div>",
            unsafe_allow_html=True
        )

    st.code(f"""
SIGNAL: {signal}
REASON: {reason}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())

    # =============================
    # MARKET BEHAVIOUR
    # =============================
    if warnings:
        st.error("🚨 Market Behaviour Alert")
        for w in warnings:
            st.write("•", w)
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



















































