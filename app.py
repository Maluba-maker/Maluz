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
    return True

if not check_password():
    st.stop()

# =============================
# IMAGE VALIDATION
# =============================
def validate_image(image):
    if image is None or image.size == 0:
        return False, "Invalid image"
    if len(image.shape) != 3:
        return False, "Image must be color"
    return True, "OK"

# =============================
# INPUT
# =============================
input_mode = st.radio("Select Input Mode", ["Upload / Drag Screenshot", "Take Photo (Camera)"])
image = None

if input_mode == "Upload / Drag Screenshot":
    uploaded = st.file_uploader("Upload OTC chart screenshot", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded))
        st.image(image, use_column_width=True)

if input_mode == "Take Photo (Camera)":
    cam = st.camera_input("Capture chart photo")
    if cam:
        image = np.array(Image.open(cam))
        st.image(image, use_column_width=True)

# =============================
# FEATURE EXTRACTION
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
    proj = np.sum(zone, axis=1)
    mean = np.mean(proj)

    return {
        "support": len(np.where(proj < mean * 0.92)[0]) > 8,
        "resistance": len(np.where(proj > mean * 1.08)[0]) > 8
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
# RULE ENGINE WITH CONFIDENCE
# =============================
def evaluate_rules(structure, sr, candle, trend):
    matches = []

    if sr["support"] and candle == "WEAK_REJECTION" and structure in ["RANGE", "BULLISH"]:
        matches.append({"dir": "BUY", "conf": 85, "reason": "Strong support reaction"})

    if sr["resistance"] and candle in ["WEAK_REJECTION", "STRONG_MOMENTUM"] and structure in ["RANGE", "BEARISH"]:
        matches.append({"dir": "SELL", "conf": 85, "reason": "Resistance rejection"})

    if sr["support"] and candle == "NEUTRAL" and structure == "BEARISH":
        matches.append({"dir": "BUY", "conf": 90, "reason": "Sell exhaustion"})

    if sr["resistance"] and candle == "NEUTRAL" and structure == "BULLISH":
        matches.append({"dir": "SELL", "conf": 92, "reason": "Buy exhaustion"})

    if structure == "BULLISH" and trend == "UPTREND":
        matches.append({"dir": "BUY", "conf": 78, "reason": "Uptrend continuation"})

    if structure == "BEARISH" and trend == "DOWNTREND":
        matches.append({"dir": "SELL", "conf": 78, "reason": "Downtrend continuation"})

    # Filter weak signals
    matches = [m for m in matches if m["conf"] >= 70]

    if not matches:
        return "WAIT", "No valid rule-set", 0

    # Sort by confidence
    matches = sorted(matches, key=lambda x: x["conf"], reverse=True)
    top = matches[0]

    if len(matches) > 1:
        second = matches[1]
        if top["dir"] != second["dir"] and abs(top["conf"] - second["conf"]) <= 2:
            return "WAIT", "Conflicting signals with similar confidence", 0

    return top["dir"], top["reason"], top["conf"]

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
        signal, reason, conf = "WAIT", "Market quality poor", 0
    else:
        structure = detect_market_structure(gray)
        sr = detect_support_resistance(gray)
        candle = analyse_candle_behaviour(gray)
        trend = confirm_trend(gray)

        signal, reason, conf = evaluate_rules(structure, sr, candle, trend)

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)
    warnings = market_behaviour_warning(gray)

    # =============================
    # OUTPUT
    # =============================
    if signal == "BUY":
        st.success(f"🟢 BUY SIGNAL ({conf}%)")
    elif signal == "SELL":
        st.error(f"🔴 SELL SIGNAL ({conf}%)")
    else:
        st.info("⚪ WAIT")

    st.code(f"""
SIGNAL: {signal}
CONFIDENCE: {conf}%
REASON: {reason}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())

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




















































