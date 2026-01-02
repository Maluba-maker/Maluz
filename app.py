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
# BRANDING
# =============================
st.markdown("## 🔹 Maluz Signal Engine")
st.caption("Maluz – a rule-based OTC market analysis.")

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

    if std > 38:
        return "IMPULSE"
    if std < 18:
        return "REJECTION"
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
# 25-PAIR RULE ENGINE (DOMINANT CONFLICT RESOLUTION)
# =============================
def evaluate_pairs(structure, sr, candle, trend):
    fired = []

    # CATEGORY A – TREND (1–5)
    if structure == "BULLISH" and candle == "IMPULSE":
        fired.append(("BUY", 88, "Pair 1: Bullish trend acceleration"))
    if structure == "BULLISH" and trend == "UPTREND" and candle == "REJECTION":
        fired.append(("BUY", 85, "Pair 2: Pullback in uptrend"))
    if structure == "BULLISH" and trend == "UPTREND" and candle == "IMPULSE":
        fired.append(("BUY", 90, "Pair 3: Breakout continuation"))
    if structure == "BEARISH" and candle == "IMPULSE":
        fired.append(("SELL", 88, "Pair 4: Bearish trend acceleration"))
    if structure == "BEARISH" and trend == "DOWNTREND" and candle == "REJECTION":
        fired.append(("SELL", 85, "Pair 5: Pullback in downtrend"))

    # CATEGORY B – SUPPORT / RESISTANCE (6–10)
    if sr["support"] and candle == "REJECTION":
        fired.append(("BUY", 87, "Pair 6: Support rejection"))
    if sr["resistance"] and candle == "REJECTION":
        fired.append(("SELL", 87, "Pair 7: Resistance rejection"))
    if sr["support"] and candle == "NEUTRAL" and structure == "BEARISH":
        fired.append(("BUY", 90, "Pair 8: Sell exhaustion / double bottom"))
    if sr["resistance"] and candle == "NEUTRAL" and structure == "BULLISH":
        fired.append(("SELL", 90, "Pair 9: Buy exhaustion / double top"))
    if sr["support"] and candle == "IMPULSE":
        fired.append(("BUY", 84, "Pair 10: Support impulse"))

    # CATEGORY C – MEAN REVERSION (11–15)
    if sr["support"] and candle == "NEUTRAL" and trend == "DOWNTREND":
        fired.append(("BUY", 86, "Pair 11: Mean reversion from lows"))
    if sr["resistance"] and candle == "NEUTRAL" and trend == "UPTREND":
        fired.append(("SELL", 86, "Pair 12: Mean reversion from highs"))
    if sr["support"] and candle == "REJECTION" and structure == "RANGE":
        fired.append(("BUY", 88, "Pair 13: Oversold snapback"))
    if sr["resistance"] and candle == "REJECTION" and structure == "RANGE":
        fired.append(("SELL", 88, "Pair 14: Overbought snapback"))
    if candle == "IMPULSE" and structure == "RANGE":
        fired.append(("BUY", 83, "Pair 15: Volatility release"))

    # CATEGORY D – MOMENTUM + STRUCTURE (16–20)
    if candle == "IMPULSE" and structure == "BULLISH" and trend == "UPTREND":
        fired.append(("BUY", 84, "Pair 16: Momentum alignment up"))
    if candle == "IMPULSE" and structure == "BEARISH" and trend == "DOWNTREND":
        fired.append(("SELL", 84, "Pair 17: Momentum alignment down"))
    if sr["support"] and structure == "BULLISH" and candle == "NEUTRAL":
        fired.append(("BUY", 89, "Pair 18: Hidden accumulation"))
    if sr["resistance"] and structure == "BEARISH" and candle == "NEUTRAL":
        fired.append(("SELL", 89, "Pair 19: Distribution"))
    if candle == "REJECTION" and trend in ["UPTREND", "DOWNTREND"]:
        fired.append(("BUY" if trend == "UPTREND" else "SELL", 83, "Pair 20: Second-leg entry"))

    # CATEGORY E – OTC / MANIPULATION (21–25)
    if sr["support"] and candle == "IMPULSE" and structure != "BEARISH":
        fired.append(("BUY", 92, "Pair 21: Stop-hunt recovery"))
    if sr["resistance"] and candle == "IMPULSE" and structure != "BULLISH":
        fired.append(("SELL", 92, "Pair 22: Stop-hunt rejection"))
    if sr["support"] and candle == "IMPULSE" and trend == "FLAT":
        fired.append(("BUY", 94, "Pair 23: Spring pattern"))
    if sr["resistance"] and candle == "IMPULSE" and trend == "FLAT":
        fired.append(("SELL", 94, "Pair 24: Upthrust pattern"))
    if candle == "IMPULSE" and structure == "RANGE":
        fired.append(("SELL", 85, "Pair 25: Wick spike fade"))

    if not fired:
        return "WAIT", "No valid pair alignment", 0, None

    fired.sort(key=lambda x: x[1], reverse=True)
    top = fired[0]

    opposing = None
    for f in fired[1:]:
        if f[0] != top[0]:
            opposing = f
            break

    if opposing:
        return (
            top[0],
            f"{top[2]} ⚠️ Conflict with {opposing[0]} ({opposing[1]}%)",
            top[1],
            opposing[1]
        )

    return top[0], top[2], top[1], None

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
        signal, reason, conf, opposing_conf = "WAIT", "Market quality poor", 0, None
    else:
        structure = detect_market_structure(gray)
        sr = detect_support_resistance(gray)
        candle = analyse_candle_behaviour(gray)
        trend = confirm_trend(gray)

        signal, reason, conf, opposing_conf = evaluate_pairs(structure, sr, candle, trend)

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)
    warnings = market_behaviour_warning(gray)

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

    if opposing_conf:
        st.warning(f"⚠️ Opposing signal confidence: {opposing_conf}%")

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






















































