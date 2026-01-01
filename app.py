import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image

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
# MARKET STRUCTURE
# =============================
def detect_market_structure(gray):
    h, w = gray.shape
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

# =============================
# SUPPORT & RESISTANCE
# =============================
def detect_support_resistance(gray):
    h = gray.shape[0]
    mid = gray[int(h*0.4):int(h*0.65), :]
    proj = np.sum(mid, axis=1)
    mean = np.mean(proj)

    return {
        "support": len(np.where(proj < mean * 0.85)[0]) > 15,
        "resistance": len(np.where(proj > mean * 1.15)[0]) > 15
    }

# =============================
# CANDLE BEHAVIOUR
# =============================
def analyse_candle_behaviour(gray):
    h, w = gray.shape
    recent = gray[int(h*0.55):int(h*0.75), int(w*0.7):]
    std = np.std(recent)

    if std > 35:
        return "STRONG_MOMENTUM"
    if std < 18:
        return "WEAK_REJECTION"
    return "NEUTRAL"

# =============================
# TREND CONFIRMATION
# =============================
def confirm_trend(gray):
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    left = np.mean(blur[:, :blur.shape[1]//3])
    right = np.mean(blur[:, blur.shape[1]//3:])

    if right > left + 3:
        return "UPTREND"
    if right < left - 3:
        return "DOWNTREND"
    return "FLAT"

# =============================
# MARKET BEHAVIOUR WARNING
# =============================
def market_behaviour_warning(gray):
    h = gray.shape[0]
    vol = np.std(gray[int(h*0.4):int(h*0.7), :])
    edges = cv2.Canny(gray, 50, 150)
    edge_strength = np.mean(edges)

    warnings = []
    if vol < 18:
        warnings.append("⚠️ Low volatility / choppy market")
    if edge_strength > 45:
        warnings.append("⚠️ Possible manipulation / abnormal spikes")

    return warnings

# =============================
# FINAL DECISION ENGINE
# =============================
def generate_signal(structure, sr, candle, trend):
    votes = []

    if structure == "BULLISH" and sr["support"]:
        votes.append("BUY")

    if structure == "BEARISH" and sr["resistance"]:
        votes.append("SELL")

    if candle == "WEAK_REJECTION" and votes:
        votes.append(votes[-1])

    if candle == "STRONG_MOMENTUM":
        return "NO TRADE"

    if trend == "FLAT":
        return "NO TRADE"

    if votes.count("BUY") >= 2:
        return "BUY"

    if votes.count("SELL") >= 2:
        return "SELL"

    return "NO TRADE"

# =============================
# UI
# =============================
st.title("📊 Maluz Image-Based Signal Engine")

uploaded = st.file_uploader("Upload chart screenshot", type=["png", "jpg", "jpeg"])

if uploaded:
    image = np.array(Image.open(uploaded))
    st.image(image, caption="Uploaded Chart", use_container_width=True)

    if st.button("🔍 Analyse Market"):
        valid, msg = validate_image(image)

        if not valid:
            st.error(msg)
            st.stop()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        structure = detect_market_structure(gray)
        sr = detect_support_resistance(gray)
        candle = analyse_candle_behaviour(gray)
        trend = confirm_trend(gray)
        warnings = market_behaviour_warning(gray)

        raw_signal = generate_signal(structure, sr, candle, trend)

        # 🔁 DISPLAY MAPPING (IMPORTANT)
        if raw_signal == "BUY":
            display_signal = "BUY"
        elif raw_signal == "SELL":
            display_signal = "SELL"
        else:
            display_signal = "WAIT"

        st.subheader("🚦 SIGNAL")

        if display_signal == "BUY":
            st.markdown(
                "<div style='background:#dcfce7;color:#166534;"
                "padding:14px;border-radius:8px;font-weight:700;'>"
                "🟢 BUY SIGNAL</div>",
                unsafe_allow_html=True
            )
        elif display_signal == "SELL":
            st.markdown(
                "<div style='background:#fee2e2;color:#991b1b;"
                "padding:14px;border-radius:8px;font-weight:700;'>"
                "🔴 SELL SIGNAL</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background:#e5e7eb;color:#374151;"
                "padding:14px;border-radius:8px;font-weight:700;'>"
                "⚪ WAIT</div>",
                unsafe_allow_html=True
            )

        if warnings:
            st.subheader("⚠️ Market Behaviour Warning")
            for w in warnings:
                st.warning(w)

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











































