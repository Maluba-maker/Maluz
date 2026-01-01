import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

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
# PAGE CONFIG (UNCHANGED)
# =============================

st.set_page_config(page_title="Maluz", layout="centered")
st.title("📊 Maluz")
st.caption("OTC Screenshot-Based Market Analysis")

# =============================
# INPUT (UNCHANGED)
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
# HELPER FUNCTIONS (NEW LOGIC)
# =============================

def extract_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def market_eligible(gray):
    return np.std(gray) >= 12

def detect_structure(gray):
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)

    if len(xs) < 120:
        return "RANGE"

    slope = np.polyfit(xs, ys, 1)[0]
    return "BEARISH" if slope > 0 else "BULLISH"

def detect_support_resistance(gray):
    h = gray.shape[0]
    mid = gray[int(h*0.4):int(h*0.65), :]
    proj = np.sum(mid, axis=1)
    mean = np.mean(proj)

    return {
        "support": len(np.where(proj < mean * 0.85)[0]) > 15,
        "resistance": len(np.where(proj > mean * 1.15)[0]) > 15
    }

def candle_behaviour(gray):
    h, w = gray.shape
    recent = gray[int(h*0.55):int(h*0.75), int(w*0.7):]
    std = np.std(recent)

    if std > 35:
        return "STRONG"
    if std < 18:
        return "REJECTION"
    return "NEUTRAL"

def trend_confirm(gray):
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    left = np.mean(blur[:, :blur.shape[1]//3])
    right = np.mean(blur[:, blur.shape[1]//3:])

    if right > left + 3:
        return "UP"
    if right < left - 3:
        return "DOWN"
    return "FLAT"

def detect_manipulation(gray):
    flags = []
    h = gray.shape[0]

    zone_a = gray[int(h*0.4):int(h*0.65), :]
    zone_b = gray[int(h*0.3):int(h*0.5), :]

    if np.std(zone_a) < 15 and np.std(zone_b) > 30:
        flags.append("Artificial volatility release")

    if np.mean(cv2.Canny(gray, 50, 150)) > 45:
        flags.append("Abnormal wick spikes")

    return flags

def generate_signal(structure, sr, candle, trend):
    votes = []

    if structure == "BULLISH" and sr["support"]:
        votes.append("BUY")
    if structure == "BEARISH" and sr["resistance"]:
        votes.append("SELL")

    if candle == "REJECTION" and votes:
        votes.append(votes[-1])

    if candle == "STRONG":
        return "WAIT", "Strong momentum – wait for pullback"

    if trend == "FLAT":
        return "WAIT", "Market undecided"

    if votes.count("BUY") >= 2:
        return "BUY", "Structure + level + rejection aligned"

    if votes.count("SELL") >= 2:
        return "SELL", "Structure + level + rejection aligned"

    return "WAIT", "Conditions not fully aligned"

# =============================
# CORE ANALYSIS (UNCHANGED FLOW)
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Invalid image.")
        st.stop()

    gray = extract_gray(image)

    manipulation_flags = detect_manipulation(gray)
    market_manipulated = len(manipulation_flags) > 0

    if not market_eligible(gray):
        signal = "WAIT"
        reason = "Market too quiet / noisy"
    else:
        structure = detect_structure(gray)
        sr = detect_support_resistance(gray)
        candle = candle_behaviour(gray)
        trend = trend_confirm(gray)

        signal, reason = generate_signal(structure, sr, candle, trend)

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    # =============================
    # OUTPUT (UNCHANGED LOOK)
    # =============================

    if signal == "BUY":
        st.markdown(
            "<div style='background:#dcfce7;color:#166534;"
            "padding:14px;border-radius:8px;font-weight:700;'>"
            "🟢 BUY SIGNAL</div>",
            unsafe_allow_html=True
        )
    elif signal == "SELL":
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

    st.code(f"""
SIGNAL: {signal}
REASON: {reason}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
""".strip())

    # =============================
    # MARKET BEHAVIOUR (ADVISORY)
    # =============================

    if market_manipulated:
        st.error("🚨 Market Behaviour Alert: Possible Manipulation Detected")
        for f in manipulation_flags:
            st.write("•", f)
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










































