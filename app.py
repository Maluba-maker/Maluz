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
# PAGE CONFIG
# =============================

st.set_page_config(page_title="Maluz", layout="centered")
st.title("📊 Maluz")
st.caption("OTC Screenshot-Based Market Analysis")

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
# HELPER FUNCTIONS
# =============================

def extract_gray_hsv(img):
    return (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    )

def region(gray, h, top, bottom):
    return gray[int(h * top):int(h * bottom), :]

def edge_strength(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.mean(edges)

def detect_manipulation(gray, h):
    flags = []

    zone_a = region(gray, h, 0.40, 0.65)
    zone_b = region(gray, h, 0.35, 0.50)

    std_a = np.std(zone_a)
    std_b = np.std(zone_b)
    edges = edge_strength(gray)

    if edges > 45:
        flags.append("Abnormal wick spikes")

    if abs(std_a - std_b) > 22:
        flags.append("Sudden volatility injection")

    if std_a < 15 and std_b > 30:
        flags.append("Artificial volatility release")

    return flags

# =============================
# THINKING-BASED ANALYSIS
# =============================

def market_eligible(gray):
    if np.std(gray) < 12:
        return False
    return True

def detect_bias(gray):
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)

    if len(xs) < 100:
        return "NONE"

    slope = np.polyfit(xs, ys, 1)[0]
    return "SELL" if slope > 0 else "BUY"

def pullback_status(gray):
    h, w = gray.shape
    left = gray[:, :w//2]
    right = gray[:, w//2:]

    std_left = np.std(left)
    std_right = np.std(right)

    if std_right < std_left * 0.85:
        return "WEAK"
    if std_right > std_left * 1.15:
        return "STRONG"
    return "NEUTRAL"

def trigger_candle(gray):
    h, w = gray.shape
    last = gray[:, int(w*0.75):int(w*0.9)]
    prev = gray[:, int(w*0.6):int(w*0.75)]

    if np.mean(last) < np.mean(prev) * 0.97:
        return "SELL"
    if np.mean(last) > np.mean(prev) * 1.03:
        return "BUY"
    return "NONE"

# =============================
# CORE ANALYSIS
# =============================

if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Invalid image.")
        st.stop()

    gray, hsv = extract_gray_hsv(image)
    h, w = gray.shape

    manipulation_flags = detect_manipulation(gray, h)
    market_manipulated = len(manipulation_flags) > 0

    # STEP 1 — Eligibility
    if not market_eligible(gray):
        signal = "WAIT"
        reason = "Market too dead / noisy"

    else:
        # STEP 2 — Bias
        bias = detect_bias(gray)

        if bias == "NONE":
            signal = "WAIT"
            reason = "No clear structure"

        else:
            # STEP 3 — Pullback quality
            pullback = pullback_status(gray)

            if pullback == "STRONG":
                signal = "WAIT"
                reason = "Strong pullback against bias"

            else:
                # STEP 4 — Trigger
                trigger = trigger_candle(gray)

                if trigger == bias:
                    signal = bias
                    reason = "Bias confirmed by trigger"
                else:
                    signal = "WAIT"
                    reason = "No valid trigger yet"

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    # =============================
    # OUTPUT
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








































