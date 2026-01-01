import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

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
# IMAGE INPUT
# =============================
st.title("📊 Market Screenshot Analysis")

uploaded = st.file_uploader("Upload chart screenshot", type=["png", "jpg", "jpeg"])

image = None
if uploaded:
    image = cv2.cvtColor(np.array(Image.open(uploaded)), cv2.COLOR_RGB2BGR)

# =============================
# ANALYSIS
# =============================
if st.button("🔍 Analyse Market"):

    if image is None or image.size == 0:
        st.error("Please upload a valid chart image.")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # =============================
    # MARKET BEHAVIOUR (NON-BLOCKING)
    # =============================
    behaviour_flags = []

    volatility = np.std(gray[int(height*0.4):int(height*0.65), :])
    if volatility < 18:
        behaviour_flags.append("Low volatility / choppy market")

    edges = cv2.Canny(gray, 50, 150)
    edge_strength = np.mean(edges)
    if edge_strength > 45:
        behaviour_flags.append("Possible spike / manipulation behaviour")

    # =============================
    # FEATURE EXTRACTION
    # =============================

    # Impulse strength (body dominance)
    recent_zone = gray[int(height*0.55):int(height*0.75), :]
    impulse_strength = np.std(recent_zone)

    # Trend bias proxy (brightness slope)
    upper = np.mean(gray[int(height*0.1):int(height*0.3), :])
    lower = np.mean(gray[int(height*0.6):int(height*0.8), :])
    trend_bias = lower - upper  # positive = bearish dominance

    # Rejection detection (wick / pause proxy)
    rejection_strength = np.std(gray[int(height*0.75):height, :])

    # =============================
    # LOGIC SCORES (NO BLOCKING)
    # =============================
    buy_score = 0
    sell_score = 0

    # BUY CONDITIONS (reaction)
    if rejection_strength > 20:
        buy_score += 1
    if impulse_strength < 22:
        buy_score += 1

    # SELL CONDITIONS (continuation)
    if trend_bias > 6:
        sell_score += 2
    if impulse_strength > 22:
        sell_score += 1

    # =============================
    # DOMINANCE DECISION (FINAL)
    # =============================
    if sell_score > buy_score:
        signal = "SELL"
        color = "#ef4444"
        reason = "Bearish continuation dominant (trend + impulse)"
    elif buy_score > sell_score:
        signal = "BUY"
        color = "#22c55e"
        reason = "Reaction bounce dominant (support + rejection)"
    else:
        signal = "WAIT"
        color = "#9ca3af"
        reason = "No clear dominance"

    # =============================
    # OUTPUT
    # =============================
    st.markdown(
        f"""
        <div style="text-align:center;">
            <h1 style="color:{color}; font-size:64px;">{signal}</h1>
            <p>{reason}</p>
            <p><small>{datetime.now().strftime('%H:%M:%S')}</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if behaviour_flags:
        st.warning("⚠️ Market Behaviour Warning:")
        for f in behaviour_flags:
            st.write(f"- {f}")
    else:
        st.success("Market behaviour appears normal.")

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

















































