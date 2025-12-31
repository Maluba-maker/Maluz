import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Maluz Market Analyst", layout="wide")

# ================================
# UI – UPLOAD
# ================================
st.markdown("### Upload OTC chart screenshot")

uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["png", "jpg", "jpeg"]
)

analyse = st.button("🔍 Analyse Market")

# ================================
# SAFETY EXIT
# ================================
if not analyse:
    st.stop()

if uploaded_file is None:
    st.warning("Please upload a chart screenshot first.")
    st.stop()

# ================================
# IMAGE LOAD
# ================================
image = np.array(Image.open(uploaded_file))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ================================
# DEFAULT VALUES (CRITICAL)
# ================================
trend = "FLAT"
momentum = "FLAT"
stochastic_state = "MID"
signal = "NO TRADE"
reason = "Context not aligned"

manipulation_flags = []  # <<< ALWAYS DEFINED

# ================================
# ---- ANALYSIS LOGIC ----
# ================================

# ---- TREND (dominant MA proxy) ----
price_slope = np.mean(np.diff(gray[-50:].mean(axis=1)))

if price_slope > 0.01:
    trend = "UP"
elif price_slope < -0.01:
    trend = "DOWN"
else:
    trend = "FLAT"

# ---- MOMENTUM (recent candles) ----
recent_move = gray[-5:].mean() - gray[-15:-10].mean()

if recent_move > 0:
    momentum = "UP"
elif recent_move < 0:
    momentum = "DOWN"

# ---- STOCHASTIC (approx proxy) ----
volatility = np.std(gray[-30:])

if volatility < 8:
    stochastic_state = "LOW"
elif volatility > 20:
    stochastic_state = "HIGH"
else:
    stochastic_state = "MID"

# ================================
# ---- MARKET BEHAVIOUR FLAGS ----
# ================================

if volatility < 6:
    manipulation_flags.append("Low volatility / choppy price action")

if trend != "FLAT" and momentum != trend:
    manipulation_flags.append("Momentum opposing dominant trend")

if abs(price_slope) < 0.005:
    manipulation_flags.append("Dominant trend flat / compressed")

# ================================
# ---- SIGNAL DECISION (LOCKED) ----
# ================================

if trend == "UP" and momentum == "UP":
    if stochastic_state in ["LOW", "MID"]:
        signal = "BUY"
        reason = "Trend continuation BUY"

elif trend == "DOWN" and momentum == "DOWN":
    if stochastic_state in ["LOW", "MID"]:
        signal = "SELL"
        reason = "Trend continuation SELL"

else:
    signal = "NO TRADE"
    reason = "Context not aligned"

# ================================
# ---- DISPLAY SIGNAL (FIRST) ----
# ================================

st.markdown("---")

if signal == "BUY":
    st.success("🟢 BUY SIGNAL")
elif signal == "SELL":
    st.error("🔴 SELL SIGNAL")
else:
    st.warning("⚪ NO TRADE")

entry_time = datetime.now().strftime("%H:%M")
expiry_time = (datetime.now() + timedelta(minutes=1)).strftime("%H:%M")

st.markdown(f"""
**SIGNAL:** {signal}  
**REASON:** {reason}  
**TREND:** {trend}  
**MOMENTUM:** {momentum}  
**STOCHASTIC:** {stochastic_state}  
**ENTRY:** {entry_time}  
**EXPIRY:** {expiry_time}
""")

# ================================
# ---- MARKET BEHAVIOUR WARNING ----
# ================================

if manipulation_flags:
    st.markdown("---")
    st.warning("⚠️ Market Behaviour Warning (Flags Only)")

    st.markdown("""
These conditions suggest **potential instability or artificial price behaviour**.
Proceed with caution — technical signals may fail under such conditions.
""")

    for flag in manipulation_flags:
        st.write("•", flag)

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



























