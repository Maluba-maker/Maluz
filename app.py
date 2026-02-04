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
st.markdown("## 🔹 Maluz Signal Engine")
st.caption("Screenshot-based • STRICT Malagna Mirror")

# =============================
# PASSWORD
# =============================
PASSWORD = "maluz123"
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

def check_password():
    def entered():
        if hashlib.sha256(st.session_state["pw"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state.auth = True
        else:
            st.session_state.auth = False

    if "auth" not in st.session_state or not st.session_state.auth:
        st.text_input("🔐 Password", type="password", key="pw", on_change=entered)
        if "auth" in st.session_state and not st.session_state.auth:
            st.error("Incorrect password")
        st.stop()

check_password()

# =============================
# INPUT
# =============================
mode = st.radio("Input Mode", ["Upload Screenshot", "Camera"])
image = None

if mode == "Upload Screenshot":
    f = st.file_uploader("Upload chart image", type=["png","jpg","jpeg"])
    if f:
        image = np.array(Image.open(f))
        st.image(image, use_column_width=True)

if mode == "Camera":
    cam = st.camera_input("Capture chart")
    if cam:
        image = np.array(Image.open(cam))
        st.image(image, use_column_width=True)

# =============================
# VISUAL DATA → MALAGNA INPUTS
# =============================
def visual_series(gray, length=40):
    h, w = gray.shape
    col = gray[:, int(w * 0.75)]
    return np.interp(
        np.linspace(0, len(col) - 1, length),
        np.arange(len(col)),
        col
    )

# ---- MOCK i5 (to satisfy Malagna logic) ----
i5 = None
ema20_slope = 0
market_active = True

if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    close = visual_series(gray)

    ema20 = np.convolve(close, np.ones(20) / 20, mode="valid")
    ema20 = np.pad(ema20, (len(close) - len(ema20), 0), constant_values=ema20[0])

    i5 = {
        "close": close,
        "ema20": ema20
    }

    if len(ema20) >= 5:
        ema20_slope = ema20[-1] - ema20[-3]

    recent_move = abs(close[-1] - close[-6])
    avg_move = np.mean(np.abs(np.diff(close[-10:])))
    if avg_move > 0 and recent_move < avg_move * 0.6:
        market_active = False

# =============================
# SUPPORT / RESISTANCE
# =============================
sr = {"support": False, "resistance": False}
if i5:
    price = i5["close"][-1]
    if abs(price - np.min(i5["close"][-20:])) / price < 0.002:
        sr["support"] = True
    if abs(price - np.max(i5["close"][-20:])) / price < 0.002:
        sr["resistance"] = True

# =============================
# CANDLE TYPE
# =============================
def candle_type_from_image(gray):
    h, w = gray.shape
    roi = gray[int(h*0.55):int(h*0.75), int(w*0.7):]
    v = np.std(roi)
    if v >= 35:
        return "IMPULSE"
    elif v <= 18:
        return "NEUTRAL"
    return "REJECTION"

candle = candle_type_from_image(gray) if image is not None else "NEUTRAL"

# =============================
# STRUCTURE & TREND
# =============================
structure = "RANGE"
trend = "FLAT"

if i5:
    if i5["close"][-1] > ema20[-1]:
        structure = "BULLISH"
        trend = "UPTREND"
    elif i5["close"][-1] < ema20[-1]:
        structure = "BEARISH"
        trend = "DOWNTREND"

# =============================
# MARKET PHASE (UNCHANGED)
# =============================
def detect_market_phase(i5, trend):
    if i5 is None:
        return "RANGE"

    price = i5["close"][-1]
    ema20 = i5["ema20"][-1]

    if trend == "UPTREND" and price > ema20:
        return "CONTINUATION"
    if trend == "DOWNTREND" and price < ema20:
        return "CONTINUATION"
    if trend == "UPTREND" and price <= ema20:
        return "PULLBACK"
    if trend == "DOWNTREND" and price >= ema20:
        return "PULLBACK"
    return "RANGE"

market_phase = detect_market_phase(i5, trend)

# =============================
# PULLBACK STATE
# =============================
def detect_pullback_state(i5, trend):
    if i5 is None or len(i5["ema20"]) < 5:
        return None

    ema20 = i5["ema20"]
    slope_now = ema20[-1] - ema20[-3]
    slope_prev = ema20[-3] - ema20[-5]

    if abs(slope_now) < abs(slope_prev):
        return "SLOWING"
    if trend == "UPTREND" and slope_now < 0:
        return "TURNING"
    if trend == "DOWNTREND" and slope_now > 0:
        return "TURNING"
    return "SLOWING"

pullback_state = detect_pullback_state(i5, trend) if market_phase == "PULLBACK" else None

# =============================
# VISUAL GATES (UNCHANGED)
# =============================
def gatekeeper(structure, trend, sr, candle):
    penalty = 0
    notes = []

    if structure == "RANGE" and trend == "FLAT":
        penalty += 12
        notes.append("Low structure clarity")

    if candle == "NEUTRAL":
        penalty += 10
        notes.append("Weak candle")

    if structure == "BULLISH" and sr["resistance"]:
        penalty += 15
        notes.append("Near resistance")

    if structure == "BEARISH" and sr["support"]:
        penalty += 15
        notes.append("Near support")

    return penalty, ", ".join(notes) if notes else "Clean setup"

# =============================
# ================= 20-RULE ENGINE =================
# =================
def evaluate_pairs(structure, sr, candle, trend, market_phase, pullback_state):

    penalty, gate_note = gatekeeper(structure, trend, sr, candle)

    fired = []
    momentum_bonus = 0

    # ================= CATEGORY A (TREND & PULLBACK) =================
    if market_phase == "CONTINUATION":

        if structure == "BULLISH" and candle == "IMPULSE":
            fired.append(("BUY", 88, "Bullish trend continuation"))

        if structure == "BEARISH" and candle == "IMPULSE":
            fired.append(("SELL", 88, "Bearish trend continuation"))

    elif market_phase == "PULLBACK" and pullback_state:

        if trend == "UPTREND":
            if pullback_state == "SLOWING":
                fired.append(("SELL", 68, "Pullback slowing"))
            else:
                fired.append(("SELL", 75, "Pullback turning"))

        elif trend == "DOWNTREND":
            if pullback_state == "SLOWING":
                fired.append(("BUY", 68, "Pullback slowing"))
            else:
                fired.append(("BUY", 75, "Pullback turning"))

    # ================= CATEGORY B (SUPPORT / RESISTANCE) =================
    if market_phase == "CONTINUATION":

        if trend == "UPTREND" and sr["support"] and candle == "REJECTION":
            fired.append(("BUY", 86, "Support hold in uptrend"))

        if trend == "DOWNTREND" and sr["resistance"] and candle == "REJECTION":
            fired.append(("SELL", 86, "Resistance hold in downtrend"))

    elif market_phase == "PULLBACK":

        if trend == "UPTREND" and sr["resistance"] and candle in ["REJECTION", "NEUTRAL"]:
            fired.append(("SELL", 72, "Pullback rejection at resistance"))

        if trend == "DOWNTREND" and sr["support"] and candle in ["REJECTION", "NEUTRAL"]:
            fired.append(("BUY", 72, "Pullback rejection at support"))

    # ================= CATEGORY C (MEAN REVERSION) =================
    if market_phase == "NO_TRADE":

        if sr["support"] and candle in ["NEUTRAL", "REJECTION"]:
            fired.append(("BUY", 75, "Range mean reversion (support)"))

        if sr["resistance"] and candle in ["NEUTRAL", "REJECTION"]:
            fired.append(("SELL", 75, "Range mean reversion (resistance)"))

    # ================= CATEGORY D (MOMENTUM) =================
    if market_phase == "CONTINUATION":

        if candle == "IMPULSE":
            momentum_bonus += 6

        if ema20_slope > 0 and trend == "UPTREND":
            momentum_bonus += 4

        if ema20_slope < 0 and trend == "DOWNTREND":
            momentum_bonus += 4

    elif market_phase == "PULLBACK":

        if candle == "REJECTION":
            momentum_bonus += 3

    # ================= DOMINANT SIDE =================
    buys = [r for r in fired if r[0] == "BUY"]
    sells = [r for r in fired if r[0] == "SELL"]

    buy_score = sum(r[1] for r in buys)
    sell_score = sum(r[1] for r in sells)

    if buy_score == sell_score:
        return "WAIT", "No dominant side", 0

    dominant_rules = buys if buy_score > sell_score else sells
    dominant_rules.sort(key=lambda x: x[1], reverse=True)
    top = dominant_rules[0]

    # ================= FINAL CONFIDENCE =================
    confidence = top[1] + (len(dominant_rules) - 1) * 3
    confidence = min(99, confidence - penalty)

    if not market_active:
        confidence -= 8

    if market_phase == "PULLBACK":
        confidence = min(confidence, 78)

    if market_phase == "CONTINUATION":
        confidence = min(confidence, 95)

    if confidence < 65:
        return "WAIT", f"Weak setup ({gate_note})", confidence

    return top[0], f"{top[2]} • {gate_note}", confidence

# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):

    signal, reason, confidence = evaluate_pairs(
        structure, sr, candle, trend, market_phase, pullback_state
    )

    entry = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    if signal == "BUY":
        st.success(f"🟢 BUY ({confidence}%)")
    elif signal == "SELL":
        st.error(f"🔴 SELL ({confidence}%)")
    else:
        st.info("⚪ WAIT")

    st.code(f"""
SIGNAL: {signal}
CONFIDENCE: {confidence}%
REASON: {reason}
ENTRY: {entry.strftime('%H:%M')}
EXPIRY: {expiry.strftime('%H:%M')}
STRUCTURE: {structure}
TREND: {trend}
PHASE: {market_phase}
CANDLE: {candle}
SUPPORT: {sr['support']}
RESISTANCE: {sr['resistance']}
""")


