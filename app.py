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
st.caption("Screenshot-based OTC analysis • Malagna-style decision logic")

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
# FEATURE EXTRACTION (UNCHANGED)
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
# MALAGNA-STYLE STATE DERIVATION (IMAGE-BASED)
# =============================
def detect_market_phase(structure, trend, candle):
    if trend in ["UPTREND", "DOWNTREND"] and candle == "IMPULSE":
        return "CONTINUATION"
    if trend in ["UPTREND", "DOWNTREND"] and candle in ["REJECTION", "NEUTRAL"]:
        return "PULLBACK"
    return "RANGE"

def detect_pullback_state(gray):
    h, _ = gray.shape
    recent = gray[int(h*0.55):int(h*0.75), :]
    older  = gray[int(h*0.35):int(h*0.55), :]

    vol_now = np.std(recent)
    vol_prev = np.std(older)

    if abs(vol_now - vol_prev) < 2:
        return "SLOWING"
    return "TURNING"

# =============================
# GATEKEEPER (FROM MALAGNA LOGIC)
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
# MALAGNA DECISION ENGINE (IMAGE VERSION)
# =============================
def evaluate_pairs_image(structure, sr, candle, trend, market_phase, pullback_state, market_active):

    penalty, gate_note = gatekeeper(structure, trend, sr, candle)
    fired = []

    # ---- CATEGORY A (TREND & PULLBACK) ----
    if market_phase == "CONTINUATION":
        if structure == "BULLISH" and candle == "IMPULSE":
            fired.append(("BUY", 88, "Bullish trend continuation"))
        if structure == "BEARISH" and candle == "IMPULSE":
            fired.append(("SELL", 88, "Bearish trend continuation"))

    elif market_phase == "PULLBACK":
        if trend == "UPTREND":
            fired.append(("SELL", 72, "Pullback against uptrend"))
        elif trend == "DOWNTREND":
            fired.append(("BUY", 72, "Pullback against downtrend"))

    # ---- CATEGORY B (SUPPORT / RESISTANCE) ----
    if market_phase == "CONTINUATION":
        if trend == "UPTREND" and sr["support"] and candle == "REJECTION":
            fired.append(("BUY", 86, "Support hold in uptrend"))
        if trend == "DOWNTREND" and sr["resistance"] and candle == "REJECTION":
            fired.append(("SELL", 86, "Resistance hold in downtrend"))

    # ---- DOMINANT SIDE ----
    buys = [r for r in fired if r[0] == "BUY"]
    sells = [r for r in fired if r[0] == "SELL"]

    buy_score = sum(r[1] for r in buys)
    sell_score = sum(r[1] for r in sells)

    if buy_score == sell_score:
        return "WAIT", "No dominant side", 0

    dominant = buys if buy_score > sell_score else sells
    dominant.sort(key=lambda x: x[1], reverse=True)
    top = dominant[0]

    confidence = top[1] + (len(dominant) - 1) * 3
    confidence = min(95 if market_phase == "CONTINUATION" else 78, confidence - penalty)

    if not market_active:
        confidence -= 8

    if confidence < 65:
        return "WAIT", f"Weak setup ({gate_note})", confidence

    return top[0], f"{top[2]} • {gate_note}", confidence

# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):

    valid, msg = validate_image(image)
    if not valid:
        st.error(msg)
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    market_active = market_quality_ok(gray)

    structure = detect_market_structure(gray)
    sr = detect_support_resistance(gray)
    candle = analyse_candle_behaviour(gray)
    trend = confirm_trend(gray)

    market_phase = detect_market_phase(structure, trend, candle)
    pullback_state = detect_pullback_state(gray) if market_phase == "PULLBACK" else None

    signal, reason, conf = evaluate_pairs_image(
        structure, sr, candle, trend, market_phase, pullback_state, market_active
    )

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
STRUCTURE: {structure}
TREND: {trend}
CANDLE: {candle}
PHASE: {market_phase}
""".strip())

    if warnings:
        st.error("🚨 Market Behaviour Alert")
        for w in warnings:
            st.write("•", w)
    else:
        st.success("✅ Market behaviour appears normal")
