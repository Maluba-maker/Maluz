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
# IMAGE HELPERS
# =============================
def candle_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    roi = hsv[int(h*0.55):int(h*0.75), int(w*0.7):]

    green = np.sum((roi[:,:,0] > 35) & (roi[:,:,0] < 85))
    red = np.sum((roi[:,:,0] < 10) | (roi[:,:,0] > 160))

    if green > red * 1.2:
        return "GREEN"
    if red > green * 1.2:
        return "RED"
    return "MIXED"

def detect_structure(gray):
    h,_ = gray.shape
    roi = gray[int(h*0.3):int(h*0.75),:]
    edges = cv2.Canny(roi,50,150)
    proj = np.sum(edges,axis=1)
    hi = np.where(proj > proj.mean()*1.2)[0]
    lo = np.where(proj < proj.mean()*0.8)[0]

    if len(hi)<2 or len(lo)<2:
        return "RANGE"
    if hi[-1]>hi[-2] and lo[-1]>lo[-2]:
        return "BULLISH"
    if hi[-1]<hi[-2] and lo[-1]<lo[-2]:
        return "BEARISH"
    return "RANGE"

def detect_sr(gray):
    h,_ = gray.shape
    zone = gray[int(h*0.45):int(h*0.75),:]
    proj = np.sum(zone,axis=1)
    m = proj.mean()
    return {
        "support": np.sum(proj < m*0.92) > 8,
        "resistance": np.sum(proj > m*1.08) > 8
    }

def detect_trend_from_ma(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,w,_ = hsv.shape

    blue = (hsv[:,:,0] > 90) & (hsv[:,:,0] < 130)
    red  = (hsv[:,:,0] < 10) | (hsv[:,:,0] > 160)

    def avg_y(mask):
        ys = np.where(mask)[0]
        return ys.mean() if len(ys)>50 else None

    left_fast  = avg_y(blue[:, :w//3])
    right_fast = avg_y(blue[:, w//3:])

    left_slow  = avg_y(red[:, :w//3])
    right_slow = avg_y(red[:, w//3:])

    if None in [left_fast,right_fast,left_slow,right_slow]:
        return "FLAT","NONE"

    prev = left_fast < left_slow
    now  = right_fast < right_slow

    if prev and not now:
        return "UPTREND","BULLISH"
    if not prev and now:
        return "DOWNTREND","BEARISH"

    return ("UPTREND" if right_fast < right_slow else "DOWNTREND"),"NONE"

def candle_strength(gray):
    h,w = gray.shape
    roi = gray[int(h*0.55):int(h*0.75),int(w*0.7):]
    v = np.std(roi)
    if v > 35:
        return "IMPULSE"
    if v < 18:
        return "REJECTION"
    return "NEUTRAL"

def market_phase(trend,candle):
    if trend in ["UPTREND","DOWNTREND"] and candle=="IMPULSE":
        return "CONTINUATION"
    if trend in ["UPTREND","DOWNTREND"]:
        return "PULLBACK"
    return "RANGE"

# =============================
# STRICT MALAGNA MIRROR ENGINE
# =============================
def evaluate_pairs(structure, sr, candle, trend, phase):
    fired = []

    # CATEGORY A — TREND
    if phase=="CONTINUATION":
        if structure=="BULLISH":
            fired.append(("BUY",88,"Bullish trend continuation"))
        if structure=="BEARISH":
            fired.append(("SELL",88,"Bearish trend continuation"))

    # CATEGORY B — SR
    if phase=="CONTINUATION":
        if trend=="UPTREND" and sr["support"]:
            fired.append(("BUY",86,"Support hold in uptrend"))
        if trend=="DOWNTREND" and sr["resistance"]:
            fired.append(("SELL",86,"Resistance hold in downtrend"))

    # CATEGORY C — PULLBACK
    if phase=="PULLBACK":
        if trend=="UPTREND":
            fired.append(("SELL",72,"Pullback against uptrend"))
        if trend=="DOWNTREND":
            fired.append(("BUY",72,"Pullback against downtrend"))

    if not fired:
        return "WAIT","No dominant side",0

    buys  = [f for f in fired if f[0]=="BUY"]
    sells = [f for f in fired if f[0]=="SELL"]

    b = sum(x[1] for x in buys)
    s = sum(x[1] for x in sells)

    if b==s:
        return "WAIT","No dominant side",0

    dom = buys if b>s else sells
    dom.sort(key=lambda x:x[1],reverse=True)
    top = dom[0]

    conf = top[1] + (len(dom)-1)*3
    conf = min(95 if phase=="CONTINUATION" else 78, conf)

    if conf < 65:
        return "WAIT","Weak setup",conf

    return top[0],top[2],conf

# =============================
# EXECUTION
# =============================
if image is not None and st.button("🔍 Analyse Market"):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    structure = detect_structure(gray)
    sr = detect_sr(gray)
    candle = candle_strength(gray)
    trend,ma_cross = detect_trend_from_ma(image)
    phase = market_phase(trend,candle)
    color = candle_color(image)

    signal,reason,conf = evaluate_pairs(structure,sr,candle,trend,phase)

    entry = datetime.now().replace(second=0,microsecond=0)+timedelta(minutes=1)
    expiry = entry + timedelta(minutes=1)

    if signal=="BUY":
        st.success(f"🟢 BUY ({conf}%)")
    elif signal=="SELL":
        st.error(f"🔴 SELL ({conf}%)")
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
PHASE: {phase}
CANDLE: {candle}
COLOR: {color}
MA CROSS: {ma_cross}
""")
