
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import platform
try:
    import winsound
except Exception:
    winsound = None
try:
    from winotify import Notification, audio
except Exception:
    Notification = None
    audio = None

st.set_page_config(page_title="Binance TR Coin Tarayıcı", layout="wide")

SYMBOLS_URL = "https://www.binance.tr/open/v1/common/symbols"
KLINE_MAIN = "https://api.binance.me/api/v1/klines"
KLINE_NEXT = "https://cloudme-tr.2meta.app/api/v1/klines"
DEPTH_MAIN = "https://api.binance.me/api/v3/depth"
DEPTH_NEXT = "https://cloudme-tr.2meta.app/api/v1/depth"

INTERVALS = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]

def clean_symbol(s):
    return s.upper().replace("/", "_").replace("-", "_").replace(" ", "")


def depth_endpoint_for_type(symbol_type):
    return DEPTH_MAIN if int(symbol_type)==1 else DEPTH_NEXT

@st.cache_data(ttl=10)
def fetch_order_book(symbol, symbol_type=1, limit=100):
    api_symbol=symbol.replace("_","") if int(symbol_type)==1 else symbol
    r=requests.get(depth_endpoint_for_type(symbol_type),params={"symbol":api_symbol,"limit":limit},timeout=10)
    r.raise_for_status()
    j=r.json()
    data=j.get("data",j) if isinstance(j,dict) else j
    bids=data.get("bids",[]); asks=data.get("asks",[])
    if not bids or not asks: raise RuntimeError("Emir defteri verisi yok")
    return bids,asks

def order_book_pressure(symbol,symbol_type=1,band_pct=1.0,limit=100):
    bids,asks=fetch_order_book(symbol,symbol_type,limit)
    best_bid=float(bids[0][0]); best_ask=float(asks[0][0]); mid=(best_bid+best_ask)/2
    lo=mid*(1-band_pct/100); hi=mid*(1+band_pct/100)
    bid_try=ask_try=0.0; bw=(0.0,0.0); aw=(0.0,0.0)
    for row in bids:
        p0,q=float(row[0]),float(row[1])
        if p0>=lo:
            v=p0*q; bid_try+=v
            if v>bw[1]: bw=(p0,v)
    for row in asks:
        p0,q=float(row[0]),float(row[1])
        if p0<=hi:
            v=p0*q; ask_try+=v
            if v>aw[1]: aw=(p0,v)
    total=bid_try+ask_try
    bp=bid_try/total*100 if total else 50; ap=ask_try/total*100 if total else 50
    ratio=bid_try/ask_try if ask_try else 99
    label="🟢 ALIM BASKISI" if bp>=60 else ("🔴 SATIŞ BASKISI" if ap>=60 else "🟡 NÖTR")
    return {"bid_pct":bp,"ask_pct":ap,"ratio":ratio,"label":label,"confirm":bp>=60,
            "spread_pct":((best_ask-best_bid)/mid*100 if mid else 0),
            "bid_wall_price":bw[0],"ask_wall_price":aw[0]}

@st.cache_data(ttl=300)
def get_try_symbols():
    r = requests.get(SYMBOLS_URL, timeout=15)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(j.get("msg", "Sembol listesi alınamadı"))
    items = j.get("data", {}).get("list", [])
    out = []
    for x in items:
        if (
            str(x.get("quoteAsset", "")).upper() == "TRY"
            and int(x.get("spotTradingEnable", 1) or 0) == 1
        ):
            out.append({
                "symbol": x.get("symbol"),
                "baseAsset": x.get("baseAsset"),
                "quoteAsset": x.get("quoteAsset"),
                "type": int(x.get("type", 1) or 1),
            })
    return sorted(out, key=lambda z: z["symbol"])

def endpoint_for_type(symbol_type):
    return KLINE_MAIN if int(symbol_type) == 1 else KLINE_NEXT

def fetch_klines(symbol, interval="1h", limit=250, symbol_type=1, timeout=12):
    api_symbol = symbol.replace("_", "") if int(symbol_type) == 1 else symbol
    url = endpoint_for_type(symbol_type)
    r = requests.get(url, params={"symbol": api_symbol, "interval": interval, "limit": int(limit)}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict):
        if j.get("code", 0) not in (0, "0", None):
            raise RuntimeError(j.get("msg", "Kline hatası"))
        raw = j.get("data", [])
    else:
        raw = j
    if not raw or not isinstance(raw, list):
        raise RuntimeError("Mum verisi yok")
    cols = [
        "open_time","open","high","low","close","volume","close_time",
        "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "quote_volume" in df:
        df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Europe/Istanbul")
    return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d.clip(upper=0))
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def indicators(df):
    d = df.copy()
    d["EMA20"] = ema(d["close"],20)
    d["EMA50"] = ema(d["close"],50)
    d["EMA200"] = ema(d["close"],200)
    d["RSI"] = rsi(d["close"])
    d["MACD"] = ema(d["close"],12)-ema(d["close"],26)
    d["MACD_SIG"] = ema(d["MACD"],9)
    d["MACD_H"] = d["MACD"]-d["MACD_SIG"]
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["BB_MID"] = mid
    d["BB_UP"] = mid + 2*std
    d["BB_LOW"] = mid - 2*std
    d["ATR"] = atr(d)
    d["VOL_MA20"] = d["volume"].rolling(20).mean()
    return d


def nearby_levels(df, current_price, lookback=160, pivot_window=3):
    """Yerel pivotlardan fiyata en yakın destek/direnç bölgelerini bulur."""
    x = df.tail(min(lookback, len(df))).copy()
    lows, highs = [], []
    for i in range(pivot_window, len(x)-pivot_window):
        lo = x["low"].iloc[i]
        hi = x["high"].iloc[i]
        if lo == x["low"].iloc[i-pivot_window:i+pivot_window+1].min():
            lows.append(float(lo))
        if hi == x["high"].iloc[i-pivot_window:i+pivot_window+1].max():
            highs.append(float(hi))

    # Yakın seviyeleri ATR/fiyat ölçeğine göre kümelendir.
    tolerance = max(current_price * 0.008, 1e-12)

    def cluster(vals):
        vals = sorted(vals)
        groups = []
        for v in vals:
            if not groups or abs(v - np.mean(groups[-1])) > tolerance:
                groups.append([v])
            else:
                groups[-1].append(v)
        return [float(np.mean(g)) for g in groups]

    lows = cluster(lows)
    highs = cluster(highs)
    supports = sorted([v for v in lows + highs if v < current_price], reverse=True)
    resistances = sorted([v for v in lows + highs if v > current_price])

    support1 = supports[0] if supports else float(x["low"].min())
    support2 = supports[1] if len(supports) > 1 else float(x["low"].min())
    resistance1 = resistances[0] if resistances else float(x["high"].max())
    resistance2 = resistances[1] if len(resistances) > 1 else float(x["high"].max())
    return support1, support2, resistance1, resistance2


def analyze(df):
    if len(df) < 60:
        raise RuntimeError("Yetersiz veri")
    d = indicators(df)
    x, p = d.iloc[-1], d.iloc[-2]
    score = 0
    why = []

    # Trend
    if x["close"] > x["EMA20"] > x["EMA50"]:
        score += 2; why.append("EMA trend +")
    elif x["close"] < x["EMA20"] < x["EMA50"]:
        score -= 2; why.append("EMA trend -")

    if len(d) >= 200:
        if x["close"] > x["EMA200"]: score += 1
        elif x["close"] < x["EMA200"]: score -= 1

    # RSI
    if x["RSI"] < 30:
        score += 2; why.append("RSI aşırı satım")
    elif 50 <= x["RSI"] <= 65:
        score += 1
    elif x["RSI"] > 70:
        score -= 2; why.append("RSI aşırı alım")
    elif 35 <= x["RSI"] < 50:
        score -= 1

    # MACD
    if x["MACD"] > x["MACD_SIG"] and p["MACD"] <= p["MACD_SIG"]:
        score += 2; why.append("MACD yukarı kesişim")
    elif x["MACD"] < x["MACD_SIG"] and p["MACD"] >= p["MACD_SIG"]:
        score -= 2; why.append("MACD aşağı kesişim")
    elif x["MACD_H"] > 0:
        score += 1
    else:
        score -= 1

    # Bollinger
    if pd.notna(x["BB_LOW"]) and x["close"] <= x["BB_LOW"]:
        score += 1
    elif pd.notna(x["BB_UP"]) and x["close"] >= x["BB_UP"]:
        score -= 1

    # Volume
    if pd.notna(x["VOL_MA20"]) and x["VOL_MA20"] > 0:
        vol_ratio = float(x["volume"]/x["VOL_MA20"])
        if vol_ratio >= 1.5:
            if x["close"] >= x["open"]:
                score += 1; why.append("Yüksek alım hacmi")
            else:
                score -= 1; why.append("Yüksek satış hacmi")
    else:
        vol_ratio = np.nan

    # Momentum
    chg_3 = (x["close"]/d.iloc[-4]["close"]-1)*100 if len(d)>=4 else 0
    if chg_3 > 2: score += 1
    elif chg_3 < -2: score -= 1

    if score >= 5: signal = "🟢 GÜÇLÜ AL"
    elif score >= 3: signal = "🟢 AL ADAYI"
    elif score <= -5: signal = "🔴 GÜÇLÜ SAT"
    elif score <= -3: signal = "🔴 SAT ADAYI"
    else: signal = "🟡 BEKLE"

    recent = d.tail(min(60,len(d)))
    support, support2, resistance, resistance2 = nearby_levels(d, float(x["close"]))
    atrv = float(x["ATR"])
    return {
        "Fiyat": float(x["close"]),
        "Puan": int(score),
        "Sinyal": signal,
        "RSI": float(x["RSI"]),
        "MACD Hist": float(x["MACD_H"]),
        "Hacim x": float(vol_ratio) if pd.notna(vol_ratio) else np.nan,
        "3 Mum %": float(chg_3),
        "ATR %": float(atrv/x["close"]*100) if x["close"] else np.nan,
        "Destek": support,
        "Destek2": support2,
        "Direnç": resistance,
        "Direnç2": resistance2,
        "Neden": ", ".join(why) if why else "Nötr"
    }, d


def scan_multi_one(item, limit=250):
    """15dk + 1s + 4s analizlerini birleştirir.
    Büyük zaman dilimlerine daha yüksek ağırlık verilir.
    """
    frames = [("15m", 0.20), ("1h", 0.35), ("4h", 0.45)]
    out = {
        "Coin": item["baseAsset"],
        "Parite": item["symbol"],
        "Tip": item["type"],
    }
    weighted = 0.0
    total_weight = 0.0
    notes = []
    last_result = None

    for iv, w in frames:
        try:
            df = fetch_klines(item["symbol"], iv, limit, item["type"])
            r, _ = analyze(df)
            last_result = r
            weighted += r["Puan"] * w
            total_weight += w
            label = {"15m":"15dk", "1h":"1S", "4h":"4S"}[iv]
            out[f"{label} Puan"] = int(r["Puan"])
            out[f"{label} RSI"] = round(float(r["RSI"]), 1)
            notes.append(f"{label}: {r['Sinyal']}")
        except Exception:
            label = {"15m":"15dk", "1h":"1S", "4h":"4S"}[iv]
            out[f"{label} Puan"] = np.nan
            out[f"{label} RSI"] = np.nan

    if total_weight == 0 or last_result is None:
        return None

    combined = weighted / total_weight
    out["Birleşik Puan"] = round(combined, 2)

    try:
        out.update(analyze_7day(item))
    except Exception:
        out.update({"7G Puan":0,"7G Trend":"⚪ 7G VERİ YOK","7G Değişim %":np.nan,
                    "7G Dip":np.nan,"7G Zirve":np.nan,"7G Aralık Konumu %":np.nan,
                    "EMA20 Üstü %":np.nan,"EMA50 Üstü %":np.nan,"EMA20 Eğimi %":np.nan,
                    "Son24S Hacim Değişim %":np.nan,"7G Maks Drawdown %":np.nan,"7G Neden":"Veri yok"})

    # Ana yön filtresi: 4S + 7G birlikte değerlendirilir.
    p4 = out.get("4S Puan", np.nan)
    p1 = out.get("1S Puan", np.nan)
    p15 = out.get("15dk Puan", np.nan)

    bullish_confirm = pd.notna(p4) and pd.notna(p1) and p4 >= 3 and p1 >= 3
    bearish_confirm = pd.notna(p4) and pd.notna(p1) and p4 <= -3 and p1 <= -3
    d7score=float(out.get("7G Puan",0))

    if combined >= 4 and bullish_confirm and d7score >= 2:
        signal = "🟢 GÜÇLÜ AL"
    elif combined >= 2.5 and pd.notna(p4) and p4 >= 1 and d7score >= 1:
        signal = "🟢 AL ADAYI"
    elif combined <= -4 and bearish_confirm:
        signal = "🔴 GÜÇLÜ SAT"
    elif combined <= -2.5 and pd.notna(p4) and p4 <= -1:
        signal = "🔴 SAT ADAYI"
    else:
        signal = "🟡 BEKLE"

    out["Sinyal"] = signal
    out["Fiyat"] = last_result["Fiyat"]
    out["4S Destek"] = last_result["Destek"]
    out["4S Destek2"] = last_result["Destek2"]
    out["4S Direnç"] = last_result["Direnç"]
    out["4S Direnç2"] = last_result["Direnç2"]
    out["4S ATR %"] = last_result["ATR %"]
    price=float(last_result["Fiyat"]); support=float(last_result["Destek"]); resistance=float(last_result["Direnç"])
    upside=((resistance/price)-1)*100 if price>0 else np.nan
    downside=((price/support)-1)*100 if support>0 and price>support else np.nan
    rr=(upside/downside) if pd.notna(downside) and downside>0 else np.nan
    out["Hedef Potansiyeli %"]=round(upside,2) if pd.notna(upside) else np.nan
    out["Desteğe Risk %"]=round(downside,2) if pd.notna(downside) else np.nan
    out["Risk/Getiri"]=round(rr,2) if pd.notna(rr) else np.nan
    out["Getiri Skoru"]=round(combined*10 + min(max(upside,0),50)*0.8 + min(max(rr if pd.notna(rr) else 0,0),5)*8 + max(min(float(out.get("7G Puan",0)),7),-7)*4,2)
    stars=0
    if p4>=3: stars+=1
    if p1>=3: stars+=1
    if p15>=2 and float(out.get("7G Puan",0))>=2: stars+=1
    if pd.notna(rr) and rr>=2: stars+=1
    if pd.notna(upside) and upside>=2: stars+=1
    out["AL Uygunluk"]=stars
    out["Yıldız"]="⭐"*stars+"☆"*(5-stars)
    out["Karar"]="🟢 ÇOK GÜÇLÜ" if stars==5 else ("🟢 GÜÇLÜ" if stars==4 else ("🟡 ORTA" if stars==3 else "⚪ BEKLE"))
    out["Teyit"] = " | ".join(notes)
    return out

def scan_one(item, interval, limit):
    try:
        df = fetch_klines(item["symbol"], interval, limit, item["type"])
        result, _ = analyze(df)
        result["Coin"] = item["baseAsset"]
        result["Parite"] = item["symbol"]
        result["Tip"] = item["type"]
        return result
    except Exception:
        return None

def fmt(v):
    if v >= 1000: return f"{v:,.2f}"
    if v >= 1: return f"{v:,.4f}"
    return f"{v:,.8f}"



def send_signal_alert(coin, action_short, price, message):
    """Windows masaüstü bildirimi + sesli uyarı. Başarısız olursa uygulama çalışmaya devam eder."""
    try:
        if winsound is not None:
            if "SAT" in action_short:
                winsound.Beep(700, 350)
                winsound.Beep(550, 450)
            elif "AL" in action_short:
                winsound.Beep(900, 300)
                winsound.Beep(1150, 400)
    except Exception:
        pass

    try:
        if Notification is not None and platform.system().lower() == "windows":
            title = f"{coin}/TRY — {action_short}"
            toast = Notification(
                app_id="Binance TR Sinyal Takip",
                title=title,
                msg=f"Fiyat: {price:.6f} TRY\n{message}",
                duration="long",
            )
            try:
                if audio is not None:
                    toast.set_audio(audio.Default, loop=False)
            except Exception:
                pass
            toast.show()
    except Exception:
        pass

def build_trade_plan(item, qty=0.0, avg_cost=0.0):
    """15dk/1S/4S verilerini birleştirip karar-destek amaçlı işlem bölgeleri üretir."""
    multi = scan_multi_one(item, 250)
    if multi is None:
        raise RuntimeError("Çoklu zaman analizi alınamadı.")

    raw4 = fetch_klines(item["symbol"], "4h", 300, item["type"])
    r4, d4 = analyze(raw4)
    raw1 = fetch_klines(item["symbol"], "1h", 250, item["type"])
    r1, d1 = analyze(raw1)
    raw15 = fetch_klines(item["symbol"], "15m", 250, item["type"])
    r15, d15 = analyze(raw15)

    price = float(r4["Fiyat"])
    atr4 = float(d4.iloc[-1]["ATR"])
    support = float(r4["Destek"])
    support2 = float(r4["Destek2"])
    resistance = float(r4["Direnç"])
    resistance2 = float(r4["Direnç2"])

    p15 = float(multi["15dk Puan"])
    p1 = float(multi["1S Puan"])
    p4 = float(multi["4S Puan"])
    combined = float(multi["Birleşik Puan"])

    # Teyit kuralları
    all_bull = p15 >= 2 and p1 >= 3 and p4 >= 3
    core_bull = p1 >= 2 and p4 >= 3
    all_bear = p15 <= -2 and p1 <= -3 and p4 <= -3
    core_bear = p1 <= -2 and p4 <= -3

    # Entry zone is near price/support, avoiding chasing far above support.
    entry_low = max(support, price - 0.60 * atr4)
    entry_high = min(price + 0.20 * atr4, resistance)
    if entry_high < entry_low:
        entry_low, entry_high = min(entry_low, price), max(entry_high, price)

    # Structural stop below nearest support with ATR buffer.
    stop = max(0.0, support - 0.35 * atr4)
    risk = max(price - stop, 1e-12)

    # Targets use nearby pivot resistances and R multiples.
    target1 = max(resistance, price + 1.0 * risk)
    target2 = max(resistance2, price + 2.0 * risk)
    target3 = price + 3.0 * risk

    # Signal state
    if all_bull and combined >= 3.0:
        action = "🟢 ALIM TEYİDİ GÜÇLÜ"
        action_short = "AL"
        confidence = "Yüksek"
        reason = "15dk, 1S ve 4S aynı yönde pozitif."
    elif core_bull and combined >= 2.0:
        action = "🟢 ALIM İÇİN TEYİT BEKLENEBİLİR"
        action_short = "AL ADAYI"
        confidence = "Orta"
        reason = "1S ve 4S pozitif; 15dk giriş zamanlamasını belirlemeli."
    elif all_bear and combined <= -3.0:
        action = "🔴 ÇIKIŞ RİSKİ YÜKSEK"
        action_short = "SAT"
        confidence = "Yüksek"
        reason = "15dk, 1S ve 4S birlikte negatif."
    elif core_bear and combined <= -2.0:
        action = "🔴 POZİSYON RİSKİ ARTIYOR"
        action_short = "SAT ADAYI"
        confidence = "Orta"
        reason = "1S ve 4S negatif; kısa vadeli tepki olsa bile ana yapı zayıf."
    else:
        action = "🟡 BEKLE / TEYİT YOK"
        action_short = "BEKLE"
        confidence = "Düşük"
        reason = "Zaman dilimleri aynı yönde yeterince teyit vermiyor."

    # Existing-position context
    pnl = None
    pnl_pct = None
    if qty > 0 and avg_cost > 0:
        pnl = qty * (price - avg_cost)
        pnl_pct = (price / avg_cost - 1) * 100

    return {
        "price": price, "p15": p15, "p1": p1, "p4": p4, "combined": combined,
        "action": action, "action_short": action_short, "confidence": confidence, "reason": reason,
        "entry_low": entry_low, "entry_high": entry_high, "stop": stop,
        "target1": target1, "target2": target2, "target3": target3,
        "support": support, "support2": support2,
        "resistance": resistance, "resistance2": resistance2,
        "atr_pct": float(r4["ATR %"]),
        "rsi15": float(r15["RSI"]), "rsi1": float(r1["RSI"]), "rsi4": float(r4["RSI"]),
        "pnl": pnl, "pnl_pct": pnl_pct,
    }


def scan_dip_one(item, limit_4h=300, limit_1d=120):
    """Dibe yakınlık + dönüş teyidi taraması."""
    try:
        df4 = fetch_klines(item["symbol"], "4h", limit_4h, item["type"])
        r4, d4 = analyze(df4)
        df1d = fetch_klines(item["symbol"], "1d", limit_1d, item["type"])
        r1d, d1d = analyze(df1d)
        multi = scan_multi_one(item, 250)
        if multi is None:
            return None

        price = float(r4["Fiyat"])
        near_resistance = float(r4["Direnç"])
        target_potential = ((near_resistance / price) - 1) * 100 if price > 0 and near_resistance > price else 0.0
        low30 = float(d1d.tail(min(30, len(d1d)))["low"].min())
        low60 = float(d1d.tail(min(60, len(d1d)))["low"].min())
        high60 = float(d1d.tail(min(60, len(d1d)))["high"].max())

        dist30 = (price / low30 - 1) * 100 if low30 > 0 else np.nan
        dist60 = (price / low60 - 1) * 100 if low60 > 0 else np.nan
        range_pos = ((price-low60)/(high60-low60))*100 if high60 > low60 else 50.0

        p15 = float(multi["15dk Puan"])
        p1 = float(multi["1S Puan"])
        p4 = float(multi["4S Puan"])
        combined = float(multi["Birleşik Puan"])

        # Dibe yakınlık puanı: 60 günlük aralığın alt bölümünde olmayı ödüllendir.
        dip_score = max(0.0, 100.0 - min(max(dist60, 0.0), 50.0) * 2.0)

        # Dönüş teyidi: düşmüş olması tek başına yeterli değil.
        reversal = 0
        if p15 >= 1: reversal += 1
        if p1 >= 2: reversal += 1
        if p4 >= 2: reversal += 1
        if float(r4["RSI"]) >= 45: reversal += 1

        # 5. teyit: 4 saatlik son mum hacmi, önceki 20 mum ortalamasının üzerinde mi?
        vols = d4["volume"].astype(float)
        vol_now = float(vols.iloc[-1])
        vol_avg20 = float(vols.iloc[-21:-1].mean()) if len(vols) >= 21 else float(vols.iloc[:-1].mean())
        vol_ratio = (vol_now / vol_avg20) if vol_avg20 > 0 else 0.0
        volume_confirm = vol_ratio >= 1.20
        if volume_confirm:
            reversal += 1

        try:
            ob=order_book_pressure(item["symbol"],item["type"],1.0,100)
            if ob["confirm"]: reversal += 1
        except Exception:
            ob={"bid_pct":50.0,"ask_pct":50.0,"ratio":1.0,"label":"⚪ VERİ YOK","confirm":False,
                "spread_pct":0.0,"bid_wall_price":0.0,"ask_wall_price":0.0}

        # Yaklaşık 24 saatlik TRY işlem hacmi:
        # 4S grafikte son 6 mumun quote_volume toplamı varsa onu kullan,
        # yoksa base volume * tipik fiyat ile yaklaşıkla.
        if "quote_volume" in d4.columns and d4["quote_volume"].notna().any():
            qv = pd.to_numeric(d4["quote_volume"], errors="coerce").fillna(0)
            volume_24h_try = float(qv.tail(min(6, len(qv))).sum())
        else:
            recent6 = d4.tail(min(6, len(d4))).copy()
            volume_24h_try = float((recent6["volume"] * recent6["close"]).sum())

        # 24s hacmin kendi 7 günlük ortalamasına göre değişimi (4S veriden yaklaşık)
        if "quote_volume" in d4.columns and d4["quote_volume"].notna().any():
            qv_all = pd.to_numeric(d4["quote_volume"], errors="coerce").fillna(0)
            blocks = []
            vals = qv_all.tail(min(42, len(qv_all))).tolist()
            for i in range(0, len(vals), 6):
                chunk = vals[i:i+6]
                if chunk:
                    blocks.append(sum(chunk))
            avg_24h = float(np.mean(blocks[:-1])) if len(blocks) > 1 else volume_24h_try
        else:
            avg_24h = volume_24h_try

        volume_24h_change = ((volume_24h_try / avg_24h) - 1) * 100 if avg_24h > 0 else 0.0

        if dist60 <= 5 and reversal >= 5:
            label = "🎯 DİBE ÇOK YAKIN + DÖNÜŞ"
        elif dist60 <= 10 and reversal >= 5:
            label = "🟢 DİBE YAKIN + DÖNÜŞ"
        elif dist60 <= 15 and reversal >= 4:
            label = "🟡 İZLE"
        else:
            label = "⚪ TEYİT YOK"

        # Dip avcısı skoru: dip yakınlığı + trend dönüş teyidi.
        hunter_score = dip_score * 0.55 + reversal * 8 + max(min(combined, 6), -6) * 2.2

        return {
            "Coin": item["baseAsset"],
            "Parite": item["symbol"],
            "Fiyat": price,
            "Yakın 4S Direnç": round(near_resistance, 8),
            "Hedef Potansiyeli %": round(target_potential, 2),
            "30G Dip": low30,
            "30G Dipten Uzaklık %": round(dist30, 2),
            "60G Dip": low60,
            "60G Dipten Uzaklık %": round(dist60, 2),
            "60G Aralık Konumu %": round(range_pos, 1),
            "15dk Puan": p15,
            "1S Puan": p1,
            "4S Puan": p4,
            "Birleşik Puan": combined,
            "4S RSI": round(float(r4["RSI"]), 1),
            "Dönüş Teyidi": f"{reversal}/6",
            "Emir Defteri": ob["label"],
            "Alış Baskısı %": round(ob["bid_pct"],1),
            "Satış Baskısı %": round(ob["ask_pct"],1),
            "Alış/Satış Oranı": round(ob["ratio"],2),
            "Spread %": round(ob["spread_pct"],4),
            "Alış Duvarı": round(ob["bid_wall_price"],8),
            "Satış Duvarı": round(ob["ask_wall_price"],8),
            "Hacim Oranı": round(vol_ratio, 2),
            "Hacim Teyidi": "✅" if volume_confirm else "❌",
            "24S Hacim TRY": round(volume_24h_try, 2),
            "24S Hacim Değişim %": round(volume_24h_change, 1),
            "Dip Avcısı Skoru": round(hunter_score, 1),
            "Durum": label,
        }
    except Exception:
        return None


def capital_plan(capital, risk_pct, entry, stop, target1, target2, target3, max_alloc_pct=35):
    risk_cash = capital * risk_pct / 100.0
    stop_pct = ((entry-stop)/entry)*100 if entry > 0 and stop < entry else 0.0
    if stop_pct <= 0:
        return None
    qty_by_risk = risk_cash / (entry-stop)
    max_alloc = capital * max_alloc_pct / 100.0
    qty_by_alloc = max_alloc / entry
    qty = min(qty_by_risk, qty_by_alloc)
    position_value = qty * entry
    actual_risk = qty * (entry-stop)
    def scenario(px):
        profit = qty * (px-entry)
        pct = ((px/entry)-1)*100 if entry else 0
        rr = profit/actual_risk if actual_risk > 0 else 0
        return profit, pct, rr
    p1=scenario(target1); p2=scenario(target2); p3=scenario(target3)
    return {
        "qty":qty, "position_value":position_value, "risk_cash":actual_risk,
        "risk_pct_capital":actual_risk/capital*100 if capital else 0,
        "stop_pct":stop_pct, "t1":p1, "t2":p2, "t3":p3
    }


def trading_costs(entry_price, exit_price, qty, buy_fee_pct=0.10, sell_fee_pct=0.10, slippage_pct=0.0):
    """Alış/satış komisyonu ve isteğe bağlı kayma dahil net sonuç."""
    buy_fee_rate = buy_fee_pct / 100.0
    sell_fee_rate = sell_fee_pct / 100.0
    slip_rate = slippage_pct / 100.0

    effective_entry = entry_price * (1 + slip_rate)
    effective_exit = exit_price * (1 - slip_rate)

    gross_buy = effective_entry * qty
    buy_fee = gross_buy * buy_fee_rate
    total_cost = gross_buy + buy_fee

    gross_sell = effective_exit * qty
    sell_fee = gross_sell * sell_fee_rate
    net_sell = gross_sell - sell_fee

    net_pnl = net_sell - total_cost
    net_pnl_pct = (net_pnl / total_cost * 100) if total_cost > 0 else 0.0

    # Başabaş: satış komisyonu ve kayma sonrası net satış = toplam alış maliyeti
    denom = qty * (1 - sell_fee_rate) * (1 - slip_rate)
    breakeven = (total_cost / denom) if denom > 0 else entry_price

    return {
        "effective_entry": effective_entry,
        "effective_exit": effective_exit,
        "gross_buy": gross_buy,
        "buy_fee": buy_fee,
        "total_cost": total_cost,
        "gross_sell": gross_sell,
        "sell_fee": sell_fee,
        "net_sell": net_sell,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "breakeven": breakeven,
        "total_fees": buy_fee + sell_fee,
    }


def analyze_7day(item):
    df = fetch_klines(item["symbol"], "1h", 168, item["type"])
    d = indicators(df)
    if len(d) < 48:
        raise RuntimeError("7 günlük analiz için yeterli veri yok.")
    first=float(d.iloc[0]["close"]); last=float(d.iloc[-1]["close"])
    ret7=(last/first-1)*100 if first>0 else 0.0
    high7=float(d["high"].max()); low7=float(d["low"].min())
    range_pos=((last-low7)/(high7-low7))*100 if high7>low7 else 50.0
    above20=float((d["close"]>d["EMA20"]).mean()*100)
    above50=float((d["close"]>d["EMA50"]).mean()*100)
    ema20_now=float(d.iloc[-1]["EMA20"]); ema50_now=float(d.iloc[-1]["EMA50"])
    ema20_old=float(d.iloc[-24]["EMA20"]) if len(d)>=24 else float(d.iloc[0]["EMA20"])
    ema_slope=(ema20_now/ema20_old-1)*100 if ema20_old>0 else 0.0
    vol_recent=float(d["volume"].tail(24).mean())
    vol_old=float(d["volume"].iloc[:-24].mean()) if len(d)>24 else vol_recent
    vol_change=(vol_recent/vol_old-1)*100 if vol_old>0 else 0.0
    closes=d["close"].astype(float); rolling_max=closes.cummax()
    max_dd=float((closes/rolling_max-1).min()*100)

    score=0; reasons=[]
    if ret7>=3: score+=2; reasons.append("7G getiri pozitif")
    elif ret7>=0: score+=1
    elif ret7<=-8: score-=2; reasons.append("7G trend zayıf")
    else: score-=1

    if last>ema20_now>ema50_now: score+=2; reasons.append("EMA20>EMA50 ve fiyat üstünde")
    elif last<ema20_now<ema50_now: score-=2; reasons.append("EMA yapısı negatif")

    if above20>=60: score+=1
    elif above20<=40: score-=1

    if ema_slope>1: score+=1; reasons.append("EMA20 eğimi yukarı")
    elif ema_slope<-1: score-=1

    if vol_change>=20: score+=1; reasons.append("Son 24S hacim artıyor")
    elif vol_change<=-30: score-=1

    if range_pos>=92: score-=1; reasons.append("7G zirvesine çok yakın")
    if range_pos<=10 and score<2: score-=1

    label="🟢 7G GÜÇLÜ POZİTİF" if score>=5 else ("🟢 7G POZİTİF" if score>=2 else ("🔴 7G NEGATİF" if score<=-3 else "🟡 7G NÖTR"))
    return {
        "7G Puan":int(score),"7G Trend":label,"7G Değişim %":round(ret7,2),
        "7G Dip":low7,"7G Zirve":high7,"7G Aralık Konumu %":round(range_pos,1),
        "EMA20 Üstü %":round(above20,1),"EMA50 Üstü %":round(above50,1),
        "EMA20 Eğimi %":round(ema_slope,2),"Son24S Hacim Değişim %":round(vol_change,1),
        "7G Maks Drawdown %":round(max_dd,2),"7G Neden":", ".join(reasons) if reasons else "Nötr"
    }


STABLE_BASES = {"USDT","USDC","FDUSD","TUSD","DAI","EUR","TRY"}

def get_24h_try_volume(item):
    df = fetch_klines(item["symbol"], "4h", 12, item["type"])
    if "quote_volume" in df.columns and df["quote_volume"].notna().any():
        qv = pd.to_numeric(df["quote_volume"], errors="coerce").fillna(0)
        vol = float(qv.tail(min(6, len(qv))).sum())
    else:
        tail = df.tail(min(6, len(df))).copy()
        vol = float((tail["volume"] * tail["close"]).sum())
    return vol, float(df.iloc[-1]["close"])

def rank_by_24h_volume(symbols, workers=6):
    rows = []
    selected = [x for x in symbols if x["baseAsset"].upper() not in STABLE_BASES]
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futmap = {ex.submit(get_24h_try_volume, item): item for item in selected}
        for fut in as_completed(futmap):
            item = futmap[fut]
            try:
                vol, price = fut.result()
                rows.append({
                    "Coin": item["baseAsset"],
                    "Parite": item["symbol"],
                    "Tip": item["type"],
                    "24S Hacim TRY": vol,
                    "Fiyat": price,
                    "_item": item
                })
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("24S Hacim TRY", ascending=False).reset_index(drop=True)

st.title("📱 Binance TR v20 Cloud Mobile")
st.caption("☁️ Bulutta çalışır • iPhone’dan bilgisayarsız kullanılabilir.")
st.caption("TRY paritelerini otomatik bulur, teknik göstergelerle puanlar. Gerçek emir göndermez.")

try:
    symbols = get_try_symbols()
except Exception as e:
    st.error(f"Binance TR işlem çiftleri alınamadı: {e}")
    st.stop()

tabs = st.tabs(["📡 Canlı Sinyal", "✅ Emir Onayı", "📉 3 Aylık Dip Avcısı", "🏆 Yüksek Hacimli AL", "💰 Sermaye Yönetimi", "🎯 Dip Avcısı", "🔥 Otomatik Tarayıcı", "💼 Pozisyonum", "📊 Detaylı Grafik", "🧮 Risk Hesabı"])


with tabs[0]:
    st.subheader("📡 Canlı Alım / Satım Sinyal Motoru")
    st.caption("Trend 15dk + 1 saat + 4 saat birlikte değerlendirilir. Ekran otomatik yenilenir.")

    c1,c2,c3,c4 = st.columns(4)
    live_coin = c1.text_input("Coin", value="COTI", key="live_coin").upper().strip()
    refresh_sec = c2.selectbox("Otomatik yenileme", [30, 60, 120, 300], index=1, format_func=lambda x: f"{x} sn")
    live_qty = c3.number_input("Pozisyon miktarı (isteğe bağlı)", min_value=0.0, value=232044.0, step=1.0)
    live_avg = c4.number_input("Ortalama maliyet (isteğe bağlı)", min_value=0.0, value=0.5793, format="%.8f")

    n1,n2,n3 = st.columns(3)
    alerts_enabled = n1.toggle("🔔 AL/SAT uyarıları", value=True)
    alert_mode = n2.selectbox("Uyarı seviyesi", ["Tüm AL/SAT adayları", "Sadece güçlü sinyaller"], index=0)
    repeat_min = n3.selectbox("Aynı sinyali tekrar uyar", [5, 15, 30, 60], index=1, format_func=lambda x: f"{x} dk sonra")

    live_matches = [x for x in symbols if x["baseAsset"].upper() == live_coin]

    if not live_matches:
        st.error(f"{live_coin}/TRY aktif paritesi bulunamadı.")
    else:
        live_item = live_matches[0]

        @st.fragment(run_every=refresh_sec)
        def live_signal_panel():
            try:
                plan = build_trade_plan(live_item, live_qty, live_avg)

                # AL/SAT uyarı motoru
                if alerts_enabled:
                    signal = plan["action_short"]
                    is_trade_signal = ("AL" in signal or "SAT" in signal) and signal != "BEKLE"
                    if alert_mode == "Sadece güçlü sinyaller":
                        is_trade_signal = signal in ("AL", "SAT") and plan["confidence"] == "Yüksek"

                    key_signal = f"last_signal_{live_coin}"
                    key_time = f"last_alert_time_{live_coin}"
                    now_ts = time.time()
                    last_signal = st.session_state.get(key_signal)
                    last_time = float(st.session_state.get(key_time, 0.0))
                    signal_changed = signal != last_signal
                    repeat_due = (now_ts - last_time) >= repeat_min * 60

                    if is_trade_signal and (signal_changed or repeat_due):
                        send_signal_alert(
                            live_coin,
                            signal,
                            plan["price"],
                            f"{plan['reason']} | Birleşik puan {plan['combined']:+.2f}"
                        )
                        if "SAT" in signal:
                            st.toast(f"🔴 {live_coin}: {signal} uyarısı! Fiyat ₺{plan['price']:.4f}", icon="⚠️")
                        else:
                            st.toast(f"🟢 {live_coin}: {signal} uyarısı! Fiyat ₺{plan['price']:.4f}", icon="🔔")
                        st.session_state[key_time] = now_ts

                    st.session_state[key_signal] = signal

                st.markdown("### Ana karar")
                a,b,c,d = st.columns(4)
                a.metric("Anlık fiyat", f"₺{plan['price']:,.4f}")
                b.metric("Sinyal", plan["action_short"])
                c.metric("Birleşik puan", f"{plan['combined']:+.2f}")
                d.metric("Teyit gücü", plan["confidence"])

                if "AL" in plan["action_short"]:
                    st.success(plan["action"] + " — " + plan["reason"])
                elif "SAT" in plan["action_short"]:
                    st.error(plan["action"] + " — " + plan["reason"])
                else:
                    st.warning(plan["action"] + " — " + plan["reason"])

                st.markdown("### Zaman dilimi teyidi")
                t1,t2,t3 = st.columns(3)
                t1.metric("15 dakika", f"{plan['p15']:+.0f}", f"RSI {plan['rsi15']:.1f}")
                t2.metric("1 saat", f"{plan['p1']:+.0f}", f"RSI {plan['rsi1']:.1f}")
                t3.metric("4 saat", f"{plan['p4']:+.0f}", f"RSI {plan['rsi4']:.1f}")

                st.markdown("### Son 7 gün ana trend filtresi")
                try:
                    d7_live=analyze_7day(live_item)
                    g1,g2,g3,g4=st.columns(4)
                    g1.metric("7G trend",d7_live["7G Trend"])
                    g2.metric("7G puan",f"{d7_live['7G Puan']:+d}")
                    g3.metric("7G fiyat değişimi",f"%{d7_live['7G Değişim %']:+.2f}")
                    g4.metric("7G aralık konumu",f"%{d7_live['7G Aralık Konumu %']:.1f}")
                    h1,h2,h3=st.columns(3)
                    h1.metric("EMA20 üstünde",f"%{d7_live['EMA20 Üstü %']:.1f}")
                    h2.metric("Son24S hacim değişimi",f"%{d7_live['Son24S Hacim Değişim %']:+.1f}")
                    h3.metric("7G max drawdown",f"%{d7_live['7G Maks Drawdown %']:.2f}")
                    st.caption("7G analiz: "+d7_live["7G Neden"])
                except Exception as e:
                    st.warning(f"7 günlük analiz alınamadı: {e}")

                st.markdown("### Emir defteri baskısı")
                try:
                    ob_live=order_book_pressure(live_item["symbol"],live_item["type"],1.0,100)
                    o1,o2,o3,o4=st.columns(4)
                    o1.metric("Alış baskısı",f"%{ob_live['bid_pct']:.1f}")
                    o2.metric("Satış baskısı",f"%{ob_live['ask_pct']:.1f}")
                    o3.metric("Alış/Satış",f"{ob_live['ratio']:.2f}x")
                    o4.metric("Spread",f"%{ob_live['spread_pct']:.4f}")
                    if ob_live["bid_pct"]>=60: st.success(ob_live["label"])
                    elif ob_live["ask_pct"]>=60: st.error(ob_live["label"])
                    else: st.warning(ob_live["label"])
                    st.caption(f"Yakın ±%1 bant • Alış duvarı ₺{ob_live['bid_wall_price']:.4f} • Satış duvarı ₺{ob_live['ask_wall_price']:.4f}")
                except Exception as e:
                    st.warning(f"Emir defteri okunamadı: {e}")

                st.markdown("### İşlem bölgeleri")
                x1,x2,x3,x4 = st.columns(4)
                x1.metric("Alım bölgesi alt", f"₺{plan['entry_low']:,.4f}")
                x2.metric("Alım bölgesi üst", f"₺{plan['entry_high']:,.4f}")
                x3.metric("Yapısal stop", f"₺{plan['stop']:,.4f}")
                x4.metric("4S volatilite", f"%{plan['atr_pct']:.2f}")

                y1,y2,y3 = st.columns(3)
                y1.metric("Hedef 1", f"₺{plan['target1']:,.4f}")
                y2.metric("Hedef 2", f"₺{plan['target2']:,.4f}")
                y3.metric("Hedef 3", f"₺{plan['target3']:,.4f}")

                st.markdown("### Yakın destek / direnç")
                z1,z2,z3,z4 = st.columns(4)
                z1.metric("Destek 1", f"₺{plan['support']:,.4f}")
                z2.metric("Destek 2", f"₺{plan['support2']:,.4f}")
                z3.metric("Direnç 1", f"₺{plan['resistance']:,.4f}")
                z4.metric("Direnç 2", f"₺{plan['resistance2']:,.4f}")

                if plan["pnl"] is not None:
                    st.markdown("### Mevcut pozisyon")
                    q1,q2 = st.columns(2)
                    q1.metric("Anlık K/Z", f"₺{plan['pnl']:,.2f}", f"{plan['pnl_pct']:+.2f}%")
                    q2.metric("Maliyet", f"₺{live_avg:,.4f}")

                st.caption(
                    f"Son otomatik kontrol: {pd.Timestamp.now(tz='Europe/Istanbul').strftime('%H:%M:%S')} • "
                    f"Her {refresh_sec} saniyede yeniden analiz edilir."
                )
            except Exception as e:
                st.error(f"Canlı analiz alınamadı: {e}")

        live_signal_panel()

    st.info(
        "🔔 Uyarılar açıksa sinyal AL/SAT yönüne geçtiğinde Windows masaüstü bildirimi ve ses verir. "
        "Programın siyah penceresi açık kalmalıdır. Teknik analiz garanti vermez; gerçek emir göndermez."
    )





with tabs[1]:
    st.subheader("✅ Emir Onayı — İşlem Planı")
    st.caption("Gerçek emir göndermez. Sinyal, miktar, stop, hedef ve maliyetleri hazırlayıp sana kontrol ettirir.")

    e1,e2,e3,e4 = st.columns(4)
    order_coin = e1.text_input("Coin", value="COTI", key="order_coin").upper().strip()
    order_capital = e2.number_input("Toplam sermaye (TL)", min_value=1000.0, value=130000.0, step=1000.0, key="order_capital")
    order_risk_pct = e3.selectbox("İşlem başına risk", [0.5,1.0,1.5,2.0], index=1, format_func=lambda x:f"%{x}", key="order_risk")
    order_max_alloc = e4.selectbox("Tek coine maksimum", [20,25,35,50], index=2, format_func=lambda x:f"%{x}", key="order_alloc")

    o1,o2,o3 = st.columns(3)
    order_buy_type = o1.selectbox("Alış tipi", ["Maker (Limit)","Taker (Piyasa)"], index=0, key="order_buy_type")
    order_sell_type = o2.selectbox("Satış tipi", ["Maker (Limit)","Taker (Piyasa)"], index=1, key="order_sell_type")
    order_slip = o3.number_input("Tahmini slippage (%)", min_value=0.0, max_value=5.0, value=0.00, step=0.01, format="%.3f", key="order_slip")

    order_buy_fee = 0.075 if order_buy_type.startswith("Maker") else 0.1125
    order_sell_fee = 0.075 if order_sell_type.startswith("Maker") else 0.1125

    matches = [x for x in symbols if x["baseAsset"].upper() == order_coin]
    if not matches:
        st.error(f"{order_coin}/TRY aktif paritesi bulunamadı.")
    else:
        try:
            item = matches[0]
            plan = build_trade_plan(item, 0, 0)
            cp = capital_plan(order_capital, order_risk_pct, plan["price"], plan["stop"],
                              plan["target1"], plan["target2"], plan["target3"], order_max_alloc)
            if cp is None:
                st.warning("Geçerli stop mesafesi oluşmadı.")
            else:
                entry=float(plan["price"]); qty=float(cp["qty"]); stop=float(plan["stop"])
                t1=float(plan["target1"]); t2=float(plan["target2"]); t3=float(plan["target3"])

                stop_result = trading_costs(entry, stop, qty, order_buy_fee, order_sell_fee, order_slip)
                be = trading_costs(entry, entry, qty, order_buy_fee, order_sell_fee, order_slip)

                st.markdown("### Sinyal ve trend")
                a,b,c,d = st.columns(4)
                a.metric("Sinyal", plan["action_short"])
                b.metric("Birleşik puan", f"{plan['combined']:+.2f}")
                c.metric("15dk / 1S / 4S", f"{plan['p15']:+.0f} / {plan['p1']:+.0f} / {plan['p4']:+.0f}")
                try:
                    d7 = analyze_7day(item)
                    d.metric("7G trend", d7["7G Trend"])
                except Exception:
                    d.metric("7G trend", "Veri yok")

                st.markdown("### Hazır emir bileti")
                b1,b2,b3,b4=st.columns(4)
                b1.metric("Yön","AL")
                b2.metric("Referans giriş",f"₺{entry:,.6f}")
                b3.metric("Miktar",f"{qty:,.4f} {order_coin}")
                b4.metric("Pozisyon",f"₺{cp['position_value']:,.2f}")

                c1,c2,c3,c4=st.columns(4)
                c1.metric("Stop",f"₺{stop:,.6f}")
                c2.metric("Stopta net sonuç",f"₺{stop_result['net_pnl']:,.2f}")
                c3.metric("Net başa baş",f"₺{be['breakeven']:,.6f}")
                c4.metric("Sermaye riski",f"%{cp['risk_pct_capital']:.2f}")

                rows=[]
                for name,px in [("Hedef 1",t1),("Hedef 2",t2),("Hedef 3",t3)]:
                    r=trading_costs(entry,px,qty,order_buy_fee,order_sell_fee,order_slip)
                    rows.append({"Hedef":name,"Fiyat":px,"Net Kâr (TL)":r["net_pnl"],"Net Kâr %":r["net_pnl_pct"],"Komisyon (TL)":r["total_fees"]})
                st.markdown("### Hedefler — net")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                try:
                    ob=order_book_pressure(item["symbol"], item["type"], 1.0, 100)
                    ob_ok=ob["bid_pct"]>=55
                    ob_text=ob["label"]
                except Exception:
                    ob_ok=False; ob_text="Veri yok"
                try:
                    d7=analyze_7day(item); d7_ok=d7["7G Puan"]>=1
                except Exception:
                    d7_ok=False

                checks=[
                    ("4S trend pozitif",plan["p4"]>=3),
                    ("1S trend pozitif",plan["p1"]>=2),
                    ("15dk giriş teyidi",plan["p15"]>=1),
                    ("7G trend uygun",d7_ok),
                    ("Emir defteri alış baskısı yeterli",ob_ok),
                    ("Stop riski sınır içinde",cp["risk_pct_capital"]<=order_risk_pct+0.01),
                ]
                passed=sum(int(v) for _,v in checks)
                st.markdown("### Onay kontrolü")
                for name,ok in checks:
                    st.write(f"{'✅' if ok else '❌'} {name}")
                if passed==len(checks):
                    st.success("Tüm kontroller geçti. Bu yine de kâr garantisi değildir.")
                elif passed>=4:
                    st.warning(f"{passed}/{len(checks)} koşul geçti.")
                else:
                    st.error(f"{passed}/{len(checks)} koşul geçti. Risk yüksek.")
                st.caption(f"Emir defteri: {ob_text}")

                ticket={
                    "coin":order_coin,"pair":item["symbol"],"side":"BUY",
                    "buy_type":order_buy_type,"sell_type":order_sell_type,
                    "entry_try":round(entry,8),"quantity":round(qty,8),
                    "position_value_try":round(cp["position_value"],2),
                    "stop_try":round(stop,8),"target1_try":round(t1,8),
                    "target2_try":round(t2,8),"target3_try":round(t3,8),
                    "buy_fee_pct":order_buy_fee,"sell_fee_pct":order_sell_fee,
                    "slippage_pct":order_slip,"signal":plan["action_short"],
                    "combined_score":round(plan["combined"],2),
                    "checks_passed":f"{passed}/{len(checks)}"
                }
                st.download_button("📄 Emir planını JSON indir",
                                   data=json.dumps(ticket,ensure_ascii=False,indent=2),
                                   file_name=f"{order_coin.lower()}_emir_plani.json",
                                   mime="application/json",
                                   use_container_width=True)

                st.warning("Gerçek emir gönderme kapalıdır. Emir planını Binance TR ekranında sen kontrol edip uygularsın.")
        except Exception as e:
            st.error(f"Emir planı hazırlanamadı: {e}")


with tabs[2]:
    st.subheader("📉 3 Aylık Dip Avcısı")
    st.caption(
        "Aktif TRY paritelerinde son 90 günlük en düşük fiyatı bulur. "
        "Dibe yakınlık + 24S hacim + 15dk/1S/4S + 7G dönüş teyidini birlikte değerlendirir."
    )

    q1,q2,q3,q4 = st.columns(4)
    dip3m_max = q1.selectbox(
        "Maksimum 3A dip uzaklığı",
        [3,5,7,10,15,20,30],
        index=4,
        format_func=lambda x: f"%{x}"
    )
    dip3m_min_vol = q2.selectbox(
        "Minimum 24S hacim",
        [0,10,25,50,100,250],
        index=2,
        format_func=lambda x: "Filtre yok" if x == 0 else f"{x} milyon TL"
    )
    dip3m_top = q3.selectbox("Gösterilecek coin", [10,20,30,50], index=1, key="dip3m_top")
    dip3m_workers = q4.selectbox("Tarama hızı", [3,4,6,8], index=2, key="dip3m_workers")

    if st.button("📉 SON 3 AYIN DİBİNDEKİLERİ TARA", type="primary", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()
        stable_bases = {"USDT","USDC","FDUSD","TUSD","DAI","EUR","TRY"}
        candidates = [x for x in symbols if x["baseAsset"].upper() not in stable_bases]
        rows = []
        total = len(candidates)

        def scan_3m(item):
            d90 = fetch_klines(item["symbol"], "1d", 95, item["type"])
            if len(d90) < 30:
                return None
            d90 = d90.tail(90).copy()

            current = float(d90.iloc[-1]["close"])
            low90 = float(d90["low"].min())
            high90 = float(d90["high"].max())
            dist = (current / low90 - 1) * 100 if low90 > 0 else np.nan
            range_pos = ((current - low90) / (high90 - low90) * 100) if high90 > low90 else 50.0

            # Yaklaşık 24S TRY işlem hacmi
            d4 = fetch_klines(item["symbol"], "4h", 12, item["type"])
            if "quote_volume" in d4.columns and d4["quote_volume"].notna().any():
                qv = pd.to_numeric(d4["quote_volume"], errors="coerce").fillna(0)
                vol24 = float(qv.tail(min(6, len(qv))).sum())
            else:
                tail = d4.tail(min(6, len(d4)))
                vol24 = float((tail["volume"] * tail["close"]).sum())

            try:
                tech = scan_multi_one(item, 250)
            except Exception:
                tech = None

            row = {
                "Coin": item["baseAsset"],
                "Parite": item["symbol"],
                "Fiyat": current,
                "3A Dip": low90,
                "3A Zirve": high90,
                "3A Dip Uzaklığı %": round(dist, 2),
                "3A Aralık Konumu %": round(range_pos, 1),
                "24S Hacim TRY": vol24
            }

            if tech:
                for k in [
                    "Sinyal","7G Puan","7G Trend","7G Değişim %",
                    "15dk Puan","1S Puan","4S Puan","Birleşik Puan",
                    "Hedef Potansiyeli %","Risk/Getiri","4S Destek","4S Direnç"
                ]:
                    if k in tech:
                        row[k] = tech[k]

            return row

        with ThreadPoolExecutor(max_workers=int(dip3m_workers)) as ex:
            futs = [ex.submit(scan_3m, item) for item in candidates]
            done = 0
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if r:
                        rows.append(r)
                except Exception:
                    pass
                done += 1
                prog.progress(done / max(total, 1))
                if done % 10 == 0 or done == total:
                    status.info(f"{done}/{total} parite tarandı...")

        if not rows:
            status.error("90 günlük veri alınamadı.")
        else:
            df = pd.DataFrame(rows)

            if dip3m_min_vol > 0:
                df = df[df["24S Hacim TRY"] >= dip3m_min_vol * 1_000_000]

            df = df[df["3A Dip Uzaklığı %"] <= dip3m_max].copy()

            def dip_label(x):
                if x <= 3:
                    return "🔴 DİBİN DİBİNDE"
                if x <= 7:
                    return "🟠 DİBE ÇOK YAKIN"
                if x <= 15:
                    return "🟡 DİBE YAKIN"
                return "⚪ DİPTEN UZAK"

            df["3A Durum"] = df["3A Dip Uzaklığı %"].apply(dip_label)

            for c in ["15dk Puan","1S Puan","4S Puan","7G Puan","Risk/Getiri","Birleşik Puan"]:
                if c not in df.columns:
                    df[c] = 0
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            # Dibe yakın ama dönüş teyidi de alanları öne çıkar
            df["Dönüş Skoru"] = (
                (dip3m_max - df["3A Dip Uzaklığı %"]).clip(lower=0) * 1.5
                + df["15dk Puan"] * 2
                + df["1S Puan"] * 3
                + df["4S Puan"] * 4
                + df["7G Puan"] * 2
                + df["Risk/Getiri"].clip(0, 5) * 2
                + df["Birleşik Puan"] * 2
                + np.log10(df["24S Hacim TRY"].clip(lower=1)) * 2
            )

            # Önce dibe yakınlık, eşit durumda dönüş kalitesi
            df = df.sort_values(
                ["3A Dip Uzaklığı %","Dönüş Skoru","24S Hacim TRY"],
                ascending=[True,False,False]
            ).head(dip3m_top)

            st.session_state["dip3m_df"] = df
            status.success(f"Tarama tamamlandı. {len(df)} coin gösteriliyor.")

    if "dip3m_df" in st.session_state:
        df = st.session_state["dip3m_df"].copy()
        if df.empty:
            st.info("Seçilen kriterlerde 3 aylık dibine yakın coin bulunamadı.")
        else:
            df.insert(0, "Sıra", range(1, len(df)+1))
            cols = [
                "Sıra","Coin","3A Durum","3A Dip Uzaklığı %","Fiyat","3A Dip","3A Zirve",
                "24S Hacim TRY","Sinyal","Dönüş Skoru","7G Trend","7G Değişim %",
                "15dk Puan","1S Puan","4S Puan","Birleşik Puan",
                "Risk/Getiri","Hedef Potansiyeli %","4S Destek","4S Direnç"
            ]
            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
            st.caption(
                "Dibe yakın olmak tek başına AL sinyali değildir. "
                "Dönüş Skoru; dibe yakınlık, hacim, 15dk/1S/4S/7G ve risk/getiri teyidini birlikte kullanır."
            )

with tabs[3]:
    st.subheader("🏆 Yüksek Hacimli Coinler — Önce Likidite, Sonra AL Sinyali")
    st.caption("Önce 24S TRY işlem hacmine göre sıralar; sonra yalnızca en yüksek hacimli coinlerde 7G + 4S + 1S + 15dk analiz yapar.")

    h1,h2,h3,h4 = st.columns(4)
    top_volume_n = h1.selectbox("Hacimde ilk kaç coin?", [10,20,30,50,75,100], index=3)
    min_vol_m = h2.selectbox("Minimum 24S hacim", [0,10,25,50,100,250,500], index=2,
                             format_func=lambda x: "Filtre yok" if x==0 else f"{x} milyon TL")
    hv_top = h3.selectbox("Sonuç sayısı", [10,20,30,50], index=1, key="hv_top")
    hv_workers = h4.selectbox("Tarama hızı", [3,4,6,8], index=2, key="hv_workers")

    if st.button("🏆 EN ÇOK İŞLEM YAPILANLARI ANALİZ ET", type="primary", use_container_width=True):
        phase = st.empty()
        prog = st.progress(0)
        phase.info("1/2 — 24 saatlik işlem hacimleri sıralanıyor...")
        vol_df = rank_by_24h_volume(symbols, workers=hv_workers)

        if vol_df.empty:
            st.error("24 saatlik hacim verisi alınamadı.")
        else:
            if min_vol_m > 0:
                vol_df = vol_df[vol_df["24S Hacim TRY"] >= min_vol_m * 1_000_000]
            selected = vol_df.head(top_volume_n).copy()
            st.session_state["top_volume_table"] = selected.drop(columns=["_item"], errors="ignore")

            phase.info(f"2/2 — En yüksek hacimli {len(selected)} coin teknik olarak analiz ediliyor...")
            rows = []
            total = len(selected)
            with ThreadPoolExecutor(max_workers=int(hv_workers)) as ex:
                futs = [ex.submit(scan_multi_one, row["_item"], 250) for _, row in selected.iterrows()]
                done = 0
                for fut in as_completed(futs):
                    try:
                        r = fut.result()
                        if r:
                            rows.append(r)
                    except Exception:
                        pass
                    done += 1
                    prog.progress(done/max(total,1))

            if rows:
                res = pd.DataFrame(rows)
                volume_map = dict(zip(selected["Coin"], selected["24S Hacim TRY"]))
                res["24S Hacim TRY"] = res["Coin"].map(volume_map)

                buy = res[(res["Birleşik Puan"] >= 2.0) & (res["4S Puan"] >= 2) & (res["7G Puan"] >= 1)].copy()
                if not buy.empty:
                    buy["Hacim Skoru"] = np.log10(buy["24S Hacim TRY"].clip(lower=1))
                    ch7 = pd.to_numeric(buy["7G Değişim %"], errors="coerce").fillna(0)

                    buy["7G Momentum Bonusu"] = 0.0
                    ideal = (ch7 >= 10) & (ch7 <= 30)
                    buy.loc[ideal, "7G Momentum Bonusu"] = 6.0

                    buy["7G Aşırılık Cezası"] = 0.0
                    m35 = ch7 > 35
                    buy.loc[m35, "7G Aşırılık Cezası"] = (ch7[m35] - 35) * 0.70
                    buy.loc[ch7 > 50, "7G Aşırılık Cezası"] += 8.0
                    buy.loc[ch7 > 75, "7G Aşırılık Cezası"] += 12.0

                    buy["Aşırılık Uyarısı"] = "🟢 NORMAL"
                    buy.loc[(ch7 > 30) & (ch7 <= 35), "Aşırılık Uyarısı"] = "🟡 ISINMIŞ"
                    buy.loc[(ch7 > 35) & (ch7 <= 50), "Aşırılık Uyarısı"] = "🟠 GERİ ÇEKİLME BEKLE"
                    buy.loc[ch7 > 50, "Aşırılık Uyarısı"] = "🔴 AŞIRI YÜKSELMİŞ"

                    if "Getiri Skoru" in buy.columns:
                        buy["Nihai Skor"] = buy["Getiri Skoru"] + buy["Hacim Skoru"]*6 + buy["7G Momentum Bonusu"] - buy["7G Aşırılık Cezası"]
                    else:
                        buy["Nihai Skor"] = buy["Birleşik Puan"]*10 + buy["7G Puan"]*4 + buy["Hacim Skoru"]*6 + buy["7G Momentum Bonusu"] - buy["7G Aşırılık Cezası"]

                    buy.loc[ch7 > 35, "Sinyal"] = "🟠 GERİ ÇEKİLME BEKLE"

                    buy = buy.sort_values(
                        ["Nihai Skor","24S Hacim TRY","7G Puan","4S Puan"],
                        ascending=[False,False,False,False]
                    ).head(hv_top)
                st.session_state["high_volume_buy"] = buy
                phase.success(f"Tarama tamamlandı. {len(res)} yüksek hacimli coin analiz edildi.")
            else:
                st.error("Teknik analiz sonucu alınamadı.")

    if "top_volume_table" in st.session_state:
        st.markdown("### 💧 24S işlem hacmi en yüksek coinler")
        vt = st.session_state["top_volume_table"].copy()
        vt.insert(0, "Hacim Sırası", range(1, len(vt)+1))
        st.dataframe(vt[["Hacim Sırası","Coin","Parite","24S Hacim TRY","Fiyat"]],
                     use_container_width=True, hide_index=True)

    if "high_volume_buy" in st.session_state:
        buy = st.session_state["high_volume_buy"].copy()
        st.markdown("### 🟢 Yüksek hacim + teknik AL adayları")
        if buy.empty:
            st.info("Seçilen yüksek hacimli coinlerde yeterli teknik AL teyidi oluşmadı.")
        else:
            buy.insert(0, "Sıra", range(1, len(buy)+1))
            cols = ["Sıra","Coin","Sinyal","Aşırılık Uyarısı","24S Hacim TRY","Nihai Skor","Getiri Skoru",
                    "Hedef Potansiyeli %","Risk/Getiri","7G Puan","7G Trend","7G Değişim %",
                    "7G Aşırılık Cezası","15dk Puan","1S Puan","4S Puan","Birleşik Puan",
                    "Fiyat","4S Destek","4S Direnç"]
            cols = [c for c in cols if c in buy.columns]
            st.dataframe(buy[cols], use_container_width=True, hide_index=True)
            st.caption("Yüksek hacim likiditeyi artırır; AL sinyali için 7G ve çoklu zaman teyidi ayrıca aranır.")

with tabs[4]:
    st.subheader("💰 Sermaye Yönetimi — İşlem Başına Risk")
    st.caption("Pozisyon büyüklüğünü hedef kâra göre değil, stop olursa kaybedilecek maksimum tutara göre hesaplar.")

    a,b,c,d=st.columns(4)
    cm_capital=a.number_input("Toplam sermaye (TL)",min_value=1000.0,value=130000.0,step=1000.0)
    cm_risk=b.selectbox("İşlem başına maksimum risk",[0.5,1.0,1.5,2.0,3.0],index=1,format_func=lambda x:f"%{x}")
    cm_alloc=c.selectbox("Tek coine maksimum sermaye",[20,25,35,50],index=2,format_func=lambda x:f"%{x}")
    cm_coin=d.text_input("Coin",value="COTI",key="cm_coin").upper().strip()

    fc1,fc2,fc3=st.columns(3)
    buy_order_type=fc1.selectbox("Alış emir tipi",["Maker (Limit)","Taker (Piyasa)"],index=0)
    sell_order_type=fc2.selectbox("Satış emir tipi",["Maker (Limit)","Taker (Piyasa)"],index=1)
    slippage_pct=fc3.number_input("Tahmini kayma / slippage (%)",min_value=0.0,max_value=5.0,value=0.00,step=0.01,format="%.3f")
    buy_fee_pct = 0.075 if buy_order_type.startswith("Maker") else 0.1125
    sell_fee_pct = 0.075 if sell_order_type.startswith("Maker") else 0.1125
    st.caption(f"Gümüş tarife • Maker: %0,075 • Taker: %0,1125 • Seçili alış: %{buy_fee_pct:.4f} • satış: %{sell_fee_pct:.4f}")

    matches=[x for x in symbols if x["baseAsset"].upper()==cm_coin]
    if not matches:
        st.error(f"{cm_coin}/TRY bulunamadı.")
    else:
        try:
            plan=build_trade_plan(matches[0],0,0)
            entry=plan["price"]
            stop=plan["stop"]
            cp=capital_plan(cm_capital,cm_risk,entry,stop,plan["target1"],plan["target2"],plan["target3"],cm_alloc)
            if cp is None:
                st.warning("Geçerli stop mesafesi oluşmadı; işlem planı üretilemedi.")
            else:
                st.markdown("### Önerilen pozisyon boyutu")
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Anlık fiyat",f"₺{entry:,.4f}")
                m2.metric("Alınabilecek adet",f"{cp['qty']:,.2f}")
                m3.metric("Pozisyon tutarı",f"₺{cp['position_value']:,.2f}")
                m4.metric("Sermayenin kullanılanı",f"%{cp['position_value']/cm_capital*100:.1f}")

                n1,n2,n3=st.columns(3)
                n1.metric("Stop",f"₺{stop:,.4f}",f"-{cp['stop_pct']:.2f}%")
                n2.metric("Stop olursa zarar",f"₺{cp['risk_cash']:,.2f}")
                n3.metric("Sermayeye risk",f"%{cp['risk_pct_capital']:.2f}")

                st.markdown("### Alım / satım maliyeti")
                cost_now = trading_costs(
                    entry, entry, cp["qty"],
                    buy_fee_pct=buy_fee_pct,
                    sell_fee_pct=sell_fee_pct,
                    slippage_pct=slippage_pct
                )
                cst1,cst2,cst3,cst4=st.columns(4)
                cst1.metric("Alış komisyonu",f"₺{cost_now['buy_fee']:,.2f}")
                cst2.metric("Tahmini satış komisyonu",f"₺{cost_now['sell_fee']:,.2f}")
                cst3.metric("Toplam işlem maliyeti",f"₺{cost_now['total_fees']:,.2f}")
                cst4.metric("Net başa baş fiyatı",f"₺{cost_now['breakeven']:,.6f}")

                st.markdown("### Hedef senaryoları — NET")
                rows=[]
                for name,px,key in [("Hedef 1",plan["target1"],"t1"),("Hedef 2",plan["target2"],"t2"),("Hedef 3",plan["target3"],"t3")]:
                    profit,pct,rr=cp[key]
                    net = trading_costs(
                        entry, px, cp["qty"],
                        buy_fee_pct=buy_fee_pct,
                        sell_fee_pct=sell_fee_pct,
                        slippage_pct=slippage_pct
                    )
                    rows.append({
                        "Hedef":name,
                        "Fiyat":px,
                        "Brüt Kâr (TL)":profit,
                        "Toplam Komisyon (TL)":net["total_fees"],
                        "Net Kâr (TL)":net["net_pnl"],
                        "Net Kâr %":net["net_pnl_pct"],
                        "Fiyat Artışı %":pct,
                        "Risk/Getiri":rr
                    })
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

                st.markdown("### İşleme giriş filtresi")
                checks=[
                    ("4S trend",plan["p4"]>=3),
                    ("1S trend",plan["p1"]>=2),
                    ("Birleşik puan",plan["combined"]>=2),
                    ("Hedef 1 risk/getiri ≥ 1",cp["t1"][2]>=1),
                ]
                passed=sum(int(v) for _,v in checks)
                st.write(" • ".join([f"{'✅' if v else '❌'} {k}" for k,v in checks]))
                if passed==4:
                    st.success("Teknik ve risk filtresi geçildi. Yine de işlem sonucu garanti değildir.")
                else:
                    st.warning(f"Filtre {passed}/4. Tüm koşullar oluşmadan pozisyon büyütmek daha yüksek risk taşır.")

                st.info(
                    "130.000 TL'yi 30 günde 1.000.000 TL yapma hedefi bu hesaplamada kullanılmaz. "
                    "Pozisyon boyutu yalnızca sermayeyi korumaya yönelik maksimum zarar sınırından hesaplanır."
                )
        except Exception as e:
            st.error(f"Sermaye planı hesaplanamadı: {e}")

with tabs[5]:
    st.subheader("🎯 Dip Avcısı — Dibe Yakın + Dönüş Teyidi")
    st.caption(
        "Sadece fiyatı düşmüş coinleri değil; 30/60 günlük dip bölgesine yakın olup "
        "15dk + 1S + 4S tarafında toparlanma işareti verenleri öne çıkarır."
    )

    d1,d2,d3 = st.columns(3)
    dip_max = d1.selectbox("Maksimum 60G dip uzaklığı", [5,10,15,20,30], index=2, format_func=lambda x: f"%{x}")
    dip_top = d2.selectbox("Gösterilecek coin", [10,20,30,50], index=1, key="dip_top")
    dip_workers = d3.selectbox("Tarama hızı", [2,3,4,6], index=2, key="dip_workers")

    f1,f2,f3 = st.columns(3)
    min_volume_m = f1.selectbox(
        "Minimum 24S işlem hacmi",
        [0, 1, 5, 10, 25, 50, 100, 250],
        index=3,
        format_func=lambda x: "Filtre yok" if x == 0 else f"{x} milyon TL"
    )
    min_target_pct = f2.selectbox(
        "Minimum hedef potansiyeli",
        [0, 2, 3, 5, 7, 10, 15],
        index=3,
        format_func=lambda x: "Filtre yok" if x == 0 else f"%{x}"
    )
    min_confirm = f3.selectbox(
        "Minimum dönüş teyidi",
        [2,3,4,5,6],
        index=3,
        format_func=lambda x: f"{x}/6"
    )

    if st.button("🎯 DİBE YAKIN COİNLERİ TARA", type="primary", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()
        rows = []
        total = len(symbols)
        done = 0
        with ThreadPoolExecutor(max_workers=int(dip_workers)) as ex:
            futs = [ex.submit(scan_dip_one, item) for item in symbols]
            for f in as_completed(futs):
                r = f.result()
                if r:
                    rows.append(r)
                done += 1
                prog.progress(done/max(total,1))
                if done % 3 == 0 or done == total:
                    status.caption(f"Dip taraması: {done}/{total} — başarılı: {len(rows)}")

        if rows:
            st.session_state["dip_results"] = pd.DataFrame(rows)
            status.success(f"Dip taraması tamamlandı: {len(rows)} coin.")
        else:
            st.error("Dip taraması sonucu alınamadı.")

    if "dip_results" in st.session_state:
        dipres = st.session_state["dip_results"].copy()
        filtered = dipres[dipres["60G Dipten Uzaklık %"] <= dip_max].copy()

        # Yakın 4S dirençten hedef potansiyeli hesapla.
        # Tarama sonucunda 4S Direnç yoksa, yaklaşık hedef olarak 30G dibinden +20% değil;
        # mevcut trend verisinden güvenli biçimde NaN bırakmak yerine pivot tabanlı direnç için
        # scan_dip_one içinde r4["Direnç"] kullanılıyor.
        if "Hedef Potansiyeli %" not in filtered.columns:
            filtered["Hedef Potansiyeli %"] = np.nan

        # 24S hacim filtresi
        if min_volume_m > 0:
            filtered = filtered[filtered["24S Hacim TRY"] >= min_volume_m * 1_000_000]

        # Dönüş teyidi filtresi
        filtered["Teyit Sayısı"] = filtered["Dönüş Teyidi"].str.split("/").str[0].astype(int)
        filtered = filtered[filtered["Teyit Sayısı"] >= min_confirm]

        # Hedef potansiyeli filtresi
        if min_target_pct > 0:
            filtered = filtered[filtered["Hedef Potansiyeli %"] >= min_target_pct]

        # Likidite + potansiyel + teyit ortak skoru
        filtered["Likidite Skoru"] = np.log10(filtered["24S Hacim TRY"].clip(lower=1))
        filtered["Fırsat Skoru"] = (
            filtered["Dip Avcısı Skoru"]
            + filtered["Likidite Skoru"] * 5
            + filtered["Hedef Potansiyeli %"].fillna(0) * 1.5
            + filtered["Teyit Sayısı"] * 4
        )

        filtered = filtered.sort_values(
            ["Fırsat Skoru","24S Hacim TRY","Dönüş Teyidi","60G Dipten Uzaklık %"],
            ascending=[False,False,False,True]
        ).head(dip_top)

        if filtered.empty:
            st.info("Seçilen dip uzaklığı filtresinde coin bulunamadı.")
        else:
            filtered.insert(0, "Sıra", range(1, len(filtered)+1))
            st.dataframe(
                filtered[[
                    "Sıra","Coin","Durum","Fırsat Skoru","Dip Avcısı Skoru","Fiyat",
                    "24S Hacim TRY","24S Hacim Değişim %","Hedef Potansiyeli %","Yakın 4S Direnç",
                    "30G Dip","30G Dipten Uzaklık %","60G Dip","60G Dipten Uzaklık %",
                    "60G Aralık Konumu %","Dönüş Teyidi","Hacim Teyidi","Hacim Oranı",
                    "15dk Puan","1S Puan","4S Puan","Birleşik Puan","4S RSI"
                ]],
                use_container_width=True, hide_index=True
            )
            st.caption(
                "Dipten Uzaklık % ne kadar düşükse fiyat son 60 günlük dibe o kadar yakındır. "
                "Fırsat Skoru; yüksek 24S TRY hacmi, hedef potansiyeli, dip yakınlığı ve dönüş teyidini birlikte kullanır. "
                "Düşük hacimli coinleri minimum hacim filtresiyle elemek için 10–50 milyon TL gibi bir alt sınır seçebilirsin."
            )

with tabs[6]:
    st.subheader("15dk + 1S + 4S Çoklu Zaman Tarayıcı")
    st.caption("Ağırlıklar: 15dk %20 • 1 saat %35 • 4 saat %45. 4 saatlik grafik ana yön filtresidir.")

    c1,c2,c3 = st.columns(3)
    candle_count = c1.selectbox("Her zaman diliminde mum sayısı", [120,200,250,300], index=2)
    top_n = c2.selectbox("Gösterilecek sonuç", [10,20,30,50], index=1)
    workers = c3.selectbox("Tarama hızı", [2,3,4,6], index=2)

    st.write(f"Bulunan aktif TRY paritesi: **{len(symbols)}**")
    gps_matches = [x for x in symbols if x["baseAsset"].upper()=="GPS"]
    if gps_matches:
        st.success("GPS/TRY bulundu. GPS de 15dk + 1S + 4S birlikte analiz edilecek.")

    if st.button("🔎 ÇOKLU ZAMAN TARAMASINI BAŞLAT", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        rows = []
        total = len(symbols)
        done = 0
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futs = [ex.submit(scan_multi_one, item, candle_count) for item in symbols]
            for f in as_completed(futs):
                r = f.result()
                if r:
                    rows.append(r)
                done += 1
                progress.progress(done/max(total,1))
                if done % 3 == 0 or done == total:
                    status.caption(f"Taranıyor: {done}/{total} — başarılı: {len(rows)}")

        if not rows:
            st.error("Tarama sonucu alınamadı.")
        else:
            res = pd.DataFrame(rows).sort_values(
                ["Birleşik Puan","4S Puan","1S Puan"], ascending=[False,False,False]
            )
            st.session_state["multi_results"] = res
            status.success(f"Çoklu zaman taraması tamamlandı: {len(res)} coin.")

    if "multi_results" in st.session_state:
        res = st.session_state["multi_results"].copy()

        st.markdown("### 🏆 En yüksek getiri potansiyelli AL adayları")
        ranked=res[(res["Birleşik Puan"]>=2.0)&(res["4S Puan"]>=2)&(res["Hedef Potansiyeli %"]>0)].copy()
        if ranked.empty:
            st.info("Şu anda filtreleri geçen getiri potansiyelli AL adayı bulunmadı.")
        else:
            ranked=ranked.sort_values(["AL Uygunluk","Getiri Skoru","Risk/Getiri"],ascending=[False,False,False]).head(top_n)
            ranked.insert(0,"Sıra",range(1,len(ranked)+1))
            st.dataframe(ranked[["Sıra","Coin","Yıldız","Karar","Sinyal","Getiri Skoru","Hedef Potansiyeli %","Desteğe Risk %","Risk/Getiri","Birleşik Puan","7G Puan","7G Trend","7G Değişim %","7G Aralık Konumu %","15dk Puan","1S Puan","4S Puan","Fiyat","4S Destek","4S Direnç"]],use_container_width=True,hide_index=True)
            st.caption("⭐⭐⭐⭐⭐: 4S ve 1S pozitif + 15dk teyidi + en az 2:1 risk/getiri + en az %2 hedef potansiyeli.")
            best=ranked[ranked["AL Uygunluk"]>=4]
            if not best.empty:
                st.success("Öne çıkan teyitli adaylar: "+", ".join(best["Coin"].head(5).astype(str).tolist()))

        st.markdown("### 🟢 Çoklu zaman AL adayları")
        buy = res[res["Birleşik Puan"] >= 2.5].head(top_n)
        if buy.empty:
            st.info("15dk + 1S + 4S birlikte teyit edilen güçlü AL adayı çıkmadı.")
        else:
            st.dataframe(
                buy[["Coin","Parite","Sinyal","Birleşik Puan","15dk Puan","1S Puan","4S Puan",
                     "15dk RSI","1S RSI","4S RSI","Fiyat","4S Destek","4S Direnç","4S ATR %","Teyit"]],
                use_container_width=True, hide_index=True
            )

        st.markdown("### 🔴 Çoklu zaman SAT / zayıf adaylar")
        sell = res.sort_values(["Birleşik Puan","4S Puan"]).head(min(top_n,len(res)))
        st.dataframe(
            sell[["Coin","Parite","Sinyal","Birleşik Puan","15dk Puan","1S Puan","4S Puan",
                  "Fiyat","4S Destek","4S Direnç","Teyit"]],
            use_container_width=True, hide_index=True
        )

        st.markdown("### 🔍 Belirli coin")
        search_coin = st.text_input("Coin adı (örn. COTI, GPS, BTC)", value="COTI").upper().strip()
        if search_coin:
            found = res[res["Coin"].str.upper() == search_coin]
            if not found.empty:
                st.dataframe(found, use_container_width=True, hide_index=True)
                row = found.iloc[0]
                st.info(
                    f"{search_coin}: 15dk {row['15dk Puan']:+.0f} | "
                    f"1S {row['1S Puan']:+.0f} | 4S {row['4S Puan']:+.0f} | "
                    f"Birleşik {row['Birleşik Puan']:+.2f} → {row['Sinyal']}"
                )
            else:
                st.warning(f"{search_coin} için sonuç bulunamadı.")

        csv = res.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Çoklu zaman sonuçlarını CSV indir", csv, "binance_tr_coklu_zaman_tarama.csv", "text/csv")


with tabs[7]:
    st.subheader("💼 Pozisyonum — Anlık Risk ve Hedef Analizi")
    c1,c2,c3 = st.columns(3)
    pos_coin = c1.text_input("Coin", value="COTI", key="pos_coin").upper().strip()
    qty = c2.number_input("Elimdeki coin miktarı", min_value=0.0, value=232044.0, step=1.0)
    avg = c3.number_input("Ortalama maliyet (TRY)", min_value=0.00000001, value=0.5793, format="%.8f")

    pf1,pf2,pf3=st.columns(3)
    pos_buy_type=pf1.selectbox("Alış emir tipi",["Maker (Limit)","Taker (Piyasa)"],index=0,key="pos_buy_type")
    pos_sell_type=pf2.selectbox("Satış emir tipi",["Maker (Limit)","Taker (Piyasa)"],index=1,key="pos_sell_type")
    pos_slippage=pf3.number_input("Tahmini slippage (%)",min_value=0.0,max_value=5.0,value=0.00,step=0.01,format="%.3f",key="pos_slip")
    pos_buy_fee = 0.075 if pos_buy_type.startswith("Maker") else 0.1125
    pos_sell_fee = 0.075 if pos_sell_type.startswith("Maker") else 0.1125
    st.caption(f"Gümüş tarife • Maker: %0,075 • Taker: %0,1125 • Seçili alış: %{pos_buy_fee:.4f} • satış: %{pos_sell_fee:.4f}")

    matches = [x for x in symbols if x["baseAsset"].upper() == pos_coin]
    if not matches:
        st.error(f"{pos_coin}/TRY aktif paritesi bulunamadı.")
    else:
        item = matches[0]
        try:
            # Ana risk/level analizi 4 saatlik grafikten.
            raw4 = fetch_klines(item["symbol"], "4h", 300, item["type"])
            r4, _ = analyze(raw4)
            current = float(r4["Fiyat"])

            # 15m + 1h + 4h teyidi.
            multi = scan_multi_one(item, 250)
            if multi is None:
                raise RuntimeError("Çoklu zaman analizi alınamadı.")

            cost = qty * avg
            value = qty * current
            pnl = value - cost
            pnl_pct = (current / avg - 1) * 100 if avg else 0
            fee_calc = trading_costs(
                avg, current, qty,
                buy_fee_pct=pos_buy_fee,
                sell_fee_pct=pos_sell_fee,
                slippage_pct=pos_slippage
            )
            breakeven_move = (fee_calc["breakeven"] / current - 1) * 100 if current else 0
            one_kurus = qty * 0.01

            st.markdown("### Pozisyon özeti")
            a,b,c,d,e = st.columns(5)
            a.metric("Anlık fiyat", f"₺{current:,.4f}")
            b.metric("Pozisyon değeri", f"₺{value:,.2f}")
            c.metric("Net Kâr / Zarar", f"₺{fee_calc['net_pnl']:,.2f}", f"{fee_calc['net_pnl_pct']:+.2f}%")
            d.metric("Net başa başa uzaklık", f"{breakeven_move:+.2f}%")
            e.metric("Birleşik sinyal", multi["Sinyal"])

            st.caption(
                f"Brüt alış tutarı: ₺{cost:,.2f} • Alış komisyonu: ₺{fee_calc['buy_fee']:,.2f} • "
                f"Tahmini satış komisyonu: ₺{fee_calc['sell_fee']:,.2f} • "
                f"Net başa baş fiyatı: ₺{fee_calc['breakeven']:,.6f} • "
                f"Fiyattaki her 0,01 TL hareket pozisyonu yaklaşık ₺{one_kurus:,.2f} değiştirir."
            )

            st.markdown("### 15dk + 1S + 4S teyidi")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("15dk puan", f"{multi['15dk Puan']:+.0f}", f"RSI {multi['15dk RSI']:.1f}")
            t2.metric("1S puan", f"{multi['1S Puan']:+.0f}", f"RSI {multi['1S RSI']:.1f}")
            t3.metric("4S puan", f"{multi['4S Puan']:+.0f}", f"RSI {multi['4S RSI']:.1f}")
            t4.metric("Birleşik puan", f"{multi['Birleşik Puan']:+.2f}")

            s1 = float(r4["Destek"])
            s2 = float(r4["Destek2"])
            r1 = float(r4["Direnç"])
            r2 = float(r4["Direnç2"])

            st.markdown("### Yakın teknik bölgeler")
            l1,l2,l3,l4 = st.columns(4)
            l1.metric("Yakın destek", f"₺{s1:,.4f}", f"{(s1/current-1)*100:+.2f}%")
            l2.metric("2. destek", f"₺{s2:,.4f}", f"{(s2/current-1)*100:+.2f}%")
            l3.metric("Yakın direnç", f"₺{r1:,.4f}", f"{(r1/current-1)*100:+.2f}%")
            l4.metric("2. direnç", f"₺{r2:,.4f}", f"{(r2/current-1)*100:+.2f}%")

            # Scenario table: not prescriptive orders, just position impact.
            levels = [
                ("2. destek", s2),
                ("Yakın destek", s1),
                ("Anlık", current),
                ("Maliyet", avg),
                ("Yakın direnç", r1),
                ("2. direnç", r2),
            ]
            rows = []
            for name, px in levels:
                val = qty * px
                pp = val - cost
                rows.append({
                    "Seviye": name,
                    "Fiyat": px,
                    "Pozisyon Değeri": val,
                    "Kâr/Zarar": pp,
                    "Kâr/Zarar %": (px/avg-1)*100 if avg else 0
                })
            st.markdown("### Fiyat senaryoları")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.info(
                "Yakın destek/dirençler artık son mumların yalnızca en düşük/en yüksek değerinden değil, "
                "4 saatlik grafikteki yerel pivotlardan hesaplanır. Bunlar kesin dönüş noktaları değildir."
            )
        except Exception as e:
            st.error(f"Pozisyon analizi alınamadı: {e}")

with tabs[8]:
    st.subheader("Tek Coin Detaylı Analiz")
    names = [x["symbol"] for x in symbols]
    default_idx = names.index("BTC_TRY") if "BTC_TRY" in names else 0
    selected = st.selectbox("Parite", names, index=default_idx)
    detail_interval = st.selectbox("Grafik zaman dilimi", INTERVALS, index=5, key="detail_iv")
    item = next(x for x in symbols if x["symbol"] == selected)

    try:
        raw = fetch_klines(selected, detail_interval, 500, item["type"])
        result, d = analyze(raw)
        x = d.iloc[-1]
        a,b,c,d1,e = st.columns(5)
        a.metric("Sinyal", result["Sinyal"])
        b.metric("Puan", f"{result['Puan']:+d}")
        c.metric("Fiyat", f"₺{fmt(result['Fiyat'])}")
        d1.metric("RSI", f"{result['RSI']:.1f}")
        e.metric("ATR", f"%{result['ATR %']:.2f}")

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=.025,
            row_heights=[.56,.14,.15,.15],
            subplot_titles=("Fiyat / EMA / Bollinger","Hacim","RSI","MACD")
        )
        fig.add_trace(go.Candlestick(
            x=d["time"], open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Mum"
        ), row=1,col=1)
        for col,name in [("EMA20","EMA20"),("EMA50","EMA50"),("EMA200","EMA200"),("BB_UP","BB Üst"),("BB_LOW","BB Alt")]:
            fig.add_trace(go.Scatter(x=d["time"], y=d[col], mode="lines", name=name), row=1,col=1)
        fig.add_hline(y=result["Destek"], line_dash="dot", row=1,col=1)
        fig.add_hline(y=result["Direnç"], line_dash="dot", row=1,col=1)
        fig.add_trace(go.Bar(x=d["time"], y=d["volume"], name="Hacim"), row=2,col=1)
        fig.add_trace(go.Scatter(x=d["time"], y=d["RSI"], mode="lines", name="RSI"), row=3,col=1)
        fig.add_hline(y=70,line_dash="dot",row=3,col=1)
        fig.add_hline(y=30,line_dash="dot",row=3,col=1)
        fig.add_trace(go.Scatter(x=d["time"], y=d["MACD"], mode="lines", name="MACD"), row=4,col=1)
        fig.add_trace(go.Scatter(x=d["time"], y=d["MACD_SIG"], mode="lines", name="Signal"), row=4,col=1)
        fig.add_trace(go.Bar(x=d["time"], y=d["MACD_H"], name="Histogram"), row=4,col=1)
        fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Analiz: " + result["Neden"])
    except Exception as e:
        st.error(f"Detay verisi alınamadı: {e}")

with tabs[9]:
    st.subheader("Pozisyon / Stop / Hedef Hesabı")
    price = st.number_input("Alış fiyatı (₺)", min_value=0.00000001, value=100.0, format="%.8f")
    balance = st.number_input("Toplam sermaye (₺)", min_value=0.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("İşlem başına risk (%)", .25, 5.0, 1.0, .25)
    stop_pct = st.slider("Stop mesafesi (%)", .5, 15.0, 3.0, .25)
    rr = st.slider("Risk / kazanç oranı", 1.0, 5.0, 2.0, .25)

    max_loss = balance*risk_pct/100
    stop = price*(1-stop_pct/100)
    unit_risk = price-stop
    qty = max_loss/unit_risk if unit_risk>0 else 0
    pos = qty*price
    target = price + rr*unit_risk

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Maksimum zarar", f"₺{max_loss:,.2f}")
    c2.metric("Stop", f"₺{fmt(stop)}")
    c3.metric("Hedef", f"₺{fmt(target)}")
    c4.metric("Teorik pozisyon", f"₺{pos:,.2f}")
    st.caption(f"Teorik coin miktarı: {qty:.8f}")

st.warning(
    "Teknik göstergeler garanti vermez. Bu panel yatırım tavsiyesi değildir ve bu sürüm Binance TR hesabına otomatik emir göndermez."
)
