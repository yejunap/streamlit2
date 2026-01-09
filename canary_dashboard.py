import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Canary Dashboard: XLY/KRE/ITB", layout="wide")

DEFAULT_TICKERS = ["XLY", "KRE", "ITB", "XLP", "SPY"]
STAGE_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "RE-ENTRY", "UNKNOWN"]

# -----------------------------
# Helpers
# -----------------------------
def to_weekly_close(df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily OHLCV to weekly close/ohlc."""
    if df.empty:
        return df
    
    # 복사본 생성
    df = df.copy()
    
    # MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 컬럼명 정규화
    df.columns = [col.capitalize() for col in df.columns]
    
    # 필요한 컬럼 확인
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.warning(f"⚠️ 누락된 컬럼: {missing}")
        return pd.DataFrame()
    
    # Ensure timezone-naive index
    df.index = pd.to_datetime(df.index).tz_localize(None, ambiguous="NaT", nonexistent="NaT")
    
    # Weekly (Fri) bars
    ohlc = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    return ohlc

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

def safe_last(series: pd.Series):
    if series is None or series.empty:
        return np.nan
    return float(series.dropna().iloc[-1]) if not series.dropna().empty else np.nan

def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:.2f}%"

def fmt_num(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.2f}"

def higher_low_detection(lows: pd.Series, lookback_weeks: int = 12) -> dict:
    """
    Simple Higher Low: compare the minimum low in the latest lookback window
    vs minimum low in the preceding lookback window.
    """
    lows = lows.dropna()
    if len(lows) < lookback_weeks * 2 + 5:
        return {"enough_data": False, "higher_low": False, "recent_min": np.nan, "prev_min": np.nan}

    recent = lows.iloc[-lookback_weeks:]
    prev = lows.iloc[-2*lookback_weeks:-lookback_weeks]
    recent_min = recent.min()
    prev_min = prev.min()
    return {
        "enough_data": True,
        "higher_low": bool(recent_min > prev_min),
        "recent_min": float(recent_min),
        "prev_min": float(prev_min),
    }

def above_ma(close: pd.Series, ma: pd.Series) -> bool:
    c = safe_last(close)
    m = safe_last(ma)
    if np.isnan(c) or np.isnan(m):
        return False
    return c >= m

def trend_down_2w(close: pd.Series) -> bool:
    s = close.dropna()
    if len(s) < 3:
        return False
    return (s.iloc[-1] < s.iloc[-2]) and (s.iloc[-2] < s.iloc[-3])

def stage_logic(weekly: dict) -> dict:
    """
    weekly: dict[ticker] -> weekly OHLCV DF
    Returns stage + reasons + recommended actions.
    Rule intent (weekly):
      - GREEN: XLY/KRE/ITB 모두 50MA & 200MA 위
      - YELLOW: XLY 또는 ITB가 50MA 아래가 2주 지속(간단히 '최근 2주 하락 + 50MA 아래'로 근사)
      - ORANGE: KRE가 50MA 아래 + 최근 2주 하락(=반등 실패 근사)
      - RED: XLY/KRE/ITB 모두 50MA 아래 (또는 200MA 아래까지 포함해 더 보수적으로)
      - RE-ENTRY: KRE Higher Low + XLY/XLP 상대강도 상승(최근 4주 기울기 +)
    """
    req = ["XLY", "KRE", "ITB", "XLP", "SPY"]
    if any(t not in weekly or weekly[t].empty for t in req):
        return {"stage": "UNKNOWN", "reasons": ["데이터 부족/로드 실패"], "actions": []}

    def calc(t):
        w = weekly[t]
        c = w["Close"]
        ma50 = sma(c, 50)
        ma200 = sma(c, 200)
        return w, c, ma50, ma200

    xly_w, xly_c, xly_50, xly_200 = calc("XLY")
    kre_w, kre_c, kre_50, kre_200 = calc("KRE")
    itb_w, itb_c, itb_50, itb_200 = calc("ITB")

    xlp_w, xlp_c, _, _ = calc("XLP")

    # Relative strength XLY/XLP (weekly)
    rs = (xly_c / xlp_c).dropna()
    rs_slope_4w = np.nan
    if len(rs) >= 6:
        # simple slope using last 5 points
        y = rs.iloc[-5:].values
        x = np.arange(len(y))
        rs_slope_4w = np.polyfit(x, y, 1)[0]  # slope

    # Higher low for KRE
    hl = higher_low_detection(kre_w["Low"], lookback_weeks=12)

    # Conditions
    xly_above50 = above_ma(xly_c, xly_50)
    kre_above50 = above_ma(kre_c, kre_50)
    itb_above50 = above_ma(itb_c, itb_50)

    xly_above200 = above_ma(xly_c, xly_200)
    kre_above200 = above_ma(kre_c, kre_200)
    itb_above200 = above_ma(itb_c, itb_200)

    xly_down2 = trend_down_2w(xly_c)
    itb_down2 = trend_down_2w(itb_c)
    kre_down2 = trend_down_2w(kre_c)

    # Stage determination (priority: RED > ORANGE > YELLOW > RE-ENTRY > GREEN)
    reasons = []

    # RED: all below 50MA
    if (not xly_above50) and (not kre_above50) and (not itb_above50):
        stage = "RED"
        reasons.append("XLY/KRE/ITB 모두 50주 이동평균 하단 (침체/리스크오프 가능성↑)")
    # ORANGE: KRE stress proxy
    elif (not kre_above50) and kre_down2:
        stage = "ORANGE"
        reasons.append("KRE 50주 이동평균 하단 + 최근 2주 연속 하락 (금융 스트레스/반등 실패 근사)")
    # YELLOW: XLY or ITB weakening
    elif ((not xly_above50 and xly_down2) or (not itb_above50 and itb_down2)):
        stage = "YELLOW"
        reasons.append("XLY 또는 ITB 약세(50주 MA 하단 + 단기 하락) (경기/금리 부담 경고)")
    else:
        # RE-ENTRY: higher low + RS rising
        if hl.get("enough_data") and hl.get("higher_low") and (not np.isnan(rs_slope_4w) and rs_slope_4w > 0):
            stage = "RE-ENTRY"
            reasons.append("KRE Higher Low + XLY/XLP 상대강도 상승(리스크온 복귀 신호)")
        elif xly_above50 and kre_above50 and itb_above50 and xly_above200 and kre_above200 and itb_above200:
            stage = "GREEN"
            reasons.append("XLY/KRE/ITB 모두 50·200주 이동평균 상단 (정상/리스크온)")
        else:
            stage = "GREEN"
            reasons.append("치명 신호 없음(기본 GREEN 유지)")

    # Action playbook
    actions = []
    if stage == "GREEN":
        actions = [
            "주식 익스포저 70~80% 유지",
            "고베타는 분할로만(레버리지는 제한)",
            "현금 10~20% 유지(기회자금)"
        ]
    elif stage == "YELLOW":
        actions = [
            "사이클/고베타/테마 비중 15%p 축소",
            "신규 공격적 매수 중단('떨어지면 산다' 일시 중지)",
            "현금/단기채 비중 +15%p 확보"
        ]
    elif stage == "ORANGE":
        actions = [
            "주식 익스포저 누적 -35%p까지 축소",
            "소형주/고PER/레버리지 전면 중단",
            "방어(배당·저변동) + 단기채로 이동",
            "(고급) 지수 풋 스프레드/헤지 검토"
        ]
    elif stage == "RED":
        actions = [
            "주식 비중 40~50% 이하로 강제 축소",
            "커버드콜·배당·단기채·현금 중심으로 재편",
            "배당 재투자(자동DRIP)는 '일시 중지' → 현금 축적",
            "재진입은 'KRE 안정 + RS 회복' 확인 후 단계적으로"
        ]
    elif stage == "RE-ENTRY":
        actions = [
            "현금에서 주식으로 +10%p씩 단계 재진입(주 단위)",
            "1) 시장 ETF → 2) 퀄리티/대형 → 3) 고베타 순서",
            "커버드콜 비중은 즉시 줄이지 말고 '상승 추세 확정' 후 축소"
        ]
    else:
        actions = ["데이터 상태 확인(티커/네트워크/야후 제한)"]

    return {
        "stage": stage,
        "reasons": reasons,
        "actions": actions,
        "rs_slope_4w": rs_slope_4w,
        "kre_hl": hl,
        "flags": {
            "xly_above50": xly_above50,
            "kre_above50": kre_above50,
            "itb_above50": itb_above50,
            "xly_above200": xly_above200,
            "kre_above200": kre_above200,
            "itb_above200": itb_above200,
            "xly_down2": xly_down2,
            "kre_down2": kre_down2,
            "itb_down2": itb_down2,
        }
    }

def plot_price_ma(weekly_df: pd.DataFrame, title: str):
    c = weekly_df["Close"]
    ma50 = sma(c, 50)
    ma200 = sma(c, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.index, y=c, name="Close"))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="SMA 50W"))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="SMA 200W"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_ratio(rs: pd.Series, title: str):
    ma20 = sma(rs, 20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name="XLY/XLP"))
    fig.add_trace(go.Scatter(x=ma20.index, y=ma20, name="SMA 20W"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def load_ticker_with_retry(ticker, start, end, max_retries=3):
    """개별 티커를 재시도 로직으로 다운로드"""
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=15
            )
            
            if df.empty:
                st.warning(f"⚠️ {ticker}: 데이터가 비어있음 (시도 {attempt+1}/{max_retries})")
                time.sleep(1)
                continue
            
            # MultiIndex 처리 (단일 티커일 때도 발생 가능)
            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex를 flat하게 변경
                df.columns = df.columns.get_level_values(0)
            
            # 컬럼명 표준화
            df.columns = [col.capitalize() for col in df.columns]
            
            # 데이터 검증
            if len(df) < 100:
                st.warning(f"⚠️ {ticker}: 데이터가 너무 적음 ({len(df)}개 행, 시도 {attempt+1}/{max_retries})")
                time.sleep(1)
                continue
            
            # 필수 컬럼 확인
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required):
                st.warning(f"⚠️ {ticker}: 필수 컬럼 누락 (시도 {attempt+1}/{max_retries})")
                st.write(f"사용 가능한 컬럼: {list(df.columns)}")
                time.sleep(1)
                continue
            
            return df.dropna(how="all")
            
        except Exception as e:
            st.warning(f"⚠️ {ticker} 다운로드 실패 (시도 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 재시도 전 대기
            continue
    
    st.error(f"❌ {ticker}: {max_retries}번 시도 후 실패")
    return pd.DataFrame()

def load_data(tickers, start, end):
    """개선된 데이터 로드 - 개별 티커별로 재시도"""
    out = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"📥 {ticker} 데이터 다운로드 중... ({idx+1}/{len(tickers)})")
        
        df = load_ticker_with_retry(ticker, start, end, max_retries=3)
        out[ticker] = df
        
        if not df.empty:
            st.success(f"✅ {ticker}: {len(df)}개 행 로드 완료")
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    return out

# -----------------------------
# UI
# -----------------------------
st.title("🕊️ Canary Dashboard — XLY / KRE / ITB (주간 시그널 + 행동 지침)")

with st.sidebar:
    st.header("설정")
    tickers = st.multiselect("티커", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    years = st.slider("조회 기간(년)", min_value=5, max_value=25, value=15, step=1)
    st.caption("주간 200MA 계산을 위해 최소 5년 이상 권장")
    show_debug = st.checkbox("디버그(플래그/계산값 표시)", value=False)
    
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 자동 갱신 (4시간)", value=False, 
                                help="4시간마다 자동으로 데이터 갱신")
    
    if st.button("🔄 지금 새로고침", use_container_width=True):
        st.cache_data.clear()
        if 'last_refresh_time' in st.session_state:
            del st.session_state.last_refresh_time
        st.rerun()

end_date = datetime.today().date() + timedelta(days=1)
start_date = datetime.today().date() - timedelta(days=365 * years)

# 자동 갱신 로직
if auto_refresh:
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    elapsed = time.time() - st.session_state.last_refresh_time
    hours_elapsed = elapsed / 3600
    hours_remaining = max(0, 4 - hours_elapsed)
    
    # 4시간이 지났으면 자동 갱신
    if hours_elapsed >= 4:
        st.info("⏰ 4시간이 경과하여 자동 갱신합니다...")
        st.cache_data.clear()
        st.session_state.last_refresh_time = time.time()
        time.sleep(1)
        st.rerun()
    else:
        # 남은 시간 표시
        mins_remaining = int(hours_remaining * 60)
        st.info(f"🔄 자동 갱신 활성화 - 다음 갱신까지 {int(hours_remaining)}시간 {mins_remaining % 60}분")
        
        # 1분마다 체크 (페이지 자동 새로고침)
        time.sleep(60)
        st.rerun()

# 데이터 로드 (캐시 없이 매번 새로 로드)
with st.spinner("📊 Yahoo Finance에서 데이터 다운로드 중..."):
    daily = load_data(tickers, start=start_date, end=end_date)

weekly = {t: to_weekly_close(daily[t]) for t in tickers}

# Ensure required tickers exist
for needed in DEFAULT_TICKERS:
    if needed not in weekly:
        weekly[needed] = pd.DataFrame()

# 데이터 상태 체크
st.divider()
st.subheader("📊 데이터 상태")
data_status_cols = st.columns(len(DEFAULT_TICKERS))
for idx, ticker in enumerate(DEFAULT_TICKERS):
    with data_status_cols[idx]:
        df = weekly.get(ticker, pd.DataFrame())
        if df.empty:
            st.error(f"❌ {ticker}\n데이터 없음")
        else:
            st.success(f"✅ {ticker}\n{len(df)}주")

result = stage_logic(weekly)

# -----------------------------
# Top summary
# -----------------------------
st.divider()
stage = result["stage"]
stage_emoji = {
    "GREEN": "🟩",
    "YELLOW": "🟨",
    "ORANGE": "🟧",
    "RED": "🟥",
    "RE-ENTRY": "🟦",
    "UNKNOWN": "⬜"
}.get(stage, "⬜")

colA, colB, colC = st.columns([1.2, 2.2, 2.6])

with colA:
    st.metric("현재 단계", f"{stage_emoji} {stage}")
    st.caption("주간(Weekly) 기준 규칙 판정")

with colB:
    st.subheader("판정 근거")
    for r in result["reasons"]:
        st.write(f"- {r}")

with colC:
    st.subheader("지금 해야 할 행동(Playbook)")
    for a in result["actions"]:
        st.write(f"✅ {a}")

st.divider()

# -----------------------------
# Signal table
# -----------------------------
def make_row(t):
    w = weekly.get(t, pd.DataFrame())
    if w is None or w.empty:
        return {
            "Ticker": t, "Close": np.nan, "WoW": np.nan,
            "Above 50W": False, "Above 200W": False,
            "RSI(14W)": np.nan
        }
    c = w["Close"].dropna()
    wow = np.nan
    if len(c) >= 2:
        wow = (c.iloc[-1] / c.iloc[-2]) - 1.0
    ma50 = sma(c, 50)
    ma200 = sma(c, 200)
    rsi14 = rsi(c, 14)
    return {
        "Ticker": t,
        "Close": safe_last(c),
        "WoW": wow,
        "Above 50W": above_ma(c, ma50),
        "Above 200W": above_ma(c, ma200),
        "RSI(14W)": safe_last(rsi14)
    }

rows = [make_row(t) for t in ["SPY", "XLY", "XLP", "KRE", "ITB"]]
sig_df = pd.DataFrame(rows)

# Pretty formatting
sig_view = sig_df.copy()
sig_view["Close"] = sig_view["Close"].map(fmt_num)
sig_view["WoW"] = sig_view["WoW"].map(fmt_pct)
sig_view["RSI(14W)"] = sig_view["RSI(14W)"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}")

st.subheader("📋 핵심 시그널 테이블 (주간)")
st.dataframe(sig_view, use_container_width=True)

# Relative strength section
xly_c = weekly["XLY"]["Close"] if not weekly["XLY"].empty else pd.Series(dtype=float)
xlp_c = weekly["XLP"]["Close"] if not weekly["XLP"].empty else pd.Series(dtype=float)
rs = (xly_c / xlp_c).dropna()
st.caption("※ XLY/XLP 상대강도: 상승이면 Risk-on 복귀 가능성, 하락이면 방어 선호 강화")

# KRE higher low details
hl = result.get("kre_hl", {})
if hl.get("enough_data"):
    st.info(
        f"KRE Higher Low 검사(12주 윈도우): "
        f"이전 저점 {hl['prev_min']:.2f} → 최근 저점 {hl['recent_min']:.2f} "
        f"({'Higher Low ✅' if hl['higher_low'] else 'Higher Low ❌'})"
    )
else:
    st.warning("KRE Higher Low 검사는 데이터가 더 필요합니다(주간 데이터 길이 부족).")

st.divider()

# -----------------------------
# Charts
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    if not weekly["XLY"].empty:
        st.plotly_chart(plot_price_ma(weekly["XLY"], "XLY (Weekly) — Close & 50/200W MA"), use_container_width=True)
    else:
        st.warning("XLY 데이터 없음")

with c2:
    if not weekly["KRE"].empty:
        st.plotly_chart(plot_price_ma(weekly["KRE"], "KRE (Weekly) — Close & 50/200W MA"), use_container_width=True)
    else:
        st.warning("KRE 데이터 없음")

with c3:
    if not weekly["ITB"].empty:
        st.plotly_chart(plot_price_ma(weekly["ITB"], "ITB (Weekly) — Close & 50/200W MA"), use_container_width=True)
    else:
        st.warning("ITB 데이터 없음")

c4, c5 = st.columns([1.2, 1.8])
with c4:
    if len(rs) > 0:
        st.plotly_chart(plot_ratio(rs, "XLY/XLP 상대강도 (Weekly)"), use_container_width=True)
    else:
        st.warning("XLY/XLP 상대강도 계산 불가(데이터 부족)")

with c5:
    st.subheader("🧭 단계 정의(요약)")
    st.write("- **GREEN**: XLY/KRE/ITB가 50·200W MA 상단(정상/리스크온)")
    st.write("- **YELLOW**: XLY 또는 ITB 약세(경기/금리 부담 경고)")
    st.write("- **ORANGE**: KRE 약세 + 단기 하락(금융 스트레스 경고)")
    st.write("- **RED**: XLY/KRE/ITB 모두 50W MA 하단(침체/리스크오프 가능성↑)")
    st.write("- **RE-ENTRY**: KRE Higher Low + XLY/XLP 상대강도 상승(단계적 재진입 조건)")

# -----------------------------
# Debug
# -----------------------------
if show_debug:
    st.divider()
    st.subheader("🔧 디버그 정보(플래그/계산값)")
    st.json({
        "stage": result["stage"],
        "reasons": result["reasons"],
        "flags": result.get("flags", {}),
        "rs_slope_4w": result.get("rs_slope_4w", None),
        "kre_higher_low": result.get("kre_hl", None),
    })

st.caption("데이터 출처: Yahoo Finance(yfinance). 주간 규칙은 '노이즈 감소' 목적의 근사이며, 사용자는 최종 의사결정 책임을 가집니다.")
