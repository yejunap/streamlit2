"""Jita 4-4 <-> Amarr VIII 아비 대시보드 (Streamlit).

실행: streamlit run app.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import freshen                                             # noqa: E402
freshen.freshen_modules(_HERE)                              # 고친 코드 바로 반영

import esi                                                 # noqa: E402
from arb_core import (PRESETS, analyze_type, candidate_scores, filter_rows,  # noqa: E402
                      internal_inversions, pair_summary)

try:
    st.set_page_config(page_title="EVE 허브 아비 — Jita↔Amarr·Dodixie·Rens", layout="wide", page_icon="📈")
except Exception:              # 테스트용 가짜 스텁 모듈이면 넘어간다
    pass

# ---- 🔒 잠금 ----------------------------------------------------------------------------------------
# 잠금 암호는 여기서 고치지 않는다 — 숫자는 리포에 안 올라간다.
#   올린곳:  대시보드 → Settings → Secrets 에   ARB_PW = "새 암호"
#   로컬:    .streamlit/secrets.toml            ARB_PW = "..."   (깃 무시됨)
#            또는 환경변수 ARB_PW 도 받는다.
# 시크릿도 env도 없으면 열지 않는다 — 예 값은 없다.
_PW = ""
try:
    _PW = str(st.secrets.get("ARB_PW") or "").strip()
except Exception:
    pass
_PW = _PW or os.environ.get("ARB_PW", "").strip()
if not _PW:
    st.error("잠금 암호가 없습니다. `ARB_PW`를 거세요 —\n\n"
            "* 구름: Settings → Secrets → `ARB_PW = \"...\"`\n"
            "* 로컬: `.streamlit/secrets.toml` 에 `ARB_PW = \"...` 또는 환경변수 `ARB_PW`\n\n"
            "그것만 걸면 됩니다.")
    st.stop()
if not st.session_state.get("auth_ok"):
    st.title("🔒 EVE 허브 아비 대시보드")
    pw = st.text_input("암호", type="password", key="_pw", placeholder="4칙")
    if st.button("들어가기"):
        if (pw or "") == _PW:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("암호가 다릅니다.")
    st.stop()

# ---- 🖐 클릭 후관용 -----------------------------------------------------------------------------
# 버튼이랑 셀렉트 항목을 키운다 — 꼭 징에 안 눌러도 눌리게.
st.markdown("""
<style>
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
  padding: .9rem 1.6rem;
  min-height: 3.4rem;
  font-size: 1.05rem;
}
div[data-baseweb="menu"] li { padding-top: .55rem; padding-bottom: .55rem; }
</style>
""", unsafe_allow_html=True)


class Bar:
    """호출 카운터 — Streamlit 진행줄에 얹는다."""

    def __init__(self, total: float, label: str):
        self.n, self.total = 0.0, float(total)
        self.p = st.sidebar.progress(0.0, label)

    def tick(self, _=None):
        self.n += 1
        self.p.progress(min(1.0, self.n / self.total))
        if self.n >= self.total:
            self.p.empty()

    def done(self):
        self.p.empty()


# --------------------------------------------------------------- 스캔
# ---- Heavy Fighter 확정 목록 (data/fighters.csv, build_fighters.py) ------------------------------
_FIGHTERS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fighters.csv")
try:
    with open(_FIGHTERS_CSV, encoding="utf-8") as _f:
        FIGHTER_TIDS = [int(r["type_id"]) for r in csv.DictReader(_f)]
except FileNotFoundError:
    FIGHTER_TIDS = []

def run_scan(hubs: list[str], workers: int, max_candidates: int, log) -> dict:
    """Jita + 선택 허브들의 주문판을 받아 후보 종목만 남긴다 (첫 실행은 수 분)."""
    names = ["Jita"] + [h for h in hubs if h != "Jita"]
    try:      # 진행 칸 수는 각 리전 페이지 합 — 세는 실패해도 스캔은 돈다
        total = sum(esi.region_pages(esi.HUBS[h]["region"]) for h in names)
    except Exception as e:
        log(f"페이지 수 세기 실패({e}) — 기본값으로 진행")
        total = 500 * len(names)
    bar = Bar(max(total, 1), "스테이션 주문판 다운로드")
    done = bar.tick

    books: dict[str, dict] = {}
    for h in names:
        hb = esi.HUBS[h]
        log(f"{h} 주문판…")
        books[h] = esi.fetch_hub_orders(hb["region"], hb["station"],
                                        workers=workers, progress=done, log=log)
    bar.done()

    # 후보는 쌍별로 상위 max_candidates를 모은다 — 한 쌍에만 남는 종목을 아끼지 않는다.
    keep: list[int] = []
    scores: dict[int, float] = {}
    jita_inverted = internal_inversions(books["Jita"])
    keep_n = "전체" if max_candidates >= 10 ** 8 else f"상위 {max_candidates}"
    for h in names[1:]:
        sc = candidate_scores(books["Jita"], books[h])
        top = [t for _, t in sorted(((s, t) for t, s in sc.items()), reverse=True)][:max_candidates]
        keep += top
        for t, s in sc.items():
            scores[t] = max(scores.get(t, 0.0), s)
        # 내부 스프레드는 교차 종목이 아니어도 사납무므로 무조건 데리고 있다.
        keep += list(jita_inverted | internal_inversions(books[h]))
        log(f"Jita↔{h}: 교차 {len(sc)} → 조마진금액 {keep_n} + 내부역전")
    # Heavy Fighter 16종(build_fighters.py)은 점수순과 무관하게 항상 계산 지킨다.
    keep += FIGHTER_TIDS
    keep = list(dict.fromkeys(keep))
    log(f"{len(names) - 1}쌍 합산 후보 → {len(keep)}종목")
    return {"hubs": names,
            "books": {t: {h: books[h].get(t, {"sells": [], "buys": []}) for h in names}
                      for t in keep},
            "scores": scores,
            "at": datetime.now(timezone.utc)}


def enrich(scan: dict, log) -> tuple[dict, dict]:
    """부피(m³/개), 이름, 일평균 거래량을 채운다."""
    ids = list(scan["books"])
    bar = Bar(2 * len(ids), "부피/거래량 조회")
    upd = bar.tick

    meta = esi.fetch_type_meta(ids, progress=upd)
    log("부피/이름 확보")
    hist = esi.fetch_histories(esi.JITA_REGION, ids, progress=upd)
    log("일평균 거래량 확보")
    bar.done()
    return meta, hist


# --------------------------------------------------------------- 표
COLUMNS = ["종목", "쌍", "패턴", "방향", "매집가", "청산가", "조마진%", "순이익%", "체결량",
           "순이익 ISK", "묶인 자본", "단위부피 m³", "총부피 m³", "ISK/m³", "필요 운항",
           "1회 운항 이익", "일평균 거래량", "소요일", "매도호가 유지 h", "매수호가 유지 h"]
SRC = ["item", "pair", "pattern", "direction", "buy_avg", "sell_avg", "gross_pct", "net_pct",
       "qty", "profit", "capital", "unit_volume", "volume_m3", "isk_per_m3", "trips",
       "profit_per_trip", "daily_volume", "days_to_sell", "ask_age_h", "bid_age_h"]
# 표는 **숫자를 숫자로** 실는다 — 문자(1,141)로 포맷하면 UI 헤더클릭 정렬이
# 문자열 정렬로 새버려서 순서가 이상해진다. 표시 자릿수만 여기서 잡는다.
DECIMALS = {
    "매집가": 2, "청산가": 2, "조마진%": 2, "순이익%": 2,
    "체결량": 0, "순이익 ISK": 0, "묶인 자본": 0,
    "단위부피 m³": 3, "총부피 m³": 0, "ISK/m³": 0,
    "필요 운항": 0, "1회 운항 이익": 0, "일평균 거래량": 0,
    "소요일": 2, "매도호가 유지 h": 0, "매수호가 유지 h": 0,
}


def to_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    view = df[SRC].copy()
    view.columns = COLUMNS
    for c, nd in DECIMALS.items():
        view[c] = pd.to_numeric(view[c], errors="coerce").round(nd)
    return view


def render_basket(box, rows_map: dict, names: set):
    """장바구니 합산을 placeholder에 채운다. 체크가 바뀐 런에도 즉시 호출된다."""
    box.empty()
    if not names:
        with box:
            st.caption("🛒 장바구니 비어 있음 — 아래 표에서 체크박스로 종목을 담아오세요")
        return
    best: dict[str, dict] = {}
    for lst in rows_map.values():
        for r in lst:
            if r["item"] in names and (r["item"] not in best or r["profit"] > best[r["item"]]["profit"]):
                best[r["item"]] = r
    if not best:
        return
    tot_v = sum(r["volume_m3"] or 0 for r in best.values())
    tot_c = sum(r["capital"] for r in best.values())
    tot_p = sum(r["profit"] for r in best.values())
    with box:
        st.markdown(f"**🛒 장바구니 {len(best)}종목** — 합산")
        a, b, c, d = st.columns(4)
        a.metric("총부피", f"{tot_v:,.0f} m³")
        b.metric("총 묶인 자본", f"{tot_c:,.0f} ISK")
        c.metric("예상 순이익", f"{tot_p:,.0f} ISK")
        cargo_m3 = st.session_state.get("cargo_m3")
        if not isinstance(cargo_m3, (int, float)) or cargo_m3 <= 0:
            cargo_m3 = 534_000
        d.metric(f"운항 횟수 (1회 {cargo_m3:,.0f}m³)", f"{-(-tot_v // cargo_m3):,.0f}")


def show_rows(rows: list[dict], title: str):
    if not rows:
        st.info(f"{title}: 조건을 통과한 아비가 없습니다.")
        return
    # 3단계 고정: ① ISK/m³ 내림차순 ② 순이익 내림차순 ③ 묶인 자본 오름차순
    rows = sorted(rows, key=lambda d: (-(d["isk_per_m3"] or 0), -d["profit"], d["capital"]))
    rows = [r for r in rows if r["isk_per_m3"] is not None]
    st.caption(f"{len(rows)}행 전체 · 정렬: **① ISK/m³↓ ② 순이익↓ ③ 묶인자본↑** — "
               "**아무 열 머리말이나 누르면 그 열로 바로 정렬됩니다**")
    st.caption("🛒 **장바구니**: 표 왼쪽 체크박스를 고르거나 **행을 아무 데나 클릭** — "
               "머리칸 체크박스는 전체 담기입니다")
    tbl = to_table(rows)
    names = list(tbl["종목"])
    pinned = st.session_state.get("pinned", set())
    cfg = {c: st.column_config.NumberColumn(format="localized") for c in DECIMALS}
    # 행 선택형 표 — 체크박스 칸이 아니라 행 통째가 클릭 면적이고
    # 헤 칸 체크박스는 전체 선택이다. 클릭이 관대하고 첫 클릭이 안 새는 일도 없다.
    key = f"tbl_{title}"
    hit = f"selhit_{title}"

    def _on_sel():
        # 선택 이벤트가 일어난 런만 표식을 세운다 — 새로고침·재스캔 후
        # 빈 선택이 보여져 장바구니를 뒤덮는 일을 막는다.
        st.session_state[hit] = True

    # 첫 실행에서만 표를 장바구니로 채운다 — 새로고침 후 체크표시가 돌아온다.
    seed = [i for i, n in enumerate(names) if n in pinned]
    if seed and f"seeded_{title}" not in st.session_state:
        st.session_state[key] = {"selection": {"rows": seed}}
        st.session_state[f"seeded_{title}"] = True
    st.dataframe(
        tbl, key=key, hide_index=True, width="stretch",
        height=min(1200, 34 * (len(tbl) + 2)),
        column_config=cfg, selection_mode="multi-row", on_select=_on_sel)
    if st.session_state.pop(hit, False):
        # 위젯 상태는 dict — 키("rows")로 읽어야 한다 (속성 아님!)
        sel = (st.session_state.get(key) or {}).get("selection") or {}
        rows_idx = sel.get("rows") or []
        sel_names = {names[i] for i in rows_idx if isinstance(i, int) and i < len(names)}
        was = pinned & set(names)         # 이 표에 담겨 있던 것
        if sel_names != was:              # 이 표가 실제로 바뀜 — 나머지 표 기여는 살린다
            merged = (pinned - set(names)) | sel_names
            st.session_state["pinned"] = merged
            # 강제 새로 고침(rerun)은 표를 맨 앞으로 튕긴다 — 함 상자 제때 채운다.
            bb, rr = globals().get("basket_box"), globals().get("rows")
            if bb is not None and rr is not None:
                render_basket(bb, rr, merged)
    st.download_button(
        f"CSV ({len(rows)}행)", tbl.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"arb_{title}_{datetime.now():%m%d_%H%M}.csv", mime="text/csv")
    return tbl


# --------------------------------------------------------------- 사이드바
def sidebar():
    st.sidebar.header("⚙️ 스캔")
    st.sidebar.caption("Jita 4-4 ↔ Amarr / Dodixie / Rens — 주문판 전량 → 창두께 상위 종목")
    hub_sel = st.sidebar.multiselect(
        "Jita ↔ 허브 쌍", esi.SECONDARIES, default=list(esi.SECONDARIES),
        help="계산할 쌍만 고른다. 적게 고를수록 주문판 다운로드가 줄어 첫 스캔이 빨라진다.")
    if not hub_sel:
        hub_sel = list(esi.SECONDARIES)      # 하나도 안 고르면 전부
    scan_now = st.sidebar.button("🔄 주문판 전체 스캔", use_container_width=True)
    if st.sidebar.button("🧹 ESI 캐시 지우기", use_container_width=True):
        import glob
        for p in glob.glob(f"{esi.CACHE_DIR}/*.json"):
            os.remove(p)
        st.sidebar.toast("캐시를 비웠습니다")
    workers = st.sidebar.slider("ESI 워커", 2, 16, 8, help="크면 ESI 트로틀링에 걸릴 수 있다")
    all_cand = st.sidebar.toggle("전체 후보 (무제한)", value=True,
                                 help="조마진금액이 있는 교차 종목 + 내부 역전을 전부 후보에 담는다")
    max_cand = 10 ** 9 if all_cand else st.sidebar.slider("후보 종목 수", 100, 5000, 1000, step=100)
    get_meta = st.sidebar.toggle("부피·일평균 거래량 조회", value=True)

    st.sidebar.header("💰 비용")
    st.sidebar.caption("즉시판매는 브로커 수수료 없음 (확인: EVE Uni Wiki — 'immediate'가 아닌 지정가에만 부과). "
                       "대신 판매세는 무조건 붙는다 — 2025-03부터 기본 7.5% (Accounting Lv5면 3.4%)")
    sell_market = st.sidebar.toggle("판매 창을 즉시판매 (시장가)로 가정", value=True,
                                    help="켜면 판매세만, 끄면 지정가 대분으로 브로커 수수료가 더해진다")
    fees = dict(PRESETS)
    custom = st.sidebar.checkbox("직접 지정")
    if custom:
        c1, c2, c3 = st.sidebar.columns(3)
        # 기준치: Accounting Lv5 (판매세 3.375%), 구매는 즉시구매(세/수수료 0)
        buy = c1.number_input("구매 요율 %", 0.0, 10.0, 0.0, step=0.1) / 100
        stax = c2.number_input("판매세 %", 0.0, 10.0, 3.375, step=0.1) / 100
        sbrk = c3.number_input("판매 브로커 %", 0.0, 10.0, 3.0, step=0.1) / 100
        fees[f"사용자 ({buy*100:.1f}%/세{stax*100:.1f}%/브{sbrk*100:.1f}%)"] = {
            "buy": buy, "sell_tax": stax, "broker_sell": sbrk}
    # 기준 표는 Accounting Lv5 프리셋 하나로 (나머지는 비교용으로 체크해서 �쳐보기)
    acct5 = next(k for k in PRESETS if k.startswith("Accounting"))
    presets = st.sidebar.multiselect("수수료 프리셋", list(fees), default=[acct5])
    cargo = st.sidebar.number_input("1회 적재 가능 부피 (m³)", 100, 1_000_000, 534_000, step=1000,
                                    key="cargo_m3",
                                    help="탈 배의 화물함 용량")

    st.sidebar.header("🔻 필터")
    min_pct = st.sidebar.number_input("최소 순이익 %", 0.0, 100.0, 1.0, step=0.5)
    min_profit = st.sidebar.number_input("최소 순이익 ISK", 0, 10_000_000_000, 0, step=100_000)
    min_cap = st.sidebar.number_input("최소 묶인 자본 ISK", 0, 10_000_000_000, 0, step=1_000_000)
    min_dv = st.sidebar.number_input("최소 일평균 거래량", 0, 1_000_000, 0, step=100)
    patterns = st.sidebar.multiselect("패턴", list("ABCD"), default=list("ABCD"))
    return dict(scan_now=scan_now, workers=workers, max_cand=max_cand, get_meta=get_meta,
                fees=fees, presets=presets, cargo=cargo, min_pct=min_pct,
                min_profit=min_profit, market_sell=sell_market, hubs=hub_sel,
                min_cap=min_cap, min_dv=min_dv, patterns=set(patterns))


def show_book(scan: dict, meta: dict, tid: int):
    """선택 종목의 허브별 4면 호가창(상위 5칸)과 주문 유지 시간을 보여준다."""
    b = scan["books"][tid]
    hubs = scan.get("hubs", list(b))
    cols = st.columns(max(len(hubs), 1))
    for i, hub in enumerate(hubs):
        bk = b[hub]
        with cols[i]:
            mv = meta.get(tid, {}).get("volume")
            st.markdown(f"**{hub}** — 매도 {len(bk['sells'])}칸 / 매수 {len(bk['buys'])}칸"
                        + (f" · {mv:,.3f} m³/개" if mv else ""))
            for side, label in (("sells", "매도"), ("buys", "매수")):
                rows = bk[side][:5]
                if not rows:
                    st.caption(f"{label} 창 없음")
                    continue
                st.dataframe(pd.DataFrame([
                    {"구분": label, "호가": f"{p:,.2f}", "잔량": f"{q:,.0f}",
                     "유지(h)": "-" if d is None else round(
                         (datetime.now(timezone.utc) - d).total_seconds() / 3600)}
                    for p, q, d in rows]), hide_index=True, width="stretch")
    st.caption("유지 시간 = 창 최우상 호가가 시장에서 버틴 시간. 오래 버틴 왜곡이 범인이다.")



# --------------------------------------------------------------- 본문
st.title("📈 Jita 4-4 ↔ Amarr · Dodixie · Rens 아비 대시보드")
st.caption("패턴 A/B: 허브 간 운송 아비 · C/D: 같은 스테이션 내부 스프레드 (운송 0, 부피 무의미)")

cfg = sidebar()
LOG: list[str] = []
log = lambda m: LOG.append(m)

if cfg["scan_now"]:
    with st.spinner("주문ware 받는 중…"):
        st.session_state["scan"] = run_scan(cfg["hubs"], cfg["workers"], cfg["max_cand"], log)
        if cfg["get_meta"]:
            meta, hist = enrich(st.session_state["scan"], log)
            st.session_state["meta"], st.session_state["hist"] = meta, hist

scan = st.session_state.get("scan")
if not scan:
    st.warning("측정된 주문이 없습니다. 사이드바에서 **주문판 전체 스캔**을 누르세요.")
    st.stop()

meta = st.session_state.get("meta", {})
hist = st.session_state.get("hist", {})
st.sidebar.success(f"스캔 {scan['at'].astimezone().strftime('%H:%M:%S')} · {len(scan['books'])}종목")

# 즉시판매(시장가)면 판매 브로커 수수료 없음 — 판매세만 붙는다
sell_fee_of = {p: (cfg["fees"][p]["sell_tax"] if cfg["market_sell"]
                   else cfg["fees"][p]["sell_tax"] + cfg["fees"][p]["broker_sell"])
               for p in cfg["presets"]}
rows: dict[str, list] = {p: [] for p in cfg["presets"]}
for tid, books in scan["books"].items():
    mv = meta.get(tid, {}).get("volume", 0.0)
    if not mv:
        continue
    for hub in scan["hubs"][1:]:                 # Jita와 짝을 이룰 허브들
        b_hub = books.get(hub, {"sells": [], "buys": []})     # 예전 스캔 잔존 시 비어있음
        for pname in cfg["presets"]:
            fees = cfg["fees"][pname]
            rows[pname] += analyze_type(
                tid, meta[tid]["name"], mv, books["Jita"], b_hub,
                fees["buy"], sell_fee_of[pname], hist.get(tid, 0.0), cfg["cargo"],
                hub_b=hub)

# ---- 🛒 장바구니 합산기 (체크한 종목) ----------------------------------------------------------
# placeholder: 체크가 바뀐 바로 그 런에 show_rows가 채워 넣는다 — 강제 리런 없이 실시간 갱신.
basket_box = st.empty()
if st.button("🧹 장바구니 비우기", key="clear_basket",
             disabled=not st.session_state.get("pinned")):
    # 클릭한 이 런에 상태를 비우면 그 아래 표가 그릴 때 체크가 풀린다 — 리런 없음
    st.session_state["pinned"] = set()
render_basket(basket_box, rows, st.session_state.get("pinned", set()))

body = st.tabs(["🔀 크로스 아비 (A/B)", "🏪 내부 스프레드 (C/D)", "🃀 전체 통합",
                "📊 페어 비교", "🔬 종목 상세"])


def filt(rows_: list, pat: set) -> list:
    return filter_rows(rows_, cfg["min_pct"], cfg["min_profit"], cfg["min_cap"],
                       cfg["min_dv"], pat)


with body[0]:
    for pname, v in rows.items():
        st.subheader(pname)
        show_rows(filt(v, {"A", "B"} & cfg["patterns"]), f"cross_{pname[:12]}")
with body[1]:
    st.caption("운송이 없어 부피 제약이 걸리지 않는다. 같은 스테이션에서 매수호가가 매도호가보다 올라갈 때만 뜬다.")
    if not any(filter_rows(v, cfg["min_pct"], cfg["min_profit"], cfg["min_cap"],
                           cfg["min_dv"], {"C", "D"} & cfg["patterns"]) for v in rows.values()):
        st.info("지금 수수료 설정에서 내부 역전가 없다. 수수료 프리셋을 『검토(0%)』로 바꾸거나 "
                "최소 순이익 %를 0으로 내려라.")
    for pname, v in rows.items():
        st.subheader(pname)
        show_rows(filt(v, {"C", "D"} & cfg["patterns"]), f"internal_{pname[:12]}")
with body[2]:
    for pname, v in rows.items():
        st.subheader(pname)
        show_rows(filt(v, set(cfg["patterns"])), f"all_{pname[:12]}")
with body[3]:
    st.caption("운송 아비(A/B)만 세었습니다 — C/D는 한 스테이션 내부 일이라 쌍의 수익에 넣지 않았습니다. "
               "합순이익 내림차순이라, 어느 쌍이 실제로 남는지 맨 위가 말합니다.")
    for pname, v in rows.items():
        summ = pair_summary(filt(v, {"A", "B"} & cfg["patterns"]))
        st.subheader(pname)
        if not summ:
            st.info("조건을 통과한 운송 아비가 없습니다.")
            continue
        st.dataframe(pd.DataFrame([
            {"쌍": p, "행": s["행"], "합순이익 ISK": round(s["합순이익"]),
             "묶인 자본": round(s["묶인자본"]), "최고 ISK/m³": round(s["최고_ISK_m3"]),
             "최고 효율 종목": s["최고_종목"], "최고 순익 종목": s["최고_순익종목"]}
            for p, s in sorted(summ.items(), key=lambda kv: -kv[1]["합순이익"])]),
            hide_index=True, width="stretch")
with body[4]:
    names = {tid: meta.get(tid, {}).get("name", f"type {tid}") for tid in scan["books"]}
    pick = st.selectbox("종목", list(names), format_func=lambda t: names[t])
    show_book(scan, meta, pick)

with st.expander("📋 진단 로그"):
    st.text("\n".join(LOG) or "없음")
    st.json({"scanned_at": scan["at"].isoformat(), "candidates": len(scan["books"]),
             "rows": {k: len(v) for k, v in rows.items()}})
