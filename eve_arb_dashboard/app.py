"""Jita 4-4 <-> Amarr VIII 아비 대시보드 (Streamlit).

실행: streamlit run app.py
"""
from __future__ import annotations

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
                      internal_inversions)

try:
    st.set_page_config(page_title="EVE 허브 아비 — Jita↔Amarr", layout="wide", page_icon="📈")
except Exception:              # 테스트용 가짜 스텁 모듈이면 넘어간다
    pass


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
def run_scan(workers: int, max_candidates: int, log) -> dict:
    """두 허브의 주문판을 받아 후보 종목만 남긴다 (첫 실행은 수 분)."""
    bar = Bar(587, "스테이션 주문판 다운로드")
    done = bar.tick

    log("Jita IV - Moon 4 주문판…")
    books_j = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION,
                                   workers=workers, progress=done, log=log)
    log("Amarr VIII (Oris) 주문판…")
    books_a = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION,
                                   workers=workers, progress=done, log=log)
    bar.done()

    scores = candidate_scores(books_j, books_a)
    keep = [t for _, t in sorted(((s, t) for t, s in scores.items()), reverse=True)][:max_candidates]
    # 내부 스프레드는 교차 종목이 아니어도 사나운 것이므로 무조건 데리고 있다.
    keep += list(internal_inversions(books_j) | internal_inversions(books_a))
    keep_n = "전체" if max_candidates >= 10 ** 8 else f"상위 {max_candidates}"
    log(f"교차 {len(scores)} → 조마진금액 {keep_n} + 내부역전 → {len(keep)}종목")
    return {"books": {t: {"Jita": books_j.get(t, {"sells": [], "buys": []}),
                          "Amarr": books_a.get(t, {"sells": [], "buys": []})} for t in keep},
            "scores": {t: scores.get(t, 0) for t in keep},
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
COLUMNS = ["종목", "패턴", "방향", "매집가", "청산가", "조마진%", "순이익%", "체결량",
           "순이익 ISK", "묶인 자본", "단위부피 m³", "총부피 m³", "ISK/m³", "필요 운항",
           "1회 운항 이익", "일평균 거래량", "소요일", "매도호가 유지 h", "매수호가 유지 h"]
SRC = ["item", "pattern", "direction", "buy_avg", "sell_avg", "gross_pct", "net_pct",
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
        d.metric("운항 횟수 (캐런 57,500m³)", f"{-(-tot_v // 57500):,.0f}")


def show_rows(rows: list[dict], title: str):
    if not rows:
        st.info(f"{title}: 조건을 통과한 아비가 없습니다.")
        return
    # 3단계 고정: ① ISK/m³ 내림차순 ② 순이익 내림차순 ③ 묶인 자본 오름차순
    rows = sorted(rows, key=lambda d: (-(d["isk_per_m3"] or 0), -d["profit"], d["capital"]))
    rows = [r for r in rows if r["isk_per_m3"] is not None]
    st.caption(f"{len(rows)}행 전체 · 정렬: **① ISK/m³↓ ② 순이익↓ ③ 묶인자본↑** — "
               "**아무 열 머리말이나 누르면 그 열로 바로 정렬됩니다**")
    tbl = to_table(rows)
    # 쇼핑 체크: 앞으로 체크된 종목은 종목명에 🟩 (표 배경색 커스텀은 Streamlit 한계)
    pinned = st.session_state.get("pinned", set())
    marked = tbl.copy()
    checked = marked["종목"].isin(pinned)
    marked["✓"] = checked
    marked.loc[checked, "종목"] = "🟩 " + marked.loc[checked, "종목"]
    marked = marked[["✓"] + COLUMNS]
    cfg = {c: st.column_config.NumberColumn(format="localized") for c in DECIMALS}
    cfg["✓"] = st.column_config.CheckboxColumn("✓", default=False,
                                               help="체크하면 장바구니 — 맨 위 합산 창에 총부피·총뭉인 자본이 뜬다")
    ed = st.data_editor(
        marked, key=f"ed_{title}", hide_index=True, width="stretch",
        height=min(1200, 34 * (len(marked) + 2)),
        column_config=cfg, disabled=[c for c in marked.columns if c != "✓"])
    if isinstance(ed, pd.DataFrame):
        now = {n.replace("🟩 ", "") for n in ed.loc[ed["✓"].fillna(False), "종목"]}
        was = pinned & set(tbl["종목"])   # 이 표에 체크된 채로 그려かった 것
        if now != was:                   # 이 표가 실제로 바뀜 — 남은 표의 기여는 살리고 합친다
            merged = (pinned - set(tbl["종목"])) | now
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
    st.sidebar.caption("Jita 4-4 ↔ Amarr VIII (Oris) — 주문판 전량 → 창두께 상위 종목")
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
    cargo = st.sidebar.number_input("1회 적재 가능 부피 (m³)", 100, 500_000, 57_500, step=1000,
                                    help="탈 배의 화물함 용량")

    st.sidebar.header("🔻 필터")
    min_pct = st.sidebar.number_input("최소 순이익 %", 0.0, 100.0, 1.0, step=0.5)
    min_profit = st.sidebar.number_input("최소 순이익 ISK", 0, 10_000_000_000, 0, step=100_000)
    min_cap = st.sidebar.number_input("최소 묶인 자본 ISK", 0, 10_000_000_000, 0, step=1_000_000)
    min_dv = st.sidebar.number_input("최소 일평균 거래량", 0, 1_000_000, 0, step=100)
    patterns = st.sidebar.multiselect("패턴", list("ABCD"), default=list("ABCD"))
    return dict(scan_now=scan_now, workers=workers, max_cand=max_cand, get_meta=get_meta,
                fees=fees, presets=presets, cargo=cargo, min_pct=min_pct,
                min_profit=min_profit, market_sell=sell_market,
                min_cap=min_cap, min_dv=min_dv, patterns=set(patterns))


def show_book(scan: dict, meta: dict, tid: int):
    """선택 종목의 양변 4면 호가창(상위 5칸)과 주문 유지 시간을 보여준다."""
    b = scan["books"][tid]
    cols = st.columns(2)
    for i, hub in enumerate(("Jita", "Amarr")):
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
st.title("📈 Jita 4-4 ↔ Amarr VIII 아비 대시보드")
st.caption("패턴 A/B: 허브 간 운송 아비 · C/D: 같은 스테이션 내부 스프레드 (운송 0, 부피 무의미)")

cfg = sidebar()
LOG: list[str] = []
log = lambda m: LOG.append(m)

if cfg["scan_now"]:
    with st.spinner("주문ware 받는 중…"):
        st.session_state["scan"] = run_scan(cfg["workers"], cfg["max_cand"], log)
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

rows: dict[str, list] = {p: [] for p in cfg["presets"]}
for tid, books in scan["books"].items():
    mv = meta.get(tid, {}).get("volume", 0.0)
    if not mv:
        continue
    for pname in cfg["presets"]:
        fees = cfg["fees"][pname]
        # 즉시판매(시장가)면 판매 브로커 수수료 없음 — 판매세만
        sell_fee = fees["sell_tax"] if cfg["market_sell"] else fees["sell_tax"] + fees["broker_sell"]
        rows[pname] += analyze_type(
            tid, meta[tid]["name"], mv, books["Jita"], books["Amarr"],
            fees["buy"], sell_fee, hist.get(tid, 0.0), cfg["cargo"])

# ---- 🛒 장바구니 합산기 (체크한 종목) ----------------------------------------------------------
# placeholder: 체크가 바뀐 바로 그 런에 show_rows가 채워 넣는다 — 강제 리런 없이 실시간 갱신.
basket_box = st.empty()
if st.button("🧹 장바구니 비우기", key="clear_basket",
             disabled=not st.session_state.get("pinned")):
    # 클릭한 이 런에 상태를 비우면 그 아래 표가 그릴 때 체크가 풀린다 — 리런 없음
    st.session_state["pinned"] = set()
render_basket(basket_box, rows, st.session_state.get("pinned", set()))

body = st.tabs(["🔀 크로스 아비 (A/B)", "🏪 내부 스프레드 (C/D)", "🃀 전체 통합", "🔬 종목 상세"])


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
    names = {tid: meta.get(tid, {}).get("name", f"type {tid}") for tid in scan["books"]}
    pick = st.selectbox("종목", list(names), format_func=lambda t: names[t])
    show_book(scan, meta, pick)

with st.expander("📋 진단 로그"):
    st.text("\n".join(LOG) or "없음")
    st.json({"scanned_at": scan["at"].isoformat(), "candidates": len(scan["books"]),
             "rows": {k: len(v) for k, v in rows.items()}})
