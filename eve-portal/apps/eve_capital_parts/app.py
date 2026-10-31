"""캐피탈 부품 생산 이익 — 독립 실행 앱.

사 이브 대시보드_apps/eve_arb_dashboard의 포맷을 본뜬다: 사이드바에 허브·할인·세금,
본문에 이익 표. 데이터는 물리지 않고 ESI에서 바로 받는다 — 계산은 capital_core 에 산다.

실행: streamlit run apps/eve_capital_parts/app.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "eve_arb_dashboard")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capital_core  # noqa: E402
import capital_data  # noqa: E402
import capital_view  # noqa: E402
import esi           # noqa: E402

st.set_page_config(page_title="EVE 캐피탈 부품 이익", layout="wide", page_icon="🏭")

# 허브 (리전 ID, 스테이션 ID) — 사 이브 대시 보드가 기 용하 는 값 이다.
# esi.HUBs 에 없는 것은 스테이션 ID 로 붙인다 (Heita 는 신규 허브).
HUBS = {
    "Jita": (10000002, 60003760),
    "Amarr": (10000043, 60008494),
    "Dodixie": (10000032, 60011866),
    "Rens": (10000030, 60004588),
    "Heita": (10000038, 60002526),
}


def _dur(v) -> str:
    if not v or (isinstance(v, float) and pd.isna(v)):
        return "-"
    v = int(v)
    h, rem = divmod(v, 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


_fmt = capital_view.fmt         # 재료 내음 표시용 — 표와 같은 접씨


st.title("🏭 캐피탈·채굴선 생산 이익")
st.caption("재료비는 창가 그대로, 파는 값은 최저 셀오가에 할인을 입혀 센다 — "
           "계산 전부는 이 폴더의 capital_core 에 산다.")

with st.sidebar:
    st.header("⚙️ 설정")
    hub_name = st.selectbox("판매 허브", list(HUBS))
    discount = st.slider("판매 할인 (%)", 0.0, 30.0, 10.0, step=2.5) / 100.0
    sell_tax = st.slider("판매세 (%)", 0.0, 5.0, 3.4, step=0.1) / 100.0
    broker = st.slider("브로커 수수료 (%)", 0.0, 5.0, 1.5, step=0.1) / 100.0
    install = st.slider("설치비 (재료비 대비 %)", 0.0, 10.0, 3.0, step=0.1) / 100.0
    tier_pick = st.multiselect("등급/군", list(capital_data.GROUPS),
                               default=list(capital_data.GROUPS))
    st.divider()
    # 정렬 — ISK 열은 표에서 글자(11B)로 바뀌니 머리칸 클릭은 세지 않는다. 여기서 산다.
    st.write("**정렬**")
    _sort_cols = list(capital_view.ISK_COLS) + list(capital_view.FORMATS)
    sort_col = st.selectbox("기준 열", _sort_cols, index=capital_view.ISK_COLS.index("일이익(ISK)"))
    sort_asc = st.toggle("올림차순", value=False)
    st.divider()
    if st.button("🗑 캐시 지우고 새로 고침", use_container_width=True):
        for f in glob.glob(os.path.join(esi.CACHE_DIR, "*")):
            os.remove(f)
        st.rerun()

if not st.button("📦 집합 돌리기", use_container_width=True, type="primary"):
    st.info("버튼을 눌러 첫 집합을 돌린다. ESI 캐시(ETag·TTL) 덕분에 같은 버튼은 "
            "더해 받을수록 빨라진다.")
    st.stop()

region, station = HUBS[hub_name]
logs: list[str] = []
log = logs.append
bar = st.progress(0.0, text="목록 받는 중")

try:
    t0 = time.time()
    parts = capital_data.fetch_parts(log=log)
    parts = {t: p for t, p in parts.items() if p.get("tier") in tier_pick}

    bar.progress(0.2, text=f"레시피 — {len(parts)}종")
    recipes = capital_data.load_recipes(parts, log=log)

    bar.progress(0.5, text="시장 창가 받는 중")
    book = capital_data.sell_book(region, station, log=log)
    volumes = capital_data.daily_volumes(region, parts, log=log)
    bar.progress(1.0, text="계산")
except Exception as e:                              # 무슨 일엔가 무너지면 로그를 까낸다
    st.error(f"집합 실패: {e}")
    with st.expander("진단 로그"):
        st.code("\n".join(logs) or "없는 일")
    st.stop()

rows = [capital_core.analyze(tid, parts[tid], recipes.get(tid), book, book,
                             discount=discount, sell_tax=sell_tax, broker=broker,
                             install=install, volume=volumes.get(tid))
        for tid in parts]
done = pd.DataFrame(rows)

names: dict[int, str] = {t: (p.get("name") or f"type {t}") for t, p in parts.items()}
mat_ids = {m["material"] for r in rows for m in r["material_rows"]} - set(names)
if mat_ids:
    for t, m in esi.fetch_type_meta(sorted(mat_ids)).items():
        names[t] = m.get("name") or f"type {t}"

# 걸러 정렬은 숫자가 살아있는 데 먼저 — 글자(11B)로 바뀌고 나면 순서가 뒤집힌다
# 편일(일당 이익) 순 — 안 파는 물건은 순익이 커도 소용없다
work = done[done["profit_total"].notna()]
table = capital_view.build(work)
table = capital_view.sort_by(table, sort_col, ascending=sort_asc)
shown = capital_view.display(table)         # ISK 접씨는 마지막에 — CSV는 숫판로 간다
# 글자는 여기서만 입힌다 — 남는 숫 열은 column_config 로 뜬다
col_cfg = {c: st.column_config.NumberColumn(format=f)
           for c, f in capital_view.FORMATS.items()}

st.subheader(f"📊 {hub_name} — {len(table)}종")
st.caption("**정렬은 사이드 바** — ISK 열은 접씨(11B · 450M)로 뜨니 머리칸 클릭은 세지 않는다")
st.dataframe(shown, hide_index=True, width="stretch", column_config=col_cfg,
             height=min(38 * (len(table) + 2), 900))

with st.expander("재료 내음"):
    for _, r in work.head(12).iterrows():
        st.markdown(f"**{r['name']}** — 재료비 {_fmt(r['unit_cost'])}, "
                    f"이익 {_fmt(r['profit_total'])}, 발시간 {_dur(r['time'])}")
        mats = pd.DataFrame([{**m, "name": names.get(m["material"], m["material"])}
                             for m in r["material_rows"]])
        st.dataframe(mats[["name", "qty", "price", "cost"]], hide_index=True)
        st.divider()

with st.expander(f"진단 로그 ({len(logs)}줄)"):
    st.code("\n".join(logs) or "없는 일")

st.download_button(
    "CSV로 받기",
    table.to_csv(index=False).encode("utf-8-sig"),
    f"capital_parts_{hub_name}_{time.strftime('%Y-%m-%d')}.csv",
    "text/csv",
)
st.caption(f"{time.time() - t0:.1f}초")