"""표 세우는 일 — 계산 끝난 행을 streamlit 표로 옮긴다 (표시 자체는 여기서 모른다).

규칙 하나: **숫자는 숫자로 싣고, 접씨(11B · 450M)는 마지막에 입힌다.**
표에 글자를 먼저 입히면(1.2B · 1h 20m) streamlit 은 그 칸을 글자로 대우한다 —
머리칸을 눌러 정렬하면 글자 순서대로 센다 (11B 가 900M 앞에 온다). 그래서 순서를
지킨다: build(숫판) → sort_by(숫 순서 확정) → display(ISK 접씨) 순으로 부른다.
정렬은 표 머리칸 대신 app 쪽 select box 에 산다.

점검: python3 test_capital_view.py
"""
from __future__ import annotations

import pandas as pd

# 글자로 남는 열 — 나머지 열은 전부 숫자로 둔다
TEXT_COLS = ("품목", "등급")

# 어떤 열을 어떤 모습으로 뜨는 지 — st.column_config.NumberColumn(format=…) 에 쓴다.
# ISK 열은 여기 없고 display 에서 글자로 바꾼다 (11B · 450M 접씨).
#   %,d       → 천 단위를 세어 읽히는 수
#   %.1f/%.0f → 작은 수 · 비율 (단위는 열 이름에 적었다)
FORMATS: dict[str, str] = {
    # 셈하는 수
    "최저물량": "%,d", "셀총량": "%,d", "주문수": "%,d", "개수": "%,d",
    "재료모름": "%,d",
    # 작은 수 — 하루 체결량 · 하루 생산력 · 소요 며칠 · 생산 몇 시간
    "체결/일": "%.1f", "일산용": "%.1f", "소화소요(일)": "%.1f", "생산시간(h)": "%.1f",
    # 비율 — 단위는 이미 열 이름에 (%)
    "ROI(%)": "%.0f", "순이익률(%)": "%.0f",
}

# ISK로 접어지는(글자로 바뀌는) 열 — 표시 이름
ISK_COLS = ("재료비(ISK)", "셀오가(ISK)", "할인후(ISK)", "이익(ISK)", "일이익(ISK)", "ISK/시간")


def fmt(v) -> str:
    """돈을 영문 접씨로 — 1,234,567,000 → 1.2B. 빈칸은 빈 글자로."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if abs(v) >= 1e12:
        return f"{v / 1e12:,.1f}T"
    if abs(v) >= 1e9:
        return f"{v / 1e9:,.1f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.1f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:,.0f}K"
    return f"{v:,.0f}"


def build(done: pd.DataFrame) -> pd.DataFrame:
    """analyze 행들(capital_core.analyze) → 열 이름 걸린 표. 숫자는 그대로 산다.

    레시피 없거나 창가 없는 행은 값이 비어간 채 온다 — 그 칸은 비워 둔다 (글자 "-"를
    박는 순간 그 열은 글자 열이 되어 정렬이 깨진다).
    """
    t = pd.DataFrame({
        "품목": done["name"],
        "등급": done["tier"],
        "재료비(ISK)": done["unit_cost"],
        "셀오가(ISK)": done["sell"],
        "할인후(ISK)": done["sell_net_after_fee"],
        "최저물량": done["sell_qty"],
        "셀총량": done["sell_total_qty"],
        "주문수": done["sell_orders"],
        "체결/일": done["daily_volume"],
        "소화소요(일)": done["days_to_sell"],
        "일산용": done["daily_capacity"],
        # 발시간은 초로 온다 — 표에서는 시간으로 본다
        "생산시간(h)": done["time"].map(lambda v: v / 3600.0 if v else None),
        "개수": done["qty_out"],
        "재료모름": done["missing_materials"].map(len),
        "이익(ISK)": done["profit_total"],
        "일이익(ISK)": done["profit_daily"],
        "ROI(%)": done["roi_pct"],
        "순이익률(%)": done["net_margin_pct"],
        "ISK/시간": done["hourly"],
    })
    for c in t.columns:
        if c in TEXT_COLS:
            continue
        # 빈 칸은 NaN — 글자로 바꾸지 말 것 (정렬이 글자 순으로 샌다)
        t[c] = pd.to_numeric(t[c], errors="coerce").astype("float64")
    return t.reset_index(drop=True)         # 줄번호가 뒤죽박죽되지 않게


def sort_by(t: pd.DataFrame, column: str, ascending: bool = False) -> pd.DataFrame:
    """표의 한 열을 숫자로 세운다 — 빈 칸은 맨 뒤로 (display 전에 불러야 산다)."""
    return t.sort_values(column, ascending=ascending, na_position="last",
                         kind="stable")


def display(t: pd.DataFrame) -> pd.DataFrame:
    """숫판 정렬이 끝난 표에 ISK 접씨를 입힌다 — 표에 보여주기만 한 판.

    부르는 순서가 옳은 길: 먼저 sort_by 로 숫 순서를 정한 뒤 이것을 부른다.
    글자로 바뀐 열은 머리칸 눌러 정렬하면 글 순서대로 샌다 — 그러니 정렬은
    app 쪽 select box 에 맡긴다.
    """
    d = t.copy()
    for c in ISK_COLS:
        d[c] = d[c].map(fmt)
    return d
