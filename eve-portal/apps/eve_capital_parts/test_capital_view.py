"""표 세우는 일 점검 — streamlit 없이 돈다.

  python3 test_capital_view.py
"""
import pandas as pd

import capital_view
from capital_core import analyze


def _rows():
    """analyze 를 데리고 온다 — 큰돈·작은돈·빈칸을 섞었다."""
    def rec(n, t=3600):
        return {"materials": [(1, n)], "product_qty": 1, "time": t}
    book = {1: {"price": 500.0}}
    zby = dict(discount=0.10, sell_tax=0.05, broker=0.0, install=0.0)
    big = analyze(11, {"name": "빅", "tier": "고급"}, rec(2000),
                  {11: {"price": 6_000_000.0, "qty": 5, "total_qty": 40, "orders": 7}},
                  book, volume=2.0, **zby)
    small = analyze(12, {"name": "스몰", "tier": "기본"}, rec(1),
                    {12: {"price": 900.0, "qty": 9}}, book, volume=30.0, **zby)
    novol = analyze(13, {"name": "노권", "tier": "기본"}, rec(3),
                    {13: {"price": 800.0}}, book, volume=None, **zby)
    return pd.DataFrame([big, small, novol])


def test_columns_stay_numeric():
    """숫 열은 build 에서 끝까지 숫 — display 가 마지막에 ISK만 글자로 바꾼다."""
    t = capital_view.build(_rows())
    for c in t.columns:
        is_num = pd.api.types.is_numeric_dtype(t[c])
        if c in capital_view.TEXT_COLS:
            assert not is_num, f"{c} 는 글자 열이어 한다"
        else:
            assert is_num, f"{c} 가 글자 열이다: {t[c].dtype}"
    # 없는 칸은 '-' 글자가 아니라 NaN 으로 남는다
    assert t["일이익(ISK)"].isna().sum() == 1


def test_formats_cover_every_numeric_column():
    """FORMATS + ISK 열 합이 숫 열 全体를 감싼다 — 새 열은 빼두지 말 것."""
    t = capital_view.build(_rows())
    nums = {c for c in t.columns if c not in capital_view.TEXT_COLS}
    assert set(capital_view.FORMATS) | set(capital_view.ISK_COLS) == nums
    assert not set(capital_view.FORMATS) & set(capital_view.TEXT_COLS)


def test_sort_is_numeric_not_text():
    """글자로 세우면 1.0M 이 500 앞에 온다 — 표는 숫으로 서야 한다."""
    t = capital_view.build(_rows())
    asc = capital_view.sort_by(t, "재료비(ISK)", ascending=True)
    assert list(asc["품목"]) == ["스몰", "노권", "빅"], list(asc["품목"])
    # 글자로 세우면 이런다 — 순서가 뒤바뀐다
    as_text = t["재료비(ISK)"].map(capital_view.fmt)
    wrong = list(t.loc[as_text.sort_values().index, "품목"])
    assert wrong != list(asc["품목"]), "글자 순서가랑 같다 — 숫으로 안 서고 있다"


def test_display_applies_compact_after_sort():
    """display 는 접씨(11B · 450M)만 입힌다 — 순서는 손대지 않는다."""
    t = capital_view.build(_rows())
    t = capital_view.sort_by(t, "재료비(ISK)", ascending=True)
    d = capital_view.display(t)
    assert list(d["품목"]) == ["스몰", "노권", "빅"]          # 순서 유지
    assert d["재료비(ISK)"].tolist() == ["500", "2K", "1.0M"]   # 1500 은 K 선에서 업힌다
    # ISK가 아닌 열은 그대로 숫이다
    assert pd.api.types.is_numeric_dtype(d["ROI(%)"])
    # 접씨 생김 — 영문 T · B · M · K
    assert capital_view.fmt(11_234_567_890) == "11.2B"
    assert capital_view.fmt(450_000_000) == "450.0M"
    assert capital_view.fmt(1_234_567_890_000) == "1.2T"
    assert capital_view.fmt(float("nan")) == ""


def test_time_becides_hours_and_index_resets():
    t = capital_view.build(_rows())
    assert list(t.index) == [0, 1, 2]                      # 줄번호가 뒤죽박죽 안 된다
    assert "소화소요(일)" in t.columns and "생산시간(h)" in t.columns
    assert t.loc[0, "생산시간(h)"] == 1.0                  # 3600 초는 한 시간


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("capital_view — 모두 지나갔다")
