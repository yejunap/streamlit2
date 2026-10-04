"""아비 계산 커널 (streamlit non-dependant).

패턴 4 가지를 계산한다.
  A  Amarr sellers → Jita buyers      (운송 O)
  B  Jita  sellers → Amarr buyers     (운송 O)
  C  Jita  sellers → Jita  buyers     (운송 X, 내부 스프레드)
  D  Amarr sellers → Amarr buyers     (운송 X, 내부 스프레드)
"""
from __future__ import annotations

from datetime import datetime, timezone

# 수수료 (% 비례). 프리셋 값:
#   buy          — 살 때 원금에 얹히는 요율 (브로커+세 합산)
#   sell_tax     — 팔 때 판매세 (시장가/지정가 무관 발생)
#   broker_sell  — 팔 때 브로커 수수료. **시장가(즉시판매)면 안 붙는다** — 앱에서 토글
# 수수료 (% 비례). 프리셋 값 (2026-08 EVE University Wiki 기준):
#   sell_tax    — 판매세. 팔면 무조건 발생. 2025-03 패치로 기본 7.5%,
#                 Accounting Lv5면 3.375% (레벨당 -11%)
#   broker_sell — 브로커 수수료. **'immediate'가 아닌 지정가를 걸 때만** 내는 돈
#                 (NPC 스테이션 기본 3%, 스킬/평판으로 ~1.9%까지) — 즉시판매엔 없음
#   buy         — 살 때 원금에 얹히는 요율. 즉시구매(매도호가 소진)면 0이 맞고,
#                 매수지정가를 걸어두면 그 물량 기준의 브로커 수수료가 생긴다
PRESETS = {
    "Accounting5 (세 3.4%)": {"buy": 0.0, "sell_tax": 0.03375, "broker_sell": 0.0188},
    "무스킬 (세 7.5% / 브로커 3%)": {"buy": 0.03, "sell_tax": 0.075, "broker_sell": 0.03},
    "검토 (수수료 0% — 호가 역전만)": {"buy": 0.0, "sell_tax": 0.0, "broker_sell": 0.0},
}


PATTERNS = {
    "A": {"label": "Amarr→Jita", "transport": True},
    "B": {"label": "Jita→Amarr", "transport": True},
    "C": {"label": "Jita 내부", "transport": False},
    "D": {"label": "Amarr 내부", "transport": False},
}


def _now():
    return datetime.now(timezone.utc)


def side_isk(side) -> float:
    return sum(p * q for p, q, _ in side)


def depth_score(book: dict) -> float:
    """한 스테이션에서의 창 두께(ISK) — 적게 얇은 쪽을 따른다."""
    if not book or not book.get("sells") or not book.get("buys"):
        return 0.0
    return min(side_isk(book["sells"]), side_isk(book["buys"]))


def candidate_scores(books_a: dict, books_b: dict) -> dict[int, float]:
    """교차 종목의 후보 점수 = 두 방향으로 벌 수 있는 최고 조마진 금액(ISK).

    예전엔 창 두께를 점수로 썼는데, 그 순서가 시장과 정반다이므로
    조마진이 큰 것을 자른다. 거래량이 큰 것이 이득의 순이라 금액으로 한다.
    """
    out = {}
    for t in set(books_a) & set(books_b):
        best = 0.0
        for asks, bids in (
            (books_b[t].get("sells", []), books_a[t].get("buys", [])),   # A: Amarr 販→Jita 販
            (books_a[t].get("sells", []), books_b[t].get("buys", [])),   # B: Jita 販→Amarr 販
        ):
            best = max(best, match_ladder(asks, bids, 0.0, 0.0)["profit"])
        if best > 0:
            out[t] = best
    return out


def internal_inversions(books: dict) -> set[int]:
    """한 스테이션 안에서 사자 > 팔기인 종목 — C/D 패턴은 교차 종목이 아니어도 산다."""
    return {t for t, b in books.items()
            if b.get("sells") and b.get("buys") and b["buys"][0][0] > b["sells"][0][0]}


def match_ladder(asks: list, bids: list, buy_fee: float, sell_fee: float) -> dict:
    """매도 호가와 매수 호가를 한 칸씩 채운다.

    각 칸의 한계이익은 감소한다며 단조로우므로, 한계이익이 0 이하는 바로 멈춘다.
    """
    ai = bi = aq = bq = 0
    qty = cost = rev = 0.0
    n_asks, n_bids = len(asks), len(bids)
    while ai < n_asks and bi < n_bids:
        pa, qa, _ = asks[ai]
        pb, qb, _ = bids[bi]
        if pb * (1 - sell_fee) <= pa * (1 + buy_fee):
            break
        take = min(qa - aq, qb - bq)
        if take <= 0:
            if aq >= qa:
                ai += 1
                aq = 0
            if bq >= qb:
                bi += 1
                bq = 0
            continue
        qty += take
        cost += take * pa
        rev += take * pb
        aq += take
        bq += take
        if aq >= qa:
            ai += 1
            aq = 0
        if bq >= qb:
            bi += 1
            bq = 0

    if qty <= 0:
        return {"qty": 0.0, "buy_avg": 0.0, "sell_avg": 0.0, "cost": 0.0,
                "revenue": 0.0, "profit": 0.0, "net_pct": 0.0, "gross_pct": 0.0}

    buy_avg, sell_avg = cost / qty, rev / qty
    net_rev, net_cost = rev * (1 - sell_fee), cost * (1 + buy_fee)
    gross = (sell_avg / buy_avg - 1) * 100 if buy_avg else 0.0
    net = (net_rev / net_cost - 1) * 100 if net_cost else 0.0
    return {
        "qty": qty, "buy_avg": buy_avg, "sell_avg": sell_avg,
        "cost": net_cost, "revenue": net_rev, "profit": net_rev - net_cost,
        "gross_pct": gross, "net_pct": net,
    }


def _age_h(orders) -> float | None:
    if not orders or orders[0][2] is None:
        return None
    return (_now() - orders[0][2]).total_seconds() / 3600


def analyze_type(tid: int, name: str, unit_volume: float, a_book: dict, b_book: dict,
                buy_fee: float, sell_fee: float, daily_vol: float,
                cargo_m3: float) -> list[dict]:
    """한 종목의 패턴 A/B/C/D 4 행을 만든다."""
    legs = {
        "A": (b_book.get("sells", []), a_book.get("buys", [])),   # Amar 판 → Jita 판
        "B": (a_book.get("sells", []), b_book.get("buys", [])),   # Jita 판 → Amarr 판
        "C": (a_book.get("sells", []), a_book.get("buys", [])),   # Jita 내부
        "D": (b_book.get("sells", []), b_book.get("buys", [])),   # Amarr 내부
    }
    rows = []
    for key, (asks, bids) in legs.items():
        m = match_ladder(asks, bids, buy_fee, sell_fee)
        if m["qty"] <= 0:
            continue
        vol_m3 = m["qty"] * unit_volume
        row = {
            "type_id": tid, "item": name, "pattern": key,
            "direction": PATTERNS[key]["label"],
            "transport": PATTERNS[key]["transport"],
            "buy_avg": m["buy_avg"], "sell_avg": m["sell_avg"],
            "gross_pct": m["gross_pct"], "net_pct": m["net_pct"],
            "qty": m["qty"], "capital": m["cost"], "profit": m["profit"],
            "unit_volume": unit_volume, "volume_m3": vol_m3,
            "isk_per_m3": (m["profit"] / vol_m3) if vol_m3 > 0 else None,
            "trips": (-(-vol_m3 // cargo_m3)) if vol_m3 > 0 else None,
            "profit_per_trip": (m["profit"] / max(1, -(-vol_m3 // cargo_m3))) if vol_m3 > 0 else None,
            "ask_age_h": _age_h(asks), "bid_age_h": _age_h(bids),
            "daily_volume": daily_vol,
            "days_to_sell": (m["qty"] / daily_vol) if daily_vol else None,
        }
        rows.append(row)
    return rows


RANKERS = {
    "ISK/m³": lambda d: d["isk_per_m3"] if d["isk_per_m3"] is not None else -1,
    "순이익 ISK": lambda d: d["profit"],
    "순이익 %": lambda d: d["net_pct"],
    "1회 운항 이익": lambda d: d["profit_per_trip"] if d["profit_per_trip"] is not None else -1,
    # 규모 자체가 큰 것도 순서대로 볼 수 있게
    "총부피 m³": lambda d: d["volume_m3"] if d["volume_m3"] is not None else -1,
    "체결량": lambda d: d["qty"],
}


def filter_rows(rows: list[dict], min_net_pct: float, min_profit: float,
                min_capital: float, min_daily_volume: float,
                patterns: set[str]) -> list[dict]:
    out = []
    for r in rows:
        if r["pattern"] not in patterns:
            continue
        if r["net_pct"] < min_net_pct or r["profit"] < min_profit:
            continue
        if r["capital"] < min_capital:
            continue
        if r["daily_volume"] < min_daily_volume:
            continue
        out.append(r)
    return out
