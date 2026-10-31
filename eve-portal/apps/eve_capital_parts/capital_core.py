"""캐피탈 부품 생산 이익 계산 커널 — 네트워크 없이 표 계산만 한다.

받는 것:
  parts   {tid: {"name", "tier", "capacity"}}          — 부품 메타
  recipe  {"materials": [(소재 tid, 수량), ...], "product_qty": 생산량,
           "time": 생산 시간 초, "install_time": 설치 시간 초}
  books   {tid: {"price": 최저 셀오가, "qty": 그 칸 물량}} — 그 허브의 팔리던 창

가격을 못 본 소재는 에러로 세지 않는다 — 없애라. 없는 걸로 돌린다.
모든 계산은 여기 고이고, 데이터 받는 법은 capital_data 에 있다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Price:
    """한 종목의 파는 창 — 최저 셀오가, 그 칸 물량, 전체 셀 물량·주문 수."""
    sell: float | None = None
    qty: int = 0
    total_qty: int = 0
    orders: int = 0


def price_of(book: dict, tid: int) -> Price:
    """창에서 한 종목 가격만 꺼 낸다. 가격만 준 것도 받고 dict 도 받는다."""
    v = (book or {}).get(tid)
    if v is None:
        return Price()
    if isinstance(v, (int, float)):
        return Price(sell=float(v))
    if isinstance(v, dict):
        p = v.get("price", v.get("sell"))
        return Price(sell=float(p) if p else None, qty=int(v.get("qty") or 0),
                     total_qty=int(v.get("total_qty") or 0), orders=int(v.get("orders") or 0))
    return Price()


def recipe_cost(recipe: dict, book: dict) -> tuple[float, list[dict], list[int]]:
    """전체 재료비 — (합계, 재료별 내역, 가격을 못 본 소재 id)."""
    total = 0.0
    rows, missing = [], []
    for tid, qty in recipe.get("materials", []):
        p = price_of(book, tid)
        if p.sell is None:
            missing.append(tid)
            rows.append({"material": tid, "qty": qty, "price": None, "cost": None})
            continue
        cost = p.sell * qty
        total += cost
        rows.append({"material": tid, "qty": qty, "price": p.sell, "cost": cost})
    return total, rows, missing


def unit_cost(recipe: dict, book: dict) -> tuple[float, list[dict], list[int]]:
    """재료비를 부품 한 개 당으로 나눈다 — (단가, 내역, 모르는 것)."""
    total, rows, missing = recipe_cost(recipe, book)
    out = max(1, int(recipe.get("product_qty") or 1))
    return total / out, rows, missing


def analyze(tid: int, part: dict, recipe: dict | None, sell_book: dict,
            material_book: dict, *, discount: float = 0.10,
            sell_tax: float = 0.034, broker: float = 0.015, install: float = 0.03,
            volume: float | None = None) -> dict:
    """한 종목의 생산 이익 한 행.

    파는 값 = 최저 셀오가 × (1 - discount) — 미리 할인을 심는다.
    파는 편에만 손비를 얹고(판매세·중수수), 사는 재료는 창가 그대로 센다.
    install 은 재료비 대비 설치비 — 재료비를 (1+install)로 불린다 (eveind2: 3%).
    volume 은 최근 10일 하루 평균 체결량 — 하루 생산 여력과 작은 게 진짜 귀찮다.
    레시피가 없거나 창가가 없으면 값은 비우고 표시만 한다 — 표에서 걸러진다.
    """
    price = price_of(sell_book, tid)
    row = {
        "type_id": tid,
        "name": (part or {}).get("name") or f"type {tid}",
        "tier": (part or {}).get("tier") or "",
        "capacity": (part or {}).get("capacity"),
        "sell": price.sell,
        "sell_qty": price.qty,
        "sell_total_qty": price.total_qty,
        "sell_orders": price.orders,
        "daily_volume": None,
        "daily_capacity": None,
        "days_to_sell": None,
        "profit_daily": None,
        "sell_net": None,
        "sell_net_after_fee": None,
        "discount": discount,
        "qty_out": None,
        "time": None,
        "install_time": None,
        "unit_cost": None,
        "material_rows": [],
        "missing_materials": [],
        "has_recipe": bool(recipe),
        "profit_unit": None,
        "profit_total": None,
        "roi_pct": None,
        "gross_margin_pct": None,
        "net_margin_pct": None,
        "hourly": None,
    }
    if price.sell is None or not recipe:
        return row

    net = price.sell * (1.0 - discount)
    net_after = net * (1.0 - (sell_tax + broker))

    unit, rows, missing = unit_cost(recipe, material_book)
    if install:                          # 재료비 대비 설치비 — 내역은 원창가
        unit *= (1.0 + install)
    out_qty = max(1, int(recipe.get("product_qty") or 1))
    row.update({
        "sell_net": net,
        "sell_net_after_fee": net_after,
        "qty_out": out_qty,
        "time": recipe.get("time"),
        "install_time": recipe.get("install_time"),
        "unit_cost": unit,
        "material_rows": rows,
        "missing_materials": missing,
    })
    if missing:                       # 재료 값 모조리 못 본 건 표로 올린다
        return row

    material_total = unit * out_qty
    profit_total = (net_after - unit) * out_qty
    row["profit_unit"] = net_after - unit
    row["profit_total"] = profit_total
    if material_total:                 # 순익 / 재료비 — 묶는 돈 기준 이익률
        row["roi_pct"] = profit_total / material_total * 100.0
        row["gross_margin_pct"] = (net - unit) / net * 100.0
        row["net_margin_pct"] = (net_after - unit) / net_after * 100.0
    t = recipe.get("time") or 0
    if t and profit_total:
        row["hourly"] = profit_total / (t / 3600.0)
    # 하루 생산 가능 수(24시간 돌린다 가정)과 10일 평균 체결량 — 작은 놈이 진짜 돈이다
    cap = (86400.0 / t) * out_qty if t else None
    row["daily_capacity"] = cap
    row["daily_volume"] = volume
    if volume:
        if price.total_qty:
            row["days_to_sell"] = price.total_qty / volume
        row["profit_daily"] = row["profit_unit"] * (min(cap, volume) if cap else volume)
    return row