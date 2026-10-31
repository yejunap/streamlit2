"""capital_core 오프라인 점검 — 네트워크 없이 돌린다.

  python3 test_capital_core.py
"""
from capital_core import analyze, price_of, recipe_cost, unit_cost


def _recipe(mats, qty=1, t=3600):
    return {"materials": mats, "product_qty": qty, "time": t}


def test_price_of_shapes():
    book = {1: {"price": 10.0, "qty": 5}, 2: 20}
    assert price_of(book, 1).sell == 10.0 and price_of(book, 1).qty == 5
    assert price_of(book, 2).sell == 20.0
    assert price_of(book, 3).sell is None          # 없는 종목은 없는 채 로


def test_recipe_cost_and_missing():
    rec = _recipe([(1, 2), (2, 1), (99, 7)])       # 99 는 창가 없음
    book = {1: {"price": 10.0}, 2: {"price": 5.0}}
    total, rows, missing = recipe_cost(rec, book)
    assert total == 25.0 and len(rows) == 3 and missing == [99]
    # 단위 가 는 전 량 기准 — 없 는 재 료 는 0 으 로 못 세 고 그냥 제 친다
    u, _, _ = unit_cost(_recipe([(1, 2)]), book)   # 20 / 1개
    assert u == 20.0


def test_analyze_profit_math():
    # 팔 가 1000, 할인 10% → 900, 판매세 7.5% → 832.5; 재 료 비 1×500 (폐 럼 없음 새)
    rec = _recipe([(1, 1)], qty=1, t=1800)
    book = {1: {"price": 500.0}}
    sell = {2: {"price": 1000.0, "qty": 3}}
    r = analyze(2, {"name": "X", "tier": "기본"}, rec, sell, book,
                discount=0.10, sell_tax=0.075, broker=0.0, install=0.0)
    assert r["sell_net"] == 900.0
    assert abs(r["sell_net_after_fee"] - 832.5) < 1e-6
    assert r["unit_cost"] == 500.0
    assert r["profit_unit"] == 332.5
    assert r["profit_total"] == 332.5
    assert r["roi_pct"] == 332.5 / 500 * 100       # 묶 는 재 료 비 대 비
    assert r["hourly"] == 332.5 / 0.5               # 1800초 = 반 시 간


def test_analyze_without_recipe_or_price():
    assert analyze(7, {}, None, {7: {"price": 9.9}}, {})["profit_total"] is None
    assert analyze(7, {}, _recipe([(1, 1)]), {}, {1: {"price": 1}})["profit_total"] is None


def test_analyze_missing_material_blocks_profit():
    rec = _recipe([(99, 1)])                       # 창 가 에 없는 재 룁
    r = analyze(2, {}, rec, {2: {"price": 100}}, {})
    assert r["profit_total"] is None
    assert r["missing_materials"] == [99]


def test_analyze_volume_absorption():
    # 팔 가 1000, 할인 10%, 세 금 7.5% → 순 832.5; 재 료 비 500/개; 발 86400초 → 여 렴 1개/날
    rec = _recipe([(1, 1)], qty=1, t=86400)
    book = {1: {"price": 500.0}}
    sell = {2: {"price": 1000.0, "qty": 3, "total_qty": 10, "orders": 4}}
    zby = dict(discount=0.10, sell_tax=0.075, broker=0.0, install=0.0)
    r = analyze(2, {}, rec, sell, book, volume=5.0, **zby)
    assert r["sell_total_qty"] == 10 and r["sell_orders"] == 4
    assert r["daily_volume"] == 5.0
    assert r["daily_capacity"] == 1.0              # 하루에 1개 밖에 못 만 든 다
    assert r["days_to_sell"] == 2.0                # 현 재 파 는 것 은 이 틀 면 두 날
    assert r["profit_daily"] == 332.5 * 1.0        # 여 렀 을 쫓 아 한 개 뿐
    # 대 용 인 경 우 : 체 결 0.2/날 → 그 것 대 로 만 판 다
    r2 = analyze(2, {}, rec, sell, book, volume=0.2, **zby)
    assert abs(r2["profit_daily"] - 332.5 * 0.2) < 1e-6
    # 체 결 없 는 것 — 일 익 도 모 르 고 소 요 날 도 모 른 것
    r3 = analyze(2, {}, rec, sell, book, volume=None, **zby)
    assert r3["profit_daily"] is None and r3["days_to_sell"] is None


def test_waste_inflates_material_cost():
    # eveind2 식 : 재료비에 설치비 덫 (3%) — 팔 가 1000 손비 없음 재 료 500
    rec = _recipe([(1, 1)], qty=1, t=3600)
    book = {1: {"price": 500.0}}
    r = analyze(2, {}, rec, {2: {"price": 1000.0}}, book,
                discount=0.0, sell_tax=0.0, broker=0.0, install=0.03)
    assert r["unit_cost"] == 515.0                 # 500 × 1.03
    assert r["profit_unit"] == 485.0
    assert r["roi_pct"] == 485.0 / 515.0 * 100.0


if __name__ == "__main__":
    for fn in sorted(globals().items()):
        if fn[0].startswith("test_"):
            fn[1]()
            print("ok", fn[0])
    print("capital_core — 모 두 지나다")