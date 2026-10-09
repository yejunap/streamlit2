"""arb_core 스무테스트 — 라이브 ESI 없이 돈다. `python3 -m test_arb_core`"""
from arb_core import (analyze_type, candidate_scores, depth_score, internal_inversions,
                      match_ladder)

# 1) 한계이익이 음수가 되면 멈춘다
asks = [(90.0, 100, None), (95.0, 100, None), (99.0, 100, None)]
bids = [(120.0, 50, None), (110.0, 50, None)]
m = match_ladder(asks, bids, buy_fee=0.0, sell_fee=0.0)
assert m["qty"] == 100.0, m        # 50@120 + 50@90
assert m["profit"] == 50 * 120 + 50 * 110 - (50 * 90 + 50 * 90), m
assert m["gross_pct"] > 0

# 2) 수수료가 크면 체결이 안 난다
m2 = match_ladder([(100.0, 100, None)], [(101.0, 100, None)], 0.05, 0.05)
assert m2["qty"] == 0

# 3) 후보 점수 = 창 두께가 아니라 실행 가능한 조마진 '금액'
book_low = {"sells": [(10.0, 100, None)], "buys": [(8.0, 100, None)]}     # Amarr
book_high = {"sells": [(20.0, 50, None)], "buys": [(15.0, 200, None)]}    # Jita
assert depth_score(book_low) == 800.0   # 얇은 쪽: 8×100
# A: Amarp 販(10)를 사 Jita 販(15)에 판다 → 100개 × 5 = 500
assert candidate_scores({1: book_high}, {1: book_low}) == {1: 500.0}
# 스프레드가 없으면(안치는 살살/팔팔) 후보에서 빠진다
flat_l = {"sells": [(10.0, 100, None)], "buys": [(5.0, 100, None)]}
flat_h = {"sells": [(30.0, 50, None)], "buys": [(9.0, 200, None)]}
assert candidate_scores({1: flat_l}, {1: flat_h}) == {}

# 4) 같은 스테이션 내부 역전 탐지 (C/D 후보로 살린다)
inv = {"sells": [(8.0, 1, None)], "buys": [(9.0, 1, None)]}   # 사자 > 팔기
assert internal_inversions({1: flat_l, 2: inv}) == {2}

# 5) 판매세만 붙일 때(즉시판매)와 브로커를 붙일 때(지정가)의 순이익이 달라야 한다
asks5 = [(100.0, 10, None)]
bids5 = [(110.0, 10, None)]
tax_only = match_ladder(asks5, bids5, 0.0, 0.01)
with_broker = match_ladder(asks5, bids5, 0.0, 0.01 + 0.015)
assert tax_only["profit"] > with_broker["profit"] > 0, (tax_only, with_broker)

# 6) 4 패턴 행이 그려지는지
rows = analyze_type(34, "Tritanium", 0.01,
                    {"sells": [(3.0, 1_000_000, None)], "buys": [(3.4, 1_000_000, None)]},
                    {"sells": [(3.3, 1_000_000, None)], "buys": [(3.1, 1_000_000, None)]},
                    0.0, 0.0, daily_vol=10_000, cargo_m3=40_000)
pat = {r["pattern"] for r in rows}
assert "A" in pat and "C" in pat, pat          # Amarr 山 → Jita 販 / Jita 內部
a = next(r for r in rows if r["pattern"] == "A")
assert a["volume_m3"] == a["qty"] * 0.01
assert a["isk_per_m3"] == a["profit"] / a["volume_m3"]
assert a["transport"] is True

# 7) 정렬: 총부피가 크면 앞으로, 부피 없음은 맨 뒤로
from arb_core import RANKERS
big = {"qty": 1, "volume_m3": 2166.0, "profit": 1, "isk_per_m3": None, "profit_per_trip": None}
small = {"qty": 1, "volume_m3": 709.0, "profit": 2, "isk_per_m3": None, "profit_per_trip": None}
none_ = {"qty": 1, "volume_m3": None, "profit": 3, "isk_per_m3": None, "profit_per_trip": None}
f = RANKERS["총부피 m³"]
assert f(big) > f(small) > f(none_), (f(big), f(small), f(none_))

# 8) 통합 앱 — 다른 짚의 라벨이 붙는다 (hub_b 인수)
r = analyze_type(34, "Trit", 0.01,
                 {"sells": [(3.0, 10, None)], "buys": [(3.4, 10, None)]},
                 {"sells": [(3.3, 10, None)], "buys": [(3.1, 10, None)]},
                 0.0, 0.0, 10_000, 40_000, hub_b="Rens")[0]
assert r["direction"] == "Rens→Jita", r
assert r["pair"] == "Jita↔Rens" and r["hub"] == "Rens", r
print("ok")
