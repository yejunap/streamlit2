"""arb_core 스무테스트 — 라이브 ESI 없이 돈다. `python3 -m test_arb_core`"""
from arb_core import (analyze_type, candidate_scores, depth_score, internal_inversions,
                      is_blocked, match_ladder, parse_block_add)

# 0) 낚시물 차단 — id랑 이름 둘 다 걸린다, 이름은 대소문자 모른다
_bl = {"ids": {35}, "names": {"junk mail bundle"}}
assert is_blocked(35, "Tritanium", _bl)
assert is_blocked(34, "Junk Mail Bundle", _bl)
assert not is_blocked(34, "Tritanium", _bl)
_ids, _names = parse_block_add("35, Junk Mail Bundle, 9")
assert _ids == {35, 9} and _names == {"junk mail bundle"}, (_ids, _names)

# 0-2) 🧹 장바구니 비우기 — pinned만 아니라 표의(tbl_) 체크도 함께 비운다
from arb_core import clear_basket_state
_ss = {"pinned": {"A"}, "tbl_cross_X": {"selection": {"rows": [0, 1]}},
       "seeded_cross_X": True, "selhit_cross_X": True, "block_off": True}
clear_basket_state(_ss)
assert _ss["pinned"] == set()
assert _ss["tbl_cross_X"]["selection"]["rows"] == []
assert "seeded_cross_X" not in _ss and "selhit_cross_X" not in _ss
assert _ss["block_off"] is True                      # 풀어 둔 표시 같은 건 건드리지 않는다
print("🧹 장바구니 비우기 ok — 체크도 함께 비워짐")

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

# 3) 후보 점수 = 창 두께가 아니라 실행 가능한 조마진 '금액' — 네 갈퀴(A/B/C/D) 돈다
book_low = {"sells": [(10.0, 100, None)], "buys": [(8.0, 100, None)]}     # Amarr
book_high = {"sells": [(20.0, 50, None)], "buys": [(15.0, 200, None)]}    # Jita
assert depth_score(book_low) == 800.0   # 얇은 쪽: 8×100
# A: Amarr 販(10)를 사 Jita 販(15)에 판다 → 100개 × 5 = 500 (C: Jita 販 20에 팔아도 500)
assert candidate_scores({1: book_high}, {1: book_low}) == {1: 500.0}
# 매도↔매도 — A/B가 아무것도 못 내면 매도창 차익만 남는다
#  Jita 販 10에 사서 허브 販 30에 판다 (D) → 50개 × 20
flat_l = {"sells": [(10.0, 100, None)], "buys": [(5.0, 100, None)]}
flat_h = {"sells": [(30.0, 50, None)], "buys": [(9.0, 200, None)]}
assert candidate_scores({1: flat_l}, {1: flat_h}) == {1: 1000.0}
# 네 갈퀴 다 안 나면 후보에서 빠진다 — 반대 방향도 매도창 차익(C)이 난다
assert candidate_scores({1: flat_h}, {1: flat_l}) == {1: 1000.0}   # 허브 10에 사 Jita 30에 판다 (C)
# 네 갈퀴가 겉돈다 — B(10→10.5)보다 D(10→11)가 크므로 큰 걸 점수로 쓴다
assert candidate_scores({1: {"sells": [(10.0, 5, None)], "buys": [(9.0, 5, None)]}},
                        {1: {"sells": [(11.0, 5, None)], "buys": [(10.5, 5, None)]}}) == {1: 5.0}

# 3-2) 매도 ↔ 매도 계산기
from arb_core import match_sell_ladder
# 싸게 사서 비싸 판다 — 파는 창이 얇으면 그 칸 물량만 체결된다
s = match_sell_ladder([(100.0, 500, None)], [(130.0, 100, None)], 0.0, 0.0)
assert s["qty"] == 100.0, s                       # 사던 양은 500이지만 팔 데가 100뿐
assert s["profit"] == 100 * (130.0 - 100.0), s
assert s["buy_avg"] == 100.0 and s["sell_avg"] == 130.0
# 두 창을 번갈아 갚는다 — 사던 창·파는 창 다 한 칸씩 내려간다
s = match_sell_ladder([(100.0, 100, None), (110.0, 100, None)],
                      [(130.0, 100, None), (140.0, 100, None)], 0.0, 0.0)
assert s["qty"] == 200.0, s
assert s["profit"] == 100 * 30 + 100 * 30, s      # (130-100) + (140-110)
# 매도호가보다 매수호가가 비싸면(살 데가 없으면) 체결 없음
assert match_sell_ladder([(130.0, 100, None)], [(100.0, 100, None)], 0.0, 0.0)["qty"] == 0
# 수수료 15%를 견기지 못하면 멈춘다 — 100에 사서 110에 팔아도 남는 게 없다
assert match_sell_ladder([(100.0, 100, None)], [(110.0, 100, None)], 0.0, 0.15)["qty"] == 0
# 판매세를 넣으면 순이익이 줄어든다
gross = match_sell_ladder([(100.0, 100, None)], [(200.0, 100, None)], 0.0, 0.0)
taxed = match_sell_ladder([(100.0, 100, None)], [(200.0, 100, None)], 0.0, 0.075)
assert taxed["profit"] < gross["profit"] and taxed["profit"] > 0, (gross, taxed)
assert gross["net_pct"] == 100.0 and abs(taxed["net_pct"] - (200 * 0.925 - 100)) < 1e-6

# 같은 스테이션 내부 역전 탐지는 이제 패턴이 아니라 후보 안전줄이다
inv = {"sells": [(8.0, 1, None)], "buys": [(9.0, 1, None)]}   # 사자 > 팔기
assert internal_inversions({1: flat_l, 2: inv}) == {2}

# 5) 판매세만 붙일 때(즉시판매)와 브로커를 붙일 때(지정가)의 순이익이 달라야 한다
asks5 = [(100.0, 10, None)]
bids5 = [(110.0, 10, None)]
tax_only = match_ladder(asks5, bids5, 0.0, 0.01)
with_broker = match_ladder(asks5, bids5, 0.0, 0.01 + 0.015)
assert tax_only["profit"] > with_broker["profit"] > 0, (tax_only, with_broker)

# 6) 네 패턴이 그려지는지 — Jita가 싸면 B·D, 허브가 싸면 A·C
rows = analyze_type(34, "Tritanium", 0.01,
                    {"sells": [(3.0, 1_000_000, None)], "buys": [(3.4, 1_000_000, None)]},
                    {"sells": [(3.3, 1_000_000, None)], "buys": [(3.1, 1_000_000, None)]},
                    0.0, 0.0, daily_vol=10_000, cargo_m3=40_000)
pat = {r["pattern"] for r in rows}
assert pat == {"A", "B", "D"}, pat            # A/B는 예전 그대로, D(Jita 販 3.0 → Amarr 販 3.3)이 늘었다
d = next(r for r in rows if r["pattern"] == "D")
assert d["buy_avg"] == 3.0 and d["sell_avg"] == 3.3, d      # 매도창과 매도창을 센다
assert d["transport"] is True and d["volume_m3"] == d["qty"] * 0.01
assert d["direction"] == "Jita→Amarr (매도→매도)", d["direction"]
# 거꾸로 허브 매도창이 싸면 C(허브 → Jita 매도창)가 뜬다
rows_c = analyze_type(34, "Tritanium", 0.01,
                     {"sells": [(3.3, 1_000_000, None)], "buys": [(3.0, 1_000_000, None)]},
                     {"sells": [(3.0, 1_000_000, None)], "buys": [(2.7, 1_000_000, None)]},
                     0.0, 0.0, daily_vol=10_000, cargo_m3=40_000)
c = next(r for r in rows_c if r["pattern"] == "C")
assert c["buy_avg"] == 3.0 and c["sell_avg"] == 3.3, c      # Amarr 3.0에 사 Jita 3.3 매도창에 팔기
assert c["direction"] == "Amarr→Jita (매도→매도)", c["direction"]
a = next(r for r in rows if r["pattern"] == "A")
assert a["isk_per_m3"] == a["profit"] / a["volume_m3"]
assert a["transport"] is True

# 6-2) C/D는 파는 편이 매도 지정가 — 브로커 수수료를 무조건 물린다 (판매세와 별개)
#      C와 D는 한 판에 함께 서지 못하니(Jita가 싸냐 비싸냐) 판을 두 개 쓴다
F1 = ({"sells": [(3.3, 10, None), (3.4, 10, None)], "buys": [(3.29, 10, None)]},        # Jita 비쌈 → C
      {"sells": [(3.0, 10, None), (3.1, 10, None)], "buys": [(3.0, 10, None)]})         # Amarr 쌈
F2 = ({"sells": [(3.0, 10, None), (3.1, 10, None)], "buys": [(3.0, 10, None)]},         # Jita 쌈 → D
      {"sells": [(3.3, 10, None), (3.4, 10, None)], "buys": [(3.29, 10, None)]})        # Amarr 비쌈


def _four(fee_cd):
    out = {}
    for a, b in (F1, F2):
        out.update({r["pattern"]: r for r in analyze_type(
            34, "Trit", 0.01, a, b, 0.0, 0.03375, 0, 40_000, hub_b="Amarr",
            sell_fee_cd=fee_cd)})
    return out


base = _four(0.03375)
fee4 = _four(0.03375 + 0.04)
assert set(base) == {"A", "B", "C", "D"}, sorted(base)
for p in ("C", "D"):
    # 브로커 4%는 팔물 전체에 준다 → 빠진 돈은 4% × (체결량 × 팔가)
    rev = base[p]["qty"] * base[p]["sell_avg"]
    assert round(fee4[p]["profit"], 2) == round(base[p]["profit"] - rev * 0.04, 2), p
for p in ("A", "B"):
    assert fee4[p]["profit"] == base[p]["profit"], p      # A/B는 만지지자

# 7) 정렬: 총부피가 크면 앞으로, 부피 없음은 맨 뒤로
from arb_core import RANKERS
big = {"qty": 1, "volume_m3": 2166.0, "profit": 1, "isk_per_m3": None, "profit_per_trip": None}
small = {"qty": 1, "volume_m3": 709.0, "profit": 2, "isk_per_m3": None, "profit_per_trip": None}
none_ = {"qty": 1, "volume_m3": None, "profit": 3, "isk_per_m3": None, "profit_per_trip": None}
f = RANKERS["총부피 m³"]
assert f(big) > f(small) > f(none_), (f(big), f(small), f(none_))

# 7-2) 셀오더 총량/거래량 — 파는 쪽 허브의 매도잔을 하루 거래로 나눈 값
r = analyze_type(34, "Trit", 0.01,
                 {"sells": [(3.0, 10, None)], "buys": [(3.4, 10, None)]},
                 {"sells": [(3.3, 400, None), (3.5, 100, None)], "buys": [(3.1, 10, None)]},
                 0.0, 0.0, daily_vol=100, cargo_m3=40_000)
d = next(x for x in r if x["pattern"] == "D")
assert d["sell_queue_qty"] == 500.0, d          # D는 Amarr 매도창 총량 400+100
assert d["sell_queue_days"] == 5.0, d           # 하루 100개면닷새치
# 반대로 Jita가 비싸면 C — 파는 편은 Jita 매도창이다
rc = analyze_type(34, "Trit", 0.01,
                  {"sells": [(3.3, 10, None), (3.4, 40, None)], "buys": [(3.0, 10, None)]},
                  {"sells": [(3.0, 10, None)], "buys": [(2.7, 10, None)]},
                  0.0, 0.0, daily_vol=10, cargo_m3=40_000)
c = next(x for x in rc if x["pattern"] == "C")
assert c["sell_queue_qty"] == 50.0, c           # Jita 매도창 10+40
assert c["sell_queue_days"] == 5.0, c
# 거래량을 모르면(0) 셀오더 대비는 비어있되 앞을 막지 않는다
r0 = analyze_type(34, "Trit", 0.01,
                  {"sells": [(3.0, 10, None)], "buys": [(3.4, 10, None)]},
                  {"sells": [(3.3, 400, None)], "buys": [(3.1, 10, None)]},
                  0.0, 0.0, daily_vol=0, cargo_m3=40_000)
assert next(x for x in r0 if x["pattern"] == "D")["sell_queue_days"] is None

# 7-3) 본 순서 — ① 순이익↓ ② 묶인 돈↓ ③ 부피↓ ④ 셀오더/거래량↓
def _row(profit, capital, vol, qdays, name="x"):
    return {"item": name, "pattern": "D", "pair": "Jita↔Amarr", "profit": profit,
            "capital": capital, "volume_m3": vol, "sell_queue_days": qdays,
            "isk_per_m3": None, "profit_per_trip": None}

from arb_core import sort_rows
order = sort_rows([
    _row(5, 9, 100, 1.0, "a 순이익 밀림"),
    _row(9, 9, 100, 1.0, "d 아무것 같음"),
    _row(9, 3, 100, 1.0, "b 묶인돈 적음"),
    _row(9, 9, 10, 1.0, c := "c 부피 적음"),
    _row(9, 9, 100, 0.2, "e 셀오더대비 적음"),
    _row(9, 9, None, 1.0, "f 부피 모름"),
])
# ① 순이익 같으면 → ② 묶인 돈 적게 → ③ 부피 작게 → ④ 셀오더/거래량 작게 (모르면 뒤)
assert [x["item"] for x in order] == ["b 묶인돈 적음", c, "e 셀오더대비 적음",
                                      "d 아무것 같음", "f 부피 모름", "a 순이익 밀림"], \
    [x["item"] for x in order]
# 예전 순서는 ISK/m³가 앞
vp = _row(9, 9, 100, 1.0, "부피효율 낮"); vp["isk_per_m3"] = 1.0
vq = _row(1, 9, 100, 1.0, "부피효율 앞"); vq["isk_per_m3"] = 100.0
assert [x["item"] for x in sort_rows([vp, vq], "① ISK/m³↓ ② 순이익↓ ③ 묶인 자본↓")] \
    == ["부피효율 앞", "부피효율 낮"]

# 7-4) 순이익을 근사치로 밀면(간격%) 아래 자리가 실제로 고른다
#  순이익은 앞서는데 묶인 돈은 뒤인 행과, 반대인 행 — 간격 0이면 앞이 앞에, 간격 10%면 뒤가 앞에
exp = _row(105, 9_000, 100, 1.0, "더 번다(비싸다)")
cheap = _row(100, 1_000, 100, 1.0, "좀 번다(싸다)")
assert [x["item"] for x in sort_rows([exp, cheap])] == ["더 번다(비싸다)", "좀 번다(싸다)"]
tight = [x["item"] for x in sort_rows([exp, cheap], tol=0.10)]
assert tight == ["좀 번다(싸다)", "더 번다(비싸다)"], tight
# 간격 밖(30%% 차이)은 그대로 순이익이 앞
far = _row(140, 9_000, 100, 1.0, "가장 번다")
assert [x["item"] for x in sort_rows([exp, cheap, far], tol=0.10)] == \
    ["가장 번다", "좀 번다(싸다)", "더 번다(비싸다)"]

# 8) 통합 앱 — 다른 짚의 라벨이 붙는다 (hub_b 인수)
r = analyze_type(34, "Trit", 0.01,
                 {"sells": [(3.0, 10, None)], "buys": [(3.4, 10, None)]},
                 {"sells": [(3.3, 10, None)], "buys": [(3.1, 10, None)]},
                 0.0, 0.0, 10_000, 40_000, hub_b="Rens")[0]
assert r["direction"] == "Rens→Jita", r
assert r["pair"] == "Jita↔Rens" and r["hub"] == "Rens", r
print("ok")
