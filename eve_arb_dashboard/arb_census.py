"""Jita/Amarr 창 전체에서 아비 후보를 자르지 않고 세어 본다."""
import sys
sys.path.insert(0, ".")
import esi
from arb_core import candidate_scores, depth_score, match_ladder

bj = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=8)
ba = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION, workers=8)
print(f"종목: Jita {len(bj)} / Amarr {len(ba)}")

inter = set(bj) & set(ba)
print(f"교차(두 창에 다 있음): {len(inter)}")

def gross(t):
    out = {}
    for key, (asks, bids) in {
        "A": (ba[t].get("sells", []), bj[t].get("buys", [])),
        "B": (bj[t].get("sells", []), ba[t].get("buys", [])),
        "C": (bj[t].get("sells", []), bj[t].get("buys", [])),
        "D": (ba[t].get("sells", []), ba[t].get("buys", [])),
    }.items():
        m = match_ladder(asks, bids, 0.0, 0.0)
        if m["qty"] > 0:
            out[key] = m
    return out

census = {p: [] for p in "ABCD"}
for t in sorted(inter):
    for p, m in gross(t).items():
        census[p].append((m["gross_pct"], t, m["qty"], m["cost"]))
for p in "ABCD":
    g = sorted(census[p], reverse=True)
    over5 = sum(1 for x in g if x[0] > 5)
    over15 = sum(1 for x in g if x[0] > 15)
    print(f"패턴 {p}: 조마진(수수료0) 0%+ {len(g)}건 | 5%+ {over5} | 15%+ {over15}", end="")
    if g:
        top = g[0]
        print(f" | 최고 {top[0]:.1f}% (type {top[1]}, {top[2]:,.0f}개, 원금 {top[3]/1e6:,.1f}M)")
    else:
        print()
# 그 중 얇은 효재 걸러짐 확인: 창두께 600위 밖의 최대 조마진은?
scores = {t: min(depth_score(bj[t]), depth_score(ba[t])) for t in inter}
keep = {t for _, t in sorted(((s, t) for t, s in scores.items()), reverse=True)[:600]}
for p in "AB":
    g_in = [x for x in census[p] if x[1] in keep]
    g_out = [x for x in census[p] if x[1] not in keep]
    print(f"패턴 {p}: 600위 안={len(g_in)}건 최고 {max([x[0] for x in g_in], default=0):.1f}%"
          f" | 600위 밖(잘림)={len(g_out)}건 최고 {max([x[0] for x in g_out], default=0):.1f}%")
