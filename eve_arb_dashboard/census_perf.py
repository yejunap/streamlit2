"""성능 확인: 전체 교차에서 조마진금액 선별이 스캔 안에서 감당되는가."""
import sys, time
sys.path.insert(0, ".")
import esi
from arb_core import candidate_scores
bj = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=8)
ba = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION, workers=8)
t0 = time.time()
sc = candidate_scores(bj, ba)
top = sorted(sc.values(), reverse=True)
print(f"교차 {len(set(bj) & set(ba))} → 조마진금액 0+ {len(sc)}건, {time.time()-t0:.1f}s")
if len(top) >= 600:
    print(f"600위 컷의 최저 금액 {top[599]:,.0f} ISK / 최고 {top[0]:,.0f}")
