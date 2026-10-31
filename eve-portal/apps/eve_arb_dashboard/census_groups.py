"""후보 643개가 무슨 그룹인지, 리그/T2에 자리가 있는지 세어 본다."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
import esi
from arb_core import candidate_scores

bj = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=8)
ba = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION, workers=8)
sc = candidate_scores(bj, ba)
print(f"교차 {len(set(bj) & set(ba))} → 조마진금액 후보 {len(sc)}")

def type_info(tid):
    try:
        body, _ = esi.get(f"/latest/universe/types/{tid}/", cache_key=f"type_{tid}")
        return tid, {"gid": body.get("group_id"), "name": body.get("name")}
    except esi.ESIRateLimited:
        return tid, {"gid": -1, "name": f"type {tid}"}

infos = {}
with ThreadPoolExecutor(max_workers=10) as pool:
    for tid, v in pool.map(type_info, list(sc)):
        infos[tid] = v

gids = {v["gid"] for v in infos.values() if v["gid"] and v["gid"] > 0}

def group_name(gid):
    try:
        body, _ = esi.get(f"/latest/universe/groups/{gid}/", cache_key=f"group_{gid}")
        return gid, body.get("name", f"그룹 {gid}")
    except esi.ESIRateLimited:
        return gid, f"그룹 {gid}"

gnames = {}
with ThreadPoolExecutor(max_workers=10) as pool:
    for gid, nm in pool.map(group_name, list(gids)):
        gnames[gid] = nm

per = {}
for tid, v in infos.items():
    nm = gnames.get(v["gid"], "?")
    per.setdefault(nm, []).append(sc[tid])
rows = sorted(per.items(), key=lambda kv: -sum(kv[1]))
print(f"\n{'그룹':<38} {'개수':>4} {'총금액(ISK)':>14}")
for nm, vals in rows[:30]:
    print(f"{nm:<38} {len(vals):>4} {sum(vals):>14,.0f}")
print(f"\n나머지 {len(rows)-30} 그룹 생략")

import re
rigs = {nm: vals for nm, vals in per.items() if re.search(r"rig|amplifier|array|core control|stress|turbine|stabili[sz]er", nm, re.I)}
print("\n— 리그(rig)로 보이는 그룹 —")
for nm, vals in sorted(rigs.items(), key=lambda kv: -sum(kv[1]))[:10]:
    print(f"  {nm}: {len(vals)}건 {sum(vals):,.0f} ISK")
t2 = {nm: vals for nm, vals in per.items() if re.search(r"II$|matrix|tower|amplifier|cruise missile|destructor|destroyer|hyperspatial|drone|covert|strategic|link|sensor|scanner|thermal|kinetic|solar|s_core|cap control|remote|capacitor|tracking|magnetic|power|stasis|meltup", nm, re.I)}
print("\n— T2/모듈성 그룹(대략) —")
for nm, vals in sorted(t2.items(), key=lambda kv: -sum(kv[1]))[:10]:
    print(f"  {nm}: {len(vals)}건 {sum(vals):,.0f} ISK")
