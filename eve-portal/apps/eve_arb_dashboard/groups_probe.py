"""groups 엔드포인트 브루트포스 — 이름에 fight/rig 있는 그룹 찾기 (캐시됨)."""
import concurrent.futures as cf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esi

def probe(gid):
    try:
        b, _ = esi.get(f"/latest/universe/groups/{gid}/", cache_key=f"group_{gid}")
        return gid, b.get("name", ""), b.get("types", [])
    except Exception:
        return gid, None, []

hits = []
with cf.ThreadPoolExecutor(max_workers=12) as pool:
    for gid, nm, types in pool.map(probe, range(1, 1600)):
        if nm and ("fight" in nm.lower() or nm.lower().startswith("rig") or "rig " in nm.lower()):
            hits.append((gid, nm, len(types)))
            print("HIT", gid, nm, len(types))
print("합계", len(hits))
with open("/tmp/groups_fight.txt", "w") as f:
    for h in hits:
        f.write(str(h) + "\n")
