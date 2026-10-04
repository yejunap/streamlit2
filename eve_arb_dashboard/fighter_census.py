"""Fighter 그룹(legacy light/medium/heavy fighter)이 두 허브 창에서 어떤姿かかかているか."""
import sys
sys.path.insert(0, ".")
import requests
import esi

r = requests.post("https://esi.evetech.net/latest/universe/ids/",
                  json={"groups": ["Fighter", "Light Fighter", "Medium Fighter", "Heavy Fighter"]},
                  timeout=15)
j = r.json()
print("그룹:", j.get("groups", []))
alltypes = []
for g in j.get("groups", []):
    b, _ = esi.get(f"/latest/universe/groups/{g['id']}/types/", cache_key=f"grptype_{g['id']}")
    alltypes += b
print("그룹 내 타입 수:", len(alltypes))

bj = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=8)
ba = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION, workers=8)
inb = [t for t in alltypes if (t in bj) or (t in ba)]
print("두 창 중 어딘가 호가 존재:", len(inb))

names = {}
def nm(tid):
    if tid not in names:
        try:
            b, _ = esi.get(f"/latest/universe/types/{tid}/", cache_key=f"type_{tid}")
            names[tid] = b.get("name", f"?{tid}")
        except esi.ESIRateLimited:
            names[tid] = f"?{tid}"
    return names[tid]

def bid(b):
    return max((o.price for o in b.buys), default=0) if b else 0

def ask(b):
    return min((o.price for o in b.sells), default=0) if b and b.sells else 0

for t in inb:
    jb, ab = bj.get(t), ba.get(t)
    print(f"  {nm(t)[:28]:28} Jita {bid(jb):>12,.0f}/{ask(jb):<12,.0f}  Amarr {bid(ab):>12,.0f}/{ask(ab):<12,.0f}")

