"""전체 거래 가능 아이템 목록 만들기.

evemarketbrowser.com은 별도 DB가 아니라 ESI를 그대로 보여주는 뷰어다
(사이트에itemap도 API도 없음이 확인됨). 같은 원천 ESI에서 전체 목록을 만든다:

  1. GET /markets/prices/      → 시장 통계가 있는(=거래 가능한) 전체 type_id
  2. POST /universe/names/     → type_id 묶음 이름 (한 번에 1000개)
  3. (점진) GET /universe/types/{tid} → 부피/group — fetch_type_meta가 이어서 캐시

출력: data/types.csv  type_id,name
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

print("market prices…")
prices = esi.get("/latest/markets/prices/")[0]
ids = {p["type_id"] for p in prices}
print(f"통계 {len(ids)}종목")

# 통계가 없어도 두 허브 창에 주문이 있는 종목까지 — 이들이 없으면 목록이 새진다
print("hub books…")
bj = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=8)
ba = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION, workers=8)
ids |= set(bj) | set(ba)
print(f"창 포함 전체 {len(ids)}종목")
ids = sorted(ids)

def names(chunk):
    import requests
    r = requests.post("https://esi.evetech.net/latest/universe/names/", json=chunk, timeout=30)
    r.raise_for_status()
    return r.json()

out = {}
chunks = [ids[i:i+1000] for i in range(0, len(ids), 1000)]
for n, chunk in enumerate(names(c) for c in chunks):
    for e in chunk:
        out[e["id"]] = e["name"]
    print(f"names {n+1}/{len(chunks)} — 누적 {len(out)}")

with open(os.path.join(DATA, "types.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["type_id", "name"])
    for tid in sorted(out):
        w.writerow([tid, out[tid]])
print(f"완료: data/types.csv ({len(out)}행)")
