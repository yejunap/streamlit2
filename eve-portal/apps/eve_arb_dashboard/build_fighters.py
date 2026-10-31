"""Heavy Fighter 16종 카탈로그 — 사용자 확정 목록(T1/T2 8쌍, Shadow 제외).

출력: data/fighters.csv  tier,name,type_id,packed_m3,jita,amarr
esi.fetch_type_meta가 packaged_volume(격납 부피) 우선 — 컨테이너 채운 그대로.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esi

# (가족, T1 tid, T2 tid) — types.csv 대조로 확정된 값
PAIRS = [
    ("Ametat", 40362, 40560),
    ("Antaeus", 40364, 40562),
    ("Cyclops", 32325, 40563),
    ("Gungnir", 40365, 40564),
    ("Malleus", 32340, 40561),
    ("Mantis", 32344, 40567),
    ("Termite", 40363, 40566),
    ("Tyrfing", 32342, 40565),
]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "fighters.csv")

def main():
    ids = [t for _, a, b in PAIRS for t in (a, b)]
    meta = esi.fetch_type_meta(ids)
    bj = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=8)
    ba = esi.fetch_hub_orders(esi.AMARR_REGION, esi.AMARR_STATION, workers=8)

    rows = []
    for fam, a, b in PAIRS:
        for tier, tid in (("T1", a), ("T2", b)):
            m = meta.get(tid, {"volume": 0.0, "name": f"?{tid}"})
            rows.append({"tier": tier, "name": m["name"], "type_id": tid,
                         "packed_m3": m["volume"], "jita": tid in bj, "amarr": tid in ba})

    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tier", "name", "type_id", "packed_m3", "jita", "amarr"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'가족':<9} {'T1 부피':>10} {'T2 부피':>10}   창가(J/A)")
    for fam, a, b in PAIRS:
        va, vb = meta[a]['volume'], meta[b]['volume']
        print(f"{fam:<9} {va:>9,.0f} {vb:>10,.0f}   "
              f"T1 {int(a in bj)}/{int(a in ba)}  T2 {int(b in bj)}/{int(b in ba)}")
    print(f"완료: {OUT} ({len(rows)}행)")


if __name__ == "__main__":
    main()
