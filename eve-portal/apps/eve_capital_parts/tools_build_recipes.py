"""S데 CSV 에서 부품 65종+프레이터 10종의 레씨피를 추려 data/recipes.json 에 적는다.

제품 목록은 capital_data.fetch_parts() — 부품(873·913)에 프레이터(513)·점프(902)까지.

출처: https://www.fuzzwork.co.uk/dump/latest/csv/ (CCP SDE 변환)
  industryActivity.csv            (typeID=블루프린트, activityID, time초)
  industryActivityMaterials.csv   (블루프린트, activityID, materialTypeID, quantity)
  industryActivityProducts.csv    (블루프린트, activityID, productTypeID, quantity)
"""
import csv
import json
import sys

sys.path.insert(0, "/home/slreflex2/.cline/data/workspaces/chat/streamlit2"
                   "/eve-industry/apps/eve_capital_parts")
import capital_data as cd
import esi

OUT = ("/home/slreflex2/.cline/data/workspaces/chat/streamlit2/eve-industry/"
       "apps/eve_capital_parts/data/recipes.json")


def rows(name):
    with open(f"/tmp/{name}.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            yield r


parts = cd.fetch_parts()
tids = set(parts)
tier_of = {t: p.get("tier") for t, p in parts.items()}

# 제품 -> (블루프린트, activity) 짝
by_product = {}
for r in rows("industryActivityProducts"):
    pt = int(r["productTypeID"])
    if pt in tids:
        by_product.setdefault(pt, []).append((int(r["typeID"]), int(r["activityID"]),
                                              int(r["quantity"])))

times = {(int(r["typeID"]), int(r["activityID"])): int(r["time"])
         for r in rows("industryActivity")}
mats = {}
for r in rows("industryActivityMaterials"):
    mats.setdefault((int(r["typeID"]), int(r["activityID"])), []).append(
        [int(r["materialTypeID"]), int(r["quantity"])])

recipes, weird = {}, []
for pt, acts in by_product.items():
    # 반응(activity 8)이면 그것, 없으면 처음 거 — 하나에 붙는다
    bp, act, q = next((a for a in acts if a[1] == 8), acts[0])
    if len(acts) > 1:
        weird.append((pt, [a[1] for a in acts]))
    got = mats.get((bp, act))
    if not got:
        continue
    recipes[str(pt)] = {"materials": [[m, q2] for m, q2 in got],
                        "product_qty": q, "time": times.get((bp, act))}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(recipes, f, ensure_ascii=False, indent=1)

print(f"레씨피 {len(recipes)}/{len(tids)}종 — 모는 것 {sorted(tids - {int(k) for k in recipes})}")
print("활동 여러 개인 제품:", weird[:8], "…") if weird else None

# 사람 검증: 표본 재료 이름 (기본 부품은 고급 단위, 고급은 티투 부속이려야 한다)
sample = ["21009"] + [t for t in list(tids)[:1] if str(t) in recipes]
sample = [int(s) for s in list(recipes)[:2]]
for pt in sample:
    mids = [m for m, _ in recipes[str(pt)]["materials"]][:6]
    names = {t: esi.fetch_type_meta([t]).get(t, {}).get("name", t) for t in mids}
    print(f"\n{pt} {parts[pt]['name']} ({tier_of[pt]}) x{recipes[str(pt)]['product_qty']} "
          f"{recipes[str(pt)]['time']}s")
    for t, n in names.items():
        mark = " ← 913/873 부품" if t in tids else ""
        print("   -", n, mark)

# 재료의 부품-간-의존 통계
dep = sum(1 for k, r in recipes.items() for m, _ in r["materials"] if int(m) in tids)
tot = sum(len(r["materials"]) for r in recipes.values())
print(f"\n재료중 부품 상호 의젼 {dep}/{tot}")
