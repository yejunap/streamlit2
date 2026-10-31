"""캐피탈 부품·배의 목록·레시피·창가를 ESI 에서 받는다.

목록은 ESI 타입 그룹에서 안다 (실 조회로 못 박은 값):
  873  Capital Construction Components            — 기본 부품 25종
  913  Advanced Capital Construction Components   — 고급 부품 40종
  513  Freighter                                  — 프레이터 6종
  902  Jump Freighter                             — 점프프레이터 4종
  463  Mining Barge                               — T1 바지선
  543  Exhumer                                    — T2 바지선 (Hulk/Mackinaw/Skiff)
  464  Strip Miner                                — 천층채굴선 (ORE·어비셜 포함)
  380  Deep Space Transport                       — DST 5종
  941  Industrial Command Ship                    — 산업지휘선 (Orca·Porpoise)

레시피는 ESI에 없다 — universe/blueprints 처런 건 존재하지 않는다 (openapi-3.0 전수
확인: blueprint 엔드는 캐릭터/보유 목록뿐). 유일한 곳이 data/recipes.json 이다 —
게임 제조 판에서 베기거나 SDE에서 짚어 넣는다. 형식은 README 에 있다.

창가는 리전 주문판 한 번 통째로 받아 적는다 — 사십 접은 대시보드의 esi.py 를 그대로
쓴다 (ETag·TTL·429 대기가 다 되어 있다).
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASH = os.path.join(os.path.dirname(_HERE), "eve_arb_dashboard")
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)
import esi                                              # noqa: E402  (사 이 브 앱 의 클 라 언 트)

DATA = os.path.join(_HERE, "data")

# 등급(군)별 타입 그룹 — 진영은 곧 표의 등급 열 값
GROUPS = {"기본": 873, "고급": 913, "프리거": 513, "점프": 902,
          "T1바지": 463, "T2바지": 543, "천층": 464, "DST": 380, "ICS": 941}
PART_GROUPS = {k: GROUPS[k] for k in ("기본", "고급")}   # 부품 두 그룹만
TTL_PARTS = 7 * 86400                   # 목록은 확장 통보 전엔 안 바뀐다


# ---------------- 목록 ------------------------------------------------------------
def fetch_parts(groups: dict | None = None, log=None) -> dict:
    """{tid: {"name", "tier", "capacity"}} — GROUPS 전부를 한 장으로.

    groups 로 아무것이나 걸러 물을 수 있다 (기본은 전부) — 부품뿐 아니라
    프레이터(513)·점프프레이터(902) 도 같은 생김새로 온다.
    """
    parts: dict[int, dict] = {}
    for tier, gid in (groups or GROUPS).items():
        body, _ = esi.get(f"/latest/universe/groups/{gid}/", cache_key=f"grp_{gid}",
                          ttl=TTL_PARTS)
        tids = [t for t in (body.get("types") or []) if t not in parts]
        for t in tids:
            parts[t] = {"tier": tier}
        if log:
            log(f"{tier} 그룹 {gid}: {len(tids)}종")
    if not parts:
        raise esi.ESIError("캐피탈 부품 목록을 받지 못했다")
    meta = esi.fetch_type_meta(list(parts), workers=10)
    for t, m in meta.items():
        parts[t].update({"name": m.get("name"), "capacity": m.get("volume")})
    return parts



# ---------------- 레시피 ----------------------------------------------------------
def load_recipes(parts: dict, log=None) -> dict:
    """data/recipes.json 을 읽어 이 부품들에 해당한 것만 채운다.

    없는 종목은 그냥 모른다 — 표에서 레시피 없음으로 표에지고 이익 계산에서 빠진다.
    """
    recipes: dict[int, dict] = {}
    try:
        with open(os.path.join(DATA, "recipes.json"), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        if log:
            log(f"recipes.json 을 못 읽었다({e}) — 레시피 빈칸으로 돈다")
        data = {}
    for k, v in data.items():
        try:
            tid = int(k)
        except ValueError:
            continue                           # 설명 같은 키는 건덕지 없다
        if tid in parts:
            recipes[tid] = {**v, "source": "file"}
    if log:
        log(f"recipes.json — {len(recipes)}/{len(parts)}종 레씨피 확보")
    return recipes


# ---------------- 시장 창가 --------------------------------------------------------
def sell_book(region: int, station: int, progress=None, log=None) -> dict:
    """{tid: {"price": 최 싼 셀 오 가, "qty": 그 칸 물 량}} — 리전 판 한 번 으 로 통 채 로."""
    orders = esi.fetch_hub_orders(region, station, progress=progress, log=log)
    out = {}
    for tid, b in orders.items():
        sells = b.get("sells") or []
        if not sells:
            continue
        price = sells[0][0]                    # sell 는 정렬 됌 — 가 장 싼 한 칸
        qty = sum(q for p, q, _ in sells if p == price)
        out[tid] = {"price": price, "qty": qty,
                    "total_qty": sum(q for _, q, _ in sells),   # 판 에 온 제 물 량
                    "orders": len(sells)}                        # 팔 자 주 문 수
    return out


def daily_volumes(region: int, tids, log=None) -> dict:
    """10 날 하 평 균 체 결 량 — 사 이 브 대 시 보드의 fetch_histories (캐시 새 하루)."""
    out = esi.fetch_histories(region, list(tids), days=10)
    if log:
        log(f"체 결 량 — {len(out)}/{len(tids)}종 받음")
    return out
