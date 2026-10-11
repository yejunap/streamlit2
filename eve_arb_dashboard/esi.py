"""ESI (EVE Swagger Interface) 클라이언트 — 허브 간 아비 대시보드용.

- ESI 권장 정책(User-Agent, ETag 조건부 요청, 420/429 대기)을 지킨다.
- 전체 주문판(리전별 페이지)을 여러 스레드로 당겨온다.
- 부피/이름/거래량은 타입별 GET 로 때운다 (ESI 에 벌크 엔드포인트가 없다).
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

ESI = "https://esi.evetech.net"
UA = "eve-arb-dashboard/1.0 (+local analysis tool)"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Jita IV - Moon 4 - Caldari Navy Assembly Plant (Jita 4-4)
JITA_SYSTEM, JITA_REGION, JITA_STATION = 30000142, 10000002, 60003760
# Amarr VIII (Oris) - Emperor Family Academy
AMARR_SYSTEM, AMARR_REGION, AMARR_STATION = 30002187, 10000043, 60008494
# Dodixie IX - Moon 20 - Federation Navy Assembly Plant (Sinq Laison, 0.9)
DODIXIE_SYSTEM, DODIXIE_REGION, DODIXIE_STATION = 30002659, 10000032, 60011866
# Rens VI - Moon 8 - Brutor Tribe Treasury (Heimatar, 0.9)
RENS_SYSTEM, RENS_REGION, RENS_STATION = 30002510, 10000030, 60004588

HUBS = {
    "Jita": {"system": JITA_SYSTEM, "region": JITA_REGION, "station": JITA_STATION},
    "Amarr": {"system": AMARR_SYSTEM, "region": AMARR_REGION, "station": AMARR_STATION},
    "Dodixie": {"system": DODIXIE_SYSTEM, "region": DODIXIE_REGION, "station": DODIXIE_STATION},
    "Rens": {"system": RENS_SYSTEM, "region": RENS_REGION, "station": RENS_STATION},
}

# Jita를 중심으로 붙이는 쪽이 기본 (모두 두 번 계산하면 서로 같으므로 한 번씩)
SECONDARIES = ["Amarr", "Dodixie", "Rens"]


def region_pages(region: int) -> int:
    """리전 주문판의 전체 페이지 수 — 진행 칸 세는 데만 쓰고, 본문은 버린다."""
    _, hdrs = get(f"/latest/markets/{region}/orders/", params={"page": 1})
    return int(hdrs.get("X-Pages", 1))


class ESIRateLimited(Exception):
    pass


_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        import requests as rq
        s = rq.Session()
        s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
        _SESSION = s
    return _SESSION


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _load_etag(key: str):
    p = _cache_path(f"etag_{key}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_etag(key: str, etag: str, body):
    with open(_cache_path(f"etag_{key}.json"), "w", encoding="utf-8") as f:
        json.dump({"etag": etag, "body": body}, f)

def get(path: str, params: dict | None = None, cache_key: str | None = None,
        retries: int = 4):
    """GET 하고 (json, headers) 반환. cache_key 를 주면 ETag 조건부 요청을 쓴다."""
    url = f"{ESI}{path}"
    headers = {}
    if cache_key:
        cached = _load_etag(cache_key)
        if cached and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
    last_err = None
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, headers=headers, timeout=45)
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 304 and cache_key:
            return _load_etag(cache_key)["body"], {}
        if r.status_code in (420, 429):
            # 요청이 많을 때 이 속하면 한 번 걸린답쳐 전체 스캔이 깨지면 안 된다 —
            # Retry-After 를 들어보되 10/20/30초로 점진적으로 머무르고 참는다.
            asked = int(r.headers.get("Retry-After") or 0)
            time.sleep(min(max(asked, 10 * (attempt + 1)), 30))
            last_err = ESIRateLimited(f"ESI rate limited ({r.status_code})")
            continue
        if r.status_code == 503:
            time.sleep(2 + attempt * 2)
            continue
        r.raise_for_status()
        body = r.json()
        if cache_key and r.headers.get("ETag"):
            _save_etag(cache_key, r.headers["ETag"], body)
        return body, dict(r.headers)
    raise ESIRateLimited(f"ESI failed for {path}: {last_err}")


def _tick(progress) -> None:
    """progress 는 호가 함수나 .update() 객체 둘 다 된다. 없으면 참는다."""
    if not progress:
        return
    if callable(progress):
        progress(1)
        return
    upd = getattr(progress, "update", None)
    if upd:
        upd(1)


def _iso(v):
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def fetch_hub_orders(region: int, station: int, workers: int = 8,
                     progress=None, log=None) -> dict:
    """리전 전체 주문을 받아 지정한 스테이션 하나만 남긴다.

    반환: {type_id: {'sells': [(price, qty, issued)], 'buys': [...]}}
    """
    first, hdrs = get(f"/latest/markets/{region}/orders/", params={"page": 1})
    total = int(hdrs.get("X-Pages", 1))

    def one(pg: int):
        body, _ = get(f"/latest/markets/{region}/orders/", params={"page": pg},
                      cache_key=f"r{region}_p{pg}")
        time.sleep(random.random() * 0.06)
        return body

    pages_body = [first]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for body in pool.map(one, range(2, total + 1)):
            pages_body.append(body)
            _tick(progress)
    if log:
        log(f"region {region}: {total} 페이지 수신")

    books: dict[int, dict] = {}
    for body in pages_body:
        for o in body:
            if o.get("location_id") != station:
                continue
            price = o.get("price")
            if not price or price <= 0.011:
                continue  # 0.01 스팸 호가는 노이즈로 제외
            t = o["type_id"]
            b = books.setdefault(t, {"sells": [], "buys": []})
            (b["buys"] if o.get("is_buy_order") else b["sells"]).append(
                (price, o.get("volume_remain", 0), _iso(o.get("issued"))))
    for b in books.values():
        b["sells"].sort(key=lambda x: x[0])
        b["buys"].sort(key=lambda x: -x[0])
    if log:
        log(f"station {station}: {len(books)}종목 호가 확보")
    return books

def fetch_type_meta(type_ids: list[int], progress=None) -> dict[int, dict]:
    """m^3/개 부피를 라벨링용 타입별 GET. (벌크 엔드포인트 미지원)"""
    out: dict[int, dict] = {}

    def one(tid: int):
        for _ in range(3):                      # 레이트리밋 맞으면 재시도 — volume 없음 처리로 새면 안 된다
            try:
                body, _ = get(f"/latest/universe/types/{tid}/", cache_key=f"type_{tid}")
                # 시장은 무조건 **포장(packaged)** 상태로 오고 간다. 타입의 volume은
                # 조립(assembled) 부피라 예컨드 리퍼 같은 항성은 5000m³로 잡혀 물량이 왜곡된다.
                # packaged_volume이 있으면 그것이 실제 운송 부피다.
                pv = float(body.get("packaged_volume") or 0.0)
                vol = float(body.get("volume") or 0.0)
                return tid, {"volume": pv if pv > 0 else vol,
                             "name": body.get("name", f"type {tid}")}
            except ESIRateLimited:
                time.sleep(2.0)
        return tid, {"volume": 0.0, "name": f"type {tid}"}

    with ThreadPoolExecutor(max_workers=10) as pool:
        for tid, v in pool.map(one, type_ids):   # 진행률은 메인 루프에서 더딘다
            out[tid] = v
            _tick(progress)
    return out


def fetch_histories(region: int, type_ids: list[int], days: int = 10,
                    progress=None) -> dict[int, float]:
    """최근 days 일 평균 거래량 (전 리전 체결 기준). 기본은 열흘 새 평균."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    out: dict[int, float] = {}

    def one(tid: int):
        try:
            body, _ = get(
                f"/latest/markets/{region}/history/",
                params={"type_id": tid, "dates_from": start.isoformat(),
                        "dates_to": end.isoformat()},
                cache_key=f"hist_{days}_{region}_{tid}")
        except ESIRateLimited:
            return tid, []
        return tid, [h.get("volume", 0) for h in body if h.get("volume")]

    with ThreadPoolExecutor(max_workers=10) as pool:
        for tid, vols in pool.map(one, type_ids):
            if vols:
                out[tid] = sum(vols) / len(vols)
            _tick(progress)
    return out

