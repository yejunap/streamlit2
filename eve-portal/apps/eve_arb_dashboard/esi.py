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

# ---- 캐시 보존 (TTL) ----------------------------------------------------------------------------
# ETag 조건부 요청은 본문을 아끼지만 왕복 자체는 그대로 남는다. 그 왕복이 만여 번을 쌓아
# 첫 실행이 몇 분으로 기인다. 안 바뀔 것은 아예 맞게 한다 (get의 ttl 인자).
# 주문판만 늘 새것으로 받고, 나머지 둘은 오래 시켜 둔다.
TTL_TYPE = 7 * 86400        # 타입 이름·부피·그룹 — 확장 발표에나 바뀐다
TTL_HISTORY = 24 * 3600     # 날이 지난 체결량은 고정 — 하루는 다시 묻지 않는다
TTL_RPAGES = 86400          # 리전 페이지 수는 워낙 천천히 바뀐다

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


def _num_get(key: str, ttl: float):
    """새어 둔 숫자가 ttl 초 안쪽이면 그 것을 그대로 쓴다. 없거나 오래졌으면 None."""
    p = _cache_path(f"num_{key}.txt")
    try:
        if time.time() - os.path.getmtime(p) < ttl:
            with open(p, encoding="utf-8") as f:
                return int(f.read().strip())
    except (OSError, ValueError):
        pass
    return None


def _num_put(key: str, value: int) -> None:
    try:
        with open(_cache_path(f"num_{key}.txt"), "w", encoding="utf-8") as f:
            f.write(str(value))
    except OSError:
        pass


def _xpages(region: int):
    """HEAD 로 1페이지 머리카지만 받아 오늘의 페이지 수를 세운다 — 본문은 오지 않는다.

    페이지는 하루 안에도 줄어든다 — 주문이 만료되면서 뒤가져 잘려 나간다. 이렇게 세어 둔
    숫자를 그대로 쫓아가면 없는 뒤쪽 페이지를 물게 되고 ESI 는 'Requested page does not
    exist!' 로 랑는다. 세 는 내내 본문을 나르지 않는다.
    """
    try:
        r = session().head(f"{ESI}/latest/markets/{region}/orders/",
                           params={"page": 1}, timeout=30)
        if r.status_code == 200:
            return int(r.headers.get("X-Pages", 1))
    except (requests.RequestException, ValueError):
        pass
    return None


def region_pages(region: int, refresh: bool = False) -> int:
    """리전 주문판의 전체 페이지 수 — 진행 칸 세는 데만 쓰고, 본문은 버린다.

    예전엔 매 번 1페이지를 통채로 받아 그 긴 숫자만 뽑고 버렸다 — 그 한 장이 수 백 KB다.
    세어 둔 것을 하루는 간직한다 (주문판 페이지는 fetch_hub_orders 가 고대로 받는다).

    refresh 는 세어 둔 것을 믿지 않고 HEAD 로 세운다 — 주문판을 받아 들일 때 이 한 번이
    있다. 없는 페이지를 물으면 404 가 나고, 그 404 는 에러 한도에 몰려 420 으로 간다.
    """
    kept = _num_get(f"rpages_{region}", TTL_RPAGES)
    if kept and not refresh:
        return kept
    if refresh:                     # 받아 들일 일이다 — 낡은 숫자를 믿고 못 간다
        live = _xpages(region)
        if live is not None:
            _num_put(f"rpages_{region}", live)
            return live
        if kept:                             # HEAD 가 지면 — 옛 숫자라도 쓴다
            return kept
    _, hdrs = get(f"/latest/markets/{region}/orders/", params={"page": 1})
    pages = int(hdrs.get("X-Pages", 1))
    _num_put(f"rpages_{region}", pages)
    return pages


class ESIError(Exception):
    """ESI 를 끝까지 못 받았다."""


class ESIRateLimited(ESIError):
    """420/429 · 5xx — 걸린 것이다. 뒤에 다시 하면 된다."""


class ESINoData(ESIError):
    """404 — 그 종의 데이터가 아 없다 (예: 거래 실적이 없는 종의 history).
    재시도로 안 좋아지니 그냥 없는 셈 친다."""


_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        import requests as rq
        from requests.adapters import HTTPAdapter
        s = rq.Session()
        s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
        # 커넥션이 워터 수보다 모자라면 매 요청이 사뿐이 새 악셀을 닮는다 — 워커와 같은 크기로 튼다
        s.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=32))
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


def _etag_age(key: str) -> float:
    """캐시 본문을 심은 뒤 지난 초 — 없는 것은 무지막지하게 크게 된다."""
    try:
        return time.time() - os.path.getmtime(_cache_path(f"etag_{key}.json"))
    except OSError:
        return float("inf")


def get(path: str, params: dict | None = None, cache_key: str | None = None,
        retries: int = 4, ttl: float | None = None):
    """GET 하고 (json, headers) 반환. cache_key 를 주면 ETag 조건부 요청을 쓴다.

    ttl 도 주면 — 본문을 심은 지 ttl 초 안쪽은 요청을 아예 내지 않는다. 조건부 요청은
    304로 본문을 아끼지만 왕복은 물리기 십이므로, 만 번을 세는 데는 이 한 칸이 지대하다.
    """
    url = f"{ESI}{path}"
    headers = {}
    if cache_key:
        cached = _load_etag(cache_key)
        if cached and cached.get("etag"):
            if ttl and _etag_age(cache_key) < ttl:
                return cached["body"], {"X-Cache": "ttl"}      # 왕복을 아낀다
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
        if r.status_code >= 500:
            # 500/502/503/504 — ESI 단골이다. 참았다 다시 한다 (429와 같이 기운을 둔다)
            time.sleep(2 + attempt * 2)
            last_err = ESIRateLimited(f"ESI {r.status_code} for {path}")
            continue
        if r.status_code == 404:
            # 그 종의 것이 아 없다 — history 없는 종목이 달하는 자리다. 세게 안 당기고 제린다.
            raise ESINoData(f"ESI 404 for {path}")
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise ESIRateLimited(f"ESI failed for {path}: {e}") from e
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


def _orders_page(region: int, pg: int):
    """주문판 한 장 — (편 수, 본문 또는 빈칸, 모른 이유) 를 돌려 준다.

    한 장 없다고 전체 스캔이 서면 안 된다 — 못 받은 건 그 이유와 함께 적 어 둔다.
    """
    try:
        body, _ = get(f"/latest/markets/{region}/orders/", params={"page": pg},
                      cache_key=f"r{region}_p{pg}")
    except ESINoData:
        # 없는 페이지 — 세어 둔 수보다 줄었거나(404) 받는 도중에 만료로 잘렸다. 빈 칸으로 지난다.
        return pg, None, "gone"
    except ESIError as e:
        # 429·5xx 를 끝까지 참지 못했다 — 지난 자리는 진단 로그에 남긴다
        return pg, None, str(e)
    time.sleep(random.random() * 0.06)
    return pg, body, None


def fetch_hub_orders(region: int, station: int, workers: int = 8,
                     progress=None, log=None) -> dict:
    """리전 전체 주문을 받아 지정한 스테이션 하나만 남긴다.

    반환: {type_id: {'sells': [(price, qty, issued)], 'buys': [...]}}

    1페이지도 2페이지와 똑같이 ETag 를 태워 받는다 — 예전엔 첫 장만 캐시 없이 전문을
    받아 오던 자시다. 페이지 수는 region_pages 이 사절로 세어 온다.

    그 리 고 페이지 수는 어 것이다 — 주문이 만료되면서 뒤가져 잘려 나간다. 그래서
    1) 받는 앞에 HEAD 로 다시 세운다(region_pages 의 refresh). 2)그래도 도중에 밀리는
    것은 없는 칸으로 지난다 — 없는 페이지는 404 를 랑하고, 그 한 장이 전체를 세우면 안 된다.
    """
    total = region_pages(region, refresh=True)

    def one(pg: int):
        return _orders_page(region, pg)

    pages_body, gone, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pg, body, err in pool.map(one, range(1, total + 1)):
            if body is None:
                if err == "gone":
                    gone += 1
                else:
                    failed += 1
                    if log and failed <= 3:
                        log(f"region {region}: {pg}페이지 실패({err})")
            else:
                pages_body.append(body)
            _tick(progress)
    if log:
        msg = f"region {region}: {len(pages_body)}/{total} 페이지 수신"
        if gone:
            msg += f" · 사라진 페이지 {gone}장 (주문이 만료 되어 창이 줄었다)"
        if failed:
            msg += f" · 못 받은 페이지 {failed}장 — 그 스테이션 창가는 얇을 수 있다"
        log(msg)
    if not pages_body:
        # 하나도 못 받았다 — 빈 표를 올리는 것보단 에러 소리를 낸다.
        raise ESIError(f"리전 {region} 주문판을 한 장도 받지 못했다 (시도 {total} 페이지)")

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

def fetch_type_meta(type_ids: list[int], progress=None, workers: int = 10) -> dict[int, dict]:
    """m^3/개 부피를 라벨링용 타입별 GET. (벌크 엔드포인트 미지원)

    타입 이름·부피·그룹은 확장 발표에나 바뀃다. 그래서 캐시를 TTL 로 싣고 오면 맞되 물린다
    — 만 회를 넘어 붙는 왕복을 통째로 아낀다. (esi.TTL_TYPE)
    """
    out: dict[int, dict] = {}

    def one(tid: int):
        for _ in range(3):                      # 레이트리밋 맞으면 재시도 — volume 없음 처리로 새면 안 된다
            try:
                body, _ = get(f"/latest/universe/types/{tid}/", cache_key=f"type_{tid}",
                              ttl=TTL_TYPE)
                # 시장은 무조건 **포장(packaged)** 상태로 오고 간다. 타입의 volume은
                # 조립(assembled) 부피라 예컨드 리퍼 같은 항성은 5000m³로 잡혀 물량이 왜곡된다.
                # packaged_volume이 있으면 그것이 실제 운송 부피다.
                pv = float(body.get("packaged_volume") or 0.0)
                vol = float(body.get("volume") or 0.0)
                # 그룹을 나란히 실는다 — 어패럴(is_apparel) 가려내기에 이 한 칸이 제일 싸다
                # (type 은 맞는데 카테고리는 그룹에서 따로 놀므로 되짶자면 그룹 GET을 또 탄다).
                return tid, {"volume": pv if pv > 0 else vol,
                             "name": body.get("name", f"type {tid}"),
                             "group": body.get("group_id")}
            except ESIRateLimited:
                time.sleep(2.0)
            except ESIError:
                break		# 404 — 없다니 다시 따질 것 없다
        return tid, {"volume": 0.0, "name": f"type {tid}", "group": None}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tid, v in pool.map(one, type_ids):   # 진행률은 메인 루프에서 더딘다
            out[tid] = v
            _tick(progress)
    return out


def fetch_histories(region: int, type_ids: list[int], days: int = 10,
                    progress=None, workers: int = 10) -> dict[int, float]:
    """최근 days 일 평균 거래량 (전 리전 체결 기준). 기본은 열흘 새 평균.

    거래량은 하루 안쪽은 어차피 제자리다 — 캐시를 TTL 로 싣고 오면 왕복을 태우지 않는다."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    out: dict[int, float] = {}

    def one(tid: int):
        try:
            body, _ = get(
                f"/latest/markets/{region}/history/",
                params={"type_id": tid, "dates_from": start.isoformat(),
                        "dates_to": end.isoformat()},
                cache_key=f"hist_{days}_{region}_{tid}", ttl=TTL_HISTORY)
        except ESIError:
            # 404(거래 실적 없는 종) · 429 · 5xx — 하나가 죽었다 고 전체 스캔이 서면 안 된다.
            # 거래량 모름으로 두고 표에서 맨 뒤로 가게 한다.
            return tid, []
        return tid, [h.get("volume", 0) for h in body if h.get("volume")]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tid, vols in pool.map(one, type_ids):
            if vols:
                out[tid] = sum(vols) / len(vols)
            _tick(progress)
    return out

