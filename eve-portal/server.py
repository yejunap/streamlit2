#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVE Online 코퍼레이션 산업/블루프린트 모니터링 대시보드

- EVE SSO(PKCE, 네이티브 앱) 로그인 → 토큰은 tokens.json 에 저장 (여러 캐릭터 가능)
- 바인딩: 기본 자동 = 테일스케일 주소(100.64/10), 없으면 127.0.0.1.
  0.0.0.0 같은 공개 바인딩은 config 의 allow_public_bind:true 없이는 불가.
- 배경 루프가 ESI 를 주기 갱신:
    * 캐릭터/코퍼레이션 산업 작업 (industry jobs)
    * 캐릭터/코퍼레이션 자산 목록에서 블루프린트 만 추려 재고로 표시
- 대시보드: http://<테일스케일-IP>:<port>  (기본 8421)

실행: python3 server.py         # python3 표준 라이브러리만 사용 (추가 의존성 없음)
"""

import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import app_proxy
import apps_api

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
TOKENS_PATH = os.path.join(BASE, "tokens.json")
NAMES_PATH = os.path.join(BASE, "esi_names.json")
WALLET_PATH = os.path.join(BASE, "wallet_history.json")

ESI_HOST = "https://esi.evetech.net/latest"
SSO_AUTHORIZE = "https://login.eveonline.com/v2/oauth/authorize/"
SSO_TOKEN = "https://login.eveonline.com/v2/oauth/token"
SSO_SCOPES = [
    # ── 전부 ESI openapi.json securityScheme 원본 대조 (2026-10-17 검증)
    # 산업/캐릭터 — 작업, 채굴, 포맬 목록
    "esi-industry.read_character_jobs.v1",
    "esi-industry.read_corporation_jobs.v1",
    "esi-industry.read_character_mining.v1",
    "esi-industry.read_corporation_mining.v1",
    "esi-assets.read_assets.v1",
    "esi-assets.read_corporation_assets.v1",
    "esi-characters.read_blueprints.v1",
    "esi-corporations.read_blueprints.v1",
    # 우주/구조물 — 이름, 시설(생산효율), 검색
    "esi-universe.read_structures.v1",
    "esi-corporations.read_structures.v1",
    "esi-corporations.read_facilities.v1",
    "esi-search.search_structures.v1",
    # 지갑 탭 — ISK 지갑(캐릭터/코퍼), 주문(캐릭터/코퍼/구조물)
    #   주의: 코퍼 지갑의 공식명은 read_corporation_wallets(복수)다!
    "esi-wallet.read_character_wallet.v1",
    "esi-wallet.read_corporation_wallets.v1",
    "esi-markets.read_character_orders.v1",
    "esi-markets.read_corporation_orders.v1",
    "esi-markets.structure_markets.v1",
    # 계약·기술·클론
    "esi-contracts.read_character_contracts.v1",
    "esi-contracts.read_corporation_contracts.v1",
    "esi-skills.read_skills.v1",
    "esi-skills.read_skillqueue.v1",
    "esi-clones.read_clones.v1",
    "esi-clones.read_implants.v1",
    # 위치·일정·피팅·PI·통지
    "esi-location.read_location.v1",
    "esi-location.read_online.v1",
    "esi-calendar.read_calendar_events.v1",
    "esi-fittings.read_fittings.v1",
    "esi-planets.manage_planets.v1",
    "esi-characters.read_notifications.v1",
]

DEFAULT_CFG = {
    "port": 8421,
    "client_id": "",
    # 개발자포털에 등록하는 리다이렉트 URL 접두 (/oauth/callback 로 자동 연결됨)
    "redirect_base": "http://localhost:8421",
    # 바인딩: "auto" = 테일스케일(100.64/10) 주소 → 없으면 127.0.0.1.
    # 그 외 IP면 테일스케일/루프백만 허용. 부득이 0.0.0.0 등은
    # allow_public_bind:true 를 켜야 한다.
    "bind": "auto",
    "allow_public_bind": False,
    # EVE SSO 인증 후 이 목록에 이름이 있는 캐릭터만 대시보드이 접근할 수 있다.
    # 비어 있으면 잠금 없음(호환). 이름은 대소문자 무시. 예: ["SooJack"]
    "allowed_characters": ["SooJack"],
    "datasource": "tranquility",
    "user_agent": "eve-industry-dashboard/1.0",
    # 갱신 주기(초): 산업 작업 / 자산(블루프린트)
    "jobs_refresh": 60,
    "assets_refresh": 600,
    # 지갑 표본 주기(초) — 10분
    "wallet_refresh": 600,
    # 호환성 날짜 (ESI X-Compatibility-Date)
    "compatibility_date": "2025-09-01",
    # 최신 ESI(2025-09 산업 개편) activity_id 실측 기반:
    # 1=생산(제품≠블프), 5=연구(제품==블프), 8=연구(발명)(서로 다른 블프)
    "activity_names": {
        "1": "생산", "5": "연구(카피)", "8": "연구(발명)",
        "4": "반응(Reaction)", "6": "가속(Acceleration)", "7": "타이밍(Timing)",
    },
}
lock = threading.Lock()
REFRESH_NOW = threading.Event()


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({k: DEFAULT_CFG[k] for k in
                       ("port", "client_id", "redirect_base", "jobs_refresh",
                        "assets_refresh", "bind", "allow_public_bind",
                        "allowed_characters")},
                      f, indent=2)
    return cfg


CFG = load_cfg()
PORT = int(CFG.get("port", 8421))
BIND = "127.0.0.1"  # main() 에서 resolve_bind() 로 결정됨

# ---- 접근 제어: EVE SSO 인증 캐릭터 이름 허용 목록 --------------------------
# config.json 의 allowed_characters 에 이름을 새기면, 그 캐릭터로 EVE SSO 를
# 마친 브라우저만 대시보드와 API 를 볼 수 있다. 비어 있으면 잠금 없음(호환).
# 이름은 대소문자를 접어 비교한다 (config 에 "SooJack" 또는 "soojack" 이나 같지 않다).
_sessions = {}          # 세션 토큰 -> {"cid": int, "name": str, "ts": float}
SESSION_LOCK = threading.Lock()
SESSION_COOKIE = "eveind_sid"


def allowed_names():
    raw = CFG.get("allowed_characters")
    if isinstance(raw, str):
        raw = [raw]
    return {str(n).strip().lower() for n in (raw or []) if str(n).strip()}


def is_gate_on():
    """allowed_characters 가 하나이라도 걸려 있으면 잠금이 켜진다."""
    return bool(allowed_names())


def _name_allowed(name):
    allowed = allowed_names()
    if not allowed:
        return True
    return (name or "").strip().lower() in allowed


# ---------------------------------------------------------------- ESI 클라이언트


class EsiError(Exception):
    def __init__(self, status, why):
        super().__init__("%s: %s" % (status, why))
        self.status = status
        self.why = why


_etag_cache = {}  # url -> {"etag":..., "body": json, "expires": ts}


def _req(url, token=None, method="GET", body=None):
    """ESI(GET/POST) 호출. ETag/Expires 재증증 캐지 사용. → (json|None, headers)"""
    headers = {
        "User-Agent": CFG.get("user_agent", "eve-industry-dashboard/1.0"),
        "Accept": "application/json",
        "X-Compatibility-Date": CFG.get("compatibility_date"),
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif method == "GET":
        with lock:
            c = _etag_cache.get(url)
        if c and c.get("etag") and c["expires"] > time.time():
            return c["body"], {"X-Pages": c.get("pages", "1"),
                               "x-cached": "1"}
        if c and c.get("etag"):
            headers["If-None-Match"] = c["etag"]
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        # EVE 정복 시간대에 30초 목걸이가 겹물리면 표정이 '멜겠다' — 짧게 자른다
        with urllib.request.urlopen(req, timeout=8) as r:
            payload = r.read()
            out = json.loads(payload) if (payload and r.status == 200) else None
            with lock:
                _etag_cache[url] = {
                    "etag": r.headers.get("ETag"),
                    "body": out,
                    "pages": r.headers.get("X-Pages") or "1",
                    # Expires까지 최장 5분은 재증증 없이 살려둔다(호출량 억제)
                    "expires": min(_expires_ts(r.headers), time.time() + 300),
                }
            return out, dict(r.headers)
    except urllib.error.HTTPError as e:
        if e.code == 304:
            with lock:
                c = _etag_cache.get(url)
            if c:
                return c["body"], {"X-Pages": c.get("pages", "1"),
                                   "x-cached": "revalidated"}
            raise EsiError(304, "cache lost")
        raise EsiError(e.code, (e.read()[:200].decode(errors="replace")
                                if e.fp else str(e)))
    except urllib.error.URLError as e:
        raise EsiError(0, str(e))

def _expires_ts(headers):
    exp = headers.get("Expires")
    if exp:
        try:
            import email.utils
            return email.utils.parsedate_to_datetime(exp).timestamp()
        except Exception:
            pass
    return time.time() + 60


def esi_url(path, params=None):
    params = dict(params or {})
    params.setdefault("datasource", CFG.get("datasource", "tranquility"))
    return ESI_HOST + path + "?" + urllib.parse.urlencode(params)


def esi_get(path, token=None, params=None):
    return _req(esi_url(path, params), token=token)


def esi_post(path, body, token=None, params=None):
    return _req(esi_url(path, params), token=token, method="POST", body=body)


# ---------------------------------------------------------------- 토큰 저장/SSO

_tokens = {}          # char_id(int) -> dict
_tokens_lock = threading.Lock()
_pending = {}         # state -> code_verifier


def _load_tokens():
    global _tokens
    if os.path.exists(TOKENS_PATH):
        try:
            with open(TOKENS_PATH, encoding="utf-8") as f:
                _tokens.update({int(k): v for k, v in json.load(f).items()})
        except Exception as e:
            sys.stderr.write("[warn] tokens.json 로드 실패: %s\n" % e)
            _tokens = {}
    else:
        _tokens = {}


def _save_tokens():
    tmp = TOKENS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _tokens.items()}, f, indent=1)
    os.replace(tmp, TOKENS_PATH)
    try:
        os.chmod(TOKENS_PATH, 0o600)
    except OSError:
        pass


def _jwt_claims(jwt):
    """서명 검증 없이 페이로드만 디코딩 (iss/aud/sub 형식만 확인)."""
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as e:
        raise EsiError(0, "jwt decode failed: %s" % e)


def _jwt_check(claims):
    ok_iss = ("https://login.eveonline.com",
              "https://login.eveonline.com/v2/oauth",
              "https://auth.eveonline.com")
    if claims.get("iss") not in ok_iss:
        raise EsiError(0, "invalid issuer: %s" % claims.get("iss"))
    # aud: 예전엔 "EVE Online" 단일, 요껀 [앱클라이언트ID, "EVE Online"] 배열로 옴
    aud = claims.get("aud")
    aud = set(map(str, aud if isinstance(aud, list) else [aud]))
    if not aud & {"EVE Online", "ESI WWW"}:
        raise EsiError(0, "invalid audience: %s" % claims.get("aud"))
    if not str(claims.get("sub", "")).startswith("CHARACTER:EVE:"):
        raise EsiError(0, "invalid subject")


def sso_login_url():
    client_id = (CFG.get("client_id") or "").strip()
    if not client_id:
        raise EsiError(0, "client_id 없음 — 설정 탭에서 EVE 개발자 클라이언트 ID 를 입력하세요")
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(16)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    with lock:
        _pending[state] = verifier
    q = urllib.parse.urlencode({
        "response_type": "code",
        "redirect_uri": redirect_uri(),
        "client_id": client_id,
        "scope": " ".join(SSO_SCOPES),
        # OIDC 강제값 — 살아세션의 자동통과/예전동의 재사용을 차단
        # (CCP에 logout 엔드이 없어 대신 세션 자체를 다시 열게 한다)
        "prompt": "login consent",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    })
    return SSO_AUTHORIZE + "?" + q


def redirect_uri():
    """리다이렉트 전체 URL. base에는 'http://host:port' 만 넣는데,
    실수로 콜백경로나 슬래시를 붙여도 정규화한다."""
    base = (CFG.get("redirect_base") or "http://localhost:%d" % PORT).strip()
    base = base.rstrip("/")
    for suffix in ("/oauth/callback", "/callback"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base + "/oauth/callback"


def _post_form(url, form):
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(form).encode(), method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": CFG.get("user_agent", "x")})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise EsiError(e.code, e.read()[:300].decode(errors="replace"))


def sso_complete(code, state):
    with lock:
        verifier = _pending.pop(state, None)
    if not verifier:
        raise EsiError(0, "만료된 로그인 시도(state) 입니다. 다시 로그인하세요")
    tok = _post_form(SSO_TOKEN, {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CFG.get("client_id", ""),
        "code_verifier": verifier,
        "redirect_uri": redirect_uri(),
    })
    claims = _jwt_claims(tok["access_token"])
    _jwt_check(claims)
    cid = int(claims["sub"].split(":")[-1])
    with _tokens_lock:
        _tokens[cid] = {
            "character_id": cid,
            "name": claims.get("name", ""),
            "corp_id": (claims.get("corp") or [None, None])[0],
            "corp_name": (claims.get("corp") or [None, ""])[1],
            "access": tok["access_token"],
            "refresh": tok.get("refresh_token", ""),
            "expiry": time.time() + tok.get("expires_in", 1100),
            "scopes": claims.get("scp") or [],
            "picture": claims.get("picture", ""),
        }
        _save_tokens()
    _fill_corp_info(cid, _tokens[cid])
    return cid


def _refresh_token(cid):
    t = _tokens.get(cid)
    if not t or not t.get("refresh"):
        raise EsiError(401, "재로그인 필요 (character %s)" % cid)
    try:
        tok = _post_form(SSO_TOKEN, {
            "grant_type": "refresh_token",
            "refresh_token": t["refresh"],
            "client_id": CFG.get("client_id", ""),
        })
    except EsiError as e:
        t["refresh_error"] = str(e.status)
        _save_tokens()
        raise
    t["refresh_error"] = ""
    t["access"] = tok["access_token"]
    # refresh_token 은 한 번뿐 재사용 가능 → 반드시 새 값 저장
    t["refresh"] = tok.get("refresh_token", t["refresh"])
    t["expiry"] = time.time() + tok.get("expires_in", 1100)
    _save_tokens()


def access_token(cid):
    """만료 임박 시 자동 갱신 후 액세스 토큰 반환.
    호로한 실수 — 락 안에서 리프레시가 30초를 부르면 지갑까지 줄을 선다.
    락 밖에서 �신하고 다시 읽는다."""
    with _tokens_lock:
        t = _tokens.get(cid)
        if not t:
            raise EsiError(401, "알 수 없는 캐릭터 %s" % cid)
        stale = t["expiry"] - 60 < time.time()
    if stale:
        _refresh_token(cid)
    with _tokens_lock:
        t = _tokens.get(cid) or {}
    if not t.get("access"):
        raise EsiError(401, "갱신 후 토큰 소실 %s" % cid)
    return t["access"]

# ---------------------------------------------------------------- 이름 캐시
# universe/names 결과를 파일로 오래 보관 (id → name)

_names = {}          # "34"->"Tritanium", 스테이션id, 구조체id, 캐릭터id …
_names_lock = threading.Lock()


def _load_names():
    if os.path.exists(NAMES_PATH):
        try:
            with open(NAMES_PATH, encoding="utf-8") as f:
                _names.update(json.load(f))
        except Exception as e:
            sys.stderr.write("[warn] esi_names.json 로드 실패: %s\n" % e)


def _save_names():
    try:
        tmp = NAMES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_names, f, ensure_ascii=False)
        os.replace(tmp, NAMES_PATH)
    except OSError as e:
        sys.stderr.write("[warn] esi_names.json 저장 실패: %s\n" % e)


INT32_MAX = 2 ** 31 - 1


def _fill_corp_info(cid, t):
    """JWT 의 corp 클레임이 없는 요즘 토큰용: 공개 캐릭터 정보로 코퍼 보정."""
    if t.get("corp_id"):
        return
    try:
        body, _ = esi_get("/characters/%d/" % cid)
        corp_id = body and body.get("corporation_id")
        if not corp_id:
            return
        name = ""
        try:
            cb, _ = esi_get("/corporations/%d/" % corp_id)
            name = (cb or {}).get("name", "")
        except EsiError:
            pass
        t["corp_id"], t["corp_name"] = corp_id, name
        _save_tokens()
        sys.stderr.write("[info] %s 코퍼 보정: %s %s\n" % (t.get("name"), corp_id, name))
    except EsiError as e:
        sys.stderr.write("[warn] 코퍼 정보 보정 실패 %s: %s\n" % (cid, e.status))


def _any_token():
    """로그인된 아무 캐릭터의 액세스 토큰 (공개 조회에 첨부용)."""
    with _tokens_lock:
        cids = list(_tokens)
    for cid in cids:
        try:
            return access_token(cid)
        except EsiError:
            pass
    return None


def _resolve_structures(sids):
    """구조물/아웃포스트 ID(int64)는 universe/names 에서 400 이 뜀
       → /universe/structures 개별 조회 (토큰+read_structures 스코프 필요).
       이름 없는 구조물은 403. 실패(?접두)는 캐지 안 해 재시도."""
    tok = _any_token()
    denied = []
    for sid in sorted(sids):
        try:
            body, _ = esi_get("/universe/structures/%d/" % sid, token=tok)
            nm = (body or {}).get("name")
            if nm:
                with _names_lock:
                    _names[str(sid)] = nm
        except EsiError as e:
            if e.status == 403:      # 접근 권한 없는 (이름 없는) 구조물
                denied.append(sid)
            else:                     # 401(스코프) 등 → 캐지 않고 이만
                sys.stderr.write("[warn] structures/%s 조회 실패 %s\n"
                                 % (sid, e.status))
            time.sleep(0.2)
    if denied:
        # 접근 권한 없는(보통 다른 코퍼 소유) 구조물은 ESI가 이름을 감춘다.
        # "asset locations" 로도 시스템은 받을 수 없어 ID만 표시한다.
        for sid in denied:
            with _names_lock:
                _names[str(sid)] = "이름 없는 구조물 (#%s)" % sid
    _save_names()


def resolve_ids(ids):
    """정수 id 집합 → 이름 확정.
    int64(구조물 등)는 universe/structures, 나머지(int32: 타입/스테이션/캐릭터…)는
    universe/names 로 보낸다. 한 배치가 터져도 인접 전체를 포기하지 않는다."""
    todo, structures = [], []
    for i in ids:
        if i is None:
            continue
        i = int(i)
        cached = _names.get(str(i))
        # '이름 없는 구조물'은 예전 403일 때의 오레이블 → 재조회 여지 유지
        if cached and not cached.startswith("이름 없는 구조물"):
            continue
        (structures if i > INT32_MAX else todo).append(i)
    if structures:
        _resolve_structures(set(structures))
    todo = sorted(set(todo))
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        try:
            out, _ = esi_post("/universe/names/", batch)
            time.sleep(0.3)
        except EsiError as e:
            sys.stderr.write("[warn] universe/names 실패: %s\n" % e)
            continue
        if out:
            with _names_lock:
                for item in out:
                    _names[str(item["id"])] = item.get("name", "")
    _save_names()


def name_of(i):
    if i is None:
        return "?"
    return _names.get(str(int(i)), "(%s)" % int(i))


# ---------------------------------------------------------------- 데이터 저장

# DATA["chars"][cid] / DATA["corps"][corpid] =
#   {"jobs": [...], "bps": [...], "jobs_ts": ts, "bps_ts": ts, "errors": {...}}
DATA = {"chars": {}, "corps": {}}
DATA_LOCK = threading.Lock()


def _table(scope):
    return DATA["chars"] if scope == "c" else DATA["corps"]


def _store(scope, key, field, val):
    with DATA_LOCK:
        row = _table(scope).setdefault(int(key), {})
        row[field] = val
        row[field + "_ts"] = time.time()
        row.get("errors", {}).pop(field, None)


def _note_error(scope, key, field, why):
    with DATA_LOCK:
        row = _table(scope).setdefault(int(key), {})
        row.setdefault("errors", {})[field] = "%s  (%s)" % (
            str(why)[:300], time.strftime("%H:%M:%S"))
        # 실패시각 — next_door가 몰아때리기(429 유발)를 막는 근거
        row.setdefault("fail_ts", {})[field] = time.time()


def _paged(path, token, params=None, max_pages=40):
    """X-Pages 헤어 기반 페이지순회 GET (1페이지 1000건)."""
    out, page = [], 1
    while page <= max_pages:
        body, hdrs = esi_get(path, token=token,
                             params=dict(params or {}, page=page))
        if body is None:
            break
        out.extend(body)
        try:
            # ESI는 'X-Pages' 로 온다 — 소문수로만 읽으면 1페이지로 오인됨
            total = int(hdrs.get("X-Pages") or hdrs.get("x-pages") or 1)
        except (TypeError, ValueError):
            total = 1
        if page >= total:
            break
        page += 1
    return out


# ---------------------------------------------------------------- 데이터 수집


def _asset_location_chain(a, parents):
    """컨테이너 안의 아이템이면 스테이션/구조물까지 부모를 거진다.
   Returns (실제_위치_id, [컨테타입_id...])."""
    loc_id, ltype = a.get("location_id"), a.get("location_type")
    chain = []
    while ltype == "item" and len(chain) < 4:
        p = parents.get(loc_id)
        if not p:
            break
        chain.append(p.get("type_id"))
        loc_id, ltype = p.get("location_id"), p.get("location_type")
    return loc_id, chain


def _fetch_asset_bps(scope, key, token, path):
    """자산 목록 중 블루프린트 만 추려 저장 (is_blueprint_copy 필드 보유).
    컨테이너 안(location_type=item)은 실제 담김 경로를 푸켜 저장한다."""
    try:
        items = _paged(path, token)
    except EsiError as e:
        hint = (" — 접근은 되나 비어있을 수 있음" if e.status == 404 else "")
        _note_error(scope, key, "bps", "자산 조회 실패 %s %s%s"
                    % (e.status, e.why, hint))
        return
    bps = [a for a in items if a.get("is_blueprint_copy") is not None]
    parents = {a.get("item_id"): a for a in items}
    # 지갑 탭용: 블루프린트·고유자산(장비/선체) 아닌 스택형만 재고다
    # — Myco Lab 한 대가 평균값에 떠밀려 30억 얹히는 식으로 자산이 부풀어 오른다
    item_qty = {}
    for a in items:
        if a.get("is_blueprint_copy") is None and a.get("type_id") \
                and not a.get("is_singleton"):
            q = int(a.get("quantity", 0) or 0)
            if q > 0:
                tid = int(a["type_id"])
                item_qty[tid] = item_qty.get(tid, 0) + q
    out = []
    for a in bps:
        loc_id, chain = _asset_location_chain(a, parents)
        out.append({
            "type_id": a["type_id"],
            "kind": "BPC" if a["is_blueprint_copy"] else "BPO",
            "qty": max(int(a.get("quantity", 1)), 1),
            "location_id": loc_id,
            "containers": chain,
        })
    todo = {r["type_id"] for r in out} | {r["location_id"] for r in out}
    for r in out:
        todo.update(r["containers"])
    resolve_ids(todo)
    _store(scope, key, "bps", out)
    _store(scope, key, "item_qty", item_qty)


def _fetch_jobs(scope, key, token, path):
    try:
        # 완료 행도 함께 반환된다(24h 제한은 없다 — 완료 탭이 histories로 남는 동안엔
        # 그대로 노출). 클라이언트에서 경과 시간으로 '최근/예전'을 가린다.
        if scope == "k":
            jobs = _paged(path, token, {"include_completed": "true"},
                          max_pages=10)
        else:
            jobs = esi_get(path, token=token,
                           params={"include_completed": "true"})[0] or []
        resolve_ids({j.get("product_type_id") for j in jobs}
                    | {j.get("blueprint_type_id") for j in jobs}
                    | {j.get("installer_id") for j in jobs}
                    | {j.get("location_id") for j in jobs})
        _store(scope, key, "jobs", jobs)
    except EsiError as e:
        hint = (" — 본인/본부에 Station·Item Manager 등 권한이 필요" 
                if e.status == 403 else "")
        _note_error(scope, key, "jobs",
                    "작업 조회 실패 %s%s" % (e.status, hint))


def _fetch_notifs(cid, tok):
    """통지 원본 수집 — 이름 없는 구조물 정체의 단서를 텍스트에서 찾는다.
    (600초 주자로 — 매 새로고침마다 때리면 429 난다)"""
    try:
        rows = esi_get("/characters/%d/notifications/" % cid, token=tok)[0] or []
    except EsiError as e:
        if e.status not in (401, 403):
            sys.stderr.write("[warn] 통지 조회 실패 %s\n" % e.status)
        return
    with lock:
        row = DATA["chars"].setdefault(cid, {})
        row["notifs"] = rows[:300]
        row["notifs_ts"] = time.time()


def refresh_character(cid):
    """캐릭터: 산업 작업 + 개인 블루프린트. (통지는 별도 주기 — refresher에서)"""
    tok = access_token(cid)
    _fetch_jobs("c", cid, tok, "/characters/%d/industry/jobs" % cid)
    _fetch_asset_bps("c", cid, tok, "/characters/%d/assets" % cid)


def _fetch_corp_structures(corp_id, tok):
    """소유 코퍼 구조물 전체(name 포함) — 네임 캐시에반영. 5분 내 재호출 아님."""
    row = DATA["corps"].get(corp_id) or {}
    if time.time() - row.get("structs_ts", 0) < 300:
        return
    try:
        rows, _ = esi_get("/corporations/%d/structures/" % corp_id, token=tok)
        with DATA_LOCK:
            DATA["corps"].setdefault(corp_id, {})["structs_ts"] = time.time()
    except EsiError as e:
        if e.status not in (403, 404):
            sys.stderr.write("[warn] 코퍼 구조물 %s\n" % e.status)
        return
    with _names_lock:
        for s in rows or []:
            if s.get("name"):
                _names[str(s["id"])] = s["name"]
    _save_names()


def refresh_corp(cid_owner, corp_id):
    """코퍼레이션: 산업 작업 + 블루프린트 재고."""
    tok = access_token(cid_owner)
    _fetch_corp_structures(corp_id, tok)
    _fetch_jobs("k", corp_id, tok,
                "/corporations/%d/industry/jobs" % corp_id)
    _fetch_asset_bps("k", corp_id, tok,
                     "/corporations/%d/assets" % corp_id)


def next_due(row, field, seconds):
    if row is None:
        return True
    now = time.time()
    # 최근 실패가 있으면 60초간 재시도 유예 — 지쳐 429를 타는 법이 없다
    fail = (row.get("fail_ts") or {}).get(field)
    if fail and now - fail < 60:
        return False
    return now - row.get(field + "_ts", 0) >= seconds


def refresh_once():
    """등록된 모든 캐릭터/코퍼를 주기별로 갱신 ESI를 정중하게: 실패 시 이어서."""
    with _tokens_lock:
        chars = list(_tokens.items())
    corps = {}
    for cid, t in chars:
        corps.setdefault(t.get("corp_id"), []).append((cid, t))
    for cid, t in chars:
        if t.get("refresh_error"):
            continue  # 재로그인 전까지 건드리지 않음
        try:
            row = DATA["chars"].get(cid)
            if next_due(row, "jobs", int(CFG.get("jobs_refresh", 60))):
                refresh_character(cid)
            if next_due(row, "bps", int(CFG.get("assets_refresh", 600))):
                _fetch_asset_bps("c", cid, access_token(cid),
                                 "/characters/%d/assets" % cid)
            if next_due(row, "notifs", 600):
                _fetch_notifs(cid, access_token(cid))
        except EsiError as e:
            _note_error("c", cid, "jobs", "토큰/요청 오류 %s" % e.status)
        except Exception as e:
            _note_error("c", cid, "jobs", "오류 %s" % e)
    for corp_id, members in corps.items():
        if not corp_id:
            continue
        cid_owner = members[0][0]
        row = DATA["corps"].get(corp_id)
        try:
            if next_due(row, "jobs", int(CFG.get("jobs_refresh", 60))):
                refresh_corp(cid_owner, corp_id)
            elif next_due(row, "bps", int(CFG.get("assets_refresh", 600))):
                _fetch_asset_bps("k", corp_id, access_token(cid_owner),
                                 "/corporations/%d/assets" % corp_id)
        except EsiError as e:
            _note_error("k", corp_id, "jobs", "토큰/요청 오류 %s" % e.status)
        except Exception as e:
            _note_error("k", corp_id, "jobs", "오류 %s" % e)


# ---------------------------------------------------------------- 지갑 표본 (10분 주기)
# 네 계열을 한 점: 코퍼 ISK 지갑 / 생산 재고 가치(작업 제품유형 × 평균가) /
# 판매주문 전체 / 세 것의 합 — wallet_history.json 에 병합.

WALLET = {"history": [], "errors": {}, "prices": {}, "price_ts": 0,
          "chart_png": None, "chart_key": None,
          "sell_min": {}, "sell_ts": 0}
WALLET_LOCK = threading.Lock()


def _load_wallet():
    if os.path.exists(WALLET_PATH):
        try:
            with open(WALLET_PATH, encoding="utf-8") as f:
                d = json.load(f)
            WALLET["history"] = d.get("history") or []
            WALLET["errors"] = d.get("errors") or {}
        except Exception as e:
            sys.stderr.write("[warn] wallet_history.json 열기 실패: %s\n" % e)


def _save_wallet():
    try:
        tmp = WALLET_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"history": WALLET["history"][-4000:],
                       "errors": WALLET.get("errors") or {}}, f)
        os.replace(tmp, WALLET_PATH)
    except OSError as e:
        sys.stderr.write("[warn] wallet_history.json 저장 실패: %s\n" % e)


def _corp_owner(corp_id):
    """코퍼 멤버의 토큰(갱신된 것) — 지갑/주문은 멤버 토큰으로 조회 가능.
    잠금 없이 돌린다: 안의 access_token이 같은 락을 다시 건다(무반입 잠금)."""
    for cid, t in list(_tokens.items()):
        if t.get("corp_id") == corp_id and not t.get("refresh_error"):
            try:
                return access_token(cid)
            except EsiError:
                return None
    return None


def _market_prices(type_ids):
    """평균가Dict(type_id→가격) — 30분 버티고, 요청은 100개씩 찐다."""
    now = time.time()
    with WALLET_LOCK:
        if now - WALLET["price_ts"] < 1800 and WALLET["prices"]:
            return dict(WALLET["prices"])
        WALLET["price_ts"] = now
    prices = {}
    todo = sorted({int(t) for t in type_ids})
    try:
        # 전체 평균가 한 통 — ESI의 GET /markets/prices/ (공개). POST 배치는 405 폐기됨
        rows, _ = esi_get("/markets/prices/")
        for row in rows or []:
            v = row.get("average_price") or row.get("average") or row.get("adjusted_price")
            if v:
                prices[int(row["type_id"])] = float(v)
    except EsiError as e:
        sys.stderr.write("[warn] 시세 조회 실패 %s — 지난 캐시로 버틴다\n" % e.status)
        with WALLET_LOCK:
            return dict(WALLET["prices"])      # 실패가 산(山) 캐시를 지우면 안 된다
    have = {t: prices[t] for t in todo if t in prices}
    with WALLET_LOCK:
        if have:
            WALLET["prices"] = have            # 산 종목만 �신 — 빈걸로 덮어쓰기 금지
    return prices




_wallet_last_try = 0.0


def wallet_sample(force=False):
    """네 지표 표본 — 주기(기본 10분) 지셨으면 history에 한 점 더한다."""
    now = time.time()
    hist = WALLET["history"]
    global _wallet_last_try
    if not force and now - _wallet_last_try < int(CFG.get("wallet_refresh", 600)) - 30:
        return
    _wallet_last_try = now
    corps = sorted(int(c) for c in DATA["corps"])
    if not corps:
        # refresh가 아직 안 도았어도 토큰에 박힌 코퍼로 지만 연다
        corps = sorted({int(t["corp_id"]) for t in _tokens.values()
                        if t.get("corp_id")})
    if not corps and not _tokens:
        sys.stderr.write("[info] 지갑 표본 유예 — 토큰 없음\n")
        return
    # 코퍼가 아직 태어난 참이어도 캐릭터 지갑은 어차피 개방 —
    # 전부를 기다리지 말고 부분적으로라도 무던히 한 점을 남긴다.
    sys.stderr.write("[info] 지갑 표본 시작 — 코퍼 %d체\n" % len(corps))
    # 재고 가치 = '내가 만든 것'뿐 — 산업 작업에_product로 오른 유형만 센다.
    # 창고 스택형을 다 세면 리서치 데이터코어(~2.4B)가 쌓여 리그 재고가 안 보인다.
    made = set()
    for row in DATA["corps"].values():
        for j in row.get("jobs") or []:
            if j.get("product_type_id"):
                made.add(int(j["product_type_id"]))
    need_types = set()
    for row in DATA["corps"].values():
        need_types.update(int(t) for t in (row.get("item_qty") or {}))
    need_types |= made
    # 시세는 30분 캐시 내부판 — 매 라운드 불러도 캐시면 수초, 없어야 한 번의 통신
    prices = _market_prices(need_types) if need_types else {}
    tok0 = None
    for _cid in list(_tokens):
        try:
            tok0 = access_token(_cid)
            break
        except EsiError:
            continue
    # 시가(時價): 자기 코퍼/캐릭터 매도 주문 저가 — ESI 전체 시장 주문장은
    # 2025-01 폐지(404)되어 창 자체는 없다. 미커형 유형은 평균가 폴백.
    own_min = {}
    stock = []
    wallet = product = orders = 0.0
    corp_isk = char_isk = 0.0
    seen_orders = set()      # 코퍼에서 센 order_id — 캐릭터 루트 중복 차단
    errors = {}
    for corp_id in corps:
        tok = _corp_owner(corp_id)
        if not tok:
            errors["token%d" % corp_id] = "사용 가능한 토큰 없음"
            continue
        # ① 코퍼 ISK 지갑 (internal + public 합)
        # ① 코퍼 ISK 지갑 — /wallets/ (plural; /wallet/은 legacy 404)
        try:
            rows, _ = esi_get("/corporations/%d/wallets/" % corp_id, token=tok)
            if isinstance(rows, (int, float)):
                corp_isk += float(rows)   # 호환 날짜 이후 숫자 통짜 응답도 감지
            else:
                corp_isk += sum(float(r.get("balance", 0)) for r in rows or [])
        except EsiError as e:
            errors["wallet"] = "코퍼 지갑 %s — 회계 담당(Accountant) 역할·스코프 확인" % e.status
        # ② 코퍼 매도 주문 — 코퍼로 올라간 주문은 여기가 원본이다
        #    (캐릭터 주문과 겹친다 — order_id를 적어 캐릭터 루프에서 뺀다)
        try:
            for o in _paged("/corporations/%d/orders/" % corp_id, tok):
                seen_orders.add(int(o.get("order_id", 0) or 0))
                if o.get("is_valid") is not False:
                    orders += float(o.get("price", 0)) * int(o.get("volume_remain", 0) or 0)
                    if not o.get("is_buy_order"):
                        t = int(o.get("type_id", 0) or 0)
                        px = float(o.get("price", 0) or 0)
                        if t and px > 0:
                            own_min[t] = min(own_min.get(t, px), px)
        except EsiError as e:
            errors["orders"] = "주문 조회 %s — 코퍼 주문 스코프 재로그인 필요" % e.status
        # ③ 코퍼 자산 재고 — 매도 주문 저가 수집이 끝난 뒤 ⑤에서 평가
        row = DATA["corps"].get(corp_id) or {}
        for tid, qty in (row.get("item_qty") or {}).items():
            if int(tid) in made:       # 데이터코어·잡소재는 생산 재고에서 제외
                stock.append((int(tid), int(qty)))
        # 코퍼가 안 잡혀도 캐릭터는 자기 지갑을 항상 볼 수 있다
        # (역한 멤버도 가능) — 코퍼 지갑이 403이어도 탭이 안 비어돈다.
    for cid, t in list(_tokens.items()):
        try:
            ctok = access_token(cid)
            rows, _ = esi_get("/characters/%d/wallet/" % cid, token=ctok)
            if isinstance(rows, (int, float)):
                char_isk += float(rows)
            else:
                char_isk += sum(float(r.get("balance", 0)) for r in rows or [])
        except EsiError as e:
            errors["cwallet%d" % cid] = "캐릭터 지갑 %s" % e.status
        try:
            for o in _paged("/characters/%d/orders/" % cid, ctok):
                if o.get("is_buy_order") or o.get("time_remaining") == "expired":
                    continue
                # 코퍼로 올린 주문은 코퍼 루트에서 이미 세었다 — 중복 계산 금지
                if int(o.get("order_id", 0) or 0) in seen_orders:
                    continue
                orders += float(o.get("price", 0)) * int(o.get("volume_remain", 0) or 0)
                t = int(o.get("type_id", 0) or 0)
                px = float(o.get("price", 0) or 0)
                if t and px > 0:
                    own_min[t] = min(own_min.get(t, px), px)
        except EsiError as e:
            errors["corders%d" % cid] = "캐릭터 주문 %s" % e.status
    # ⑤ 제품 재고 평가 — 내 매도 주문 최저가 → 평균가 폴백 (×0.9 — 매도 호가 변동 버퍼)
    for t, qty in stock:
        p = own_min.get(t) or prices.get(t)
        if p:
            product += float(p) * qty * 0.9
    wallet = corp_isk + char_isk
    total = wallet + product + orders
    with WALLET_LOCK:
        WALLET["history"].append({"ts": int(now), "wallet": round(wallet),
                                  "corp_isk": round(corp_isk),
                                  "char_isk": round(char_isk),
                                  "product": round(product),
                                  "orders": round(orders),
                                  "total": round(total)})
        WALLET["history"] = WALLET["history"][-4000:]
        WALLET["errors"] = errors
        WALLET["chart_key"] = None   # 그래프 재생성
    _save_wallet()
    sys.stderr.write("[info] 지갑 표본: 통장 %.0f / 생산 %.0f / 주문 %.0f\n"
                     % (wallet, product, orders))


def _chart_render(hist, fmt="%m-%d %H:%M", win=None):
    """matplotlib — Total/Corp/제품/주문 4계열. 단위 B(billion) ISK, KST축.
    (라틴 문자만 — 한글 폰트 없는 서버에서 한글은 tofu가 됨)"""
    import io
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import dates as mdates
    from datetime import datetime, timedelta, timezone
    kst = timezone(timedelta(hours=9))
    xs = [datetime.fromtimestamp(h["ts"], kst) for h in hist]
    B = 1e9
    fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=110)
    fig.patch.set_facecolor("#000208")
    ax.set_facecolor("#000208")
    series = [("Total", [h.get("total", 0) / B for h in hist], "#c48cf5", 2.2),
              ("Corp ISK", [h.get("corp_isk", 0) / B for h in hist], "#e8a33d", 1.5),
              ("Products", [h.get("product", 0) / B for h in hist], "#6ee09c", 1.5),
              ("Sell orders", [h.get("orders", 0) / B for h in hist], "#7ea6ff", 1.5)]
    dots = len(hist) < 14        # 10점 안팎이면 점을 박는다 — 창이 그런 용도다
    for label, ys, c, lw in series:
        ax.plot(xs, ys, color=c, lw=lw, marker=("o" if dots else None),
                ls="-", ms=4, label=label)
    ax.set_ylabel("B ISK", color="#c8cdd8")
    ax.grid(color="#2a2e39", lw=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors="#9aa2b1")
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    if win:
        # 데이터가 창보다 한산하면 autoscale이 축을 수식으로 벌린다 — 창을 못 박는다
        from datetime import datetime as _dt
        ax.set_xlim(_dt.fromtimestamp(win[0], kst), _dt.fromtimestamp(win[1], kst))
    fig.autofmt_xdate()
    leg = ax.legend(loc="upper left", frameon=False, ncol=2)
    for t in leg.get_texts():
        t.set_color("#c8cdd8")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def wallet_chart_bytes(rng="1d"):
    """시간 창(1h/1d/1w/1mo)별 차트 PNG — X축 점은 늘 10개 안팎으로 모은다.

    10분 표본이라 1시간은커녕 7점뿐이고 1달은 4300점이라 그대로 못 보여준다.
    창 폭을 10등분해 각 칸의 마지막 값(자산은 스냅숀이라 최신이 대표)을 취한다.
    """
    dur, fmt = {"1h": (3600, "%H:%M"), "1d": (86400, "%H:%M"),
                "1w": (604800, "%m-%d"), "1mo": (2592000, "%m-%d")} \
        .get(rng, (86400, "%H:%M"))
    now = time.time()
    with WALLET_LOCK:
        hist = [h for h in WALLET["history"] if h["ts"] >= now - dur]
    if len(hist) > 12:                       # 10칸 다운샘플 (그 이하면 원대로)
        step = dur / 10.0
        bins = [None] * 10
        for h in hist:                       # ts 오름차순 — 뒤의가 대표가 된다
            bins[min(int((h["ts"] - (now - dur)) / step), 9)] = h
        hist = [h for h in bins if h]
    key = (rng, len(hist), hist[-1]["ts"] if hist else 0)
    with WALLET_LOCK:
        if WALLET.get("chart_png") and WALLET["chart_key"] == key:
            return WALLET["chart_png"]
    if not hist:
        hist = [{"ts": int(time.time()), "wallet": 0, "product": 0,
                 "orders": 0, "total": 0}]
    png = _chart_render(hist, fmt, win=(now - dur, now))
    with WALLET_LOCK:
        WALLET["chart_png"] = png
        WALLET["chart_key"] = key
    return png


# ---------------------------------------------------------------- 대시보드용 스냅숏

STATUS_KO = {"active": "진행중", "paused": "정지", "pending": "대기",
             "done": "완료", "completed": "완료", "reversed": "취소됨",
             "reverted": "취소됨",
             # 산업 작업 실제 응답 변형(delivered/delivering 중→완료 취급)
             "delivered": "완료", "delivering": "완료",
             "cancelled": "취소됨", "canceled": "취소됨",
             "failed": "실패", "cancelled_reverse": "취소됨"}


def _fmt_left(ts):
    """3h 23m 14s 식 — 0인 상위 단위 생략, 초가 0이면 초도 생략."""
    if ts is None:
        return ""
    ts = int(ts)
    d, rem = divmod(ts, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append("%dd" % d)
    if d or h:
        parts.append("%dh" % h)
    if d or h or m:
        parts.append("%dm" % m)
    if s:
        parts.append("%ds" % s)
    return " ".join(parts) if parts else "0s"


def _job_row(j, scope, owner_name):
    now = time.time()
    end = j.get("end_date")
    ends_in = None
    dt = None
    if end:
        try:
            import datetime
            dt = datetime.datetime.strptime(
                end[:19], "%Y-%m-%dT%H:%M:%S").replace(
                    tzinfo=datetime.timezone.utc)
            ends_in = dt.timestamp() - now
        except Exception:
            dt = None
    act = CFG.get("activity_names", {}).get(str(j.get("activity_id")))
    st = STATUS_KO.get(str(j.get("status", "")).lower(), j.get("status"))
    done_age_h = None
    if j.get("completed") or st == "완료":
        st = "완료"
        if ends_in is not None:
            done_age_h = round(-ends_in / 3600.0, 1)  # end가 지났으니 +값(시간 단위)
        ends_in = None          # 완료는 잔여 시간 표기 없음
    elif ends_in is not None and ends_in <= 0:
        # 타이머가 지났는데 아직 active = 제품 생성됨 / 아직 안 받아 회수대기
        st = "회수대기"
        done_age_h = round(-ends_in / 3600.0, 1)
        ends_in = None
    elif ends_in is not None:
        ends_in = max(0, ends_in)
    # 완료시각(UTC)을 한국시간으로 — dt는 tz-aware(UTC)
    done_txt = ""
    if st in ("완료", "회수대기") and dt is not None:
        import datetime
        done_txt = (dt.astimezone(datetime.timezone(
            datetime.timedelta(hours=9)))).strftime("%Y-%m-%d %H:%M")
    elif st in ("완료", "회수대기") and end:
        done_txt = end[:16].replace("T", " ") + " (UTC)"
    return {
        "scope": "코퍼레이션" if scope == "k" else "캐릭터",
        "owner": owner_name,
        "activity": act or ("activity#%s" % j.get("activity_id")),
        "product": name_of(j.get("product_type_id")),
        "blueprint": name_of(j.get("blueprint_type_id")),
        "bp_type_id": j.get("blueprint_type_id"),
        "status": st,
        "runs": j.get("runs"), "successful_runs": j.get("successful_runs"),
        "licensed_runs": j.get("licensed_runs"),
        "probability": j.get("probability"),
        "cost": j.get("cost"),
        "location": name_of(j.get("location_id")),
        "installer": name_of(j.get("installer_id")),
        "end_date": end, "ends_in": ends_in,
        "done_age_h": done_age_h,
        "ends_left": "" if ends_in is None else _fmt_left(ends_in),
        # 완료시각 — KST로 표기 (서버/브라우저 로컬 무관하게 고정 +9)
        "done_txt": done_txt,
    }


def _bp_rows(scope, owner_name, rows):
    out = []
    for b in rows or []:
        loc = name_of(b.get("location_id"))
        if b.get("containers"):
            loc += " · " + " › ".join(name_of(t) for t in b["containers"])
        out.append(dict(b, scope="코퍼레이션" if scope == "k" else "개인",
                        owner=owner_name,
                        name=name_of(b["type_id"]),
                        location=loc))
    return out


def build_state():
    chars, jobs, blueprints = [], [], []
    # 잠금 없이 읽는다 — refresh가 잠금 안에 SSO 재호시를 하며 들고
    # 있어도 상태 조회가 얼지 않는다. 소수 경합은 조용히 용납한다.
    items = list(_tokens.items())
    for cid, t in items:
        row = DATA["chars"].get(cid, {})
        chars.append({
            "character_id": cid, "name": t.get("name"),
            "corp_id": t.get("corp_id"), "corp_name": t.get("corp_name"),
            "picture": t.get("picture", ""),
            "token_expires": t.get("expiry"),
            "refresh_error": t.get("refresh_error") or "",
            "errors": dict(row.get("errors", {})),
        })
        jobs += [_job_row(j, "c", t.get("name"))
                 for j in row.get("jobs") or []]
        blueprints += _bp_rows("c", t.get("name") or "", row.get("bps"))
    for corp_id, row in list(DATA["corps"].items()):
        corp_name = "코퍼레이션 %s" % corp_id
        for cid, t in items:
            if t.get("corp_id") == corp_id:
                corp_name = t.get("corp_name") or corp_name
                break
        jobs += [_job_row(j, "k", corp_name) for j in row.get("jobs") or []]
        blueprints += _bp_rows("k", corp_name, row.get("bps"))
    jobs.sort(key=lambda j: (j.get("ends_in") is None, j.get("ends_in") or 0))
    return {
        "ts": time.time(),
        "bind": BIND,
        "client_id_set": bool((CFG.get("client_id") or "").strip()),
        "redirect_uri": redirect_uri(),
        "config": {k: CFG.get(k) for k in
                   ("jobs_refresh", "assets_refresh")},
        "characters": chars,
        "jobs": jobs,
        "blueprints": blueprints,
        "corp_errors": [
            [cid, {k: v for k, v in (row.get("errors") or {}).items()}]
            for cid, row in DATA["corps"].items() if row.get("errors")],
    }


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    # 30초 내 후속 요청 없으면 커넥 접는다 — 죽은 브라우저가 핸들러
    # 스레드를 무한 물고 있으면 안 된다 (StreamRequestHandler.timeout)
    timeout = 30

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        # 큰 덩어리(state 2~3MB)는 지킬리 터널에서 그냥 죽는다 — 무조건 쪼여서 보낸다
        ae = self.headers.get("Accept-Encoding", "") or ""
        gz = ("gzip" in ae and len(body) > 20000
              and ctype.startswith(("application/json", "text/")))
        if gz:
            import gzip
            body = gzip.compress(body, 6)
        self.send_response(code)
        # 캐시 여지 없임 — UI 고친 뒤 새로고침이면 무조건 바로 반영
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", ctype)
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _redirect(self, url):
        self.send_response(302)
        self.send_header("Location", url)
        self.end_headers()

    # ---- 접근 제어 (SooJack 전용) ----------------------------------------
    def _session(self):
        cookie = self.headers.get("Cookie", "") or ""
        for part in cookie.split(";"):
            k, _, v = part.strip().partition("=")
            if k == SESSION_COOKIE:
                with SESSION_LOCK:
                    return _sessions.get(v.strip())
                break
        return None

    def _authed(self):
        """allowed_characters 가 걸려 있으면, 허용 캐릭터의 SSO 세션 쿠키가 있어야 통과."""
        if not is_gate_on():
            return True
        s = self._session()
        return bool(s) and _name_allowed(s.get("name"))

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        # SSO 진입/콜백은 잠금 없이 통과시킨다 (콜백에서 쿠키를 심는다)
        if u.path not in ("/oauth/callback", "/login", "/api/login") \
                and is_gate_on() and not self._authed():
            if u.path == "/":
                try:
                    self._redirect(sso_login_url())
                except EsiError as e:
                    self._json({"error": str(e)}, 400)
            else:
                self._json({"error": "인증 필요 — EVE 로그인 후 허용 캐릭터만 접근 가능"}, 401)
            return
        if u.path.startswith("/api/apps/"):     # 포털 네이티브 앱 백단 (iframe 중계 불필요)
            name = u.path[len("/api/apps/"):]
            self._json(apps_api.handle(name, q))
            return
        pfx, port = app_proxy.match(u.path)
        if pfx:                                  # /apps/… — 로컬 streamlit 중계
            app_proxy.serve(self, pfx, port)
            return
        if u.path == "/":
            with open(os.path.join(BASE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            try:
                self._json(build_state())
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif u.path == "/api/debug/notifs":
            with lock:
                out = {cid: (row.get("notifs") or [])[:300]
                       for cid, row in DATA["chars"].items()}
            self._json(out)
        elif u.path == "/api/wallet":
            with WALLET_LOCK:
                out = {"history": list(WALLET["history"]),
                       "errors": dict(WALLET.get("errors") or {})}
            out["interval"] = int(CFG.get("wallet_refresh", 600))
            self._json(out)
        elif u.path == "/api/wallet/chart.png":
            rng = (q.get("range") or ["1d"])[0]
            if rng not in ("1h", "1d", "1w", "1mo"):
                rng = "1d"
            try:
                self._send(200, wallet_chart_bytes(rng), "image/png")
            except Exception as e:
                self._send(500, ("chart error: %s" % e).encode(), "text/plain")
        elif u.path in ("/login", "/api/login"):
            try:
                self._redirect(sso_login_url())
            except EsiError as e:
                self._json({"error": str(e)}, 400)
        elif u.path == "/oauth/callback":
            code = (q.get("code") or [""])[0]
            state = (q.get("state") or [""])[0]
            try:
                cid = sso_complete(code, state)
                name = (_tokens.get(cid) or {}).get("name", "")
                if is_gate_on() and not _name_allowed(name):
                    self._send(
                        403,
                        "<meta charset='utf-8'><b>접근 거부</b> — 이 서버는 허용된 "
                        "캐릭터(%s)만 이용할 수 잇습니다. 지금 로그인한 캐릭터: %s"
                        % (", ".join(sorted(allowed_names())), name or "?"),
                        "text/html; charset=utf-8")
                    return
                # 허용 캐릭터 — 브라우저에 세션 쿠키를 심어 대시보드를 열 수 있게 한다
                stok = secrets.token_urlsafe(32)
                with SESSION_LOCK:
                    _sessions[stok] = {"cid": cid, "name": name, "ts": time.time()}
                REFRESH_NOW.set()
                self.send_response(200)
                self.send_header(
                    "Set-Cookie",
                    "%s=%s; Path=/; HttpOnly; SameSite=Lax" % (SESSION_COOKIE, stok))
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    ("<meta charset='utf-8'>접근이 승인되었습니다 (%s). 창을 닫습니다."
                     "<script>setTimeout(()=>{location.href='/'},1200)</script>"
                     % name).encode())
            except Exception as e:
                self._json({"error": str(e)}, 400)
        else:
            self._send(404, b"not found", "text/plain")

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if is_gate_on() and not self._authed():
            self._json({"error": "인증 필요 — EVE 로그인 후 허용 캐릭터만 접근 가능"}, 401)
            return
        pfx, port = app_proxy.match(u.path)
        if pfx:
            app_proxy.serve(self, pfx, port)
            return
        if u.path == "/api/logout":
            cid = int(self._body().get("character_id", 0))
            # CCP에 리프레시 토큰 폐기 — 살아세션의 자동 재발급을 태초부터 차단
            with _tokens_lock:
                rows = list(_tokens.items())
            for one_id, t in (rows if cid in (0, None) or not cid else
                              [(cid, _tokens.get(cid, {}))]):
                rt = t.get("refresh_token")
                if not rt:
                    continue
                try:
                    req = urllib.request.Request(
                        "https://login.eveonline.com/v2/oauth/revoke",
                        data=urllib.parse.urlencode({
                            "token": rt,
                            "client_id": (CFG.get("client_id") or "").strip(),
                        }).encode(), method="POST")
                    urllib.request.urlopen(req, timeout=10)
                    sys.stderr.write("[info] CCP 토큰 폐기: %s\n" % one_id)
                except Exception as e:
                    sys.stderr.write("[warn] CCP 폐기 실패 %s: %s\n" % (one_id, e))
            # 잠금 없앤다 — 배경 갱신이 잠금 안에서 SSO 부를 때 유저가 영결
            if cid:
                _tokens.pop(cid, None)
            else:
                _tokens.clear()
            _save_tokens()
            if cid:
                DATA["chars"].pop(cid, None)
            else:
                DATA["chars"].clear()
            with SESSION_LOCK:
                _sessions.clear()          # 브라우저 세션도 함께 내려낸다
            self._json({"ok": True})
        elif u.path == "/api/refresh":
            REFRESH_NOW.set()
            self._json({"ok": True})
        elif u.path == "/api/config":
            d = self._body()
            for k in ("client_id", "redirect_base", "jobs_refresh",
                      "assets_refresh"):
                if k in d and str(d[k]).strip():
                    CFG[k] = d[k]
            cfg = {k: CFG.get(k) for k in
                   ("port", "client_id", "redirect_base", "jobs_refresh",
                    "assets_refresh", "bind", "allow_public_bind",
                    "allowed_characters")}
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        pass


def background_loop():
    # 지갑 표본은 별도 실로 — refresh이 ESI 폭풍에 5분씩 기어도
    # 표본 제때 돌고, 표본이 기어도 refresh가 안 죽는다.
    def wallet_worker():
        time.sleep(5)
        import traceback
        try:
            wallet_sample(force=True)   # 부직 직후 첫 점 — 지갑 2회뿐이라fast
        except Exception as e:
            sys.stderr.write("[warn] wallet: %s\n%s\n" % (e, traceback.format_exc()))
        while True:
            time.sleep(60)
            try:
                wallet_sample()
            except Exception as e:
                import traceback
                sys.stderr.write("[warn] wallet: %s\n%s\n" % (e, traceback.format_exc()))
    threading.Thread(target=wallet_worker, daemon=True).start()

    while True:
        REFRESH_NOW.wait(timeout=20)
        REFRESH_NOW.clear()
        try:
            refresh_once()
        except Exception as e:
            sys.stderr.write("[warn] refresh: %s\n" % e)


def tailscale_ip():
    """테일스케일 IPv4(100.64/10) 주소 탐색: CLI → ip 커맨드 → 없음."""
    try:
        out = subprocess.run(["tailscale", "ip", "-4"],
                             capture_output=True, text=True, timeout=5)
        ip = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        if ip:
            return ipaddress.ip_address(ip)
    except Exception:
        pass
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        m = re.match(r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)", line)
        if m and (m.group(1).startswith("tailscale") or m.group(1).startswith("ts")):
            try:
                return ipaddress.ip_address(m.group(2))
            except ValueError:
                pass
    return None


def resolve_bind():
    """바인딩 주소 결정 — 기본은 테일스케일 전용. 공개 대내는 불가.

    bind: "auto"(기본) | "tailscale" | IP문자열
    0.0.0.0 등 테일스케일/루프백이 아닌 주소는 allow_public_bind:true 시에만 허용.
    """
    ts = tailscale_ip()
    b = str(CFG.get("bind") or "auto").strip().lower()
    if b in ("auto", "tailscale"):
        if ts is None:
            sys.stderr.write("[warn] 테일스케일 주소가 없어 127.0.0.1 에 바인딩합니다\n")
            return "127.0.0.1", False
        return str(ts), True
    try:
        addr = ipaddress.ip_address(b)
    except ValueError:
        sys.exit("오류: bind 값 '%s' 은 IP 가 아닙니다" % b)
    if addr.is_loopback or addr in ipaddress.ip_network("100.64.0.0/10"):
        return str(addr), (addr in ipaddress.ip_network("100.64.0.0/10"))
    if not CFG.get("allow_public_bind"):
        sys.exit("오류: %s 은 테일스케일(100.64/10)·로컬 주소가 아닙니다. "
                 "그래도 열려면 config 에 allow_public_bind:true" % addr)
    sys.stderr.write("[warn] 공개 대외 바인딩(%s) 활성화됨 — 위험합니다\n" % addr)
    return str(addr), False


class Server(ThreadingHTTPServer):
    """accept 커널 이벤트 의존 끊기 — 테일스케일 인터페이스에선 커넥이 걸어져도
    수락 이벤트가 커널에 안 오는 수가 있다 (블로킹 accept 그대로 말썽: 고는 애는
    수락대기접속을 recv-Q에 품은 체 코딩을 마친다). 2초 주기 폴링으로 무조건 회수.
    폴링이야말로 이걸 안 놓치는 유일한 길."""
    daemon_threads = True
    request_queue_size = 64

    def serve_forever(self, poll_interval=0.5):
        import socket as _s
        self.socket.settimeout(2.0)
        while True:
            try:
                conn, addr = self.socket.accept()
            except KeyboardInterrupt:
                break
            except (_s.timeout, TimeoutError, OSError):
                continue
            try:
                self.process_request(conn, addr)
            except OSError as e:
                sys.stderr.write("[warn] 수락 처리 실패: %s\n" % e)
            if getattr(self, "_BaseServer__shutdown_request", False):
                break


def main():
    global BIND
    _load_tokens()
    _load_names()
    _load_wallet()
    for cid, t in list(_tokens.items()):
        _fill_corp_info(cid, t)   # 요즘 토큰은 JWT에 corp가 없어서 보정
    BIND, is_ts = resolve_bind()
    if is_ts and re.match(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
                          CFG.get("redirect_base", "")):
        # 기본값이면 등록용 콜백으로 쓸 테일스케일 주소로 자동 바꿈
        CFG["redirect_base"] = "http://%s:%d" % (BIND, PORT)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({k: CFG.get(k) for k in
                       ("port", "client_id", "redirect_base", "jobs_refresh",
                        "assets_refresh", "bind", "allow_public_bind",
                        "allowed_characters")}, f, indent=2)
    threading.Thread(target=background_loop, daemon=True).start()
    srv = Server((BIND, PORT), Handler)
    print("산업/블루프린트 대시보드: http://%s:%d  [%s]"
          % (BIND, PORT, "테일스케일 전용" if is_ts else "로컬 전용"))
    print("등록용 리다이렉트 URI: %s" % redirect_uri())
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
