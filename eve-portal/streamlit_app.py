#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EVE 산업 포털 — Streamlit 런처.

기존 엔진(server.py)은 고대로 돌리고(스레드), 그 위에 같은 데이터를 얹은
간소한 얼굴을 streamlit 으로 웃는다. 엔진의 SSO·폴링·캐시는 고스란히 산다.
    streamlit run streamlit_app.py
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

import streamlit as st

st.set_page_config(page_title="EVE 산업 포털", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("EVE_APPS_DIR", os.path.join(HERE, "apps"))
import server  # 같은 자리의 복사 — __main__ 가드라 가져오기 안전

CFG = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
BASE = "http://127.0.0.1:%s" % CFG.get("port", 8421)

# Streamit Cloud — 설정 → Secrets에 **tokens_b64**로 tokens.json의 base64를 넣는다.
# (TOML 따영블에서 중첩 중괄호가 깨진다 —그래서 중괄호 없는 base64를 쓴다.
#  tokens_json(plain)도 호환되나 ''' 안에서 }가 새면 안니니 b64를 권한다.)
_secrets_note = ""
try:
    import base64 as _b64
    _tj = None
    if "tokens_b64" in st.secrets:
        _tj = _b64.b64decode(st.secrets["tokens_b64"]).decode()
        _secrets_note = "시크릿 b64 수신 — %d바이트" % len(_tj)
    elif "tokens_json" in st.secrets:
        _tj = st.secrets["tokens_json"]
        _secrets_note = "시크릿 json 수신"
    if _tj:
        if not isinstance(_tj, str):
            _tj = json.dumps(_tj)
        json.loads(_tj)   # 시크릿 JSON 형식 검증
        _token_path = os.path.join(HERE, "tokens.json")
        # Secrets는 최초 부팅 때만 초기 데이터로 사용한다.
        # 매 rerun마다 덮어쓰면 SSO 직후 저장한 새 캐릭터 토큰이 사라진다.
        if not os.path.exists(_token_path):
            with open(_token_path, "w", encoding="utf-8") as f:
                f.write(_tj)
            _secrets_note += " — 초기 토큰 복원"
        else:
            _secrets_note += " — 기존 토큰 파일 유지"
    else:
        _secrets_note = "시크릿 없음 — b64 또는 json 키를 넣으세요"
except Exception as e:
    _secrets_note = "시크릿 처리 실패: %s" % e

# 클라우드 SSO — 콜백注册地址를 시크릿이나 앱 자체 주소로 엔진에 뒤입힌다.
# EVE 개발자 포털에 등록한 콜백: https://<앱주소>/oauth/callback
try:
    _rb = st.secrets.get("redirect_base")
    if not _rb:
        _h = st.context.headers.get("host") or ""
        if "streamlit.app" in _h:
            _rb = "https://" + _h
    if _rb:
        server.CFG["redirect_base"] = _rb
    # 시크릿에 직접 두 적당: client_id / redirect_full — 포털과 한 글자용으로
    if st.secrets.get("client_id"):
        server.CFG["client_id"] = st.secrets["client_id"]
    _rfull = st.secrets.get("redirect_full")
    if _rfull:
        server.redirect_uri = lambda: _rfull
except Exception:
    pass

# OAuth callback helpers. EVE SSO returns to /oauth/callback with code/state.
import urllib.parse as _up


def _say(step):
    """Never log authorization codes, tokens, client secrets, or state values."""
    print("[sso] %s" % step)


def do_exchange(code, state):
    """Exchange a one-time authorization code; mark only successful exchanges."""
    _say("Authorization code received")
    if st.session_state.get("_cb_seen") == state:
        return st.session_state.get("_cb_msg", "이미 처리된 콜백입니다")
    try:
        _say("Token exchange starting")
        cid = server.sso_complete(code, state)
        msg = "%s님 로그인 완료 — 토큰 저장됨" % server._tokens[cid]["name"]
        st.session_state["_cb_seen"] = state
        _say("Login completed")
    except Exception as e:
        msg = "SSO 완료 실패: %s" % e
        _say("Login failed: %s" % e)
    st.session_state["_cb_msg"] = msg
    return msg


_sso_msg = ""


@st.cache_resource
def backend():
    """엔진 한 판 — 8421에 이미 다른 얼굴이 붙어 있으면 그걸 쓴다."""
    errs = []

    def run():
        # 예외를 삼키지 말자 — 목숨 없는 스레드가 제일 무섭다
        try:
            server.main()
        except Exception as e:
            errs.append("%s: %s" % (type(e).__name__, e))

    threading.Thread(target=run, daemon=True).start()
    for _ in range(25):
        try:
            urllib.request.urlopen(BASE + "/", timeout=1)
            return True, ""
        except urllib.error.HTTPError:
            return True, ""      # 401/302 도 살아 있는 얼굴
        except Exception:
            if errs:
                return False, errs[0]
            time.sleep(1)
    return False, "엔진 안 열림 (25초 대기)"


# Start the engine before handling callback so it restores pending PKCE state.
_ok, _why = backend()
if not _ok:
    st.error("SSO callback 엔진 시작 실패: %s" % _why)
    st.stop()

try:
    _q = dict(st.query_params)
except Exception:
    _q = {}

if "error" in _q:
    st.error("EVE SSO 오류: %s" % _q.get("error_description", _q["error"]))
    st.stop()

if _q.get("code") and _q.get("state"):
    _say("Callback endpoint reached")
    _sso_msg = do_exchange(_q["code"], _q["state"])
    if _sso_msg.startswith("SSO 완료 실패"):
        st.error(_sso_msg)
        st.link_button("EVE SSO 다시 로그인", server.sso_login_url())
    else:
        st.success(_sso_msg)
        st.markdown('<meta http-equiv="refresh" content="1;url=/" />',
                    unsafe_allow_html=True)
        st.link_button("포털로 이동", "/")
    st.stop()


def api(path):
    try:
        if path == "/api/state":
            # 토큰판은 방금 로그인도 디스크에 함께 적는다 — 메모리에 합쳐 입힌다.
            # (_load_tokens는 파일 없으면 통째 비우니 쓰지 말 것 — set병만)
            _tp = os.path.join(HERE, "tokens.json")
            if os.path.exists(_tp):
                with open(_tp, encoding="utf-8") as f:
                    _disk = {int(k): v for k, v in json.load(f).items()}
                with server._tokens_lock:
                    for _cid, _t in _disk.items():
                        server._tokens.setdefault(_cid, _t)
            return server.build_state()
        with urllib.request.urlopen(BASE + path, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"_login": True}
        raise
    except Exception as e:
        return {"_err": str(e)}


def login_url():
    """EVE 등록 주소로 만든 로그인 URL — 엔진이 있으면 PKCE까지 산다."""
    try:
        return server.sso_login_url()
    except Exception:
        return BASE + "/sso/login"


def manual_box():
    """리다이렉트가 안 열렸을 때 — 쿼리를 프록시가 삼켰으면 이 길이 마지막 길."""
    with st.expander("EVE 로그인하고 왔는데 안 뜨면 — 콜백 URL 붙여넣기"):
        _u = st.text_input("돌아온 리다이렉트 주소 전체", key="cb_manual",
                           placeholder="https://evemine.streamlit.app/oauth/callback?code=...&state=...")
        if st.button("이걸로 로그인 확정") and _u:
            _qs = _up.parse_qs(_up.urlparse(_u).query)
            if "code" in _qs and "state" in _qs:
                st.info(do_exchange(_qs["code"][0], _qs["state"][0]))
            else:
                st.error("주소에서 code/state를 못 찾았습니다 — 전체 복사가 대상이에요")


st.title("🛰 EVE 산업 포털")
st.caption("🔎 %s" % _secrets_note)
if _sso_msg:
    st.info(_sso_msg)
ok, why = backend()
if not ok:
    st.error("엔진 없음 — %s" % why)
    st.stop()

d = api("/api/state")
if d.get("_login"):
    st.warning("EVE 로그인 후 허용 캐릭터만 접근 가능")
    st.link_button("🔐 EVE SSO 로그인", login_url())
    manual_box()
    st.stop()
if d.get("_err"):
    st.error("엔진 응답 없음 — %s" % d["_err"])
    st.stop()

chars = d.get("characters") or []
if not chars:
    # 안전표: 상태가 비었어도 토큰판이 보이면 캐릭터만큼은 드러눕힌다
    _live = []
    try:
        _mem = dict(server._tokens)
        _tp = os.path.join(HERE, "tokens.json")
        if os.path.exists(_tp):
            with open(_tp, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    _mem.setdefault(int(k), v)
        _live = list(_mem.values())
    except Exception:
        pass
    if _live:
        st.warning("상태판은 비었음 — 토큰판으로 캐릭터만이라도 실었습니다. "
                   "(엔진 새로침/폴링 필요)")
        st.dataframe([{"이름": t.get("name"), "코퍼": t.get("corp_name"),
                       "갱신오류": t.get("refresh_error", "")} for t in _live],
                     use_container_width=True)
    else:
        st.info("캐릭터 데이터 비음 — EVE 로그인하세요. 클라우드에 토큰을 영구히 "
                "두려면 Secrets에 `tokens_b64`를 넣으세요.")
        st.link_button("🔐 EVE 로그인 — 이 뜨는 곳의 주소로 콜백 등록 필요", login_url())
        manual_box()
    # 속좁은 데까지 화면에 — 어디가 안 오는지 지가 말하게 한다
    try:
        _tok = {str(k): {"이름": v.get("name"), "갱신오류": v.get("refresh_error", "")}
                for k, v in server._tokens.items()}
    except Exception as e:
        _tok = {"오류": str(e)}
    st.expander("🩺 진단 — 붙여주시면 제가 맞춵니다").json(
        {"엔진이 쥔 토큰": _tok,
         "상태 키": list(d.keys()),
         "상태 짤줌": {k: (v if isinstance(v, (int, float, str, bool)) else f"{type(v).__name__}:{len(v) if hasattr(v, '__len__') else ''}") for k, v in d.items()},
         "서버 목옥": {"tokens 주소": id(server._tokens),
                    "함수 주소": id(server.build_state)}}
        if isinstance(d, dict) else {"상태": str(type(d))}
    )
    st.stop()

with st.sidebar:
    st.caption("기관: %d · 캐릭터 %d" % (
        len({c.get("corp_name") for c in chars}), len(chars)))
    if st.button("🔄 새로고침"):
        st.rerun()

t1, t2, t3, t4 = st.tabs(["캐릭터", "생산", "블루프린트", "지갑"])
with t1:
    st.dataframe([{
        "이름": c.get("name"), "코퍼": c.get("corp_name"),
        "직업": c.get("title", ""),
    } for c in chars], use_container_width=True)
with t2:
    rows = []
    for c in chars:
        for j in c.get("jobs", []):
            rows.append({"캐릭터": c["name"], "활동": j.get("activity"),
                         "진행": round(j.get("progress", 0), 1),
                         "완료": j.get("runs"), "장소": j.get("location", ""),
                         "끝나면": j.get("stops", "")})
    st.dataframe(rows or [{"캐릭터": "없음"}], use_container_width=True)
with t3:
    rows = []
    for c in chars:
        for bp in c.get("blueprints", [])[:400]:
            rows.append({"캐릭터": c["name"], "블프": bp.get("name"),
                         "난이도": bp.get("location", "")})
    st.dataframe(rows or [{"캐릭터": "없음"}], use_container_width=True)
with t4:
    rows = []
    for c in chars:
        for w in (c.get("wallet") or {}).get("lines", [])[:200]:
            rows.append({"캐릭터": c["name"], "소식": w.get("type"),
                         "ISK": w.get("amount"), "때": w.get("date")})
    st.dataframe(rows or [{"캐릭터": "없음"}], use_container_width=True)

st.caption("엔진 상태 — SSO 리다이렉트는 %s 로 등록" % CFG.get("redirect_base"))
