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
        json.loads(_tj)   # 목숨 없는 판 걸러내기
        with open(os.path.join(HERE, "tokens.json"), "w", encoding="utf-8") as f:
            f.write(_tj)
        _secrets_note += " — 토큰판 갱신"
    else:
        _secrets_note = "시크릿 없음 — b64 또는 json 키를 넣으세요"
except Exception as e:
    _secrets_note = "시크릿 처리 실패: %s" % e


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


def api(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"_login": True}
        raise
    except Exception as e:
        return {"_err": str(e)}


st.title("🛰 EVE 산업 포털")
st.caption("🔎 %s" % _secrets_note)
ok, why = backend()
if not ok:
    st.error("엔진 없음 — %s" % why)
    st.stop()

d = api("/api/state")
if d.get("_login"):
    st.warning("EVE 로그인 후 허용 캐릭터만 접근 가능")
    st.link_button("🔐 EVE SSO 로그인", BASE + "/sso/login")
    st.stop()
if d.get("_err"):
    st.error("엔진 응답 없음 — %s" % d["_err"])
    st.stop()

chars = d.get("characters") or []
if not chars:
    st.info("캐릭터 데이터 비음 — 로컬이면 포털 창에서 EVE SSO 로그인, "
            "**클라우드면 설정(Sidebar ⋮ → Settings → Secrets)에 `tokens_json`**으로 "
            "tokens.json 내용을 붙여 넣으세요. (이 기기엔 tokens.json이 없거나 만료됨)")
    st.link_button("🔐 포털에서 로그인 (로컬일 때만 됨)", BASE + "/sso/login")
    st.code("엔진 되준 key: %s" % list(d.keys()))
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
