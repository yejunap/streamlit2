"""캐시 만심(TTL) — 같은 것을 두 번 물지 않는다.

왕복 한 번이야 짧지만 만 회를 세면 첫 실행이 몇 분으로 기인다. 안 바뀔 것은
요청을 내지 않고 지나간다. 그 한 칸을 새는 지킨다."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esi  # noqa: E402

KEY = "ttl_합성"
P = esi._cache_path(f"etag_{KEY}.json")
assert not os.path.exists(P), f"지난 실행의 것이 남았다 지운다: {P}"


def no_network():
    """여다나면 터뜨린다 — 지나가야 할 길에서 나가지 못하기 때문이다."""
    def boom(*a, **k):
        raise AssertionError("요청을 나갔다")
    return boom


_real_session, real_get = esi.session, esi.get

# 1) 만심 안쪽 — 요청 없이 본문만는다
esi.session = no_network()
esi._save_etag(KEY, 'W/"x"', {"name": "Tritanium"})
body, hdr = esi.get("/anything", cache_key=KEY, ttl=3600)
assert body == {"name": "Tritanium"} and hdr.get("X-Cache") == "ttl", (body, hdr)

# 2) 만심이 지났다 — 그을 때 나간다
os.utime(P, (time.time() - 9999,) * 2)
try:
    esi.get("/anything", cache_key=KEY, ttl=3600)
    raise AssertionError("만심이 지났는데도 안 나갔다")
except AssertionError as e:
    assert "요청" in str(e), e

# 3) ttl 을 주지 않으면 기존처럼 ETag 로 간다 (주문판은 늘 새 것으로 받아야 한다)
try:
    esi.get("/anything", cache_key=KEY)
    raise AssertionError("ttl 없이는 무조건 나가야 한다")
except AssertionError as e:
    assert "요청" in str(e), e

# 4) 타입·거래량의 기본 만심이 주문판을 물지 않는다 — 받아 오던 크기가 다다르다
assert esi.TTL_TYPE > 0 and esi.TTL_HISTORY > 0
src = open(os.path.abspath(esi.__file__), encoding="utf-8").read()
assert "ttl=TTL_TYPE" in src and "ttl=TTL_HISTORY" in src, "fetch 에 만심이 걸이지 않았다"
orders = [l for l in src.splitlines() if "orders/" in l and "get(" in l]
assert not any("ttl=" in l for l in orders), "주문판까지 캐리면 안 된다"
os.remove(P)

# 5) 페이지 수 새김 — 세어 둔 숫자를 재 쓴다 (1페이지를 통채로 받는 일 없다)
esi.session = _real_session
calls = {"n": 0}


def fake_get(path, params=None, cache_key=None, retries=4, ttl=None):
    calls["n"] += 1
    return None, {"X-Pages": "43"}


esi.get = fake_get
try:
    assert esi.region_pages(10000003) == 43
    assert esi.region_pages(10000003) == 43      # 세어 둔 것을 쓴다
    assert calls["n"] == 1, calls                # 한 번만 나갔다
finally:
    esi.get = real_get
    if os.path.exists(esi._cache_path("num_rpages_10000003.txt")):
        os.remove(esi._cache_path("num_rpages_10000003.txt"))

print("esi 캐시 만심 ok — 있으면 묻지 않고, 없으면 나간다")
