"""없는 페이지(404)에 스캔이 서면 안 된다.

리전 주문판 페이지 수는 하루 안에도 줄어든다 (주문이 만료되면서 뒤가져 잘려 나간다).
세어 둔 숫자를 그대로 쫓아 없는 페이지를 물면 ESI 는 "Requested page does not exist!"
를 랑한다. 지난 일에 이 한 장이 전체 스캔을 세웠다.

그래서 1) 받는 앞에 HEAD 로 다시 세운다(region_pages 의 refresh).
2)그래도 도중에 밀린 것은 빈 칸으로 지난다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esi  # noqa: E402

REAL = 5       # 오늘 진짜 페이지 수
esi._num_get = lambda *a, **k: None    # 세어 둔 숫자는 타본다 (HEAD 로 센다)
esi._num_put = lambda *a, **k: None
esi._xpages = lambda region: REAL

seen = []


def order_book(path, params=None, cache_key=None, **kw):
    pg = (params or {}).get("page")
    seen.append(pg)
    if pg > REAL:
        raise esi.ESINoData("ESI 404 for " + path)      # 없는 페이지는 ESI 가 이 리 랑한다
    return ([{"location_id": esi.JITA_STATION, "type_id": 34, "price": 9.0 + pg,
              "is_buy_order": False, "volume_remain": 50}], {})


esi.get = order_book

# 1) 세어 둔 수가 낡아도 HEAD 로 세운 것까지만 안 다 — 없는 것을 물어 404 를 맞지 않는다
log = []
seen.clear()
books = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=4, log=log.append)
assert 34 in books, books
assert max(seen) == REAL, seen
assert any("5/5" in m for m in log), log


# 2) 한 장을 못 받았어도(429·5xx) 남은 것으론 간다 — 모자란 것은 진단 로그에 남긴다
def flaky(path, params=None, **kw):
    if (params or {}).get("page") == 3:
        raise esi.ESIRateLimited("ESI 429")
    return order_book(path, params=params, **kw)


esi.get = flaky
log.clear()
books = esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=4, log=log.append)
assert 34 in books, books
assert any("못 받은" in m for m in log), log


# 3) 하나도 못 받았다면 모른 체하지 않는다 — 빈 표를 올리지 말고 소리를 낸다
def nothing(path, params=None, **kw):
    raise esi.ESIRateLimited("ESI 502")


esi.get = nothing
try:
    esi.fetch_hub_orders(esi.JITA_REGION, esi.JITA_STATION, workers=2, log=lambda m: None)
    raise AssertionError("전부를 못 받았으면 넘어 가면 안 된다")
except esi.ESIError:
    pass

# 4) 주문판을 받아 들일 때 세어 둔 숫자를 믿지 않는다 — refresh 가 걸렸다
src = open(os.path.abspath(esi.__file__), encoding="utf-8").read()
assert "region_pages(region, refresh=True)" in src, "주문판 받 기 앞에 세우는 데가 없다"

print("없는 페이지 견딤 ok — 404 에 스캔이 안 서면")
