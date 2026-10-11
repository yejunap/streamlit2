"""ESI 창사 하나가 죽었다고 전체 스캔이 서면 안 된다 — 넷 안 보고 본다."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esi  # noqa: E402

_real = esi.get


def raise_(exc):
    def _f(*a, **k):
        raise exc
    return _f


# 1) history 404 — 거래 실적 없는 종목에서 난다. 없는 걸로 다니면 된다.
esi.get = raise_(esi.ESINoData("404"))
assert esi.fetch_histories(esi.JITA_REGION, [34, 35], days=10) == {}

# 2) 5xx 도 같다 — 못한 걸으로 간다.
esi.get = raise_(esi.ESIRateLimited("502"))
assert esi.fetch_histories(esi.JITA_REGION, [34], days=10) == {}


# 3) 한 종목만 404 — 나머지는 살아야 한다.
def mixed(*a, **k):
    if str((k.get("params") or {}).get("type_id")) == "99999":
        raise esi.ESINoData("404")
    return [{"volume": 7}], {}


esi.get = mixed
out = esi.fetch_histories(esi.JITA_REGION, [34, 99999], days=10)
assert 34 in out and 99999 not in out, out

# 4) 타입 정보 도 404에 지지 않는다 — 부피 모름으로 두고 지난다.
esi.get = raise_(esi.ESINoData("404"))
meta = esi.fetch_type_meta([34])
assert meta[34]["volume"] == 0.0, meta

esi.get = _real
print("esi 견딤 ok — 404/5xx에 스캔이 안 서면")
