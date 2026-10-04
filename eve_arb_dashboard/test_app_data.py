"""데이터 있는 렌더링 — 목 스캔을 심고 표가 나타나는지 본다."""
from datetime import datetime, timedelta, timezone

from streamlit.testing.v1 import AppTest

now = datetime.now(timezone.utc)


def book(ask, ask_q, bid, bid_q):
    return {"sells": [(ask, ask_q, now - timedelta(hours=3))],
            "buys": [(bid, bid_q, now - timedelta(hours=3))]}


SCAN = {
    "books": {
        34: {"Jita": book(3.0, 5_000_000, 3.9, 5_000_000),
             "Amarr": book(3.4, 4_000_000, 2.8, 4_000_000)},
        35: {"Jita": book(16.0, 2_000_000, 15.0, 2_000_000),
             "Amarr": book(20.0, 2_000_000, 14.0, 2_000_000)},
    },
    "scores": {34: 1e7, 35: 1e7},
    "at": now,
}
META = {34: {"volume": 0.01, "name": "Tritanium"}, 35: {"volume": 0.01, "name": "Pyerite"}}
HIST = {34: 1_000_000, 35: 500_000}

at = AppTest.from_file("app.py", default_timeout=90)
at.session_state["scan"] = SCAN
at.session_state["meta"] = META
at.session_state["hist"] = HIST
at.run()
assert not at.exception, [e.value for e in at.exception]
tables = list(at.dataframe)
print("실행 테이블:", len(tables))
assert tables, "표가 없다"
head = tables[0].value
cols = ["종목", "방향", "순이익%", "ISK/m³", "총부피 m³", "필요 운항"]
print(head[cols].to_string(index=False) if all(c in head.columns for c in cols) else head.head().to_string())
print("표 개수:", len(tables))
# 헤 클릭 정렬이 숫자 정렬로 되려면 컬럼이 문자면 안 된다
num = head[["총부피 m³", "ISK/m³", "매집가"]]
assert all(str(t).startswith(("int", "float")) for t in num.dtypes), num.dtypes
print("컬럼 숫자형 ok — 헤더클릭 정렬이 숫자 기준으로 동작")
