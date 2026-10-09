"""데이터 있는 렌더링 — 목 스캔을 심고 표가 나타나는지 본다."""
from datetime import datetime, timedelta, timezone

from streamlit.testing.v1 import AppTest

now = datetime.now(timezone.utc)


def book(ask, ask_q, bid, bid_q):
    return {"sells": [(ask, ask_q, now - timedelta(hours=3))],
            "buys": [(bid, bid_q, now - timedelta(hours=3))]}


SCAN = {
    # 세 쌍을 다 실는다 — Jita를 기준으로 짝마다 창이 다르다.
    "hubs": ["Jita", "Amarr", "Dodixie", "Rens"],
    "books": {
        # 34: Jita가 싸고 세 허브가 다 비싸다 → 세 쌍에서 고만 아비가 나야 한다
        34: {"Jita": book(3.0, 5_000_000, 3.9, 5_000_000),
             "Amarr": book(3.4, 4_000_000, 2.8, 4_000_000),
             "Dodixie": book(3.6, 4_000_000, 2.9, 4_000_000),
             "Rens": book(3.2, 4_000_000, 2.7, 4_000_000)},
        # 35: Rens에만 걸린 종목 — Jita↔Rens 쌍만 살아야 한다
        35: {"Jita": book(16.0, 2_000_000, 15.0, 2_000_000),
             "Rens": book(20.0, 2_000_000, 14.0, 2_000_000)},
    },
    "scores": {34: 1e7, 35: 1e7},
    "at": now,
}
META = {34: {"volume": 0.01, "name": "Tritanium"}, 35: {"volume": 0.01, "name": "Pyerite"}}
HIST = {34: 1_000_000, 35: 500_000}

at = AppTest.from_file("app.py", default_timeout=90)
at.session_state["auth_ok"] = True      # 🔒 잠금 통과 — 잠금 자체는 test_app_render가 본다
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

#---- 여러 쌍이 한 표에 섞여 실린다 — 쌍 칼럼이 없거나 한 쌍이면 통합이 안 된 것
assert "쌍" in head.columns, list(head.columns)
seen = set(head["쌍"])
assert seen <= {"Jita↔Amarr", "Jita↔Dodixie", "Jita↔Rens"}, seen
assert len(seen) >= 2, seen
print("섞인 쌍:", sorted(seen))
# 라벨은 각 쌍의 두 번째 허브 이름을 담는다 — 요약표(방향 칸 없음)는 걸러른다
dirs = {d for t in tables if "방향" in t.value.columns
        for d in set(t.value["방향"])}
assert any("Dodixie" in d for d in dirs) and any("Rens" in d for d in dirs), dirs
print("방향 라벨에 허브 이름 실림 ok:", sorted(dirs)[:4])
