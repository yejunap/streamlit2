"""데이터 있는 렌더링 — 목 스캔을 심고 표가 나타나는지 본다."""
import os

os.environ.setdefault("ARB_ID", "dumdum-id")   # 잠금이 쌍 fail-closed라 둘 다 걸어야 지나간다
os.environ.setdefault("ARB_PW", "dumdum")

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

#---- 🚫 낚시물 차단 — 세션에 쌓인 목록이 행을 지운다 (34는 표에 실제 나는 행)
at2 = AppTest.from_file("app.py", default_timeout=90)
at2.session_state["auth_ok"] = True
at2.session_state["scan"] = SCAN
at2.session_state["meta"] = META
at2.session_state["hist"] = HIST
at2.session_state["blocked"] = {"ids": [34], "names": []}
at2.run()
assert not at2.exception, [e.value for e in at2.exception]
seen_names = {n for t in at2.dataframe if "종목" in t.value.columns for n in t.value["종목"]}
assert "Tritanium" not in seen_names, seen_names
print("낚시물 차단 ok — 34(Tritanium) 걸러짐")

#---- ➕ 걸기 클릭은 쌓인다 — 두 번 걸어도 초기화되지 않는다
at3 = AppTest.from_file("app.py", default_timeout=90)
at3.session_state["auth_ok"] = True
at3.session_state["scan"] = SCAN
at3.session_state["meta"] = META
at3.session_state["hist"] = HIST
at3.run()
adder = at3.button(key="block_go")
at3.text_input(key="block_add").set_value("34")
adder.click().run()
at3.text_input(key="block_add").set_value("35")
at3.button(key="block_go").click().run()
assert at3.session_state["blocked"]["ids"] == [34, 35], at3.session_state["blocked"]
left = set()
for t in at3.dataframe:
    if "종목" in t.value.columns:
        left |= set(t.value["종목"])
assert not left, left                      # 둘 다 걸리면 남는 종목이 없다
print("➕ 누적 걸기 ok — [34, 35] 다 걸림")

#---- ⏸ 잠시 해제 — 걸어 둔 목록은 그대로, 걸리기만 않는 다
at4 = AppTest.from_file("app.py", default_timeout=90)
at4.session_state["auth_ok"] = True
at4.session_state["scan"] = SCAN
at4.session_state["meta"] = META
at4.session_state["hist"] = HIST
at4.session_state["blocked"] = {"ids": [34], "names": []}   # 표에 실제 있는 행을 걸었다
at4.run()
assert not at4.exception, [e.value for e in at4.exception]
assert not any("Tritanium" in set(t.value["종목"]) for t in at4.dataframe if "종목" in t.value.columns)
at4.button(key="block_off_btn").click().run()
assert not at4.exception, [e.value for e in at4.exception]
back = {n for t in at4.dataframe if "종목" in t.value.columns for n in set(t.value["종목"])}
assert "Tritanium" in back, back                 # 풀었으니 돌아온다
assert at4.session_state["blocked"]["ids"] == [34]   # 목록은 그대로다
print("⏸ 잠시 해제 ok — 걸림은 그대로, 표는 되돌아옴")

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

#---- 🧹 장바구니 비우기 — 게이지 지면 체크도 함께 세져 ----------
at5 = AppTest.from_file("app.py", default_timeout=90)
at5.session_state["auth_ok"] = True
at5.session_state["scan"] = SCAN
at5.session_state["meta"] = META
at5.session_state["hist"] = HIST
at5.session_state["pinned"] = {"Tritanium"}          # 체크로 담긴 것처럼
at5.run()
assert not at5.exception, [e.value for e in at5.exception]
assert at5.session_state["pinned"] == {"Tritanium"}
at5.button(key="clear_basket").click().run()
assert not at5.exception, [e.value for e in at5.exception]
assert at5.session_state["pinned"] == set()          # 바구니는 비었다
at5.run()                                            # 다시 돌려도 비어 있다 — 체크가 새면 안 산다
assert at5.session_state["pinned"] == set()
print("🧹 장바구니 비우기 ok — 체크도 함께세져")
