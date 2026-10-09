"""앱 랩퍼(run_scan/enrich)가 실제 ESI 호출로 도는지 확인 (기본 생략).

실행: ESI_LIVE=1 python3 test_app_scan.py    # 첫 실행은 수 분, 캐시되면 빠르다
"""
import os
import time

from streamlit.testing.v1 import AppTest

if __name__ == "__main__" and os.environ.get("ESI_LIVE") != "1":
    print("건너뜀: ESI 실호출 테스트는 ESI_LIVE=1 로 돌린다.")
    raise SystemExit(0)

os.environ.setdefault("ARB_PW", "dumdum")   # 잠금이 걸려야 통과한다

at = AppTest.from_file("app.py", default_timeout=900)
at.session_state["auth_ok"] = True      # 🔒 잠금 통과
at.run()                            # 위젯 트리를 먼저 채운다
t0 = time.time()
at.sidebar.button[0].click().run()   # "주문판 전체 스캔"
assert not at.exception, [e.value for e in at.exception]
scan = at.session_state["scan"]
print(f"스캔 완료 {time.time()-t0:.0f}s — {len(scan['books'])}종목")
tables = list(at.dataframe)
print("표:", len(tables), "| 첫 열:", list(tables[0].value.columns)[:6])
