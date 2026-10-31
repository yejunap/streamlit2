"""대시보드가 예 없이 뜨는지 확인한다(Streamlit AppTest).

잠금 제거 — 이 앱은 EVE 산업 대시보드(SooJack 전용 EVE SSO 를 지나는 창)에
탭으로 얹혀 쓰이므로 암호가 없다. 시크릿 쌍 없이도 바로 열려야 한다."""
import os

from streamlit.testing.v1 import AppTest

os.environ.pop("ARB_ID", None)
os.environ.pop("ARB_PW", None)

at = AppTest.from_file("app.py", default_timeout=90)
at.run()
assert not at.exception, [e.value for e in at.exception]
assert at.title[0].value.startswith("📈"), "대시보드가 바로 열려야 한다"
assert any("주문판 전체 스캔" in w.value for w in at.warning), "스캔 유도가 없다"
print("render ok:", at.title[0].value)
