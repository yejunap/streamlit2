"""대시보드가 예 없이 뜨는지 확인한다(Streamlit AppTest)."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=90)
at.run()
assert not at.exception, [e.value for e in at.exception]
assert at.title[0].value.startswith("📈")
assert any("주문판 전체 스캔" in w.value for w in at.warning), "스캔 유도가 없다"
print("render ok:", at.title[0].value)
