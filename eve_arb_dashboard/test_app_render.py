"""대시보드가 예 없이 뜨는지 확인한다(Streamlit AppTest)."""
import os

from streamlit.testing.v1 import AppTest

# 잠금 암호가 없으면 열리지 않는다 — 예 값이 새지 않았는지 fail-closed 본다.
# 단, 이 디렉터에 시크릿 파일(로컬 시드)이 있으면 그 자체가 "걸린 상태"라 이 단계는 건너뛴다.
import pathlib
_seeded = pathlib.Path(".streamlit/secrets.toml").exists()
os.environ.pop("ARB_PW", None)
if not _seeded:
    at = AppTest.from_file("app.py", default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("ARB_PW" in e.value for e in at.error), "시크릿 없다고 안 열려야 한다"

os.environ["ARB_PW"] = "5767" if _seeded else "dumdum"   # 테스트용 잠금
at = AppTest.from_file("app.py", default_timeout=90)
at.run()
assert not at.exception, [e.value for e in at.exception]
assert at.title[0].value.startswith("🔒"), "잠금이 안 걸렸다"
at.text_input[0].set_value(os.environ["ARB_PW"])
at.button[0].click().run()
assert not at.exception, [e.value for e in at.exception]
assert at.title[0].value.startswith("📈")
assert any("주문판 전체 스캔" in w.value for w in at.warning), "스캔 유도가 없다"
print("render ok:", at.title[0].value)
