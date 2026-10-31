"""freshen 회귀 — 모듈이 캐시에 박혀 있어도 고친 코드가 반영된다."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import esi         # noqa: E402  캐시에 남 있을 때를 만든다
import arb_core    # noqa: E402
import freshen     # noqa: E402

esi_get, arb_fn = esi.get, arb_core.candidate_scores
freshen.freshen_modules(HERE)                   # 첫째: 스탬프만 남긴다
freshen.freshen_modules(HERE)                 # 둘째: 그대로여야 한다
assert esi.get is esi_get and arb_core.candidate_scores is arb_fn, "안 바꾼 건 그대로"

# 파일 고친 뒤엔 함수객체가 바껴야 한다
for name in ("esi", "arb_core"):
    os.utime(f"{HERE}/{name}.py", (time.time() + 5,) * 2)
freshen.freshen_modules(HERE)
assert esi.get is not esi_get, "esi 재실행 안 됨"
assert arb_core.candidate_scores is not arb_fn, "arb_core 재실행 안 됨"
print("freshen ok — 그대로면 유지, 바뀌었으면 재실행")
