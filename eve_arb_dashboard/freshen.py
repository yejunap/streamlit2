"""코드를 고치면 새로고침 없이 반영 — 스트리밍 서버용 모듈 새로고침.

Streamlit 은 메인 스크립트(app.py)만 다시 돌기고 옆 모듈(esi, arb_core)은
메모에 박아 쓰는 버릇이 있다. 그래서 ① 앱 디렉터를 sys.path 에 심고
② 고친 티가 난다면(mtime)통에 있던 놈을 다시 올린다.
"""
import importlib
import os
import sys

WATCH = ("esi", "arb_core")


def freshen_modules(here: str) -> None:
    """here 디렉터리의 WATCH 모듈을 필요하면 다시 올린다."""
    if here not in sys.path:
        sys.path.insert(0, here)
    for name in WATCH:
        mod, stamp = sys.modules.get(name), getattr(sys, f"_fresh_{name}", None)
        path = os.path.join(here, name + ".py")
        if mod is not None and stamp is not None and os.path.getmtime(path) != stamp:
            importlib.reload(mod)
        setattr(sys, f"_fresh_{name}", os.path.getmtime(path))