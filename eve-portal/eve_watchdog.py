#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8421 포털 지킴이 — 죽으면 5초 뒤 되살리고, 메면 내리고 다시 세운다.

EVE 본서 정오(한국 08:00)에 포털이 '메어' 보이는 창이 있다 — 그때는 응답이
6초도 안 나온다. 8초 안에 /api/state 를 세 판 못 내리면 강제로 내리고 다시
세운다. 죽은 자리의 종료 코드는 남긴다. 사용:
    setsid nohup python3 eve_watchdog.py >> /tmp/eve-watchdog.log 2>&1 < /dev/null &
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def targets():
    """config.json 을 열어 붙어볼 주소들 — bind auto 면 127.0.0.1 에 안 걸린다."""
    host, port = "127.0.0.1", 8421
    try:
        import json
        cfg = json.load(open(os.path.join(HERE, "config.json")))
        port = cfg.get("port", port)
        base = cfg.get("redirect_base", "")
        if "://" in base:
            host = base.split("://", 1)[1].split("/", 1)[0].split(":")[0]
    except Exception:
        pass
    return [("127.0.0.1", port), (host, port)]


def alive():
    """붙는 자리가 있으면 산다 — 서버가 통째 말려도 접힌다."""
    import socket
    for host, port in targets():
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            continue
    return False


def responsive():
    """/api/state 가 8초 안에 얼굴을 내면 안 메었다 — 401 도 살아 있는 얼굴."""
    import urllib.error
    import urllib.request
    for host, port in targets():
        try:
            with urllib.request.urlopen("http://%s:%d/api/state" % (host, port),
                                        timeout=8) as r:
                if r.status in (200, 401):
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (200, 401):   # 401 은 예외로 던지지만 산 얼굴이다
                return True
        except Exception:
            continue
    return False


def spawn():
    log = open(os.path.join(HERE, "server.log"), "ab", 0)
    err = open(os.path.join(HERE, "server.err"), "ab", 0)
    return subprocess.Popen([sys.executable, "server.py"], cwd=HERE,
                            stdout=log, stderr=err, stdin=subprocess.DEVNULL,
                            start_new_session=True)


def main():
    p, froze = None, 0
    while True:
        if p is not None and p.poll() is not None:
            print("[watchdog] 서버 그림 내림 code=%s — 5초 뒤 다시 세운다"
                  % p.returncode, flush=True)
            p = None
        if p is None and not alive():
            p = spawn()
            froze = 0
            print("[watchdog] 세움 pid=%s" % p.pid, flush=True)
        elif p is not None and p.poll() is None:
            if responsive():
                froze = 0
            else:
                froze += 1
                print("[watchdog] 응답 없음 (%d/3)" % froze, flush=True)
                if froze >= 3:
                    print("[watchdog] 메었군 — 내리고 다시 세운다", flush=True)
                    p.kill()
                    p, froze = None, 0
        time.sleep(5)


if __name__ == '__main__':
    main()
