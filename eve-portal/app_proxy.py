#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8421 포털 안의 /apps/<이름>/ 를 127.0.0.1:85xx(streamlit)에 겹쳐 중계한다.

류: 8421 포털에 중첩해서 탭을 열어도 앱이 그대로 SSO를 타게 하기 위함이다.
그리고 이걸 그냥 리다이렉트로 할 수 없는게 streamlit이 웹소켓(/_stcore/stream)을
쓰기 때문이고 그 웹소켓에서 Host가 Origin과 다르면 거른다. 그래서 이 파일을
통해서는 Host를 고치치 않는다.

원래는 같은 기기의 포트 4개로 흩어져있다(serve_local_apps.sh).
"""

import select
import socket

# 포털의 하위경로  ← 포크되는 앱의 포트(모두 127.0.0.1에 바인드함)
APP_PORTS = {
    "/apps/crystal/": 8521,
    "/apps/fighter/": 8522,
    "/apps/arb/": 8523,
    "/apps/capital/": 8524,
}

# 요청을 보낼 때 덜어내는 헤더. 호스트 별풍(hop-by-hop)이라 연결이 바뀐다.
# 붙음문자(Host, Origin, Sec-WebSocket-*)은 여기 없고 Connection/Upgrade은 계산한다.
PUMP_TIMEOUT = 900     # 초: 두 커넥이 아무것도 안 하면 끊는다
_STRIP_REQ = frozenset((
    "connection", "keep-alive", "proxy-connection", "te", "trailer",
    "transfer-encoding", "upgrade", "content-length",
))

def _rewrite_head(head, pfx, port):
    """업스트림이 우리 접두어를 모른체 보낸다. Set-Cookie Path만 겹으로,
    Location은 접두어이 붙여 찾게 하고, X-Frame-Options는 내린다."""
    lines = head.split(b"\r\n")
    out = [lines[0]]
    for ln in lines[1:]:
        k, _, v = ln.partition(b":")
        kl = k.lower()
        if kl == b"x-frame-options":
            continue
        if kl == b"location" and v.strip():
            v = v.strip()
            for base in (b"http://127.0.0.1:%d" % port,
                         b"http://localhost:%d" % port):
                if v.startswith(base):
                    v = v[len(base):] or b"/"
            if not v.startswith(b"/"):
                v = b"/" + v
            p = pfx.encode()
            if not v.startswith(p):          # 업스트림은 접두어를 모른다
                v = p + v.lstrip(b"/") if v != b"/" else p
            v = b" " + v
        elif kl == b"set-cookie":
            v = v.replace(b" Path=/", (" Path=%s" % pfx).encode())
            # 주의: v 는 bytes, decode 해야 str 과 비교할 수 있다 (bytes in str → TypeError)
            if "Path=" not in v.decode("latin-1"):
                v = v + ("; Path=%s" % pfx).encode()
        out.append(k + b":" + v)
    return b"\r\n".join(out)

def serve(handler, pfx, port):
    """포털 핸들러를 시켜 브라우저와 붙는다. 웹소켓 확장이면 붙는다.

    이 함수가 돌아오면 포털의 연결을 끊는다 (브라우저가 다시 접속한다).
    """
    # 앱이 --server.baseUrlPath로 떠 있어서 접두어를 그대로 알아낸다 — 자르지 않는다
    path = handler.path
    body = b""
    n = handler.headers.get("Content-Length")
    if n:
        body = handler.rfile.read(int(n))
    elif (handler.headers.get("Transfer-Encoding") or "").lower() == "chunked":
        handler._send(411, "중계에서는 청김.requests만 받을 수 있다", "text/plain")
        return
    upg = (handler.headers.get("Upgrade") or "").lower() == "websocket"

    try:
        up = socket.create_connection(("127.0.0.1", port), timeout=10)
    except OSError:
        handler._send(502, "앱이 말해 잇서요: :%d — serve_local_apps.sh "
                           "실행 중?" % port, "text/plain")
        return
    try:
        lines = ["%s %s HTTP/1.1" % (handler.command, path)]
        for k, v in handler.headers.items():
            if k.lower() in _STRIP_REQ:
                continue
            lines.append("%s: %s" % (k, v))      # Host/Origin은 원문대로 보낸다
        lines.append("Connection: " + ("upgrade" if upg else "close"))
        if upg:
            lines.append("Upgrade: websocket")
        if body:
            lines.append("Content-Length: %d" % len(body))
        up.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1",
                                                           "replace") + body)
        head, rest = _read_head(up)
        if not head:
            handler._send(502, "앱이 응답을 끊었다 :%d" % port, "text/plain")
            raise OSError("업스트림 응답 없음")
        head = _rewrite_head(head, pfx, port)
        if rest:
            head += b"\r\n\r\n" + rest
        else:
            head += b"\r\n\r\n"
        handler.wfile.write(head)
        handler.wfile.flush()
        # 웹소켓이든 일반이든 남은 삶은 인라인으로 — 이 커넥은 여기가 끝.
        # 스레드에 넘기면 이 핸들러가 그만 커넥을 접어서 양쪽이 깨진다.
        _pump(up, handler.connection)
    except OSError:
        pass
    finally:
        try:
            up.close()
        except OSError:
            pass
    handler.close_connection = True


def _read_head(sock):
    """달아붙은 응답 헤더만 먼저 읽는다. (바디 프레임은 만지지 않는다)"""
    buf = b""
    while b"\r\n\r\n" not in buf:
        try:
            d = sock.recv(65536)
        except OSError:
            break
        if not d:
            break
        buf += d
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def _pump(up, cli):
    """두 커넥을 맞바꾼다. 한쪽이 먼저 끊기면 남은 쪽을 흘리고 함께 끊는다."""
    pair = {up: cli, cli: up}
    for s in (up, cli):
        try:
            s.setblocking(False)
        except OSError:
            pass
    alive = [up, cli]
    while alive:
        try:
            ready = select.select(alive, [], [], PUMP_TIMEOUT)[0]
        except (OSError, ValueError):
            break
        for s in ready:
            other = pair[s]
            try:
                data = s.recv(65536)
            except OSError:
                data = b""
            if not data:
                alive = [x for x in alive if x not in (s, other)]
                break
            try:
                other.sendall(data)
            except OSError:
                alive = [x for x in alive if x not in (s, other)]
                break
    for s in (up, cli):
        try:
            s.close()
        except OSError:
            pass



def match(path):
    """경로(쿼리 포함 원문)가 /apps/<이름>/ 영역이면 (접두어, 포드)."""
    for pfx, port in APP_PORTS.items():
        if path == pfx[:-1] or path.startswith(pfx):
            return pfx, port
    return None, None
