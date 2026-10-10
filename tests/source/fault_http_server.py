"""RN-146 测试支撑：可注入故障的本地 HTTP Range 服务（tester-only，仅 stdlib）。

由 ``tests/source/conftest.py`` 的 session fixture 自动拉起（端口动态分配、
启动后向 stdout 打印实际端口），无需人工干预。也可手工运行用于排障：

    python3 fault_http_server.py --root <夹具目录> --log /tmp/x.jsonl [--port 8080]

模式（query string）：
    mode=normal|short&n=1|chunk&max=65536&seed=..|slow&delay=0.2|stall|cut&after=1024|cut&fraction=0.5
    /status/404 /status/500 /redirect/<name>
每次请求写一行 JSONL：path/mode/status/offset/declared_length/bytes/sha1/range_header/cut
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOCK = threading.Lock()
LOG_PATH = None


def log(entry):
    with LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ROOT = None
    server_version = "rn146-fault-http/1.1"

    def log_message(self, fmt, *args):  # 静默默认 stderr 日志
        pass

    def _simple(self, code):
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _file_of(self, path):
        if not path.startswith("/f/"):
            return None
        fpath = os.path.join(self.ROOT, path[len("/f/"):])
        return fpath if os.path.isfile(fpath) else None

    def do_HEAD(self):
        path = urlparse(self.path).path
        fpath = self._file_of(path)
        if fpath is None:
            self._simple(404)
            return
        size = os.path.getsize(fpath)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        log({"path": path, "mode": "head", "status": 200, "bytes": 0})

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        mode = qs.get("mode", ["normal"])[0]

        if path.startswith("/status/"):
            try:
                code = int(path.rsplit("/", 1)[1])
            except ValueError:
                code = 500
            self._simple(code)
            log({"path": path, "mode": "status", "status": code, "bytes": 0})
            return
        if path.startswith("/redirect/"):
            target = "/f/" + path.rsplit("/", 1)[1]
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            log({"path": path, "mode": "redirect", "status": 302, "bytes": 0,
                 "location": target})
            return
        fpath = self._file_of(path)
        if fpath is None:
            self._simple(404)
            log({"path": path, "mode": mode, "status": 404, "bytes": 0})
            return
        if mode == "stall":
            log({"path": path, "mode": mode, "status": -1, "bytes": 0})
            time.sleep(600)
            return

        size = os.path.getsize(fpath)
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if not m:
                self._simple(416)
                return
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self._simple(416)
            return
        length = end - start + 1
        status = 206 if range_header else 200

        if mode == "ignore_range":
            start, length, status = 0, size, 200
        elif mode == "short":
            length = min(length, int(qs.get("n", ["1"])[0]))
        elif mode == "chunk":
            rng = random.Random(int(qs.get("seed", ["20260924"])[0]))
            length = min(length, rng.randint(1, int(qs.get("max", ["65536"])[0])))
        elif mode == "badrange" and start + 1 < size:
            start = start + 1

        cut_after = None
        if mode == "cut":
            if "fraction" in qs:
                cut_after = max(1, int(length * float(qs["fraction"][0])))
            else:
                cut_after = int(qs.get("after", ["1024"])[0])

        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header(
                "Content-Range",
                "bytes %d-%d/%d" % (start, start + length - 1, size),
            )
        self.send_header("Content-Length", str(length))
        self.end_headers()

        sent = 0
        digest = hashlib.sha1()
        try:
            with open(fpath, "rb") as fh:
                fh.seek(start)
                while sent < length:
                    piece = fh.read(min(65536, length - sent))
                    if not piece:
                        break
                    if mode == "slow":
                        time.sleep(float(qs.get("delay", ["0.2"])[0]))
                    if cut_after is not None and sent + len(piece) > cut_after:
                        piece = piece[: max(0, cut_after - sent)]
                    if not piece:
                        break
                    self.wfile.write(piece)
                    digest.update(piece)
                    sent += len(piece)
                    if cut_after is not None and sent >= cut_after:
                        self.close_connection = True
                        break
        except (BrokenPipeError, ConnectionResetError):
            pass
        log({
            "path": path, "mode": mode, "status": status, "offset": start,
            "declared_length": length, "bytes": sent,
            "sha1": digest.hexdigest(), "range_header": range_header,
            "cut": sent < length,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, default=0, help="0 = 动态分配（默认）")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    global LOG_PATH
    LOG_PATH = args.log
    with open(LOG_PATH, "w", encoding="utf-8"):
        pass
    Handler.ROOT = os.path.abspath(args.root)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    print(srv.server_address[1], flush=True)  # 拉起方据此获知实际端口
    srv.serve_forever()


if __name__ == "__main__":
    main()
