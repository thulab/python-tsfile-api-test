"""RN-146 测试支撑：HTTP Range 只读 source 与远端夹具访问（tester-only）。

服务端点由 ``conftest.py`` 的 ``fault_http_server`` fixture 启动后经
``bind_env()`` 注入；测试内用 ``url_of()`` / ``log_rows()`` / ``log_since()``
访问，不读任何进程外环境变量，保证用例自包含。
"""
import hashlib
import http.client
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

from tsfile import TsFileReader

DATA = Path(__file__).parent / "data"
TABLE = DATA / "simple_table_t1.tsfile"
TREE = DATA / "simple_tree.tsfile"
TREE_DEVICE = "root.ln.wf01.wt01"
TREE_COLS = ["temperature", "status"]


class _Env:
    def __init__(self, base, log_path, root):
        self.base = base
        self.log_path = Path(log_path)
        self.root = Path(root)


_ENV = None


def bind_env(base, log_path, root):
    global _ENV
    _ENV = _Env(base, log_path, root)


def env():
    assert _ENV is not None, "fault_http_server fixture 未初始化（conftest 未加载）"
    return _ENV


class HttpRangeFile:
    """Range 语义 HTTP file-like：seek/tell/read(size)，tester-only。"""

    def __init__(self, url, timeout=30.0):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        self.closed = False
        self.close_calls = 0
        self.read_sizes = []
        self.size = self._probe_size()

    def _probe_size(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 206:
                raise OSError("server did not honour Range probe: %d" % resp.status)
            content_range = resp.headers.get("Content-Range", "")
            return int(content_range.rsplit("/", 1)[-1])

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError("invalid whence: %r" % (whence,))
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def read(self, size=-1):
        assert size is not None and size >= 0, "TsFileReader must use explicit sizes"
        self.read_sizes.append(size)
        if size == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": "bytes=%d-%d" % (self.pos, end)}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                data = resp.read()
        except http.client.IncompleteRead as exc:
            raise OSError("incomplete read: %s" % exc) from exc
        except urllib.error.HTTPError as exc:
            raise OSError("HTTP %d %s" % (exc.code, exc.reason)) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError("request timeout: %s" % exc) from exc
        if status == 200:
            raise OSError("server ignored Range (200 full body)")
        self.pos += len(data)
        return data

    def close(self):
        self.close_calls += 1
        self.closed = True


def url_of(name, extra=""):
    return "%s/f/%s%s" % (env().base, name, extra)


def log_rows():
    path = env().log_path
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def log_since(mark, needle=None):
    rows = log_rows()[mark:]
    if needle:
        rows = [r for r in rows if needle in r.get("path", "")]
    return rows


def table_rows(source):
    rows = []
    with TsFileReader(source) as reader:
        result = reader.query_table("test", ["s0", "s2"])
        try:
            while result.next():
                rows.append(tuple(result.get_value_by_index(i) for i in range(1, 4)))
        finally:
            result.close()
    return rows


def tree_rows(source):
    rows = []
    with TsFileReader(source) as reader:
        result = reader.query_timeseries(TREE_DEVICE, TREE_COLS, 0, (1 << 63) - 1)
        try:
            while result.next():
                rows.append(tuple(result.get_value_by_index(i) for i in range(1, 4)))
        finally:
            result.close()
    return rows


_CACHE = {}


def local_table():
    if "t" not in _CACHE:
        _CACHE["t"] = table_rows(str(TABLE))
    return _CACHE["t"]


def local_tree():
    if "r" not in _CACHE:
        _CACHE["r"] = tree_rows(str(TREE))
    return _CACHE["r"]


def sha1_of(data):
    return hashlib.sha1(data).hexdigest()
