"""RN-146 tests/source 共享夹具：本地故障注入 HTTP 服务（session 级，自启自停）。

用例自包含：夹具在 tmp 目录现造，服务端口动态分配，日志写 tmp；
不读任何进程外环境变量、不依赖外部夹具目录或人工步骤。
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import http_source

HERE = Path(__file__).parent
DATA = HERE / "data"


@pytest.fixture(scope="session")
def fault_http_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("rn146-http-root")
    for name in ("simple_table_t1.tsfile", "simple_tree.tsfile"):
        shutil.copyfile(DATA / name, root / name)
    # 编号 25（多分片）：同一数据文件的两个分片副本
    shutil.copyfile(DATA / "simple_table_t1.tsfile", root / "shard_a.tsfile")
    shutil.copyfile(DATA / "simple_table_t1.tsfile", root / "shard_b.tsfile")

    log_path = tmp_path_factory.mktemp("rn146-http-log") / "server-log.jsonl"
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "fault_http_server.py"),
         "--root", str(root), "--port", "0", "--log", str(log_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line = (proc.stdout.readline() or "").strip()
        assert line.isdigit(), "fault_http_server 未报告端口: %r" % line
        http_source.bind_env("http://127.0.0.1:%s" % line, log_path, root)
        yield http_source.env()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
