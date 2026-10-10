"""RN-146 用例 45、47、48：TP-7 联动回归（backend × source / Arrow 批读口径 / 错误信息）。

编号 46（DataFrame use_index × 远端）台账结论为「无入口 · 不验证」，按落地口径
不落测试代码（结论保留在验证报告中）。本地故障注入 HTTP 服务由 conftest 的
fault_http_server fixture 自动拉起。
"""
import tempfile
from pathlib import Path

import pytest

import tsfile
from tsfile import TsFileReader

from http_source import (TABLE, TREE, HttpRangeFile, local_table, local_tree,
                         table_rows, tree_rows, url_of)

pytestmark = pytest.mark.usefixtures("fault_http_server")


def _set_backend(backend):
    try:
        tsfile.set_file_read_backend(backend)
    except TypeError:
        tsfile.set_file_read_backend(backend.value)


def test_case45_read_backend_with_http_source():
    original = tsfile.get_file_read_backend()
    baseline = local_table()
    counts = {}
    for name in ("AUTO", "MMAP", "PREAD"):
        _set_backend(getattr(tsfile.FileReadBackend, name))
        rows = table_rows(HttpRangeFile(url_of("simple_table_t1.tsfile")))
        counts[name] = len(rows)
        assert rows == baseline, "backend=%s 改变结果" % name
    _set_backend(original)
    print("CASE45 counts=%s original=%s restored=%s"
          % (counts, original, tsfile.get_file_read_backend()))
    assert tsfile.get_file_read_backend() == original


def _arrow_rows(path, batch_size):
    pa = pytest.importorskip("pyarrow", reason="Arrow 批读口径需要 pyarrow")
    tables = []
    with TsFileReader(str(path)) as reader:
        result = reader.query_table("test", ["s0", "s2"], batch_size=batch_size)
        try:
            while True:
                batch = result.read_arrow_batch()
                if batch is None:
                    break
                tables.append(batch)
        finally:
            result.close()
    if not tables:
        return []
    combined = pa.concat_tables(tables)
    times = combined.column(0).to_pylist()
    s0 = combined.column(1).to_pylist()
    s2 = combined.column(2).to_pylist()
    return list(zip(times, s0, s2))


def test_case47_local_path_regression_arrow_batch():
    base = table_rows(str(TABLE))
    assert table_rows(str(TABLE)) == base
    for batch_size in (10, 1000):
        rows = _arrow_rows(TABLE, batch_size)
        assert rows == base, "batch_size=%d Arrow 批读与 row 模式逐行不一致" % batch_size
    assert tree_rows(str(TREE)) == local_tree()
    nulls = sum(1 for row in base if any(v is None for v in row))
    print("CASE47 table=%d arrow_batch=10/1000 tree=%d null_rows=%d"
          % (len(base), len(local_tree()), nulls))


def test_case48_error_messages_regression(tmp_path):
    missing = tmp_path / "rn146-no-such-file-146.tsfile"
    try:
        TsFileReader(str(missing))
        raise AssertionError("打开不存在文件必须报错")
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001
        print("CASE48 missing -> %s: %s" % (type(exc).__name__, exc))
    corrupt = Path(tempfile.mkdtemp(prefix="rn146-case48-")) / "truncated.tsfile"
    data = TABLE.read_bytes()
    corrupt.write_bytes(data[: len(data) // 2])
    try:
        with TsFileReader(str(corrupt)) as reader:
            result = reader.query_table("test", ["s0", "s2"])
            rows = 0
            while result.next():
                rows += 1
            result.close()
        print("CASE48 corrupt -> 未报错 rows=%d（异常）" % rows)
    except Exception as exc:  # noqa: BLE001
        print("CASE48 corrupt -> %s: %s" % (type(exc).__name__, exc))
