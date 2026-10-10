"""RN-146 用例 29：tree 模型 data block 读失败必须传播（不得静默 partial/空）。

判定口径：注入读失败（第 N 次 read 抛错 / 累计读满 N 字节后断流）后，
要么显式抛错，要么返回与基线一致的全量结果；返回短行数即视为「静默吞」。
"""
from pathlib import Path

from tsfile import TsFileReader

from helpers import FailingBytesIO

DATA = Path(__file__).parent / "data"
TREE = DATA / "simple_tree.tsfile"
TREE_DEVICE = "root.ln.wf01.wt01"
TREE_COLS = ["temperature", "status"]


def _tree_rows(source):
    rows = []
    with TsFileReader(source) as reader:
        result = reader.query_timeseries(TREE_DEVICE, TREE_COLS, 0, (1 << 63) - 1)
        try:
            while result.next():
                rows.append(tuple(result.get_value_by_index(i) for i in range(1, 4)))
        finally:
            result.close()
    return rows


def _run(data, fail_at=None, fail_after_bytes=None):
    src = FailingBytesIO(data, fail_at=fail_at, fail_after_bytes=fail_after_bytes)
    try:
        return _tree_rows(src), None
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def test_c29_tree_data_block_failures_do_not_silently_truncate():
    data = TREE.read_bytes()
    base, err = _run(data)
    assert err is None and base, (err, base)
    swallow = []
    for n in range(1, 20):
        rows, err = _run(data, fail_at=n)
        if err is None and len(rows) < len(base):
            swallow.append(("fail_at", n, len(rows)))
    for cut in range(256, len(data), 256):
        rows, err = _run(data, fail_after_bytes=cut)
        if err is None and len(rows) < len(base):
            swallow.append(("cut", cut, len(rows)))
    print("c29 baseline=%d injected_points=%d swallowed=%d"
          % (len(base), 19 + len(range(256, len(data), 256)), len(swallow)))
    assert not swallow, "tree 读失败被静默吞: %s" % swallow
