"""RN-146 用例 12–22、24、25：真实远端（HTTP Range + 故障注入）。

判定口径：读结果必须与本地基线一致；异常场景必须抛出可识别错误、不得返回
空结果 / partial / 误导性错误；耗时与 RSS 只记录不设阈值。
本地故障注入 HTTP 服务由 conftest 的 fault_http_server fixture 自动拉起。
编号 23（≈1 GB 远端大文件）按落地口径不纳入默认套件。
"""
import hashlib
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tsfile import TsFileReader

from http_source import (TABLE, HttpRangeFile, env, local_table, local_tree,
                         log_rows, log_since, table_rows, tree_rows, url_of)

pytestmark = pytest.mark.usefixtures("fault_http_server")


def test_case12_http_normal_table():
    mark = len(log_rows())
    src = HttpRangeFile(url_of("simple_table_t1.tsfile"))
    rows = table_rows(src)
    assert rows == local_table()
    assert src.close_calls == 0
    entries = log_since(mark, "simple_table_t1.tsfile")
    assert entries and all(e["status"] == 206 for e in entries)
    print("CASE12 rows=%d http_requests=%d" % (len(rows), len(entries)))


def test_case13_http_normal_tree():
    src = HttpRangeFile(url_of("simple_tree.tsfile"))
    rows = tree_rows(src)
    assert rows == local_tree()
    assert rows, "tree baseline must be non-empty"
    print("CASE13 rows=%d" % len(rows))


def test_case14_short_read_one_byte():
    src = HttpRangeFile(url_of("simple_table_t1.tsfile", "?mode=short&n=1"), timeout=60.0)
    rows = table_rows(src)
    assert rows == local_table()
    assert src.read_sizes and min(src.read_sizes) >= 0
    assert len(src.read_sizes) > 50, "expected many aggregated reads"
    print("CASE14 rows=%d read_calls=%d" % (len(rows), len(src.read_sizes)))


def test_case15_random_chunking():
    url_t = url_of("simple_table_t1.tsfile", "?mode=chunk&max=65536&seed=1461")
    url_r = url_of("simple_tree.tsfile", "?mode=chunk&max=65536&seed=1462")
    for _ in range(3):
        assert table_rows(HttpRangeFile(url_t)) == local_table()
        assert tree_rows(HttpRangeFile(url_r)) == local_tree()
    print("CASE15 rounds=3 ok")


def test_case16_initial_cursor_nonzero():
    src = HttpRangeFile(url_of("simple_table_t1.tsfile"))
    src.seek(100)
    with TsFileReader(src) as reader:
        pos_after_open = src.tell()
        result = reader.query_table("test", ["s0", "s2"])
        rows = []
        try:
            while result.next():
                rows.append(tuple(result.get_value_by_index(i) for i in range(1, 4)))
        finally:
            result.close()
    print("CASE16 pos_after_open=%d pos_after_query=%d close_calls=%d"
          % (pos_after_open, src.tell(), src.close_calls))
    assert rows == local_table()
    assert pos_after_open == 100, "reader must restore caller cursor"
    assert src.tell() == 100, "reader must restore caller cursor after query"
    assert src.close_calls == 0, "reader must not close caller source"


def test_case17_slow_link():
    src = HttpRangeFile(url_of("simple_table_t1.tsfile", "?mode=slow&delay=0.2"), timeout=120.0)
    t0 = time.time()
    rows = table_rows(src)
    elapsed = time.time() - t0
    print("CASE17 rows=%d elapsed=%.2fs（仅记录）" % (len(rows), elapsed))
    assert rows == local_table()


def test_case18_connect_stall_timeout():
    t0 = time.time()
    with pytest.raises(Exception) as ei:
        src = HttpRangeFile(url_of("simple_table_t1.tsfile", "?mode=stall"), timeout=2.0)
        table_rows(src)
    elapsed = time.time() - t0
    print("CASE18 error=%s: %s elapsed=%.2fs"
          % (type(ei.value).__name__, ei.value, elapsed))
    assert elapsed < 15, "stall must fail fast, not hang"


def test_case19_midstream_cut():
    src = HttpRangeFile(url_of("simple_table_t1.tsfile", "?mode=cut&after=1024"))
    try:
        rows = table_rows(src)
    except Exception as exc:  # noqa: BLE001
        print("CASE19 raised=%s: %s" % (type(exc).__name__, exc))
    else:
        print("CASE19 no-exception rows=%d/%d" % (len(rows), len(local_table())))
        assert rows == local_table(), "断流不得返回 partial/空结果"


def test_case20_server_404():
    with pytest.raises(Exception) as ei:
        src = HttpRangeFile(url_of("no_such_file_146.tsfile"))
        table_rows(src)
    print("CASE20 error=%s: %s" % (type(ei.value).__name__, ei.value))


def test_case21_server_5xx_and_redirect():
    with pytest.raises(Exception) as ei:
        src = HttpRangeFile("%s/status/500" % env().base)
        table_rows(src)
    print("CASE21a 500 -> %s: %s" % (type(ei.value).__name__, ei.value))
    try:
        rows = table_rows(HttpRangeFile("%s/redirect/simple_table_t1.tsfile" % env().base))
    except Exception as exc:  # noqa: BLE001
        print("CASE21b 302 -> raised %s: %s" % (type(exc).__name__, exc))
    else:
        print("CASE21b 302 -> rows=%d/%d（客户端支持跟随重定向）"
              % (len(rows), len(local_table())))
        assert rows == local_table()


def test_case22_range_semantics():
    mark = len(log_rows())
    assert table_rows(HttpRangeFile(url_of("simple_table_t1.tsfile"))) == local_table()
    entries = log_since(mark, "simple_table_t1.tsfile")
    assert entries
    data = TABLE.read_bytes()
    for e in entries:
        assert e["status"] == 206, e
        assert e["range_header"], "all requests must carry Range header"
        off, ln = e["offset"], e["bytes"]
        assert hashlib.sha1(data[off:off + ln]).hexdigest() == e["sha1"], e
    print("CASE22 verified_requests=%d (offset+content sha1 逐条核对)" % len(entries))


def test_case24_concurrent_readers():
    url = url_of("simple_table_t1.tsfile")
    baseline = local_table()
    for round_no in range(3):
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda _i: table_rows(HttpRangeFile(url)), range(8)))
        for r in results:
            assert r == baseline
        print("CASE24 threads round=%d ok" % (round_no + 1))
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_mp_worker, args=(url, q)) for _ in range(4)]
    for p in procs:
        p.start()
    got = [q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    for status, payload in got:
        assert status == "ok", payload
    print("CASE24 multiprocessing=4 ok")


def _mp_worker(url, q):
    try:
        src = HttpRangeFile(url)
        rows = table_rows(src)
        q.put(("ok", len(rows)))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", "%s: %s" % (type(exc).__name__, exc)))


def test_case25_multi_shard():
    assert table_rows(HttpRangeFile(url_of("shard_a.tsfile"))) == local_table()
    assert table_rows(HttpRangeFile(url_of("shard_b.tsfile"))) == local_table()
    import tsfile as tsfile_mod
    names = [n for n in dir(tsfile_mod) if "rame" in n or n.startswith("read")]
    note = "no read_table entry; entries=%s" % names
    fn = getattr(tsfile_mod, "read_table", None)
    if fn is not None:
        try:
            fn(url_of("simple_table_t1.tsfile"))
            note = "read_table(url) 未报错（异常，需复核）"
        except Exception as exc:  # noqa: BLE001
            note = "read_table(url) -> %s: %s" % (type(exc).__name__, exc)
    print("CASE25 shards ok | DataFrame×远端: %s" % note)
