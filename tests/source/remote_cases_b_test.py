"""RN-146 用例 19（重做）：半包断流（cut&fraction=0.5）不得返回 partial/空结果。

原批次 3b 重做版；编号 23 的 Tag 过滤补测依赖 ≈1 GB 夹具，按落地口径不纳入。
本地故障注入 HTTP 服务由 conftest 的 fault_http_server fixture 自动拉起。
"""
import pytest

from http_source import HttpRangeFile, local_table, log_rows, log_since, table_rows, url_of

pytestmark = pytest.mark.usefixtures("fault_http_server")


def test_case19_midstream_cut_half():
    mark = len(log_rows())
    src = HttpRangeFile(url_of("simple_table_t1.tsfile", "?mode=cut&fraction=0.5"))
    try:
        rows = table_rows(src)
    except Exception as exc:  # noqa: BLE001
        rows = None
        outcome = "raised %s: %s" % (type(exc).__name__, exc)
    else:
        outcome = "no-exception rows=%d/%d" % (len(rows), len(local_table()))
    entries = [e for e in log_since(mark, "simple_table_t1.tsfile") if e.get("mode") == "cut"]
    cut_true = [e for e in entries if e.get("cut")]
    print("CASE19b %s | cut_requests=%d cut_true=%d" % (outcome, len(entries), len(cut_true)))
    assert cut_true, "半包断流注入必须真实触发（服务端 cut=True）"
    if rows is not None:
        assert rows == local_table(), "半包断流不得返回 partial/空结果"
