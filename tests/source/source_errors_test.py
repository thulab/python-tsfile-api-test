"""RN-146 用例 26–41：错误语义、partial 行为与生命周期。

编号 26/27/28/30（跨接口缺陷「读失败被静默吞」）已随 TsFile-165 修复；
2026-10-10 复测 4 处 XPASS 确认后摘除 xfail(strict) 标记。
问题报告：``records/rn146-defect-read-failure-swallowed-20260924.md``。
"""
import gc
import weakref
from pathlib import Path

import pytest

from tsfile import TsFileReader
from tsfile.exceptions import FileOpenError, FileReadError

from helpers import FailingBytesIO, OverReturnSource, TrackingBytesIO

DATA = Path(__file__).parent / "data"
TABLE_T1 = DATA / "simple_table_t1.tsfile"


def table_rows(source):
    rows = []
    with TsFileReader(source) as reader:
        result = reader.query_table("test", ["s0", "s2"])
        try:
            while result.next():
                rows.append((result.get_value_by_index(1), result.get_value_by_index(2),
                             result.get_value_by_index(3)))
        finally:
            result.close()
    return rows


def test_c26_query_time_read_failure_maps_to_file_read_error():
    src = FailingBytesIO(TABLE_T1.read_bytes(), fail_after_bytes=1024)
    src.seek(11)
    with pytest.raises(FileReadError):
        table_rows(src)
    assert src.tell() == 11
    assert src.close_calls == 0


def test_c27_28_no_silent_empty_result_on_injected_failures():
    observed = []
    for n in range(1, 8):
        src = FailingBytesIO(TABLE_T1.read_bytes(), fail_at=n)
        try:
            rows = table_rows(src)
        except Exception as exc:  # noqa: BLE001
            observed.append((n, type(exc).__name__))
        else:
            observed.append((n, "ok:%d" % len(rows)))
    print("c27/28 fail_at matrix:", observed)
    assert not [o for o in observed if o[1] == "ok:0"], observed


def test_c30_truncation_never_returns_partial_silently():
    data = TABLE_T1.read_bytes()
    expected = table_rows(str(TABLE_T1))
    silent_partial = []
    for n in range(64, len(data), 512):
        src = FailingBytesIO(data, fail_after_bytes=n)
        try:
            rows = table_rows(src)
        except Exception:  # noqa: BLE001
            continue
        if rows != expected:
            silent_partial.append((n, len(rows), len(expected)))
    assert not silent_partial, silent_partial


def test_c31_truncated_source_rejected():
    data = TABLE_T1.read_bytes()
    try:
        rows = table_rows(TrackingBytesIO(data[: len(data) - 800]))
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ in {"FileReadError", "FileOpenError",
                                      "TsFileCorruptedError", "LibraryError"}, type(exc).__name__
    else:
        pytest.fail("截断 source 被静默接受，返回 %d 行" % len(rows))


def test_c32_non_io_exception_propagates_or_wraps():
    src = FailingBytesIO(TABLE_T1.read_bytes(), fail_at=3, exc=ValueError)
    try:
        table_rows(src)
    except ValueError as exc:
        assert "injected failure" in str(exc)
    except FileReadError as exc:
        # 设计口径：包装为统一错误即可；记录包装后的文本
        assert "read" in str(exc).lower() or "injected" in str(exc)


def test_c33_source_closed_before_query():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    reader = TsFileReader(src)
    src.close()
    with pytest.raises(Exception):  # noqa: B017
        result = reader.query_table("test", ["s0"])
        result.close()
    reader.close()


def test_c34_over_returning_source_recorded():
    outcome = "ok"
    try:
        rows = table_rows(OverReturnSource(TABLE_T1.read_bytes()))
        outcome = "returned %d rows" % len(rows)
    except Exception as exc:  # noqa: BLE001
        outcome = "raised %s: %s" % (type(exc).__name__, exc)
    print("c34 over-return:", outcome)


def test_c35_read_failure_after_open_rejected():
    # 与编号 26/30 同族的本地 source 场景（远端 5xx 中途断流见 remote_sources 用例 21）
    src = FailingBytesIO(TABLE_T1.read_bytes(), fail_after_bytes=512, exc=OSError)
    with pytest.raises((FileReadError, FileOpenError)):
        table_rows(src)


def test_c36_reader_does_not_close_source():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    table_rows(src)
    assert src.close_calls == 0
    assert table_rows(src) == table_rows(str(TABLE_T1))


def test_c37_source_released_after_close():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    ref = weakref.ref(src)
    reader = TsFileReader(src)
    assert ref() is not None
    reader.close()
    del reader, src
    gc.collect()
    assert ref() is None


def test_c38_close_is_idempotent():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    reader = TsFileReader(src)
    reader.close()
    reader.close()
    assert src.close_calls == 0


def test_c39_source_closed_before_reader_close_is_safe():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    reader = TsFileReader(src)
    src.close()
    reader.close()


def test_c40_reader_closed_then_source_closed():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    reader = TsFileReader(src)
    reader.close()
    src.close()
    assert src.close_calls == 1


def test_c41_reader_without_close_does_not_break_process():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    TsFileReader(src)
    del src
    gc.collect()
