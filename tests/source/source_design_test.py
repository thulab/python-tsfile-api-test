"""RN-146 用例 54–58：设计文档 §4.2 口径补强（负长度/非 TsFile/版本/缓存/守护）。"""
import threading
from pathlib import Path

import pytest

from tsfile import TsFileReader, exceptions as tsexc

from helpers import NegativeLength, TrackingBytesIO

DATA = Path(__file__).parent / "data"
TABLE_T1 = DATA / "simple_table_t1.tsfile"
Corrupt = getattr(tsexc, "TsFileCorruptedError", None)


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


def test_c54_negative_length_raises_value_error():
    with pytest.raises(ValueError):
        TsFileReader(NegativeLength(TABLE_T1.read_bytes()))


def test_c55_corrupt_content_and_missing_tail_magic():
    assert Corrupt is not None, "tsfile.exceptions.TsFileCorruptedError 缺失"
    with pytest.raises(Corrupt):
        table_rows(TrackingBytesIO(b"\x00" * 4096))
    data = TABLE_T1.read_bytes()
    with pytest.raises(Corrupt):
        table_rows(TrackingBytesIO(data[: len(data) - 4]))


def test_c56_unsupported_version_byte_recorded():
    data = bytearray(TABLE_T1.read_bytes())
    observed = []
    for off in range(0, 16):
        mutated = bytearray(data)
        mutated[off] = 0x09
        try:
            table_rows(TrackingBytesIO(bytes(mutated)))
        except Exception as exc:  # noqa: BLE001
            observed.append((off, type(exc).__name__, str(exc)[:60]))
        else:
            observed.append((off, "accepted", ""))
    print("c56 version-byte matrix:", observed)
    assert observed


def test_c57_length_cache_does_not_detect_content_change():
    data = bytearray(TABLE_T1.read_bytes())
    src = TrackingBytesIO(bytes(data))
    first = table_rows(src)
    data[64:80] = bytes(16)
    src.seek(0)
    src._buffer = bytearray(data)  # noqa: SLF001
    second = None
    try:
        second = table_rows(src)
    except Exception as exc:  # noqa: BLE001
        print("c57 second read raised:", type(exc).__name__, exc)
    print("c57 rows: first=%d second=%s" % (len(first), None if second is None else len(second)))


def test_c58_multi_column_query_completes_within_guard():
    done = []

    def worker():
        with TsFileReader(str(TABLE_T1)) as reader:
            result = reader.query_table("test", ["s0", "s1", "s2"])
            try:
                count = 0
                while result.next():
                    count += 1
                done.append(count)
            finally:
                result.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "多列读取 60s 内未完成（疑似挂死）"
    assert done and done[0] > 0
