"""RN-146 用例 1–11：source 输入形态与合约（str/Path/bytes/file-like/非法对象/短读/游标）。"""
import io
import os
from pathlib import Path

import pytest

from tsfile import TsFileReader
from tsfile.exceptions import FileOpenError, FileReadError

from helpers import (BytesPathLike, FailingBytesIO, NoSizeRead, ReadOnlySource,
                     TextSource, TrackingBytesIO)

DATA = Path(__file__).parent / "data"
TABLE_T1 = DATA / "simple_table_t1.tsfile"
TREE = DATA / "simple_tree.tsfile"


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


def tree_rows(source):
    rows = []
    with TsFileReader(source) as reader:
        result = reader.query_timeseries("root.ln.wf01.wt01", ["temperature", "status"],
                                         0, (1 << 63) - 1)
        try:
            while result.next():
                rows.append((result.get_value_by_index(1), result.get_value_by_index(2),
                             result.get_value_by_index(3)))
        finally:
            result.close()
    return rows


def test_c01_str_path_baseline():
    rows = table_rows(str(TABLE_T1))
    assert rows and rows[0][1] == "a"


def test_c02_pathlib_matches_str():
    assert table_rows(Path(TABLE_T1)) == table_rows(str(TABLE_T1))


def test_c03_bytes_and_pathlike_match_str():
    expected = table_rows(str(TABLE_T1))
    assert table_rows(os.fsencode(str(TABLE_T1))) == expected
    assert table_rows(BytesPathLike(os.fsencode(str(TABLE_T1)))) == expected


def test_c04_file_object_table_query():
    assert table_rows(TrackingBytesIO(TABLE_T1.read_bytes())) == table_rows(str(TABLE_T1))


def test_c05_file_object_tree_query():
    assert tree_rows(TrackingBytesIO(TREE.read_bytes())) == tree_rows(str(TREE))


def test_c06_read_only_source_rejected_with_type_error():
    with pytest.raises(TypeError) as err:
        TsFileReader(ReadOnlySource(TABLE_T1.read_bytes()))
    assert "seekable binary file object" in str(err.value)


def test_c07_read_without_size_is_rejected():
    with pytest.raises((FileReadError, TypeError)):
        table_rows(NoSizeRead(TABLE_T1.read_bytes()))


def test_c08_text_returning_source_rejected():
    with pytest.raises(FileReadError):
        table_rows(TextSource(TABLE_T1.read_bytes()))


@pytest.mark.parametrize("bad", [None, 123, {"a": 1}])
def test_c09_none_and_invalid_sources(bad):
    with pytest.raises(TypeError) as err:
        TsFileReader(bad)
    assert "seekable binary file object" in str(err.value)


def test_c10_seek_boundary_open_error_then_read_error():
    class SeekEndUnsupported(io.BytesIO):
        def seek(self, offset, whence=0):
            if whence == 2:
                raise OSError("SEEK_END unsupported")
            return super().seek(offset, whence)

    with pytest.raises(FileOpenError):
        TsFileReader(SeekEndUnsupported(TABLE_T1.read_bytes()))

    src = FailingBytesIO(TABLE_T1.read_bytes(), fail_at=4)
    with pytest.raises(FileReadError):
        table_rows(src)


def test_c11_shared_source_between_two_readers_recorded():
    src = TrackingBytesIO(TABLE_T1.read_bytes())
    first = table_rows(src)
    second = table_rows(src)
    print("c11 shared-source: first_rows=%d second_rows=%d cursor=%d read_calls=%d"
          % (len(first), len(second), src.tell(), len(src.read_sizes)))
