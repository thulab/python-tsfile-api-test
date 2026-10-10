# -*- coding: utf-8 -*-
"""RN-142 编号 46：读后端（AUTO/MMAP/PREAD）× 索引组合一致性。

判据：三种后端下冷建/热开查询结果与无索引基线逐值一致；热开不重建索引
（inode/mtime/size 与 sha256 不变）；结束后恢复原后端。
"""
import gc
import hashlib
import shutil
from pathlib import Path

import numpy as np

from tsfile import (FileReadBackend, TsFileDataFrame, get_file_read_backend,
                    set_file_read_backend)
from tsfile.dataset.index import index_path_for


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(directory, indexed):
    frame = TsFileDataFrame(str(directory), show_progress=False, use_index=indexed)
    try:
        names = sorted(str(x) for x in frame.list_timeseries())
        assert names
        result = {}
        for key, start, end in [('full', None, None), ('window', 80, 220)]:
            values = frame.loc[start:end, names]
            result[key] = (list(map(str, values.series_names)),
                           np.asarray(values.timestamps).copy(),
                           np.asarray(values.values).copy())
            del values
        return result
    finally:
        frame.close()
        gc.collect()


def equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key][0] == expected[key][0]
        np.testing.assert_array_equal(actual[key][1], expected[key][1])
        np.testing.assert_equal(actual[key][2], expected[key][2])


def test_case46_backend_matrix_matches_baseline(d1_root, tmp_path):
    work = tmp_path / "case46"
    work.mkdir()
    data = work / 'data'
    data.mkdir()
    source = Path(d1_root) / "table"
    paths = []
    for p in sorted(source.glob('*.tsfile')):
        copied = data / p.name
        shutil.copy2(p, copied)
        assert digest(p) == digest(copied)
        paths.append(str(copied))
    assert len(paths) == 3
    index = Path(index_path_for(paths)).resolve()
    assert index.parent == data
    old_backend = get_file_read_backend()
    measurements = []
    try:
        baseline = snapshot(data, False)
        assert not index.exists()
        for backend in (FileReadBackend.AUTO, FileReadBackend.MMAP, FileReadBackend.PREAD):
            # Only this batch's generated index is removed, after preserving it.
            if index.exists():
                index.unlink()
            set_file_read_backend(backend)
            assert get_file_read_backend() == backend
            cold = snapshot(data, True)
            equal(cold, baseline)
            before = index.stat()
            checksum = digest(index)
            hot = snapshot(data, True)
            equal(hot, baseline)
            after = index.stat()
            assert (before.st_ino, before.st_mtime_ns, before.st_size) == \
                (after.st_ino, after.st_mtime_ns, after.st_size)
            assert digest(index) == checksum
            measurements.append({'backend': backend.name, 'cold_equal': True, 'hot_equal': True,
                                 'index_sha256': checksum, 'rows': len(cold['full'][1]),
                                 'window_rows': len(cold['window'][1])})
            if backend == FileReadBackend.MMAP:
                set_file_read_backend(FileReadBackend.AUTO)
                equal(snapshot(data, True), baseline)
                current = index.stat()
                assert (current.st_ino, current.st_mtime_ns, current.st_size) == \
                    (after.st_ino, after.st_mtime_ns, after.st_size)
                assert digest(index) == checksum
                measurements[-1]['mmap_to_auto_hot_equal'] = True
    finally:
        set_file_read_backend(old_backend)
        gc.collect()
    assert get_file_read_backend() == old_backend
    print("[case 46] pass backends=%s index_sha_equal=%s backend_restored=%s"
          % ([m['backend'] for m in measurements],
             len({m['index_sha256'] for m in measurements}) == 1,
             old_backend.name))
