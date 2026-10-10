# -*- coding: utf-8 -*-
"""RN-142 用例 17–19、21–24、26–28、30：查询一致性（索引 vs 无索引）。

原则：只比对**公开 API** 的返回值/异常语义。
编号 20/25/31（无公开入口/夹具不可用）台账已移除，本文件不落地。
"""
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import fixtures as fx
from tsfile import TsFileDataFrame


def pair(d1_root, sub="table"):
    """返回 (无索引, 索引) 两个 DataFrame。"""
    directory = str(Path(d1_root) / sub)
    plain = TsFileDataFrame(directory, show_progress=False, use_index=False)
    indexed = TsFileDataFrame(directory, show_progress=False, use_index=True)
    return plain, indexed


def pair_on(directory):
    """针对（tmp 内）指定目录返回 (无索引, 索引) 两个 DataFrame。"""
    plain = TsFileDataFrame(str(directory), show_progress=False, use_index=False)
    indexed = TsFileDataFrame(str(directory), show_progress=False, use_index=True)
    return plain, indexed


def table_copy(d1_root, tmp_path, sub="table", name="data"):
    """把 D1 的某个子集复制到 tmp_path 再操作（避免污染 session 级夹具）。"""
    dst = tmp_path / name
    shutil.copytree(str(Path(d1_root) / sub), str(dst))
    return dst


def values(frame, name):
    series = frame[name]
    return np.asarray(series[:])


def timestamps(frame, name):
    return np.asarray(frame[name].timestamps)


def compare_all_series(plain, indexed, label):
    names_a = [str(item) for item in plain.list_timeseries()]
    names_b = [str(item) for item in indexed.list_timeseries()]
    assert names_a == names_b, (label, names_a, names_b)
    for name in names_a:
        va, vb = values(plain, name), values(indexed, name)
        assert va.shape == vb.shape, (label, name, va.shape, vb.shape)
        np.testing.assert_array_equal(va, vb, err_msg="%s %s" % (label, name))
        np.testing.assert_array_equal(timestamps(plain, name), timestamps(indexed, name))
        assert va.dtype == vb.dtype, (label, name, va.dtype, vb.dtype)
    return names_a


def test_case17_single_file_values_match(d1_root, tmp_path):
    one = tmp_path / "one"
    one.mkdir()
    shutil.copy(str(Path(d1_root) / "table" / "weather_a.tsfile"), str(one))
    plain = TsFileDataFrame(str(one), show_progress=False, use_index=False)
    indexed = TsFileDataFrame(str(one), show_progress=False, use_index=True)
    try:
        names = compare_all_series(plain, indexed, "single")
    finally:
        plain.close()
        indexed.close()
    print("[case 17] pass series=%d" % len(names))


def test_case18_multi_shard_merge_matches(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path))
    try:
        names = compare_all_series(plain, indexed, "multi")
        merged = [n for n in names if n.startswith("weather.dev_a") and n.endswith(".temperature")][0]
        assert len(plain[merged]) == len(indexed[merged]) == 300
        window_a = np.asarray(plain[merged][80:120])
        window_b = np.asarray(indexed[merged][80:120])
        np.testing.assert_array_equal(window_a, window_b)
        assert window_a.shape[0] == 40, window_a.shape
    finally:
        plain.close()
        indexed.close()
    print("[case 18] pass series=%d merged_rows=300 window_rows=%d"
          % (len(names), int(window_a.shape[0])))


def test_case19_time_range_and_empty_windows(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path))
    try:
        name = [str(item) for item in plain.list_timeseries()
                if str(item).startswith("weather.dev_a") and str(item).endswith(".temperature")][0]
        details = {}
        for label, key in (("point_0_0", slice(0, 0)), ("boundary_99_100", slice(99, 100)),
                           ("beyond_300_400", slice(300, 400))):
            a, b = np.asarray(plain[name][key]), np.asarray(indexed[name][key])
            np.testing.assert_array_equal(a, b)
            details[label] = int(a.shape[0])
        inverted = {}
        for tag, frame in (("plain", plain), ("indexed", indexed)):
            try:
                inverted[tag] = int(np.asarray(frame[name][400:300]).shape[0])
            except Exception as exc:  # noqa: BLE001
                inverted[tag] = "%s: %s" % (type(exc).__name__, str(exc)[:80])
        assert inverted["plain"] == inverted["indexed"], inverted
        details["inverted"] = inverted["plain"]
    finally:
        plain.close()
        indexed.close()
    print("[case 19] pass %s" % details)


def test_case21_tag_addressing_matches(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path))
    try:
        prefixed_a = [str(item) for item in plain.list_timeseries("weather.dev_b")]
        prefixed_b = [str(item) for item in indexed.list_timeseries("weather.dev_b")]
        assert prefixed_a == prefixed_b, (prefixed_a, prefixed_b)
        # dev_b 的 humidity 列全为空值 → 写入器不生成该序列，故 dev_b 只有 temperature
        assert prefixed_a == ["weather.dev_b.us.temperature"], prefixed_a
        dev_b = prefixed_a[0]
        np.testing.assert_array_equal(values(plain, dev_b), values(indexed, dev_b))
        dev_a = [str(item) for item in plain.list_timeseries()
                 if str(item).startswith("weather.dev_a.cn") and str(item).endswith(".temperature")][0]
        np.testing.assert_array_equal(values(plain, dev_a), values(indexed, dev_a))
        empty_tag = [str(item) for item in plain.list_timeseries()
                     if str(item).startswith("weather.dev_c") and str(item).endswith(".temperature")][0]
        np.testing.assert_array_equal(values(plain, empty_tag), values(indexed, empty_tag))
        dev_b_rows = int(values(plain, dev_b).shape[0])   # 必须在 close 前取值
    finally:
        plain.close()
        indexed.close()
    print("[case 21] pass tag_prefix_series=%s dev_b_rows=%d empty_tag_path=%s"
          % (prefixed_a, dev_b_rows, empty_tag))


def test_case22_list_and_metadata_api_matches(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path))
    try:
        assert len(plain) == len(indexed)
        assert plain.model == indexed.model == "table"
        assert [str(x) for x in plain.list_timeseries()] == [str(x) for x in indexed.list_timeseries()]
        meta_a = [repr(item) for item in plain.list_timeseries_metadata()]
        meta_b = [repr(item) for item in indexed.list_timeseries_metadata()]
        assert meta_a == meta_b, "list_timeseries_metadata 不一致"
        stats_a = [plain[str(item)].stats for item in plain.list_timeseries()]
        stats_b = [indexed[str(item)].stats for item in indexed.list_timeseries()]
        assert stats_a == stats_b, "series stats 不一致"
        series_count = int(len(plain))                    # 必须在 close 前取值
    finally:
        plain.close()
        indexed.close()
    print("[case 22] pass series_count=%d metadata_rows=%d" % (series_count, len(meta_a)))


def test_case23_loc_multi_field_matches(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path))
    try:
        requested = ["weather.dev_a.cn.temperature", "weather.dev_a.cn.humidity",
                     "weather.dev_b.us.temperature"]
        a = plain.loc[0:120, requested]
        b = indexed.loc[0:120, requested]
        assert list(a.series_names) == list(b.series_names)
        np.testing.assert_array_equal(np.asarray(a.timestamps), np.asarray(b.timestamps))
        np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))
        single = plain.loc[:5, [requested[0]]]
        single_i = indexed.loc[:5, [requested[0]]]
        np.testing.assert_array_equal(np.asarray(single.values), np.asarray(single_i.values))
        rows = int(np.asarray(a.timestamps).shape[0])
        columns = list(a.series_names)
    finally:
        plain.close()
        indexed.close()
    print("[case 23] pass rows=%d columns=%s" % (rows, columns))


def test_case24_null_values_and_tags_match(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path))
    try:
        # ① dev_b 的 humidity 列全空 → 不生成序列（与 case21 同一事实）
        dev_b_series = [str(item) for item in plain.list_timeseries() if str(item).startswith("weather.dev_b")]
        assert dev_b_series == ["weather.dev_b.us.temperature"], dev_b_series
        # ② dev_a 的 humidity 来自分片 C（每 5 行一个空值）→ 部分空值
        dev_a_humidity = [str(item) for item in plain.list_timeseries()
                          if str(item).startswith("weather.dev_a") and str(item).endswith(".humidity")][0]
        partial_a = values(plain, dev_a_humidity)
        partial_b = values(indexed, dev_a_humidity)
        np.testing.assert_array_equal(partial_a, partial_b)
        partial_null = int(np.isnan(partial_a).sum())
        assert 0 < partial_null < partial_a.shape[0], partial_null
        # ③ 空 TAG 设备 dev_c（region 为空）路径与数据一致
        dev_c = [str(item) for item in plain.list_timeseries()
                 if str(item).startswith("weather.dev_c") and str(item).endswith(".temperature")][0]
        np.testing.assert_array_equal(values(plain, dev_c), values(indexed, dev_c))
    finally:
        plain.close()
        indexed.close()
    print("[case 24] pass dev_b_series=%s dev_a_null_rows=%d empty_tag_path=%s"
          % (dev_b_series, partial_null, dev_c))


def test_case26_tree_model_matches(d1_root, tmp_path):
    plain, indexed = pair_on(table_copy(d1_root, tmp_path, sub="tree"))
    try:
        names = compare_all_series(plain, indexed, "tree")
        assert plain.model == indexed.model == "tree"
        assert len(names) == 3, names
    finally:
        plain.close()
        indexed.close()
    print("[case 26] pass series=%d model=tree" % len(names))


def test_case27_mixed_model_rejected(d1_root, tmp_path):
    mixed = str(table_copy(d1_root, tmp_path, sub="mixed"))
    errors = {}
    for tag, use_index in (("plain", False), ("indexed", True)):
        try:
            frame = TsFileDataFrame(mixed, show_progress=False, use_index=use_index)
            frame.close()
            errors[tag] = None
        except Exception as exc:  # noqa: BLE001
            errors[tag] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    assert errors["plain"] and errors["indexed"], errors
    assert "Mixed table-model and tree-model" in errors["plain"]
    assert "Mixed table-model and tree-model" in errors["indexed"]
    print("[case 27] pass errors=%s" % errors)


def test_case28_duplicate_timestamps_rejected(d1_root, tmp_path):
    data = tmp_path / "dup"
    data.mkdir()
    for name in ("weather_a.tsfile", "weather_b.tsfile"):
        shutil.copy(str(Path(d1_root) / "table" / name), str(data / name))
    # 把 B 的时间区间改成与 A 重叠：同设备 dev_a 出现重复时间戳
    os.remove(str(data / "weather_b.tsfile"))
    fx._write_table_shard(str(data / "weather_b.tsfile"), 50, 149, {"dev_a": ("cn", 0)})
    errors = {}
    for tag, use_index in (("plain", False), ("indexed", True)):
        try:
            frame = TsFileDataFrame(str(data), show_progress=False, use_index=use_index)
            frame.close()
            errors[tag] = None
        except Exception as exc:  # noqa: BLE001
            errors[tag] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    assert errors["plain"] and errors["indexed"], errors
    assert "Duplicate timestamp" in errors["plain"] and "Duplicate timestamp" in errors["indexed"]
    print("[case 28] pass errors=%s index_published=%s"
          % (errors, bool(glob.glob(str(data / "*.tsidx")))))


def test_case30_timeseries_idempotent_close(d1_root, tmp_path):
    table = table_copy(d1_root, tmp_path)
    frame = TsFileDataFrame(str(table), show_progress=False, use_index=True)
    name = str(frame.list_timeseries()[0])
    with frame[name] as series:
        first = len(np.asarray(series[:]))
    series.close()
    series.close()
    print("[case 30] pass rows=%d double_close_ok=True" % first)
    frame.close()
