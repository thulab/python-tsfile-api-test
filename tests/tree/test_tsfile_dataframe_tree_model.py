import concurrent.futures
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfile import (
    AlignedTimeseries,
    ColumnCategory,
    ColumnSchema,
    Field,
    RowRecord,
    TSDataType,
    TableSchema,
    Timeseries,
    TimeseriesSchema,
    TsFileDataFrame,
    TsFileTableWriter,
    TsFileWriter,
)
from tsfile.dataset.reader import TsFileSeriesReader


def _write_tree_file(path):
    """构造基础树模型文件：2 个设备、3 条数值序列、统一时间轴，用作多数正向用例的稳定样本。"""
    writer = TsFileWriter(str(path))
    writer.register_timeseries(
        "root.ln.wf01.wt01", TimeseriesSchema("status", TSDataType.INT32)
    )
    writer.register_timeseries(
        "root.ln.wf01.wt01", TimeseriesSchema("temperature", TSDataType.DOUBLE)
    )
    writer.register_timeseries(
        "root.ln.wf02.wt02", TimeseriesSchema("status", TSDataType.INT32)
    )
    for t in range(5):
        writer.write_row_record(
            RowRecord(
                "root.ln.wf01.wt01",
                t,
                [
                    Field("status", t, TSDataType.INT32),
                    Field("temperature", float(t) + 0.5, TSDataType.DOUBLE),
                ],
            )
        )
        writer.write_row_record(
            RowRecord(
                "root.ln.wf02.wt02",
                t,
                [Field("status", t * 2, TSDataType.INT32)],
            )
        )
    writer.close()


def _default_value(name, dtype, t):
    """按数据类型生成默认写入值，便于审查不同类型测点的预期值来源。"""
    if dtype == TSDataType.INT32:
        return int(t)
    if dtype == TSDataType.INT64:
        return int(t * 10)
    if dtype in (TSDataType.FLOAT, TSDataType.DOUBLE):
        return float(t) + 0.5
    if dtype in (TSDataType.STRING, TSDataType.TEXT):
        return f"{name}-{t}"
    return t


def _write_tree_rows(path, device_measurements, t_start=0, t_count=3, value_func=None):
    """按设备到测点列表的映射批量写树模型数据，支持自定义时间起点和值生成逻辑。"""
    writer = TsFileWriter(str(path))
    for device, measurements in device_measurements.items():
        for name, dtype in measurements:
            writer.register_timeseries(device, TimeseriesSchema(name, dtype))
    for t in range(t_start, t_start + t_count):
        for device, measurements in device_measurements.items():
            fields = []
            for name, dtype in measurements:
                value = (
                    value_func(device, name, dtype, t)
                    if value_func
                    else _default_value(name, dtype, t)
                )
                fields.append(Field(name, value, dtype))
            writer.write_row_record(RowRecord(device, t, fields))
    writer.close()


def _write_tree_points(path, device_measurements, points):
    """按显式点位写入树模型数据，用于构造稀疏时间轴、重叠分片等边界场景。"""
    writer = TsFileWriter(str(path))
    dtype_by_device = {}
    for device, measurements in device_measurements.items():
        dtype_by_device[device] = dict(measurements)
        for name, dtype in measurements:
            writer.register_timeseries(device, TimeseriesSchema(name, dtype))
    for device, timestamp, values in points:
        fields = [
            Field(name, value, dtype_by_device[device][name])
            for name, value in values.items()
        ]
        writer.write_row_record(RowRecord(device, timestamp, fields))
    writer.close()


def _write_case_sensitive_measurement_file(path):
    """构造同一设备下仅物理量大小写不同的树模型文件。"""

    def field_value_func(device, name, dtype, t):
        if name == "temperature":
            return float(30 + t)
        if name == "Temperature":
            return float(40 + t)
        return float(t)

    _write_tree_rows(
        path,
        {
            "root.case.d1": [
                ("temperature", TSDataType.DOUBLE),
                ("Temperature", TSDataType.DOUBLE),
            ],
        },
        value_func=field_value_func,
    )


def _write_weather_table(path, start=0):
    """构造表模型 weather 文件，用于验证树表混用检测逻辑。"""
    schema = TableSchema(
        "weather",
        [
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    df = pd.DataFrame(
        {
            "time": [start, start + 1, start + 2],
            "device": ["device_a", "device_a", "device_a"],
            "temperature": [20.0, 21.5, 23.0],
            "humidity": [50.0, 52.0, 55.0],
        }
    )
    with TsFileTableWriter(str(path), schema) as writer:
        writer.write_dataframe(df)


def _write_empty_table(path):
    """构造无数据的表模型 TsFile，用于验证零序列 reader 不参与模型类型冲突判断。"""
    schema = TableSchema(
        "empty_table",
        [
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("value", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    with TsFileTableWriter(str(path), schema):
        pass


def _assert_timeseries_metadata(ts, name, count, start_time, end_time):
    """统一校验 Timeseries 的名称、长度和统计信息。"""
    assert isinstance(ts, Timeseries)
    assert ts.name == name
    assert len(ts) == count
    assert ts.stats == {
        "start_time": start_time,
        "end_time": end_time,
        "count": count,
    }


# 用例 PYTHON-TREE-035：单文件加载为tree且表头符合规则
# ── 由 test_tree_loads_single_file_and_model_metadata 拆分而来（3 个）──
def test_tree_single_file_path_loads(tmp_path):
    """用例 1：验证传入单个 .tsfile 文件路径可加载树模型数据。"""
    # 1. 生成包含 2 个设备、3 条数值序列的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载单个树模型文件，校验序列数量与单文件展开展示
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert len(tsdf) == 3
        assert "TsFileDataFrame(tree model, 3 time series, 1 files)" in repr(tsdf)

def test_tree_model_identified_as_tree(tmp_path):
    """用例 9：验证加载集合全部为树模型文件时 model 判定为 tree。"""
    # 1. 生成包含 2 个设备、3 条数值序列的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载单个树模型文件，校验模型类型判定
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.model == "tree"

def test_tree_series_names_flatten_device_segments(tmp_path):
    """用例 15：验证树模型序列名映射为 root.{设备名分段_1}...{设备名分段_k}.{物理量名称} 扁平字符串。"""
    # 1. 生成包含 2 个设备、3 条数值序列的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件，校验序列清单按扁平路径展开
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf02.wt02.status",
        ]

        # 3. 校验展示内容使用树模型元数据列（设备名分段 + 物理量名称），不出现表模型 table 列
        rendered = repr(tsdf)
        assert "_col_1" in rendered and "_col_2" in rendered and "_col_3" in rendered
        assert "table" not in rendered.splitlines()[1]


# ── 由 test_tree_loads_file_list_and_recursive_directory 拆分而来（4 个）──
def test_tree_file_list_loads_multiple_files(tmp_path):
    """用例 2：验证传入多个 .tsfile 组成的列表可一次性加载。"""
    # 1. 构造两个树模型TsFile文件
    root_dir = tmp_path / "tree_data"
    sub_dir = root_dir / "sub"
    sub_dir.mkdir(parents=True)
    first = root_dir / "b.tsfile"
    second = sub_dir / "a.tsfile"
    _write_tree_rows(first, {"root.b.device": [("m1", TSDataType.DOUBLE)]}, 10)
    _write_tree_rows(second, {"root.a.device": [("m1", TSDataType.DOUBLE)]}, 0)

    # 2. 按文件列表加载，确认两个文件均参与合并
    with TsFileDataFrame([str(first), str(second)], show_progress=False) as listed:
        assert len(listed) == 2
        assert sorted(listed.list_timeseries()) == [
            "root.a.device.m1",
            "root.b.device.m1",
        ]

def test_tree_directory_load_finds_all_tsfiles(tmp_path):
    """用例 3：验证传入目录时递归查找目录下所有 .tsfile 并加载。"""
    # 1. 构造包含子目录的测试目录和两个树模型TsFile文件
    root_dir = tmp_path / "tree_data"
    sub_dir = root_dir / "sub"
    sub_dir.mkdir(parents=True)
    first = root_dir / "b.tsfile"
    second = sub_dir / "a.tsfile"
    _write_tree_rows(first, {"root.b.device": [("m1", TSDataType.DOUBLE)]}, 10)
    _write_tree_rows(second, {"root.a.device": [("m1", TSDataType.DOUBLE)]}, 0)

    # 2. 按目录递归加载，确认目录下所有 .tsfile 都被加载且路径被归一化
    with TsFileDataFrame(str(root_dir), show_progress=False) as from_dir:
        assert sorted(from_dir.list_timeseries()) == [
            "root.a.device.m1",
            "root.b.device.m1",
        ]
        assert all(Path(path).is_absolute() for path in from_dir._paths)

def test_tree_directory_load_searches_nested_subdirectories(tmp_path):
    """用例 4：验证目录下的多级子目录中的 .tsfile 也能被递归找到。"""
    # 1. 构造带两级子目录的测试目录，最深层放置树模型TsFile文件
    root_dir = tmp_path / "tree_data"
    nested_dir = root_dir / "level1" / "level2"
    nested_dir.mkdir(parents=True)
    nested = nested_dir / "deep.tsfile"
    _write_tree_rows(nested, {"root.deep.device": [("m1", TSDataType.DOUBLE)]})

    # 2. 按目录加载，确认深层子目录中的文件也被递归加载
    with TsFileDataFrame(str(root_dir), show_progress=False) as from_dir:
        assert from_dir.list_timeseries() == ["root.deep.device.m1"]

def test_tree_directory_load_ignores_non_tsfile_files(tmp_path):
    """用例 7：验证目录中非 .tsfile 后缀文件被忽略。"""
    # 1. 构造目录，混合放置树模型TsFile文件和多种非 .tsfile 文件
    root_dir = tmp_path / "tree_data"
    root_dir.mkdir(parents=True)
    valid = root_dir / "valid.tsfile"
    _write_tree_rows(valid, {"root.b.device": [("m1", TSDataType.DOUBLE)]})
    (root_dir / "ignore.txt").write_text("not a tsfile", encoding="utf-8")
    (root_dir / "ignore.tsfile.bak").write_text("not a tsfile", encoding="utf-8")

    # 2. 按目录加载，确认只有 .tsfile 文件参与加载
    with TsFileDataFrame(str(root_dir), show_progress=False) as from_dir:
        assert [Path(path).name for path in from_dir._paths] == ["valid.tsfile"]
        assert from_dir.list_timeseries() == ["root.b.device.m1"]


# ── 由 test_tree_load_path_errors_and_relative_path 拆分而来（3 个）──
def test_tree_empty_directory_raises(tmp_path):
    """用例 5：验证目录下无任何 .tsfile 时抛异常。"""
    # 1. 构造空目录并校验无 .tsfile 文件时报错
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .tsfile files found"):
        TsFileDataFrame(str(empty_dir), show_progress=False)

def test_tree_missing_file_path_raises(tmp_path):
    """用例 6：验证指定不存在的文件路径时抛异常。"""
    # 1. 指定不存在的文件路径并校验报错
    with pytest.raises(FileNotFoundError, match="TsFile not found"):
        TsFileDataFrame(str(tmp_path / "missing.tsfile"), show_progress=False)

def test_tree_relative_path_normalized_to_absolute(tmp_path, monkeypatch):
    """用例 8：验证相对路径被归一化为绝对路径。"""
    # 1. 生成树模型TsFile文件并切换到临时目录
    path = tmp_path / "relative.tsfile"
    _write_tree_rows(path, {"root.rel.device": [("m1", TSDataType.DOUBLE)]})
    monkeypatch.chdir(tmp_path)

    # 2. 使用相对路径加载，确认内部路径归一化为绝对路径
    with TsFileDataFrame("relative.tsfile", show_progress=False) as tsdf:
        assert tsdf._paths == [os.path.abspath("relative.tsfile")]
        assert tsdf.list_timeseries() == ["root.rel.device.m1"]


# ── 由 test_tree_rejects_mixed_table_and_tree_models 拆分而来（2 个）──
def test_tree_rejects_table_then_tree_model_mix(tmp_path):
    """用例 10：验证同一加载集合中混入表模型与树模型文件时加载阶段抛异常。"""
    # 1. 分别生成树模型和表模型TsFile文件
    tree_path = tmp_path / "tree.tsfile"
    table_path = tmp_path / "table.tsfile"
    _write_tree_file(tree_path)
    _write_weather_table(table_path)

    # 2. 表模型在前、树模型在后时，加载阶段即抛出树表混用错误
    with pytest.raises(ValueError, match="Mixed table-model and tree-model"):
        TsFileDataFrame([str(table_path), str(tree_path)], show_progress=False)

def test_tree_rejects_tree_then_table_model_mix(tmp_path):
    """用例 11：验证首个文件为树模型、后续文件为表模型时同样报混用错误。"""
    # 1. 分别生成树模型和表模型TsFile文件
    tree_path = tmp_path / "tree.tsfile"
    table_path = tmp_path / "table.tsfile"
    _write_tree_file(tree_path)
    _write_weather_table(table_path)

    # 2. 树模型在前、表模型在后时，同样抛出树表混用错误
    with pytest.raises(ValueError, match="Mixed table-model and tree-model"):
        TsFileDataFrame([str(tree_path), str(table_path)], show_progress=False)


# ── 由 test_tree_empty_and_no_numeric_inputs_report_stable_errors 拆分而来（3 个）──
def test_tree_load_without_any_numeric_series_raises(tmp_path):
    """用例 13：验证加载集合中不存在任何数值序列时抛异常。"""
    # 1. 生成仅包含非数值测点的树模型TsFile文件
    string_only = tmp_path / "string_only.tsfile"
    _write_tree_rows(
        string_only,
        {"root.a.b": [("status", TSDataType.STRING)]},
    )

    # 2. 加载非数值测点文件，校验加载集合无数值序列时报错而不是生成空视图
    with pytest.raises(ValueError, match="No numeric measurements found"):
        TsFileDataFrame(str(string_only), show_progress=False)

def test_tree_empty_file_without_device_metadata_raises(tmp_path):
    """用例 115：验证树模型文件没有任何设备名元数据时加载失败。"""
    # 1. 生成空树模型TsFile文件
    empty_tree = tmp_path / "empty_tree.tsfile"
    writer = TsFileWriter(str(empty_tree))
    writer.close()

    # 2. 加载空树文件，校验无设备元数据时报错
    with pytest.raises(ValueError, match="No devices found in tree-model TsFile"):
        TsFileDataFrame(str(empty_tree), show_progress=False)

def test_tree_string_or_text_only_measurements_are_non_numeric(tmp_path):
    """用例 116：验证树模型仅含 STRING 或 TEXT 物理量时被整体判定为无数值序列。"""
    # 1. 分别生成仅含 STRING 和仅含 TEXT 物理量的树模型TsFile文件
    string_only = tmp_path / "string_only_measurement.tsfile"
    text_only = tmp_path / "text_only_measurement.tsfile"
    _write_tree_rows(string_only, {"root.a.b": [("status", TSDataType.STRING)]})
    _write_tree_rows(text_only, {"root.a.b": [("note", TSDataType.TEXT)]})

    # 2. 加载两个文件，校验 STRING 和 TEXT 物理量都会被整体判定为无数值序列
    for path in (string_only, text_only):
        with pytest.raises(ValueError, match="No numeric measurements found"):
            TsFileDataFrame(str(path), show_progress=False)


def test_tree_rejects_multiple_root_segments_in_one_file(tmp_path):
    """用例 114：验证单个树模型文件包含多个 root 段时快速失败，避免生成部分 catalog。"""
    # 1. 生成包含 root1/root2 两个根段的树模型TsFile文件
    path = tmp_path / "multi_root.tsfile"
    _write_tree_rows(
        path,
        {
            "root1.a": [("m1", TSDataType.DOUBLE)],
            "root2.a": [("m1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载多 root 文件，校验快速失败并提示 multiple root segments
    with pytest.raises(ValueError, match="multiple root segments"):
        TsFileDataFrame(str(path), show_progress=False)


# 用例 PYTHON-TREE-041：list_timeseries前缀匹配
# ── 由 test_tree_list_timeseries_prefix_is_segment_aware 拆分而来（6 个）──
def test_tree_list_timeseries_without_prefix_returns_all(tmp_path):
    """用例 23：验证不传前缀时返回全部序列名。"""
    # 1. 生成包含 root.db 和 root.dbx 两类设备名的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.d1": [("s1", TSDataType.DOUBLE), ("s2", TSDataType.DOUBLE)],
            "root.db.d2": [("s1", TSDataType.DOUBLE)],
            "root.dbx.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件，校验不传前缀时返回全部序列名
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == [
            "root.db.d1.s1",
            "root.db.d1.s2",
            "root.db.d2.s1",
            "root.dbx.d1.s1",
        ]

def test_tree_list_timeseries_filters_by_root_prefix(tmp_path):
    """用例 24：验证按 root 前缀筛选序列。"""
    # 1. 生成包含 root.db 和 root.dbx 两类设备名的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.d1": [("s1", TSDataType.DOUBLE), ("s2", TSDataType.DOUBLE)],
            "root.db.d2": [("s1", TSDataType.DOUBLE)],
            "root.dbx.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件并按 root 前缀筛选，校验 root 下所有序列均被命中
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries("root")) == [
            "root.db.d1.s1",
            "root.db.d1.s2",
            "root.db.d2.s1",
            "root.dbx.d1.s1",
        ]

def test_tree_list_timeseries_filters_by_device_prefix(tmp_path):
    """用例 25：验证按设备名前缀筛选序列。"""
    # 1. 生成包含 root.db 和 root.dbx 两类设备名的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.d1": [("s1", TSDataType.DOUBLE), ("s2", TSDataType.DOUBLE)],
            "root.db.d2": [("s1", TSDataType.DOUBLE)],
            "root.dbx.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件并按设备名前缀筛选，只返回对应设备名下序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries("root.db.d1")) == [
            "root.db.d1.s1",
            "root.db.d1.s2",
        ]

def test_tree_list_timeseries_prefix_matches_by_segment(tmp_path):
    """用例 26：验证前缀匹配按设备名分段位置感知，不做纯字符串前缀。"""
    # 1. 生成包含 root.db 和 root.dbx 两类设备名的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.d1": [("s1", TSDataType.DOUBLE), ("s2", TSDataType.DOUBLE)],
            "root.db.d2": [("s1", TSDataType.DOUBLE)],
            "root.dbx.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件并按库名前缀筛选，校验不把 root.dbx 误判为 root.db
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries("root.db")) == [
            "root.db.d1.s1",
            "root.db.d1.s2",
            "root.db.d2.s1",
        ]

        # 3. 反向校验 root.dbx 分段只命中自身设备名下的序列
        assert tsdf.list_timeseries("root.dbx") == ["root.dbx.d1.s1"]

def test_tree_list_timeseries_unmatched_prefix_returns_empty(tmp_path):
    """用例 27：验证前缀无匹配时返回空列表。"""
    # 1. 生成包含 root.db 和 root.dbx 两类设备名的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.d1": [("s1", TSDataType.DOUBLE), ("s2", TSDataType.DOUBLE)],
            "root.db.d2": [("s1", TSDataType.DOUBLE)],
            "root.dbx.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件并校验缺失前缀不会误匹配任何序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries("root.none") == []

def test_tree_list_timeseries_invalid_prefix_returns_empty(tmp_path):
    """用例 28：验证非法/无法解析的前缀返回空列表而非抛错。"""
    # 1. 生成包含 root.db 和 root.dbx 两类设备名的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.d1": [("s1", TSDataType.DOUBLE), ("s2", TSDataType.DOUBLE)],
            "root.db.d2": [("s1", TSDataType.DOUBLE)],
            "root.dbx.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件并校验非法转义前缀返回空列表而不抛错
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries("root.db\\") == []


def test_tree_list_timeseries_filters_different_device_names(tmp_path):
    """用例 140：验证 list_timeseries 可区分同一数据库名下的不同设备名。"""
    # 1. 生成同一数据库名下包含多个设备名的树模型TsFile文件
    path = tmp_path / "list_timeseries_devices.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.device_a": [("s1", TSDataType.DOUBLE)],
            "root.db.device_b": [("s1", TSDataType.DOUBLE)],
            "root.db.device_c": [("s2", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验无前缀时返回所有设备名下的序列
        assert sorted(tsdf.list_timeseries()) == [
            "root.db.device_a.s1",
            "root.db.device_b.s1",
            "root.db.device_c.s2",
        ]

        # 4. 校验设备名前缀只返回对应设备名下的序列
        assert tsdf.list_timeseries("root.db.device_a") == ["root.db.device_a.s1"]
        assert tsdf.list_timeseries("root.db.device_b") == ["root.db.device_b.s1"]
        assert tsdf.list_timeseries("root.db.device_c") == ["root.db.device_c.s2"]


# 用例 PYTHON-TREE-043：同设备不同物理量可区分
def test_tree_list_timeseries_filters_different_measurement_names(tmp_path):
    """用例 141：验证 list_timeseries 可区分同一设备名下的不同物理量名称。"""
    # 1. 生成同一设备名下包含多个物理量名称的树模型TsFile文件
    path = tmp_path / "list_timeseries_measurements.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.device_a": [
                ("temperature", TSDataType.DOUBLE),
                ("humidity", TSDataType.DOUBLE),
                ("status", TSDataType.DOUBLE),
            ]
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验设备名前缀返回该设备名下全部物理量名称序列
        assert sorted(tsdf.list_timeseries("root.db.device_a")) == [
            "root.db.device_a.humidity",
            "root.db.device_a.status",
            "root.db.device_a.temperature",
        ]

        # 4. 校验完整序列名前缀只返回对应物理量名称序列
        assert tsdf.list_timeseries("root.db.device_a.temperature") == [
            "root.db.device_a.temperature"
        ]
        assert tsdf.list_timeseries("root.db.device_a.humidity") == [
            "root.db.device_a.humidity"
        ]


# 用例 PYTHON-TREE-044：list_timeseries按库名分段匹配
def test_tree_list_timeseries_filters_different_database_names(tmp_path):
    """用例 142：验证 list_timeseries 可区分不同数据库名，且数据库名前缀按分段匹配。"""
    # 1. 生成不同数据库名下设备名和物理量名称相同的树模型TsFile文件
    path = tmp_path / "list_timeseries_databases.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db1.d1": [("s1", TSDataType.DOUBLE)],
            "root.db2.d1": [("s1", TSDataType.DOUBLE)],
            "root.db10.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验不同数据库名均完整列出
        assert sorted(tsdf.list_timeseries()) == [
            "root.db1.d1.s1",
            "root.db10.d1.s1",
            "root.db2.d1.s1",
        ]

        # 4. 校验数据库名前缀不会按纯字符串误匹配
        assert tsdf.list_timeseries("root.db1") == ["root.db1.d1.s1"]
        assert tsdf.list_timeseries("root.db2") == ["root.db2.d1.s1"]
        assert tsdf.list_timeseries("root.db") == []


# 用例 PYTHON-TREE-045：list_timeseries按中间设备名前缀筛选
# ── 由 test_tree_list_timeseries_handles_different_series_name_depths 拆分而来（2 个）──
def test_tree_list_timeseries_lists_every_series_name_depth(tmp_path):
    """用例 16：验证不同设备名可有不同分段数(k>=1)。"""
    # 1. 生成设备名深度不同的树模型TsFile文件
    path = tmp_path / "list_timeseries_depths.tsfile"
    _write_tree_rows(
        path,
        {
            "root.depth": [("m1", TSDataType.DOUBLE)],
            "root.depth.area": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line.unit": [("m1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件，确认 k=1 到 k=4 的设备名分段数可共存且序列名全部返回
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries("root.depth")) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
            "root.depth.m1",
        ]

def test_tree_list_timeseries_filters_by_middle_device_prefix(tmp_path):
    """用例 143：验证不同序列名深度的树模型序列可完整列出并按中间设备名前缀筛选。"""
    # 1. 生成设备名深度不同的树模型TsFile文件
    path = tmp_path / "list_timeseries_depths.tsfile"
    _write_tree_rows(
        path,
        {
            "root.depth": [("m1", TSDataType.DOUBLE)],
            "root.depth.area": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line.unit": [("m1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验中间设备名前缀命中自身及更深设备名下序列
        assert sorted(tsdf.list_timeseries("root.depth.area")) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
        ]

        # 4. 校验完整序列名前缀只返回单条序列
        assert tsdf.list_timeseries("root.depth.area.line.unit.m1") == [
            "root.depth.area.line.unit.m1"
        ]


def test_tree_list_timeseries_handles_different_series_counts(tmp_path):
    """用例 144：验证 list_timeseries 在单序列和大量序列场景下均返回完整序列清单。"""
    # 1. 生成仅包含 1 条序列的树模型TsFile文件
    single_path = tmp_path / "list_timeseries_single.tsfile"
    _write_tree_rows(single_path, {"root.count.single": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载单序列文件并校验返回数量
    with TsFileDataFrame(str(single_path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.count.single.m1"]
        assert len(tsdf.list_timeseries()) == len(tsdf) == 1

    # 3. 生成包含 64 条序列的树模型TsFile文件
    many_path = tmp_path / "list_timeseries_many.tsfile"
    _write_tree_rows(
        many_path,
        {f"root.count.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(64)},
    )

    # 4. 加载大量序列文件并校验 list_timeseries 不受展示截断影响
    with TsFileDataFrame(str(many_path), show_progress=False) as tsdf:
        names = sorted(tsdf.list_timeseries())
        assert len(names) == len(tsdf) == 64
        assert names[0] == "root.count.d00.m1"
        assert names[-1] == "root.count.d63.m1"
        assert tsdf.list_timeseries("root.count.d00") == ["root.count.d00.m1"]


# 用例 PYTHON-TREE-047：metadata区分库下设备名
def test_tree_list_timeseries_metadata_filters_different_device_names(tmp_path):
    """用例 145：验证 list_timeseries_metadata 可区分同一数据库名下的不同设备名。"""
    # 1. 生成同一数据库名下包含多个设备名的树模型TsFile文件
    path = tmp_path / "metadata_devices.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.device_a": [("s1", TSDataType.DOUBLE)],
            "root.db.device_b": [("s1", TSDataType.DOUBLE)],
            "root.db.device_c": [("s2", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验无前缀时返回所有设备名下的元数据
        meta = tsdf.list_timeseries_metadata()
        assert sorted(meta.index.tolist()) == [
            "root.db.device_a.s1",
            "root.db.device_b.s1",
            "root.db.device_c.s2",
        ]

        # 4. 校验 _col_i 表达设备名分段，field 表达物理量名称
        assert meta.loc["root.db.device_a.s1", "_col_1"] == "db"
        assert meta.loc["root.db.device_a.s1", "_col_2"] == "device_a"
        assert meta.loc["root.db.device_c.s2", "field"] == "s2"

        # 5. 校验设备名前缀只返回对应设备名下的元数据
        filtered = tsdf.list_timeseries_metadata("root.db.device_a")
        assert filtered.index.tolist() == ["root.db.device_a.s1"]
        assert filtered.loc["root.db.device_a.s1", "_col_2"] == "device_a"


# 用例 PYTHON-TREE-048：metadata区分设备下物理量
def test_tree_list_timeseries_metadata_filters_different_measurement_names(tmp_path):
    """用例 146：验证 list_timeseries_metadata 可区分同一设备名下的不同物理量名称。"""
    # 1. 生成同一设备名下包含多个物理量名称的树模型TsFile文件
    path = tmp_path / "metadata_measurements.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db.device_a": [
                ("temperature", TSDataType.DOUBLE),
                ("humidity", TSDataType.DOUBLE),
                ("status", TSDataType.DOUBLE),
            ]
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验设备名前缀返回该设备名下全部物理量名称元数据
        filtered = tsdf.list_timeseries_metadata("root.db.device_a")
        assert sorted(filtered.index.tolist()) == [
            "root.db.device_a.humidity",
            "root.db.device_a.status",
            "root.db.device_a.temperature",
        ]
        assert sorted(filtered["field"].tolist()) == [
            "humidity",
            "status",
            "temperature",
        ]

        # 4. 校验完整序列名前缀只返回对应物理量名称元数据
        exact = tsdf.list_timeseries_metadata("root.db.device_a.temperature")
        assert exact.index.tolist() == ["root.db.device_a.temperature"]
        assert exact.loc["root.db.device_a.temperature", "field"] == "temperature"


# 用例 PYTHON-TREE-049：metadata按库名分段匹配
def test_tree_list_timeseries_metadata_filters_different_database_names(tmp_path):
    """用例 147：验证 list_timeseries_metadata 可区分不同数据库名，且数据库名前缀按分段匹配。"""
    # 1. 生成不同数据库名下设备名和物理量名称相同的树模型TsFile文件
    path = tmp_path / "metadata_databases.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db1.d1": [("s1", TSDataType.DOUBLE)],
            "root.db2.d1": [("s1", TSDataType.DOUBLE)],
            "root.db10.d1": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验不同数据库名均完整返回元数据
        meta = tsdf.list_timeseries_metadata()
        assert sorted(meta.index.tolist()) == [
            "root.db1.d1.s1",
            "root.db10.d1.s1",
            "root.db2.d1.s1",
        ]

        # 4. 校验数据库名前缀不会按纯字符串误匹配
        db1 = tsdf.list_timeseries_metadata("root.db1")
        assert db1.index.tolist() == ["root.db1.d1.s1"]
        assert db1.loc["root.db1.d1.s1", "_col_1"] == "db1"
        assert tsdf.list_timeseries_metadata("root.db2").index.tolist() == [
            "root.db2.d1.s1"
        ]

        empty = tsdf.list_timeseries_metadata("root.db")
        assert empty.empty
        assert list(empty.columns) == list(meta.columns)


# 用例 PYTHON-TREE-050：序列元数据不同深度
# ── 由 test_tree_list_timeseries_metadata_handles_different_series_name_depths 拆分而来（2 个）──
def test_tree_list_timeseries_metadata_pads_short_device_segments(tmp_path):
    """用例 17：验证短设备名在设备名分段列 _col_i 末尾按左对齐补 None。"""
    # 1. 生成设备名深度不同的树模型TsFile文件
    path = tmp_path / "metadata_depths.tsfile"
    _write_tree_rows(
        path,
        {
            "root.depth": [("m1", TSDataType.DOUBLE)],
            "root.depth.area": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line.unit": [("m1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件，校验不同深度序列元数据全部返回，且全局设备名分段列扩展到最深
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        meta = tsdf.list_timeseries_metadata("root.depth")
        assert sorted(meta.index.tolist()) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
            "root.depth.m1",
        ]
        assert "_col_4" in meta.columns

        # 3. 校验浅层设备名按左对齐占用分段列，末尾补 None
        assert pd.isna(meta.loc["root.depth.m1", "_col_2"])
        assert meta.loc["root.depth.area.line.unit.m1", "_col_4"] == "unit"

def test_tree_list_timeseries_metadata_filters_by_middle_device_prefix(tmp_path):
    """用例 148：验证不同序列名深度的树模型元数据可完整返回并按中间设备名前缀筛选。"""
    # 1. 生成设备名深度不同的树模型TsFile文件
    path = tmp_path / "metadata_depths.tsfile"
    _write_tree_rows(
        path,
        {
            "root.depth": [("m1", TSDataType.DOUBLE)],
            "root.depth.area": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line": [("m1", TSDataType.DOUBLE)],
            "root.depth.area.line.unit": [("m1", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验中间设备名前缀命中自身及更深设备名下序列的完整元数据
        area = tsdf.list_timeseries_metadata("root.depth.area")
        assert sorted(area.index.tolist()) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
        ]
        assert "_col_4" in area.columns

        # 4. 校验完整序列名前缀只返回单条元数据
        exact = tsdf.list_timeseries_metadata("root.depth.area.line.unit.m1")
        assert exact.index.tolist() == ["root.depth.area.line.unit.m1"]


def test_tree_list_timeseries_metadata_handles_different_series_counts(tmp_path):
    """用例 149：验证 list_timeseries_metadata 在单序列和大量序列场景下均返回完整元数据。"""
    # 1. 生成仅包含 1 条序列的树模型TsFile文件
    single_path = tmp_path / "metadata_single.tsfile"
    _write_tree_rows(single_path, {"root.count.single": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载单序列文件并校验元数据行数
    with TsFileDataFrame(str(single_path), show_progress=False) as tsdf:
        meta = tsdf.list_timeseries_metadata()
        assert meta.index.tolist() == ["root.count.single.m1"]
        assert len(meta) == len(tsdf) == 1

    # 3. 生成包含 64 条序列的树模型TsFile文件
    many_path = tmp_path / "metadata_many.tsfile"
    _write_tree_rows(
        many_path,
        {f"root.count.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(64)},
    )

    # 4. 加载大量序列文件并校验 metadata 不受展示截断影响
    with TsFileDataFrame(str(many_path), show_progress=False) as tsdf:
        meta = tsdf.list_timeseries_metadata()
        names = sorted(meta.index.tolist())
        assert len(meta) == len(tsdf) == 64
        assert names[0] == "root.count.d00.m1"
        assert names[-1] == "root.count.d63.m1"
        assert meta.loc["root.count.d00.m1", "_col_2"] == "d00"


# 用例 PYTHON-TREE-052：元数据DataFrame列结构与CSV导出
def test_tree_metadata_dataframe_columns_filter_and_export(tmp_path):
    """用例 29-37：验证元数据 DataFrame 的列结构、时间列类型、前缀过滤、空结果和 CSV 导出能力。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并读取完整元数据DataFrame
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        meta = tsdf.list_timeseries_metadata()

        # 3. 校验元数据列结构、时间列类型和序列统计信息
        assert isinstance(meta, pd.DataFrame)
        assert list(meta.columns) == [
            "field",
            "start_time",
            "end_time",
            "count",
            "_col_1",
            "_col_2",
            "_col_3",
        ]
        assert "table" not in meta.columns
        assert pd.api.types.is_datetime64_any_dtype(meta["start_time"])
        assert pd.api.types.is_datetime64_any_dtype(meta["end_time"])
        assert meta.loc["root.ln.wf01.wt01.temperature", "count"] == 5

        # 4. 校验按设备前缀过滤元数据
        filtered = tsdf.list_timeseries_metadata("root.ln.wf01")
        assert sorted(filtered.index.tolist()) == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]

        # 5. 校验按完整序列路径过滤时仅返回单条序列
        exact = tsdf.list_timeseries_metadata("root.ln.wf01.wt01.temperature")
        assert exact.index.tolist() == ["root.ln.wf01.wt01.temperature"]
        assert exact.loc["root.ln.wf01.wt01.temperature", "field"] == "temperature"

        # 6. 校验无匹配前缀时返回空DataFrame且保留列结构
        empty = tsdf.list_timeseries_metadata("root.none")
        assert empty.empty
        assert list(empty.columns) == list(meta.columns)

        # 7. 校验元数据DataFrame可正常导出CSV
        csv_path = tmp_path / "metadata.csv"
        meta.to_csv(csv_path)
        assert "root.ln.wf01.wt01.status" in csv_path.read_text(encoding="utf-8")


# 用例 PYTHON-TREE-053：元数据列投影与布尔过滤子视图
# ── 由 test_tree_metadata_column_projection_and_boolean_filter 拆分而来（8 个）──
def test_tree_metadata_field_column_projection(tmp_path):
    """用例 38：验证 df['field'] 返回各序列物理量名称的 pandas.Series。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并按元数据 field 列名投影
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert list(tsdf["field"]) == ["status", "temperature", "status"]

def test_tree_metadata_col1_column_projection(tmp_path):
    """用例 39：验证 df['_col_1'] 返回设备名第一分段的 Series。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并按设备名第一分段列名投影
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert list(tsdf["_col_1"]) == ["ln", "ln", "ln"]

def test_tree_metadata_col3_column_projection(tmp_path):
    """用例 40：验证 df['_col_3'] 返回设备名第三分段值。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并按设备名第三分段列名投影
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert list(tsdf["_col_3"]) == ["wt01", "wt01", "wt02"]

def test_tree_metadata_table_column_is_unavailable(tmp_path):
    """用例 41：验证树模型下访问表模型 df['table'] 列抛 KeyError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并校验树模型不暴露 table 元数据列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(KeyError):
            tsdf["table"]

def test_tree_metadata_time_and_count_columns_projection(tmp_path):
    """用例 42：验证 df['start_time']/['end_time']/['count'] 可投影。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并按时间列和统计列投影
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert list(tsdf["start_time"]) == [0, 0, 0]
        assert list(tsdf["end_time"]) == [4, 4, 4]
        assert list(tsdf["count"]) == [5, 5, 5]

def test_tree_metadata_unknown_column_raises_key_error(tmp_path):
    """用例 43：验证访问不存在的列名抛 KeyError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并校验未知列名按序列查询失败
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(KeyError, match="Series not found"):
            tsdf["unknown_column"]

def test_tree_metadata_boolean_filter_returns_subset_view(tmp_path):
    """用例 44：验证 df[df['_col_2']=='d1'] 按设备名分段元数据布尔过滤返回子集视图。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并使用布尔条件筛选子集视图
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        subset = tsdf[tsdf["_col_2"] == "wf01"]
        assert isinstance(subset, TsFileDataFrame)
        assert len(subset) == 2
        assert subset.list_timeseries() == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]

def test_tree_metadata_boolean_filter_without_match_returns_empty_view(tmp_path):
    """用例 45：验证布尔过滤无命中时返回空视图。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并使用无命中布尔条件筛选，返回空子集视图
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        empty_subset = tsdf[tsdf["_col_2"] == "missing"]
        assert len(empty_subset) == 0
        assert empty_subset.list_timeseries() == []


# ── 由 test_tree_lazy_metadata_projection_view_and_context_close 拆分而来（3 个）──
def test_tree_initialization_only_scans_metadata(tmp_path, monkeypatch):
    """用例 14：验证初始化仅扫描元数据、不读取实际数值。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)
    read_calls = []

    # 2. 保存原始读取方法并准备 spy，记录是否触发真实数据读取
    original_read_by_ref = TsFileSeriesReader.read_series_by_ref
    original_read_by_row = TsFileSeriesReader.read_series_by_row

    def read_by_ref_spy(self, *args, **kwargs):
        read_calls.append("time_range")
        return original_read_by_ref(self, *args, **kwargs)

    def read_by_row_spy(self, *args, **kwargs):
        read_calls.append("row")
        return original_read_by_row(self, *args, **kwargs)

    monkeypatch.setattr(TsFileSeriesReader, "read_series_by_ref", read_by_ref_spy)
    monkeypatch.setattr(TsFileSeriesReader, "read_series_by_row", read_by_row_spy)

    # 3. 加载文件，确认初始化阶段不触发数据读取
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert read_calls == []

        # 4. 访问元数据投影列，确认仍不触发数据读取
        assert list(tsdf["start_time"]) == [0, 0, 0]
        assert list(tsdf["end_time"]) == [4, 4, 4]
        assert list(tsdf["count"]) == [5, 5, 5]
        assert read_calls == []

def test_tree_subset_view_reuses_parent_readers(tmp_path):
    """用例 56：验证子集视图复用父对象 readers、不重复打开文件。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件，记录根对象已打开的 reader
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        readers = dict(tsdf._readers)

        # 3. 构造子集视图，确认复用父对象 reader 且不重复打开文件
        subset = tsdf[0:2]
        assert subset._readers is tsdf._readers
        assert subset._root is tsdf
        assert len(tsdf._readers) == len(readers) == 1

        # 4. 读取子集序列，确认子集视图结果正确且仍复用同一批 reader
        np.testing.assert_array_equal(
            subset["root.ln.wf01.wt01.temperature"][:2],
            np.array([0.5, 1.5]),
        )
        assert list(tsdf._readers) == list(readers)

def test_tree_context_exit_closes_root_and_releases_readers(tmp_path):
    """用例 108：验证 with 语句退出后自动关闭并释放文件句柄。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 进入 with 期间 reader 处于打开状态
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert not tsdf._closed
        assert tsdf._readers != {}

    # 3. 退出 with 后确认根对象已关闭并释放 reader
    assert tsdf._closed
    assert tsdf._readers == {}

    # 4. 关闭后访问序列应提示对象已关闭，避免继续使用已释放的文件句柄
    with pytest.raises(RuntimeError, match="closed"):
        tsdf["root.ln.wf01.wt01.status"]


# ── 由 test_tree_getitem_series_and_subset_validation 拆分而来（10 个）──
def test_tree_getitem_name_returns_timeseries_handle(tmp_path):
    """用例 46：验证 df[name] 按完整树路径返回 Timeseries 懒加载句柄。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 按完整树路径取单条序列，校验句柄类型与名称
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        by_name = tsdf["root.ln.wf01.wt01.temperature"]
        assert isinstance(by_name, Timeseries)
        assert by_name.name == "root.ln.wf01.wt01.temperature"

        # 3. 句柄保持懒加载：取到句柄时尚未读取该序列的时间戳
        assert by_name._timestamps is None

def test_tree_getitem_position_index_returns_timeseries(tmp_path):
    """用例 47：验证 df[0] 按位置索引取序列。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 按位置索引取首条序列，校验句柄类型与解析到的序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        by_position = tsdf[0]
        assert isinstance(by_position, Timeseries)
        assert by_position.name == "root.ln.wf01.wt01.status"

def test_tree_getitem_negative_index_returns_last_series(tmp_path):
    """用例 48：验证 df[-1] 负索引取末条序列。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 按负索引取末条序列，校验句柄类型与解析到的序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        last = tsdf[-1]
        assert isinstance(last, Timeseries)
        assert last.name == "root.ln.wf02.wt02.status"

def test_tree_getitem_index_out_of_range_raises(tmp_path):
    """用例 49：验证整数索引越界抛 IndexError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 超出正向与负向边界的整数索引均抛 IndexError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(IndexError, match="out of range"):
            tsdf[99]
        with pytest.raises(IndexError, match="out of range"):
            tsdf[-4]

def test_tree_getitem_unknown_name_raises_key_error(tmp_path):
    """用例 50：验证按不存在的序列名取值抛 KeyError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 路径不完整或测点不存在的序列名均抛 KeyError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(KeyError, match="Series not found"):
            tsdf["root.ln.missing"]
        with pytest.raises(KeyError, match="Series not found"):
            tsdf["root.ln.wf01.wt01.no_such_measurement"]

def test_tree_getitem_slice_returns_subset_view(tmp_path):
    """用例 51：验证 df[0:3] 返回 TsFileDataFrame 视图。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 前段切片返回子集视图，视图长度与展示头均反映子集信息
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        subset = tsdf[0:2]
        assert isinstance(subset, TsFileDataFrame)
        assert len(subset) == 2
        assert "subset of 3" in repr(subset)
        assert subset.list_timeseries() == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]

        # 3. 三元素切片同样返回视图，且序列清单与全量一致
        full = tsdf[0:3]
        assert isinstance(full, TsFileDataFrame)
        assert len(full) == 3
        assert full.list_timeseries() == tsdf.list_timeseries()

def test_tree_getitem_index_list_returns_positional_view(tmp_path):
    """用例 52：验证 df[[0,2,5]] 返回指定位置的视图。"""
    # 1. 生成基础树模型TsFile文件（含 3 条序列）
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 按位置列表取第 1、3 条序列，校验视图类型与序列顺序
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert isinstance(tsdf[[0, 2]], TsFileDataFrame)
        assert [ts.name for ts in tsdf[[0, 2]]] == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf02.wt02.status",
        ]

    # 3. 构造含 6 条序列的文件，校验一次取多个不连续位置
    many_path = tmp_path / "many.tsfile"
    _write_tree_rows(
        many_path,
        {
            "root.m.dev": [
                (f"m{i}", TSDataType.DOUBLE) for i in range(6)
            ]
        },
    )
    with TsFileDataFrame(str(many_path), show_progress=False) as many_df:
        picked = many_df[[0, 2, 5]]
        assert isinstance(picked, TsFileDataFrame)
        assert picked.list_timeseries() == [
            "root.m.dev.m0",
            "root.m.dev.m2",
            "root.m.dev.m5",
        ]

def test_tree_getitem_index_list_with_non_integer_raises(tmp_path):
    """用例 53：验证整数列表含非整数元素抛 TypeError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 列表中混入字符串或浮点数均抛 TypeError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(TypeError, match="List index must contain integers"):
            tsdf[[0, "bad"]]
        with pytest.raises(TypeError, match="List index must contain integers"):
            tsdf[[0, 1.5]]

def test_tree_getitem_index_list_out_of_range_raises(tmp_path):
    """用例 54：验证整数列表含越界索引抛 IndexError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 列表中含正向或负向越界索引均抛 IndexError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(IndexError, match="out of range"):
            tsdf[[0, 99]]
        with pytest.raises(IndexError, match="out of range"):
            tsdf[[0, -4]]

def test_tree_getitem_unsupported_key_type_raises(tmp_path):
    """用例 55：验证传入不支持的 key 类型抛 TypeError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. dict 与 float 等不受支持的 key 类型均抛 TypeError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(TypeError, match="Unsupported key type"):
            tsdf[{"bad": "key"}]
        with pytest.raises(TypeError, match="Unsupported key type"):
            tsdf[1.5]


def test_tree_getitem_index_forms_expose_series_metadata(tmp_path):
    """用例 47、48、151：验证正索引和负索引获取 Timeseries 后，可查看 name、len() 和 stats。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 按正索引获取第一条序列并校验元信息
        _assert_timeseries_metadata(
            tsdf[0],
            "root.ln.wf01.wt01.status",
            count=5,
            start_time=0,
            end_time=4,
        )

        # 4. 按正索引获取第二条序列并校验元信息
        _assert_timeseries_metadata(
            tsdf[1],
            "root.ln.wf01.wt01.temperature",
            count=5,
            start_time=0,
            end_time=4,
        )

        # 5. 按负索引获取最后一条序列并校验元信息
        _assert_timeseries_metadata(
            tsdf[-1],
            "root.ln.wf02.wt02.status",
            count=5,
            start_time=0,
            end_time=4,
        )


# 用例 PYTHON-TREE-057：布尔过滤子集仍可取元信息
def test_tree_getitem_metadata_filtered_view_exposes_series_metadata(tmp_path):
    """用例 44、150：验证按元数据布尔过滤后的子集视图仍可获取 Timeseries 元信息。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 按设备名第二分段过滤，只保留 wf01 设备下的两条序列
        subset = tsdf[tsdf["_col_2"] == "wf01"]
        assert isinstance(subset, TsFileDataFrame)
        assert subset.list_timeseries() == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]

        # 4. 在过滤后的子集视图中按索引获取序列并校验元信息
        _assert_timeseries_metadata(
            subset[1],
            "root.ln.wf01.wt01.temperature",
            count=5,
            start_time=0,
            end_time=4,
        )

        # 5. 在过滤后的子集视图中按名称获取序列并校验元信息
        _assert_timeseries_metadata(
            subset["root.ln.wf01.wt01.status"],
            "root.ln.wf01.wt01.status",
            count=5,
            start_time=0,
            end_time=4,
        )

        # 6. 校验过滤视图的 metadata 仍只包含子集序列
        metadata = subset.list_timeseries_metadata()
        assert metadata.index.tolist() == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]
        assert metadata.loc["root.ln.wf01.wt01.temperature", "count"] == 5


# 用例 PYTHON-TREE-058：特殊设备名可按名称查看元信息
def test_tree_getitem_named_series_variants_expose_metadata(tmp_path):
    """用例 46、152：验证普通、中文、反引号、点号转义和大小写设备名序列均可按名称查看元信息。"""
    # 1. 生成覆盖多类序列名规则的树模型TsFile文件
    path = tmp_path / "getitem_named_series_variants.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db_01.device_02": [("sensor_03", TSDataType.DOUBLE)],
            "root.区域_1.设备1": [("温度_1", TSDataType.DOUBLE)],
            "root.`sg-1`.d1": [("`m-1`", TSDataType.DOUBLE)],
            "root.a.b": [("m.dot", TSDataType.DOUBLE)],
            "root.case.DeviceA": [("s1", TSDataType.DOUBLE)],
        },
    )

    # 2. 准备按名称读取时使用的完整序列路径
    expected_names = [
        "root.db_01.device_02.sensor_03",
        "root.区域_1.设备1.温度_1",
        "root.`sg-1`.d1.`m-1`",
        "root.a.b.m\\.dot",
        "root.case.DeviceA.s1",
    ]

    # 3. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == sorted(expected_names)

        # 4. 逐个按名称获取 Timeseries，并校验 name、len() 和 stats
        for name in expected_names:
            _assert_timeseries_metadata(
                tsdf[name],
                name,
                count=3,
                start_time=0,
                end_time=2,
            )


# 用例 PYTHON-TREE-059：切片子集视图
def test_tree_getitem_slice_variants_return_metadata_ready_subsets(tmp_path):
    """用例 51、153：验证前段、中段、尾段、步长和空切片子集视图均符合预期。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 准备不同切片段和对应的序列清单
    slice_cases = [
        (slice(0, 1), ["root.ln.wf01.wt01.status"]),
        (
            slice(1, 3),
            ["root.ln.wf01.wt01.temperature", "root.ln.wf02.wt02.status"],
        ),
        (
            slice(-2, None),
            ["root.ln.wf01.wt01.temperature", "root.ln.wf02.wt02.status"],
        ),
        (
            slice(None, None, 2),
            ["root.ln.wf01.wt01.status", "root.ln.wf02.wt02.status"],
        ),
        (slice(3, 3), []),
    ]

    # 3. 加载文件并逐个校验切片视图
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        for key, expected_names in slice_cases:
            subset = tsdf[key]
            assert isinstance(subset, TsFileDataFrame)
            assert len(subset) == len(expected_names)
            assert subset.list_timeseries() == expected_names

            # 4. 非空切片视图应可继续通过索引获取 Timeseries 元信息
            if expected_names:
                _assert_timeseries_metadata(
                    subset[0],
                    expected_names[0],
                    count=5,
                    start_time=0,
                    end_time=4,
                )


# 用例 PYTHON-TREE-060：列表子集顺序/负索引/越界报错
# ── 由 test_tree_getitem_list_variants_duplicates_and_errors 拆分而来（2 个）──
def test_tree_getitem_list_respects_requested_order(tmp_path):
    """用例 154：验证列表按请求索引顺序返回子集视图，并支持负索引混用后继续查看序列元信息。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 使用不同索引顺序获取列表子集，并校验顺序和元信息
        different_order = tsdf[[2, 0]]
        assert different_order.list_timeseries() == [
            "root.ln.wf02.wt02.status",
            "root.ln.wf01.wt01.status",
        ]
        _assert_timeseries_metadata(
            different_order[0],
            "root.ln.wf02.wt02.status",
            count=5,
            start_time=0,
            end_time=4,
        )

        # 4. 列表中混用负索引时，应映射到对应序列并保留请求顺序
        mixed_negative = tsdf[[0, -1]]
        assert mixed_negative.list_timeseries() == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf02.wt02.status",
        ]

        # 5. 负索引命中的序列仍可继续查看元信息
        _assert_timeseries_metadata(
            mixed_negative[1],
            "root.ln.wf02.wt02.status",
            count=5,
            start_time=0,
            end_time=4,
        )

def test_tree_getitem_repeated_index_view_and_invalid_entries(tmp_path):
    """用例 155：验证列表全为同一个索引时保留重复序列视图，列表含越界索引或非整数时稳定报错。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 列表全为同一个索引时，应保留重复序列视图
        repeated = tsdf[[1, 1, 1]]
        assert len(repeated) == 3
        assert repeated.list_timeseries() == [
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.temperature",
        ]
        for idx in range(len(repeated)):
            _assert_timeseries_metadata(
                repeated[idx],
                "root.ln.wf01.wt01.temperature",
                count=5,
                start_time=0,
                end_time=4,
            )

        # 4. 校验非法索引和非法列表元素的错误类型
        with pytest.raises(IndexError, match="out of range"):
            tsdf[-4]
        with pytest.raises(IndexError, match="out of range"):
            tsdf[[0, 99]]
        with pytest.raises(IndexError, match="out of range"):
            tsdf[[0, -4]]
        with pytest.raises(TypeError, match="List index must contain integers"):
            tsdf[[0, "bad"]]


def test_tree_timeseries_metadata_and_indexing(tmp_path):
    """用例 57-63、69、71、72：验证 Timeseries 的名称、长度、统计信息、时间戳、正反向切片和非法索引行为。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取指定 Timeseries 对象
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        ts = tsdf["root.ln.wf01.wt01.temperature"]

        # 3. 校验 Timeseries 基础元信息和展示内容
        assert len(ts) == 5
        assert ts.stats["count"] == 5
        assert "root.ln.wf01.wt01.temperature" in repr(ts)
        np.testing.assert_array_equal(ts.timestamps, np.arange(5, dtype=np.int64))

        # 4. 校验单点、切片、反向切片和空切片读取
        assert ts[1] == 1.5
        np.testing.assert_array_equal(ts[1:4], np.array([1.5, 2.5, 3.5]))
        np.testing.assert_array_equal(ts[::-1], np.array([4.5, 3.5, 2.5, 1.5, 0.5]))
        np.testing.assert_array_equal(ts[9:9], np.array([], dtype=np.float64))

        # 5. 校验越界索引和非法索引类型报错
        with pytest.raises(IndexError, match="out of range"):
            ts[99]
        with pytest.raises(TypeError, match="Unsupported key type"):
            ts["bad"]


# 用例 PYTHON-TREE-062：负索引切片与步长
def test_tree_timeseries_additional_slice_and_aligned_indexing_forms(tmp_path):
    """用例 64-68、84-86：补充验证负索引切片、步长采样、时间戳切片，以及 AlignedTimeseries 的索引访问。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取目标序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        ts = tsdf["root.ln.wf01.wt01.temperature"]

        # 3. 校验负索引切片、步长切片和时间戳切片
        np.testing.assert_array_equal(
            ts[-10:], np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        )
        np.testing.assert_array_equal(ts[::2], np.array([0.5, 2.5, 4.5]))
        np.testing.assert_array_equal(
            ts.timestamps[1:4], np.array([1, 2, 3], dtype=np.int64)
        )

        # 4. 执行 .loc 对齐读取，得到 AlignedTimeseries
        aligned = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]
        assert len(aligned) == 5

        # 5. 校验 AlignedTimeseries 的行索引、切片和二维索引访问
        np.testing.assert_array_equal(aligned[0], np.array([0.5, 0.0]))
        np.testing.assert_array_equal(
            aligned[0:2], np.array([[0.5, 0.0], [1.5, 1.0]])
        )
        assert aligned[0, 1] == 0.0


# 用例 PYTHON-TREE-063：timestamps缓存复用
def test_tree_timeseries_timestamps_are_cached_but_values_are_read_on_demand(tmp_path):
    """用例 70：验证 Timeseries.timestamps 会缓存复用，而值读取仍按请求走底层行读取。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取目标 Timeseries
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        ts = tsdf["root.ln.wf01.wt01.temperature"]
        timestamp_calls = []
        row_calls = []

        # 3. 保存原始加载函数，并准备调用计数 spy
        original_load_timestamps = ts._load_timestamps
        original_read_by_position = ts._read_by_position

        def load_timestamps_spy():
            timestamp_calls.append("timestamps")
            return original_load_timestamps()

        def read_by_position_spy(offset, limit):
            row_calls.append((offset, limit))
            return original_read_by_position(offset, limit)

        ts._load_timestamps = load_timestamps_spy
        ts._read_by_position = read_by_position_spy

        # 4. 连续读取 timestamps，确认只触发一次时间戳加载
        np.testing.assert_array_equal(ts.timestamps, np.arange(5, dtype=np.int64))
        np.testing.assert_array_equal(ts.timestamps, np.arange(5, dtype=np.int64))
        assert timestamp_calls == ["timestamps"]

        # 5. 连续读取同一值切片，确认值读取仍按请求触发底层读取
        np.testing.assert_array_equal(ts[1:3], np.array([1.5, 2.5]))
        np.testing.assert_array_equal(ts[1:3], np.array([1.5, 2.5]))
        assert row_calls == [(1, 2), (1, 2)]


# 用例 PYTHON-TREE-064：.loc对齐、单时间戳、负索引与参数校验
# ── 由 test_tree_loc_alignment_forms_and_validation 拆分而来（14 个）──
def test_tree_loc_range_query_returns_aligned_timeseries(tmp_path):
    """用例 73：验证 df.loc[start:end, [names]] 返回 AlignedTimeseries。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 执行带闭区间时间范围和序列名列表的 .loc 对齐查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        data = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]

        # 3. 校验返回类型为对齐结果对象，且保留请求的序列
        assert isinstance(data, AlignedTimeseries)
        assert data.series_names == [
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.status",
        ]

def test_tree_loc_column_order_follows_requested_series(tmp_path):
    """用例 75：验证结果列顺序严格按请求的序列顺序。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 先请求 status 再请求 temperature，结果列必须与该顺序一致
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        ordered = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.status", "root.ln.wf01.wt01.temperature"],
        ]
        assert ordered.series_names == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]
        np.testing.assert_array_equal(
            ordered.values[:, 0], np.arange(5, dtype=np.float64)
        )
        np.testing.assert_array_equal(
            ordered.values[:, 1], np.arange(5, dtype=np.float64) + 0.5
        )

        # 3. 调换请求顺序后各列随之调换，未被按序列名重新排序
        swapped = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]
        assert swapped.series_names == [
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.status",
        ]
        np.testing.assert_array_equal(
            swapped.values[:, 0], np.arange(5, dtype=np.float64) + 0.5
        )
        np.testing.assert_array_equal(
            swapped.values[:, 1], np.arange(5, dtype=np.float64)
        )

def test_tree_loc_accepts_mixed_name_and_index_specifiers(tmp_path):
    """用例 78：验证序列指定支持名称与整数索引混用。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 同一次 .loc 请求中混用序列名与整数索引
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        mixed = tsdf.loc[0:2, ["root.ln.wf01.wt01.temperature", 0]]
        assert mixed.series_names == [
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.status",
        ]

        # 3. 两种指定形式解析出的序列都能取到正确取值
        assert mixed.shape == (3, 2)
        np.testing.assert_array_equal(
            mixed.values, np.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0]])
        )

def test_tree_loc_single_timestamp_query(tmp_path):
    """用例 79：验证 df.loc[t, [...]] 单时间戳查询。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 以整数时间戳发起单点查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        single = tsdf.loc[1, [0, "root.ln.wf01.wt01.temperature"]]
        assert isinstance(single, AlignedTimeseries)
        assert single.shape == (1, 2)
        np.testing.assert_array_equal(single.timestamps, np.array([1], dtype=np.int64))

        # 3. 单点结果的取值来自请求时间戳所在行
        np.testing.assert_array_equal(single.values, np.array([[1.0, 1.5]]))

def test_tree_loc_open_time_slices_and_empty_range(tmp_path):
    """用例 80：验证时间片开区间 [:end]、[start:]、[:] 正常。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 分别使用左开、右开和全开时间片查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        name = "root.ln.wf01.wt01.temperature"
        np.testing.assert_array_equal(
            tsdf.loc[:2, [name]].timestamps, np.array([0, 1, 2], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            tsdf.loc[2:, [name]].timestamps, np.array([2, 3, 4], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            tsdf.loc[:, [name]].timestamps, np.arange(5, dtype=np.int64)
        )

        # 3. 时间范围未命中任何数据时返回空对齐结果且形状稳定
        empty = tsdf.loc[100:200, [name]]
        assert isinstance(empty, AlignedTimeseries)
        assert empty.shape == (0, 1)
        assert empty.series_names == [name]
        np.testing.assert_array_equal(empty.timestamps, np.array([], dtype=np.int64))

def test_tree_loc_negative_series_index(tmp_path):
    """用例 81：验证 df.loc[:, [-1]] 负序列索引。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 以负索引指定序列，校验解析到末条序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        negative = tsdf.loc[:, [-1]]
        assert isinstance(negative, AlignedTimeseries)
        assert negative.series_names == ["root.ln.wf02.wt02.status"]

        # 3. 取值确实来自末条序列（root.ln.wf02.wt02.status 的值为 t*2）
        np.testing.assert_array_equal(
            negative.timestamps, np.arange(5, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            negative.values[:, 0], np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        )

def test_tree_loc_repeated_series_specifiers_do_not_raise(tmp_path):
    """用例 82：验证同一序列被重复指定(名+索引/同名两次)时去重不报错。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 同一序列分别以名称与整数索引指定，内部去重后不报错
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        name = "root.ln.wf01.wt01.temperature"
        idx = tsdf.list_timeseries().index(name)
        by_name_and_index = tsdf.loc[0:2, [name, idx]]
        assert by_name_and_index.shape == (3, 2)
        np.testing.assert_array_equal(
            by_name_and_index.values, np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        )

        # 3. 同一个序列名重复两次同样不报错，两列取值一致
        repeated = tsdf.loc[0:2, [name, name]]
        assert repeated.shape == (3, 2)
        np.testing.assert_array_equal(repeated.values[:, 0], repeated.values[:, 1])

def test_tree_loc_aligned_result_exposes_attributes(tmp_path):
    """用例 84：验证 AlignedTimeseries 的 timestamps/values/series_names/shape 属性。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 执行对齐查询并逐个校验结果对象的公开属性
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        data = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]
        assert data.series_names == [
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.status",
        ]
        assert data.shape == (5, 2)
        np.testing.assert_array_equal(data.timestamps, np.arange(5, dtype=np.int64))
        np.testing.assert_array_equal(
            data.values,
            np.array([[0.5, 0.0], [1.5, 1.0], [2.5, 2.0], [3.5, 3.0], [4.5, 4.0]]),
        )

def test_tree_loc_aligned_result_row_count(tmp_path):
    """用例 85：验证 len(data) 返回行数 N。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 对齐结果的行数等于时间戳数量，也等于矩阵首维
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        data = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]
        assert len(data) == 5
        assert len(data) == len(data.timestamps)
        assert len(data) == data.shape[0]

        # 3. 缩短时间范围后行数随之变化
        trimmed = tsdf.loc[1:3, ["root.ln.wf01.wt01.temperature"]]
        assert len(trimmed) == 3

def test_tree_loc_aligned_result_indexing(tmp_path):
    """用例 86：验证 data[0]、data[0:10]、data[0,1] 索引。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 单行索引、行切片和行列二元索引均按矩阵语义取值
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        data = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]
        np.testing.assert_array_equal(data[0], np.array([0.5, 0.0]))
        assert data[0, 1] == 0.0
        assert data[4, 0] == 4.5

        # 3. 行切片范围超过实际行数时按可用行返回
        np.testing.assert_array_equal(data[0:10], data.values)
        assert data[0:10].shape == (5, 2)

def test_tree_loc_requires_two_arguments(tmp_path):
    """用例 87：验证 loc 传参不是 (time, series) 二元组时报错。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 只传时间片或传三个参数均抛 ValueError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(ValueError, match="requires exactly 2"):
            tsdf.loc[0]
        with pytest.raises(ValueError, match="requires exactly 2"):
            tsdf.loc[0:1]
        with pytest.raises(ValueError, match="requires exactly 2"):
            tsdf.loc[0:1, [0], [0]]

def test_tree_loc_invalid_time_index_raises_type_error(tmp_path):
    """用例 88：验证时间索引非 slice/int 时抛 TypeError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 字符串和列表作为时间索引均抛 TypeError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(TypeError, match="Time index must be slice or int"):
            tsdf.loc["bad", [0]]
        with pytest.raises(TypeError, match="Time index must be slice or int"):
            tsdf.loc[[0, 1], [0]]

def test_tree_loc_invalid_series_specifier_raises_type_error(tmp_path):
    """用例 89：验证序列指定含非 int/str 元素时抛 TypeError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 任意对象或浮点数作为序列指定元素均抛 TypeError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(TypeError, match="Series specifier must be int or str"):
            tsdf.loc[0:1, [object()]]
        with pytest.raises(TypeError, match="Series specifier must be int or str"):
            tsdf.loc[0:1, [0, 1.5]]

def test_tree_loc_out_of_range_series_index_raises(tmp_path):
    """用例 90：验证 loc 内序列整数索引越界抛 IndexError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 正向与负向越界的序列索引均抛 IndexError
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        with pytest.raises(IndexError, match="out of range"):
            tsdf.loc[0:1, [999]]
        with pytest.raises(IndexError, match="out of range"):
            tsdf.loc[0:1, [-99]]

        # 3. 系列名无法解析时的 KeyError 同属序列指定错误（原多验证点函数中的同名断言）
        with pytest.raises(KeyError, match="Series not found"):
            tsdf.loc[0:1, ["root.no.such"]]


# ── 由 test_tree_loc_sparse_union_open_ranges_and_requested_bounds 拆分而来（3 个）──
def test_tree_loc_union_aligns_sparse_timestamps_with_nan(tmp_path):
    """用例 74：验证多序列按时间戳并集对齐、缺失位置填 NaN。"""
    # 1. 生成两个设备时间戳不完全重合的稀疏树模型TsFile文件
    path = tmp_path / "sparse.tsfile"
    _write_tree_points(
        path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        [
            ("root.a.d1", 0, {"m1": 10.0}),
            ("root.a.d1", 1, {"m1": 11.0}),
            ("root.a.d2", 1, {"m1": 21.0}),
            ("root.a.d2", 2, {"m1": 22.0}),
        ],
    )

    # 2. 加载文件并执行跨设备 .loc 对齐查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[0:2, ["root.a.d1.m1", "root.a.d2.m1"]]

        # 3. 校验时间戳取两个序列时间轴的并集
        np.testing.assert_array_equal(aligned.timestamps, np.array([0, 1, 2]))

        # 4. 校验缺失位置用 NaN 补齐
        assert np.isnan(aligned.values[0, 1])
        assert np.isnan(aligned.values[2, 0])

def test_tree_loc_cross_device_timelines_align_by_timestamp(tmp_path):
    """用例 77：验证跨设备且时间轴不同的序列对齐，缺失填 NaN。"""
    # 1. 生成两个设备时间戳不完全重合的稀疏树模型TsFile文件
    path = tmp_path / "sparse.tsfile"
    _write_tree_points(
        path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        [
            ("root.a.d1", 0, {"m1": 10.0}),
            ("root.a.d1", 1, {"m1": 11.0}),
            ("root.a.d2", 1, {"m1": 21.0}),
            ("root.a.d2", 2, {"m1": 22.0}),
        ],
    )

    # 2. 加载文件并执行跨设备 .loc 对齐查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[0:2, ["root.a.d1.m1", "root.a.d2.m1"]]

        # 3. 校验各设备数值按各自时间轴落到对应列，跨设备不串值
        assert aligned.values[0, 0] == 10.0
        assert aligned.values[1, 0] == 11.0
        assert aligned.values[1, 1] == 21.0
        assert aligned.values[2, 1] == 22.0

        # 4. 校验时间轴不同留下的空位仍以 NaN 表示
        assert np.isnan(aligned.values[0, 1])

def test_tree_loc_alignment_stays_within_requested_bounds(tmp_path):
    """用例 83：验证带 NaN 时对齐结果不超出请求时间范围。"""
    # 1. 生成两个设备时间戳不完全重合的稀疏树模型TsFile文件
    path = tmp_path / "sparse.tsfile"
    _write_tree_points(
        path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        [
            ("root.a.d1", 0, {"m1": 10.0}),
            ("root.a.d1", 1, {"m1": 11.0}),
            ("root.a.d2", 1, {"m1": 21.0}),
            ("root.a.d2", 2, {"m1": 22.0}),
        ],
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验左开/右开/全开时间范围不会越过请求边界（原用例 80 的断言）
        np.testing.assert_array_equal(
            tsdf.loc[:1, ["root.a.d1.m1"]].timestamps, np.array([0, 1])
        )
        np.testing.assert_array_equal(
            tsdf.loc[1:, ["root.a.d2.m1"]].timestamps, np.array([1, 2])
        )
        np.testing.assert_array_equal(
            tsdf.loc[:, ["root.a.d1.m1"]].timestamps, np.array([0, 1])
        )

        # 4. 校验显式闭区间裁剪后的时间戳和值，缺失位置为 NaN 且不越界
        clipped = tsdf.loc[1:2, ["root.a.d1.m1", "root.a.d2.m1"]]
        np.testing.assert_array_equal(clipped.timestamps, np.array([1, 2]))
        assert np.isnan(clipped.values[1, 0])


def test_tree_loc_cross_device_alignment_preserves_values(tmp_path):
    """用例 76、77：验证跨设备对齐不会串值，两个设备同名测点的数值按各自设备保持正确。"""
    # 1. 生成包含两个设备同名 status 测点的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并跨设备对齐读取 status 测点
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.status", "root.ln.wf02.wt02.status"],
        ]

        # 3. 校验跨设备时间戳和值矩阵不串列
        np.testing.assert_array_equal(aligned.timestamps, np.arange(5, dtype=np.int64))
        np.testing.assert_array_equal(
            aligned.values,
            np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
        )


# [去 skip 裸跑] 已知缺陷：跨设备且物理量名不同时 .loc 对齐查询第二列返回 NaN（失败即缺陷证据；验证期再按 AGENTS.md 挂 xfail(strict=True)）
# 用例 PYTHON-TREE-067：跨设备不同物理量.loc读取真实值
def test_tree_loc_cross_device_different_measurements_preserves_values(tmp_path):
    """用例 76、77、130：回归验证跨设备且物理量名不同时 .loc 对齐查询仍能读取各自真实值。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 跨设备读取不同物理量名称，当前产品第二列返回 NaN
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf02.wt02.status"],
        ]

        # 3. 预期两个设备的真实值均能保留
        np.testing.assert_array_equal(aligned.timestamps, np.arange(5, dtype=np.int64))
        np.testing.assert_array_equal(
            aligned.values,
            np.array([[0.5, 0.0], [1.5, 2.0], [2.5, 4.0], [3.5, 6.0], [4.5, 8.0]]),
        )


# 用例 PYTHON-TREE-068：.loc重复序列去重保留列位置
# ── 由 test_tree_loc_dedups_repeated_series_specifiers 拆分而来（3 个）──
def test_tree_loc_dedups_name_and_index_specifier(tmp_path):
    """用例 126：验证 .loc 内同一序列用名称和整数索引重复指定时内部去重但输出保留请求列。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取目标序列名称和索引
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        name = "root.ln.wf01.wt01.temperature"
        idx = tsdf.list_timeseries().index(name)

        # 3. 使用名称和索引重复指定同一序列，确认输出保留两列且数值一致
        repeated = tsdf.loc[0:2, [name, idx]]
        assert repeated.shape == (3, 2)
        np.testing.assert_array_equal(
            repeated.values, np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        )

def test_tree_loc_repeated_series_name_keeps_requested_columns(tmp_path):
    """用例 127：验证 .loc 同一个序列名重复两次时不报错且结果列数符合请求。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取目标序列名称
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        name = "root.ln.wf01.wt01.temperature"

        # 3. 同一序列名重复两次，确认不报错且结果列数与请求一致
        duplicated = tsdf.loc[0:2, [name, name]]
        assert duplicated.shape == (3, 2)
        assert duplicated.series_names == [name, name]
        np.testing.assert_array_equal(
            duplicated.values, np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        )

def test_tree_loc_repeated_specifier_keeps_column_positions(tmp_path):
    """用例 128：验证重复序列夹在其它序列中时结果列顺序和重复列位置不被去重打乱。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取目标序列名称
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        name = "root.ln.wf01.wt01.temperature"

        # 3. 将重复序列夹在其它序列中，确认输出列顺序稳定、重复列位置不被去重打乱
        mixed = tsdf.loc[0:2, [name, "root.ln.wf01.wt01.status", name]]
        assert mixed.series_names == [
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]
        np.testing.assert_array_equal(
            mixed.values, np.array([[0.5, 0.0, 0.5], [1.5, 1.0, 1.5], [2.5, 2.0, 2.5]])
        )


def test_tree_loc_duplicate_specifiers_read_underlying_device_once(tmp_path, monkeypatch):
    """用例 129：验证重复序列查询不会重复触发底层设备读取，避免同一 shard 被多次扫描。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取底层 reader
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        reader = next(iter(tsdf._readers.values()))
        original_read = reader.read_device_fields_by_time_range
        calls = []

        # 3. 给底层时间范围读取方法打 spy，统计调用次数
        def read_spy(*args, **kwargs):
            calls.append(args)
            return original_read(*args, **kwargs)

        monkeypatch.setattr(reader, "read_device_fields_by_time_range", read_spy)

        # 4. 使用名称和索引重复请求同一序列
        name = "root.ln.wf01.wt01.temperature"
        idx = tsdf.list_timeseries().index(name)
        aligned = tsdf.loc[0:2, [name, idx, name]]

        # 5. 校验输出保留 3 列，但底层只读取一次
        assert aligned.shape == (3, 3)
        assert len(calls) == 1
        np.testing.assert_array_equal(aligned.values[:, 0], aligned.values[:, 1])
        np.testing.assert_array_equal(aligned.values[:, 0], aligned.values[:, 2])


# 用例 PYTHON-TREE-070：合法树路径均可参与.loc对齐查询
def test_tree_loc_named_series_variants_align_by_tree_path_rules(tmp_path):
    """用例 19-21、73、78：验证命名规则场景下，各类合法树路径均可参与 .loc 对齐查询。"""
    # 1. 生成覆盖普通、中文、反引号、点号转义和大小写设备名的树模型TsFile文件
    path = tmp_path / "loc_named_series_variants.tsfile"
    series_offsets = {
        ("root.db_01.device_02", "sensor_03"): 10.0,
        ("root.区域_1.设备1", "温度_1"): 20.0,
        ("root.`sg-1`.d1", "`m-1`"): 30.0,
        ("root.a.b", "m.dot"): 40.0,
        ("root.case.DeviceA", "s1"): 50.0,
        ("root.case.deviceA", "s1"): 60.0,
    }

    def value_func(device, name, dtype, t):
        return series_offsets[(device, name)] + float(t)

    _write_tree_rows(
        path,
        {
            "root.db_01.device_02": [("sensor_03", TSDataType.DOUBLE)],
            "root.区域_1.设备1": [("温度_1", TSDataType.DOUBLE)],
            "root.`sg-1`.d1": [("`m-1`", TSDataType.DOUBLE)],
            "root.a.b": [("m.dot", TSDataType.DOUBLE)],
            "root.case.DeviceA": [("s1", TSDataType.DOUBLE)],
            "root.case.deviceA": [("s1", TSDataType.DOUBLE)],
        },
        value_func=value_func,
    )

    requested = [
        "root.db_01.device_02.sensor_03",
        "root.区域_1.设备1.温度_1",
        "root.`sg-1`.d1.`m-1`",
        "root.a.b.m\\.dot",
        "root.case.DeviceA.s1",
        "root.case.deviceA.s1",
    ]

    # 2. 按完整序列名发起对齐查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[0:2, requested]

        # 3. 校验列顺序、时间戳和各命名规则路径下的值均保持正确
        assert isinstance(aligned, AlignedTimeseries)
        assert aligned.series_names == requested
        np.testing.assert_array_equal(aligned.timestamps, np.array([0, 1, 2]))
        np.testing.assert_array_equal(
            aligned.values,
            np.array(
                [
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
                    [12.0, 22.0, 32.0, 42.0, 52.0, 62.0],
                ]
            ),
        )


# 用例 PYTHON-TREE-071：元数据过滤子集可.loc对齐查询
def test_tree_loc_metadata_filtered_subset_aligns_series(tmp_path):
    """用例 44、73、150：验证元数据过滤得到的子集视图仍可执行 .loc 对齐查询。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 通过树模型设备名分段列过滤出 wf01 子集
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        subset = tsdf[tsdf["_col_2"] == "wf01"]
        aligned = subset.loc[
            0:4,
            [0, "root.ln.wf01.wt01.temperature"],
        ]

        # 3. 校验子集内索引和名称混用仍按请求顺序对齐
        assert aligned.series_names == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]
        np.testing.assert_array_equal(aligned.timestamps, np.arange(5, dtype=np.int64))
        np.testing.assert_array_equal(
            aligned.values,
            np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]),
        )


# 用例 PYTHON-TREE-072：跨文件并集按时间戳补齐NaN
def test_tree_loc_cross_file_field_union_aligns_missing_values(tmp_path):
    """用例 74、95、117-119：验证跨文件合并和物理量并集场景下 .loc 对齐会按时间戳补齐 NaN。"""
    # 1. 生成两个同设备分片：m1 跨文件延续，m2/m3 分别只存在于其中一个文件
    path1 = tmp_path / "part1.tsfile"
    path2 = tmp_path / "part2.tsfile"

    def value_func(device, name, dtype, t):
        if name == "m1":
            return float(t) + 0.1
        if name == "m2":
            return float(t) + 20.0
        if name == "m3":
            return float(t) + 30.0
        return float(t)

    _write_tree_rows(
        path1,
        {"root.merge.d": [("m1", TSDataType.DOUBLE), ("m2", TSDataType.DOUBLE)]},
        t_start=0,
        value_func=value_func,
    )
    _write_tree_rows(
        path2,
        {"root.merge.d": [("m1", TSDataType.DOUBLE), ("m3", TSDataType.DOUBLE)]},
        t_start=3,
        value_func=value_func,
    )

    # 2. 跨两个文件读取物理量并集
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        aligned = tsdf.loc[
            0:5,
            ["root.merge.d.m1", "root.merge.d.m2", "root.merge.d.m3"],
        ]

        # 3. m1 合并两个时间分片，m2/m3 在缺失时间戳位置填 NaN
        np.testing.assert_array_equal(aligned.timestamps, np.arange(6, dtype=np.int64))
        np.testing.assert_allclose(
            aligned.values,
            np.array(
                [
                    [0.1, 20.0, np.nan],
                    [1.1, 21.0, np.nan],
                    [2.1, 22.0, np.nan],
                    [3.1, np.nan, 33.0],
                    [4.1, np.nan, 34.0],
                    [5.1, np.nan, 35.0],
                ]
            ),
            equal_nan=True,
        )


# 用例 PYTHON-TREE-073：同名序列按时间分片合并并累加count
# ── 由 test_tree_merges_identical_series_across_files 拆分而来（3 个）──
def test_tree_same_series_across_shards_merges_into_one(tmp_path):
    """用例 91：验证多文件含同名序列(不同日期分片)时自动合并为一条连续序列。"""
    # 1. 生成两个包含同名序列、时间范围不重叠的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=10)

    # 2. 按文件列表加载并确认同名序列合并为一条连续逻辑序列
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.a.b.m1"]

        # 3. 校验合并后的长度和跨分片连续的时间轴
        ts = tsdf["root.a.b.m1"]
        assert len(ts) == 6
        np.testing.assert_array_equal(
            ts.timestamps, np.array([0, 1, 2, 10, 11, 12], dtype=np.int64)
        )

def test_tree_merged_series_count_sums_shard_counts(tmp_path):
    """用例 92：验证合并后元数据 count 为各分片点数之和。"""
    # 1. 生成两个包含同名序列、时间范围不重叠的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=10)

    # 2. 分别加载两个分片，记录各分片点数
    with TsFileDataFrame(str(path1), show_progress=False) as first:
        first_count = first.list_timeseries_metadata().loc["root.a.b.m1", "count"]
    with TsFileDataFrame(str(path2), show_progress=False) as second:
        second_count = second.list_timeseries_metadata().loc["root.a.b.m1", "count"]
    assert first_count == 3
    assert second_count == 3

    # 3. 合并加载后确认元数据 count 为各分片点数之和
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        merged_count = tsdf.list_timeseries_metadata().loc["root.a.b.m1", "count"]
        assert merged_count == first_count + second_count == 6

def test_tree_same_device_and_measurement_across_files_merge(tmp_path):
    """用例 94：验证两树文件相同设备名和物理量名称、时间不相交时合并为一条序列。"""
    # 1. 生成两个包含同名序列、时间范围不重叠的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=10)

    # 2. 分别加载两个文件，确认设备名和物理量名称完全相同
    with TsFileDataFrame(str(path1), show_progress=False) as only_first:
        assert only_first.list_timeseries() == ["root.a.b.m1"]
    with TsFileDataFrame(str(path2), show_progress=False) as only_second:
        assert only_second.list_timeseries() == ["root.a.b.m1"]

    # 3. 同时加载两个文件，确认时间不相交时合并为一条序列并覆盖两个分片的时间段
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.a.b.m1"]
        ts = tsdf["root.a.b.m1"]
        assert len(ts) == 6
        np.testing.assert_array_equal(
            ts.timestamps, np.array([0, 1, 2, 10, 11, 12], dtype=np.int64)
        )


def test_tree_duplicate_timestamps_across_shards_raise(tmp_path):
    """用例 93：验证跨分片出现重复时间戳时，构建索引阶段即抛出明确错误。"""
    # 1. 生成两个同名序列且时间戳重叠的树模型TsFile文件
    path1 = tmp_path / "first.tsfile"
    path2 = tmp_path / "second.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=2)

    # 2. 加载文件，构建索引阶段即抛出重复时间戳错误
    with pytest.raises(ValueError, match="Duplicate timestamp 2.*across TsFile shards"):
        TsFileDataFrame([str(path1), str(path2)], show_progress=False)


# 用例 PYTHON-TREE-075：重叠时间范围分片流式归并读取
def test_tree_overlap_slice_uses_row_stream_without_full_timestamp_merge(
    tmp_path, monkeypatch
):
    """用例 97：验证重叠时间范围分片按行流式归并读取，不依赖全量时间戳物化路径。"""
    # 1. 生成两个同名序列、时间交错但不重复的树模型TsFile文件
    path1 = tmp_path / "even.tsfile"
    path2 = tmp_path / "odd.tsfile"
    _write_tree_points(
        path1,
        {"root.a.b": [("m1", TSDataType.DOUBLE)]},
        [
            ("root.a.b", 0, {"m1": 0.0}),
            ("root.a.b", 2, {"m1": 2.0}),
            ("root.a.b", 4, {"m1": 4.0}),
        ],
    )
    _write_tree_points(
        path2,
        {"root.a.b": [("m1", TSDataType.DOUBLE)]},
        [
            ("root.a.b", 1, {"m1": 1.0}),
            ("root.a.b", 3, {"m1": 3.0}),
            ("root.a.b", 5, {"m1": 5.0}),
        ],
    )

    # 2. 加载文件并替换读取方法，禁止走全量时间戳合并路径
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        row_calls = []
        for reader in tsdf._readers.values():
            original_read_by_row = reader.read_series_by_row

            # 3. 给按行读取方法打 spy，并让按时间范围读取快速失败
            def row_spy(*args, _original=original_read_by_row, **kwargs):
                row_calls.append(args)
                return _original(*args, **kwargs)

            def forbidden_time_merge(*args, **kwargs):
                raise AssertionError("full timestamp merge should not be used")

            monkeypatch.setattr(reader, "read_series_by_row", row_spy)
            monkeypatch.setattr(reader, "read_series_by_ref", forbidden_time_merge)

        # 4. 按行切片读取，确认结果正确且确实走了按行读取
        np.testing.assert_array_equal(
            tsdf["root.a.b.m1"][0:4], np.array([0.0, 1.0, 2.0, 3.0])
        )
        assert row_calls


# 用例 PYTHON-TREE-076：多文件字段并集扩展
# ── 由 test_tree_unions_field_subsets_and_preserves_local_field_mapping 拆分而来（5 个）──
def test_tree_union_takes_field_name_union_across_files(tmp_path):
    """用例 95：验证同设备名不同物理量名称子集跨文件时取物理量名称并集。"""
    # 1. 准备两个字段集合不同但设备路径相同的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"

    # 2. 定义按测点名生成不同值的函数，方便验证字段不串列
    def values(_device, name, _dtype, t):
        if name == "m1":
            return 100.0 + t
        if name == "m2":
            return 200.0 + t
        return 300.0 + t

    # 3. 写入第一个文件，包含 m1 和 m3
    _write_tree_rows(
        path1,
        {"root.a.b": [("m1", TSDataType.DOUBLE), ("m3", TSDataType.DOUBLE)]},
        t_start=0,
        value_func=values,
    )

    # 4. 写入第二个文件，包含 m2 和 m1，顺序与第一个文件不同
    _write_tree_rows(
        path2,
        {"root.a.b": [("m2", TSDataType.DOUBLE), ("m1", TSDataType.DOUBLE)]},
        t_start=10,
        value_func=values,
    )

    # 5. 加载两个文件，校验序列清单取两个文件的物理量名称并集
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == [
            "root.a.b.m1",
            "root.a.b.m2",
            "root.a.b.m3",
        ]

def test_tree_union_field_columns_expand_in_first_seen_order(tmp_path):
    """用例 117：验证多个树模型文件物理量名称集合不同时全局物理量名称列表按首次出现顺序扩展。"""
    # 1. 准备两个字段集合不同但设备路径相同的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"

    # 2. 定义按测点名生成不同值的函数，方便验证字段不串列
    def values(_device, name, _dtype, t):
        if name == "m1":
            return 100.0 + t
        if name == "m2":
            return 200.0 + t
        return 300.0 + t

    # 3. 写入第一个文件，包含 m1 和 m3
    _write_tree_rows(
        path1,
        {"root.a.b": [("m1", TSDataType.DOUBLE), ("m3", TSDataType.DOUBLE)]},
        t_start=0,
        value_func=values,
    )

    # 4. 写入第二个文件，包含 m2 和 m1，顺序与第一个文件不同
    _write_tree_rows(
        path2,
        {"root.a.b": [("m2", TSDataType.DOUBLE), ("m1", TSDataType.DOUBLE)]},
        t_start=10,
        value_func=values,
    )

    # 5. 加载两个文件，校验全局物理量名称列表按首次出现顺序扩展
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert tsdf._index.table_entries["root"].field_columns == ("m1", "m3", "m2")

def test_tree_union_maps_global_field_index_to_local_shard(tmp_path):
    """用例 118：验证多树文件物理量名称顺序或子集不同时按物理量名称映射全局索引。"""
    # 1. 准备两个字段集合不同但设备路径相同的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"

    # 2. 定义按测点名生成不同值的函数，方便验证字段不串列
    def values(_device, name, _dtype, t):
        if name == "m1":
            return 100.0 + t
        if name == "m2":
            return 200.0 + t
        return 300.0 + t

    # 3. 写入第一个文件，包含 m1 和 m3
    _write_tree_rows(
        path1,
        {"root.a.b": [("m1", TSDataType.DOUBLE), ("m3", TSDataType.DOUBLE)]},
        t_start=0,
        value_func=values,
    )

    # 4. 写入第二个文件，包含 m2 和 m1，顺序与第一个文件不同
    _write_tree_rows(
        path2,
        {"root.a.b": [("m2", TSDataType.DOUBLE), ("m1", TSDataType.DOUBLE)]},
        t_start=10,
        value_func=values,
    )

    # 5. 加载两个文件，校验全局索引按物理量名称映射到各自分片的局部字段
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        m2_ref = tsdf._resolve_series_name("root.a.b.m2")
        assert m2_ref == (0, 2)
        shards = tsdf._index.series_shards[m2_ref]
        assert len(shards) == 1
        reader, device_id, local_field_idx = shards[0]
        assert (
            reader.get_series_info_by_ref(device_id, local_field_idx)["column_name"]
            == "m2"
        )

        # 6. 分别读取 m1/m2，确认 reader-local 字段索引映射正确
        np.testing.assert_array_equal(
            tsdf["root.a.b.m1"][:],
            np.array([100.0, 101.0, 102.0, 110.0, 111.0, 112.0]),
        )
        np.testing.assert_array_equal(
            tsdf["root.a.b.m2"][:], np.array([210.0, 211.0, 212.0])
        )

def test_tree_union_of_field_subsets_loads_instead_of_rejecting(tmp_path):
    """用例 119：验证同一设备名在不同树文件中拥有不同物理量名称子集时使用物理量名称并集而非拒绝加载。"""
    # 1. 准备两个字段集合不同但设备路径相同的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"

    # 2. 定义按测点名生成不同值的函数，方便验证字段不串列
    def values(_device, name, _dtype, t):
        if name == "m1":
            return 100.0 + t
        if name == "m2":
            return 200.0 + t
        return 300.0 + t

    # 3. 写入第一个文件，包含 m1 和 m3
    _write_tree_rows(
        path1,
        {"root.a.b": [("m1", TSDataType.DOUBLE), ("m3", TSDataType.DOUBLE)]},
        t_start=0,
        value_func=values,
    )

    # 4. 写入第二个文件，包含 m2 和 m1，顺序与第一个文件不同
    _write_tree_rows(
        path2,
        {"root.a.b": [("m2", TSDataType.DOUBLE), ("m1", TSDataType.DOUBLE)]},
        t_start=10,
        value_func=values,
    )

    # 5. 加载两个文件，校验两个文件的物理量名称子集按并集合并而不是拒绝加载
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == [
            "root.a.b.m1",
            "root.a.b.m2",
            "root.a.b.m3",
        ]

        # 6. 只存在于第一个文件的物理量仍可按名称读取
        np.testing.assert_array_equal(
            tsdf["root.a.b.m3"][:], np.array([300.0, 301.0, 302.0])
        )

def test_tree_union_registers_only_fields_owned_by_device(tmp_path):
    """用例 125：验证树模型内部元数据合并存在物理量名称并集时只注册设备名真实拥有的物理量。"""
    # 1. 准备两个字段集合不同但设备路径相同的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"

    # 2. 定义按测点名生成不同值的函数，方便验证字段不串列
    def values(_device, name, _dtype, t):
        if name == "m1":
            return 100.0 + t
        if name == "m2":
            return 200.0 + t
        return 300.0 + t

    # 3. 写入第一个文件，包含 m1 和 m3
    _write_tree_rows(
        path1,
        {"root.a.b": [("m1", TSDataType.DOUBLE), ("m3", TSDataType.DOUBLE)]},
        t_start=0,
        value_func=values,
    )

    # 4. 写入第二个文件，包含 m2 和 m1，顺序与第一个文件不同
    _write_tree_rows(
        path2,
        {"root.a.b": [("m2", TSDataType.DOUBLE), ("m1", TSDataType.DOUBLE)]},
        t_start=10,
        value_func=values,
    )

    # 5. 加载两个文件，校验逻辑序列数等于并集物理量数而不是文件数乘物理量数
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert len(tsdf) == 3

        # 6. 每个逻辑序列只登记真正持有它的分片，未拥有的物理量不会挂到别的文件上
        expected_shards = {"root.a.b.m1": 2, "root.a.b.m2": 1, "root.a.b.m3": 1}
        for name, expected_count in expected_shards.items():
            shards = tsdf._index.series_shards[tsdf._resolve_series_name(name)]
            assert len(shards) == expected_count
            field_name = name.split(".")[-1]
            for reader, device_id, local_field_idx in shards:
                assert (
                    reader.get_series_info_by_ref(device_id, local_field_idx)[
                        "column_name"
                    ]
                    == field_name
                )


# ── 由 test_tree_unions_different_depths_and_short_paths_stay_resolvable 拆分而来（3 个）──
def test_tree_union_pads_shallow_device_segments_with_nan(tmp_path):
    """用例 96：验证不同最大设备名深度的树文件合并时全局设备名分段布局取并集、较浅设备名补 None。"""
    # 1. 生成浅层设备和深层设备两个树模型TsFile文件
    path1 = tmp_path / "shallow.tsfile"
    path2 = tmp_path / "deep.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]})
    _write_tree_rows(path2, {"root.a.b.c": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载两个文件并获取全局元数据
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        meta = tsdf.list_timeseries_metadata()

        # 3. 校验全局布局取并集扩展到 _col_3，浅层设备名按左对齐占用分段列、末尾补 None
        assert "_col_3" in meta.columns
        assert meta.loc["root.a.b.m1", "_col_1"] == "a"
        assert meta.loc["root.a.b.m1", "_col_2"] == "b"
        assert pd.isna(meta.loc["root.a.b.m1", "_col_3"])

def test_tree_union_extends_segment_layout_to_deepest_device(tmp_path):
    """用例 120：验证多树文件设备名深度不同时全局设备名分段布局扩展到最深设备名。"""
    # 1. 生成浅层设备和深层设备两个树模型TsFile文件
    path1 = tmp_path / "shallow.tsfile"
    path2 = tmp_path / "deep.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]})
    _write_tree_rows(path2, {"root.a.b.c": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载两个文件并校验不同深度的序列都可识别
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == ["root.a.b.c.m1", "root.a.b.m1"]

        # 3. 校验全局设备名分段布局扩展到最深设备名，最深分段落在 _col_3
        meta = tsdf.list_timeseries_metadata()
        assert "_col_3" in meta.columns
        assert meta.loc["root.a.b.c.m1", "_col_3"] == "c"

def test_tree_union_short_path_stays_resolvable(tmp_path):
    """用例 121：验证不同深度 union 后浅层设备名补 NaN 只用于展示和元数据不进入逻辑路径解析。"""
    # 1. 生成浅层设备和深层设备两个树模型TsFile文件
    path1 = tmp_path / "shallow.tsfile"
    path2 = tmp_path / "deep.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]})
    _write_tree_rows(path2, {"root.a.b.c": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载两个文件
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        # 3. 校验短路径仍可按原路径读取值，补齐的 None 不进入逻辑路径解析
        values = tsdf["root.a.b.m1"][:]
        np.testing.assert_array_equal(values, np.array([0.5, 1.5, 2.5]))
        assert not np.isnan(values).any()

        # 4. 校验前缀 root.a.b 能同时命中浅层和深层序列
        assert sorted(tsdf.list_timeseries("root.a.b")) == [
            "root.a.b.c.m1",
            "root.a.b.m1",
        ]


def test_tree_union_repr_and_show_keep_tree_headers(tmp_path, capsys):
    """用例 134：验证多文件 union 后 repr/show 仍使用树模型 _col_i 表头，不出现 table 列。"""
    # 1. 生成不同设备深度的两个树模型TsFile文件
    shallow = tmp_path / "shallow.tsfile"
    deep = tmp_path / "deep.tsfile"
    _write_tree_rows(shallow, {"root.a.b": [("m1", TSDataType.DOUBLE)]})
    _write_tree_rows(deep, {"root.a.b.c": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载两个文件并获取 repr 展示内容
    with TsFileDataFrame([str(shallow), str(deep)], show_progress=False) as tsdf:
        rendered = repr(tsdf)

        # 3. 校验 repr 使用树模型列头，不出现 table 列
        assert "TsFileDataFrame(tree model" in rendered
        assert "_col_1" in rendered and "_col_2" in rendered and "_col_3" in rendered
        assert "table" not in rendered.splitlines()[1]

        # 4. 调用 show 并校验输出仍保留树模型列头
        tsdf.show(max_rows=20)
        shown = capsys.readouterr().out
        assert "_col_3" in shown
        assert "table" not in shown.splitlines()[1]


# 用例 PYTHON-TREE-079：非数值测点不暴露，NaN保留
# ── 由 test_tree_omits_non_numeric_measurements_and_keeps_nan_values 拆分而来（3 个）──
def test_tree_numeric_measurements_read_as_float64(tmp_path):
    """用例 98：验证 BOOLEAN、INT32、INT64、FLOAT、DOUBLE、TIMESTAMP 树模型物理量均可由数值面读取并统一输出 float64。"""

    # 1. 生成同时包含六种数值物理量的树模型TsFile文件
    path = tmp_path / "numeric_types.tsfile"

    def value_func(device, name, dtype, t):
        if name == "flag":
            return bool(t % 2)
        if name == "epoch":
            return 1000 + t
        return _default_value(name, dtype, t)

    _write_tree_rows(
        path,
        {
            "root.types.dev": [
                ("flag", TSDataType.BOOLEAN),
                ("i32_m", TSDataType.INT32),
                ("i64_m", TSDataType.INT64),
                ("f32_m", TSDataType.FLOAT),
                ("f64_m", TSDataType.DOUBLE),
                ("epoch", TSDataType.TIMESTAMP),
            ]
        },
        value_func=value_func,
    )

    # 2. 加载文件，确认六种数值物理量都暴露为序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == [
            "root.types.dev.epoch",
            "root.types.dev.f32_m",
            "root.types.dev.f64_m",
            "root.types.dev.flag",
            "root.types.dev.i32_m",
            "root.types.dev.i64_m",
        ]

        # 3. 校验各类型数值面读数并统一输出 float64
        expected_values = {
            "flag": [0.0, 1.0, 0.0],
            "i32_m": [0.0, 1.0, 2.0],
            "i64_m": [0.0, 10.0, 20.0],
            "f32_m": [0.5, 1.5, 2.5],
            "f64_m": [0.5, 1.5, 2.5],
            "epoch": [1000.0, 1001.0, 1002.0],
        }
        for name, expected in expected_values.items():
            values = tsdf[f"root.types.dev.{name}"][:]
            assert values.dtype == np.float64
            np.testing.assert_array_equal(values, np.array(expected, dtype=np.float64))

def test_tree_omits_non_numeric_measurements(tmp_path):
    """用例 99：验证 TEXT、DATE、BLOB、STRING 树模型物理量均被过滤、不作为序列暴露。"""
    import datetime

    # 1. 手动生成同时包含数值测点和非数值测点的树模型TsFile文件
    path = tmp_path / "mixed.tsfile"
    writer = TsFileWriter(str(path))
    writer.register_timeseries("root.a.b", TimeseriesSchema("temp", TSDataType.DOUBLE))
    writer.register_timeseries("root.a.b", TimeseriesSchema("note", TSDataType.TEXT))
    writer.register_timeseries("root.a.b", TimeseriesSchema("status", TSDataType.STRING))
    writer.register_timeseries("root.a.b", TimeseriesSchema("day", TSDataType.DATE))
    writer.register_timeseries("root.a.b", TimeseriesSchema("blob", TSDataType.BLOB))

    # 2. 写入数值测点和各非数值测点
    for t, value in enumerate([0.5, float("nan"), 2.5]):
        writer.write_row_record(
            RowRecord(
                "root.a.b",
                t,
                [
                    Field("temp", value, TSDataType.DOUBLE),
                    Field("note", f"note-{t}", TSDataType.TEXT),
                    Field("status", "ok", TSDataType.STRING),
                    Field("day", datetime.date(2024, 1, 1 + t), TSDataType.DATE),
                    Field("blob", b"\x01\x02", TSDataType.BLOB),
                ],
            )
        )
    writer.close()

    # 3. 加载文件并确认只暴露数值测点，非数值测点均不作为序列暴露
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.a.b.temp"]
        for measurement in ("note", "status", "day", "blob"):
            with pytest.raises(KeyError):
                tsdf[f"root.a.b.{measurement}"]

def test_tree_nan_values_are_preserved(tmp_path):
    """用例 100：验证空值以 NaN 保留、不被丢弃或改写。"""
    # 1. 手动生成包含空值的数值测点树模型TsFile文件
    path = tmp_path / "nan.tsfile"
    writer = TsFileWriter(str(path))
    writer.register_timeseries("root.a.b", TimeseriesSchema("temp", TSDataType.DOUBLE))
    for t, value in enumerate([0.5, float("nan"), 2.5]):
        writer.write_row_record(
            RowRecord("root.a.b", t, [Field("temp", value, TSDataType.DOUBLE)])
        )
    writer.close()

    # 2. 读取数值测点，确认 NaN 不被丢弃：长度和时间轴均覆盖空值所在时刻
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        ts = tsdf["root.a.b.temp"]
        assert len(ts) == 3
        np.testing.assert_array_equal(ts.timestamps, np.array([0, 1, 2], dtype=np.int64))

        # 3. 校验空值原样保留为 NaN，前后数值不被改写
        values = ts[:]
        assert values[0] == 0.5
        assert np.isnan(values[1])
        assert values[2] == 2.5


def test_tree_escaped_measurement_name_can_be_resolved(tmp_path):
    """用例 18：验证测点名包含点号时会转义展示，并可使用转义后的逻辑路径读取。"""
    # 1. 生成测点名包含点号的树模型TsFile文件
    path = tmp_path / "escaped.tsfile"
    _write_tree_rows(path, {"root.a.b": [("m.dot", TSDataType.DOUBLE)]})

    # 2. 加载文件并校验 list_timeseries 中的点号被转义
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.a.b.m\\.dot"]

        # 3. 使用转义后的逻辑路径读取原始测点值
        np.testing.assert_array_equal(
            tsdf["root.a.b.m\\.dot"][:], np.array([0.5, 1.5, 2.5])
        )


# 用例 PYTHON-TREE-081：IoTDB合法节点按路径规则展示读取
def test_tree_iotdb_unquoted_identifier_names_are_preserved(tmp_path):
    """用例 19：验证 IoTDB 非反引号合法节点按 root + 设备路径 + 物理量规则展示并读取。"""
    # 1. 生成包含字母、数字、下划线、中文和最短设备路径的树模型TsFile文件
    path = tmp_path / "iotdb_unquoted_names.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db_01.device_02": [("sensor_03", TSDataType.DOUBLE)],
            "root.区域_1.设备1": [("温度_1", TSDataType.DOUBLE)],
            "root.sg2": [("pressure", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验序列名遵循 root.{seg_1}...{seg_k}.{field} 扁平路径规则
        assert sorted(tsdf.list_timeseries()) == [
            "root.db_01.device_02.sensor_03",
            "root.sg2.pressure",
            "root.区域_1.设备1.温度_1",
        ]

        # 4. 校验 k=1 的最短设备路径拆分为一个 _col_i 和一个 field
        meta = tsdf.list_timeseries_metadata()
        assert meta.loc["root.sg2.pressure", "_col_1"] == "sg2"
        assert meta.loc["root.sg2.pressure", "field"] == "pressure"

        # 5. 按完整序列路径读取，确认合法节点名不会影响数据读取
        np.testing.assert_array_equal(
            tsdf["root.db_01.device_02.sensor_03"][:], np.array([0.5, 1.5, 2.5])
        )
        np.testing.assert_array_equal(
            tsdf["root.区域_1.设备1.温度_1"][:], np.array([0.5, 1.5, 2.5])
        )


# 用例 PYTHON-TREE-082：反引号特殊节点原样展示可读取
def test_tree_iotdb_backquoted_identifier_names_are_preserved(tmp_path):
    """用例 20：验证 IoTDB 反引号包裹的特殊路径节点会原样展示并可按完整路径读取。"""
    # 1. 生成包含反引号特殊节点的树模型TsFile文件
    path = tmp_path / "iotdb_quoted_names.tsfile"
    _write_tree_rows(
        path,
        {
            "root.`sg-1`.d1": [("`m-1`", TSDataType.DOUBLE)],
            "root.sg.`123`": [("`456`", TSDataType.DOUBLE)],
            "root.`sg 1`.d1": [("`m 1`", TSDataType.DOUBLE)],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验反引号节点在序列名中原样保留
        expected = [
            "root.`sg 1`.d1.`m 1`",
            "root.`sg-1`.d1.`m-1`",
            "root.sg.`123`.`456`",
        ]
        assert sorted(tsdf.list_timeseries()) == expected

        # 4. 校验元数据中的设备路径段和 field 保留反引号文本
        meta = tsdf.list_timeseries_metadata()
        assert meta.loc["root.`sg-1`.d1.`m-1`", "_col_1"] == "`sg-1`"
        assert meta.loc["root.`sg-1`.d1.`m-1`", "field"] == "`m-1`"
        assert meta.loc["root.sg.`123`.`456`", "_col_2"] == "`123`"
        assert meta.loc["root.sg.`123`.`456`", "field"] == "`456`"

        # 5. 使用反引号完整路径读取数据
        for name in expected:
            np.testing.assert_array_equal(
                tsdf[name][:], np.array([0.5, 1.5, 2.5])
            )


# 用例 PYTHON-TREE-083：路径大小写敏感，不同视为不同序列
def test_tree_iotdb_identifier_names_are_case_sensitive(tmp_path):
    """用例 21：验证 IoTDB 路径节点大小写敏感，设备路径段和物理量名大小写不同均视为不同序列。"""
    # 1. 生成包含大小写差异设备路径段的树模型TsFile文件
    device_path = tmp_path / "iotdb_case_sensitive_device_names.tsfile"

    def device_value_func(device, name, dtype, t):
        if device == "root.case.DeviceA":
            return float(10 + t)
        if device == "root.case.deviceA":
            return float(20 + t)
        return float(t)

    _write_tree_rows(
        device_path,
        {
            "root.case.DeviceA": [("s1", TSDataType.DOUBLE)],
            "root.case.deviceA": [("s1", TSDataType.DOUBLE)],
        },
        value_func=device_value_func,
    )

    # 2. 加载设备路径大小写文件
    with TsFileDataFrame(str(device_path), show_progress=False) as tsdf:
        # 3. 校验大小写不同的设备路径段均作为独立序列返回
        assert sorted(tsdf.list_timeseries()) == [
            "root.case.DeviceA.s1",
            "root.case.deviceA.s1",
        ]
        assert len(tsdf) == 2

        # 4. 校验元数据保留设备路径段原始大小写
        meta = tsdf.list_timeseries_metadata()
        assert meta.loc["root.case.DeviceA.s1", "_col_2"] == "DeviceA"
        assert meta.loc["root.case.deviceA.s1", "_col_2"] == "deviceA"

        # 5. 按大小写不同的设备完整路径分别读取，确认不会合并或串值
        np.testing.assert_array_equal(
            tsdf["root.case.DeviceA.s1"][:], np.array([10.0, 11.0, 12.0])
        )
        np.testing.assert_array_equal(
            tsdf["root.case.deviceA.s1"][:], np.array([20.0, 21.0, 22.0])
        )

    # 6. 生成包含大小写差异物理量名的树模型TsFile文件
    field_path = tmp_path / "iotdb_case_sensitive_field_names.tsfile"
    _write_case_sensitive_measurement_file(field_path)

    # 7. 加载物理量名大小写文件
    with TsFileDataFrame(str(field_path), show_progress=False) as tsdf:
        print(tsdf)
        assert sorted(tsdf.list_timeseries()) == [
            "root.case.d1.Temperature",
            "root.case.d1.temperature",
        ]
        meta = tsdf.list_timeseries_metadata()
        assert meta.loc["root.case.d1.Temperature", "field"] == "Temperature"
        assert meta.loc["root.case.d1.temperature", "field"] == "temperature"
        # 8. 读取小写的物理量序列名
        np.testing.assert_array_equal(
            tsdf["root.case.d1.temperature"][:], np.array([30.0, 31.0, 32.0])
        )


# [去 skip 裸跑] 已知缺陷：大写物理量 root.case.d1.Temperature 按完整路径读取为空（失败即缺陷证据；验证期再按 AGENTS.md 挂 xfail(strict=True)）
# 用例 PYTHON-TREE-084：大写物理量名称读取
def test_tree_iotdb_uppercase_measurement_can_be_read_by_full_path(tmp_path):
    """用例 21：回归验证 IoTDB 合法大写物理量名称可按完整路径读取真实数据。"""
    # 1. 生成同一设备下大小写不同物理量名的树模型TsFile文件
    field_path = tmp_path / "iotdb_case_sensitive_field_names.tsfile"
    _write_case_sensitive_measurement_file(field_path)

    # 2. 按完整路径读取大写物理量，当前产品返回空数组，先以 strict xfail 保留缺陷信号
    with TsFileDataFrame(str(field_path), show_progress=False) as tsdf:
        np.testing.assert_array_equal(
            tsdf["root.case.d1.Temperature"][:], np.array([40.0, 41.0, 42.0])
        )


# [去 skip 裸跑] 已知缺陷：大写物理量 root.case.d1.Temperature 参与 .loc 对齐查询返回空结果（失败即缺陷证据；验证期再按 AGENTS.md 挂 xfail(strict=True)）
# 用例 PYTHON-TREE-085：大写物理量可参与.loc对齐查询
def test_tree_loc_uppercase_measurement_can_align_by_full_path(tmp_path):
    """用例 21、73：回归验证 IoTDB 合法大写物理量名称可参与 .loc 对齐查询。"""
    # 1. 生成同一设备下大小写不同物理量名的树模型TsFile文件
    field_path = tmp_path / "iotdb_case_sensitive_field_names.tsfile"
    _write_case_sensitive_measurement_file(field_path)

    # 2. 同时对齐读取小写和大写物理量，当前产品返回空对齐结果
    with TsFileDataFrame(str(field_path), show_progress=False) as tsdf:
        aligned = tsdf.loc[
            0:2,
            ["root.case.d1.temperature", "root.case.d1.Temperature"],
        ]

        # 3. 预期两条大小写不同的物理量均按完整路径取到真实值
        assert aligned.series_names == [
            "root.case.d1.temperature",
            "root.case.d1.Temperature",
        ]
        np.testing.assert_array_equal(aligned.timestamps, np.array([0, 1, 2]))
        np.testing.assert_array_equal(
            aligned.values,
            np.array([[30.0, 40.0], [31.0, 41.0], [32.0, 42.0]]),
        )


# 用例 PYTHON-TREE-086：list_timeseries特殊路径列出筛选
def test_tree_list_timeseries_handles_iotdb_naming_rules(tmp_path):
    """用例 22：验证 list_timeseries 对非反引号、反引号、点号转义和大小写路径均能正确列出和筛选。"""
    # 1. 生成覆盖多类 IoTDB 命名规则的树模型TsFile文件
    path = tmp_path / "iotdb_list_timeseries_names.tsfile"
    _write_tree_rows(
        path,
        {
            "root.db_01.device_02": [("sensor_03", TSDataType.DOUBLE)],
            "root.区域_1.设备1": [("温度_1", TSDataType.DOUBLE)],
            "root.`sg-1`.d1": [("`m-1`", TSDataType.DOUBLE)],
            "root.sg.`123`": [("`456`", TSDataType.DOUBLE)],
            "root.`sg 1`.d1": [("`m 1`", TSDataType.DOUBLE)],
            "root.a.b": [("m.dot", TSDataType.DOUBLE)],
            "root.case.DeviceA": [("s1", TSDataType.DOUBLE)],
            "root.case.deviceA": [("s1", TSDataType.DOUBLE)],
            "root.case.d1": [
                ("temperature", TSDataType.DOUBLE),
                ("Temperature", TSDataType.DOUBLE),
            ],
        },
    )

    # 2. 加载文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验无前缀时返回全部命名规则序列
        expected = [
            "root.`sg 1`.d1.`m 1`",
            "root.`sg-1`.d1.`m-1`",
            "root.a.b.m\\.dot",
            "root.case.DeviceA.s1",
            "root.case.d1.Temperature",
            "root.case.d1.temperature",
            "root.case.deviceA.s1",
            "root.db_01.device_02.sensor_03",
            "root.sg.`123`.`456`",
            "root.区域_1.设备1.温度_1",
        ]
        assert sorted(tsdf.list_timeseries()) == sorted(expected)

        # 4. 校验非反引号合法节点和中文节点可按前缀筛选
        assert tsdf.list_timeseries("root.db_01") == [
            "root.db_01.device_02.sensor_03"
        ]
        assert tsdf.list_timeseries("root.区域_1") == ["root.区域_1.设备1.温度_1"]

        # 5. 校验反引号特殊节点和测点名点号转义可按前缀筛选
        assert tsdf.list_timeseries("root.`sg-1`") == ["root.`sg-1`.d1.`m-1`"]
        assert tsdf.list_timeseries("root.`sg 1`") == ["root.`sg 1`.d1.`m 1`"]
        assert tsdf.list_timeseries("root.a.b") == ["root.a.b.m\\.dot"]

        # 6. 校验大小写敏感，大小写不同的设备和物理量不会互相匹配
        assert tsdf.list_timeseries("root.case.DeviceA") == ["root.case.DeviceA.s1"]
        assert tsdf.list_timeseries("root.case.deviceA") == ["root.case.deviceA.s1"]
        assert tsdf.list_timeseries("root.case.d1.temperature") == [
            "root.case.d1.temperature"
        ]
        assert tsdf.list_timeseries("root.case.d1.Temperature") == [
            "root.case.d1.Temperature"
        ]
        assert tsdf.list_timeseries("root.case.device") == []


# 用例 PYTHON-TREE-087：树模型序列名闭环验证
def test_tree_series_name_variants_round_trip_all_dataframe_apis(tmp_path):
    """用例 18-22、29-37、46、57-73、84-86、101-107、150、152：补充验证各类树模型序列名在浏览、元数据、取数和 .loc 中可闭环。"""
    # 1. 构造当前可闭环读取的序列名矩阵；大写物理量由 strict xfail 用例单独跟踪。
    name_cases = [
        ("lowercase", "root.name.lower", "value", "root.name.lower.value", "root.name.lower"),
        (
            "uppercase_device",
            "root.name.DeviceA",
            "value",
            "root.name.DeviceA.value",
            "root.name.DeviceA",
        ),
        ("chinese", "root.区域.设备", "温度", "root.区域.设备.温度", "root.区域.设备"),
        (
            "measurement_dot",
            "root.name.dotfield",
            "m.dot",
            "root.name.dotfield.m\\.dot",
            "root.name.dotfield",
        ),
        (
            "quoted_space_device",
            "root.`space node`.d1",
            "value",
            "root.`space node`.d1.value",
            "root.`space node`.d1",
        ),
        (
            "quoted_symbol_device",
            "root.sg.`sym-!@#`",
            "value",
            "root.sg.`sym-!@#`.value",
            "root.sg.`sym-!@#`",
        ),
        (
            "quoted_numeric_device_field",
            "root.sg.`123`",
            "`456`",
            "root.sg.`123`.`456`",
            "root.sg.`123`",
        ),
        (
            "quoted_space_field",
            "root.sg.d1",
            "`field space`",
            "root.sg.d1.`field space`",
            "root.sg.d1",
        ),
    ]

    # 2. 每类名字单独成文件，避免混入多设备 union 因素后影响纯序列名闭环定位。
    for case_idx, (case_name, device, raw_field, expected_name, prefix) in enumerate(name_cases):
        expected = np.array([case_idx * 10.0 + 0.5, case_idx * 10.0 + 1.5, case_idx * 10.0 + 2.5])
        path = tmp_path / f"tree_series_name_{case_name}.tsfile"
        _write_tree_points(
            path,
            {device: [(raw_field, TSDataType.DOUBLE)]},
            [(device, timestamp, {raw_field: value}) for timestamp, value in enumerate(expected)],
        )

        with TsFileDataFrame(str(path), show_progress=False) as tsdf:
            series = tsdf.list_timeseries()
            all_metadata = tsdf.list_timeseries_metadata()

            assert series == [expected_name]
            assert all_metadata.index.tolist() == [expected_name]
            assert len(tsdf) == 1
            assert tsdf.list_timeseries(prefix) == [expected_name]
            assert tsdf.list_timeseries(expected_name) == [expected_name]
            assert tsdf.list_timeseries_metadata(prefix).index.tolist() == [expected_name]
            assert tsdf.list_timeseries_metadata(expected_name).index.tolist() == [expected_name]
            assert all_metadata.loc[expected_name, "field"] == raw_field
            assert int(all_metadata.loc[expected_name, "count"]) == len(expected)

            ts = tsdf[expected_name]
            assert ts.name == expected_name
            assert len(ts) == len(expected)
            np.testing.assert_array_equal(ts.timestamps, np.arange(len(expected), dtype=np.int64))
            np.testing.assert_allclose(ts[:], expected)

            subset = tsdf[[0]]
            assert subset.list_timeseries() == [expected_name]
            assert expected_name in subset.list_timeseries_metadata().index
            assert expected_name in tsdf[tsdf["field"] == raw_field].list_timeseries()

            aligned = tsdf.loc[:, [0, expected_name]]
            assert aligned.series_names == [expected_name, expected_name]
            assert aligned.shape == (len(expected), 2)
            np.testing.assert_array_equal(aligned.timestamps, np.arange(len(expected), dtype=np.int64))
            np.testing.assert_allclose(aligned.values[:, 0], expected)
            np.testing.assert_allclose(aligned.values[:, 1], expected)
            assert "TsFileDataFrame(tree model, 1 time series" in repr(subset)

    # 3. 校验相似前缀按路径分段匹配，避免 root.prefix.a 误命中 root.prefix.ab。
    prefix_path = tmp_path / "tree_series_name_prefix_segment.tsfile"
    _write_tree_points(
        prefix_path,
        {
            "root.prefix.a": [("m", TSDataType.DOUBLE)],
            "root.prefix.ab": [("m", TSDataType.DOUBLE)],
        },
        [
            ("root.prefix.a", 0, {"m": 1.0}),
            ("root.prefix.a", 1, {"m": 2.0}),
            ("root.prefix.ab", 0, {"m": 10.0}),
            ("root.prefix.ab", 1, {"m": 20.0}),
        ],
    )
    with TsFileDataFrame(str(prefix_path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries("root.prefix.a") == ["root.prefix.a.m"]
        assert tsdf.list_timeseries_metadata("root.prefix.a").index.tolist() == ["root.prefix.a.m"]
        np.testing.assert_allclose(tsdf["root.prefix.a.m"][:], np.array([1.0, 2.0]))
        np.testing.assert_allclose(tsdf["root.prefix.ab.m"][:], np.array([10.0, 20.0]))


# 用例 PYTHON-TREE-088：路径含点号时读取给出明确异常
def test_tree_series_name_dot_device_segment_reports_clear_error_from_dataframe_api(tmp_path):
    """用例 131：补充验证设备路径段包含点号时，list_timeseries 返回路径后的读取路径给出明确异常。"""
    # 1. 构造反引号设备路径段包含点号的树模型文件。
    path = tmp_path / "tree_quoted_dot_device.tsfile"
    _write_tree_rows(path, {"root.sg.`dot.node`": [("value", TSDataType.DOUBLE)]})

    # 2. list_timeseries 和 metadata 可暴露序列；后续读取应给出明确 NotImplementedError。
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        series = tsdf.list_timeseries()
        assert series == ["root.sg.`dot\\.node`.value"]
        metadata = tsdf.list_timeseries_metadata()
        assert metadata.index.tolist() == series
        assert metadata.loc[series[0], "_col_2"] == "`dot.node`"

        with pytest.raises(NotImplementedError, match="Tree device segment with"):
            tsdf[series[0]][:]
        with pytest.raises(NotImplementedError, match="Tree device segment with"):
            tsdf.loc[:, series]


# 用例 PYTHON-TREE-089：DataFrame展示截断
# ── 由 test_tree_repr_show_and_aligned_display_are_truncated 拆分而来（8 个）──
def test_tree_repr_truncates_long_series_list(tmp_path):
    """用例 101：验证 print(df) 最多输出 20 行、超出头尾截断。"""
    # 1. 生成 30 个设备的树模型TsFile文件，用于触发展示截断
    path = tmp_path / "many.tsfile"
    devices = {f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(30)}
    _write_tree_rows(path, devices, t_count=2)

    # 2. 加载文件并校验 DataFrame repr 的截断展示
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        rendered = repr(tsdf)
        assert "30 time series" in rendered
        assert "..." in rendered

        # 3. 校验最多只渲染 20 行序列
        rendered_rows = [
            line for line in rendered.splitlines() if "1970-01-01" in line
        ]
        assert len(rendered_rows) == 20

def test_tree_show_limits_rendered_rows(tmp_path, capsys):
    """用例 102：验证 df.show(max_rows) 控制最大输出行数。"""
    # 1. 生成 30 个设备的树模型TsFile文件，用于触发展示截断
    path = tmp_path / "many.tsfile"
    devices = {f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(30)}
    _write_tree_rows(path, devices, t_count=2)

    # 2. 加载文件并调用 DataFrame.show(max_rows)，校验输出表头和截断
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        tsdf.show(max_rows=6)
        df_show = capsys.readouterr().out
        assert "TsFileDataFrame(tree model" in df_show
        assert "..." in df_show

def test_tree_display_headers_expose_tree_model_columns(tmp_path, capsys):
    """用例 103：验证树模型展示表头包含设备名分段列 _col_i 与物理量名称列 field。"""
    # 1. 生成 30 个设备的树模型TsFile文件，用于触发展示截断
    path = tmp_path / "many.tsfile"
    devices = {f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(30)}
    _write_tree_rows(path, devices, t_count=2)

    # 2. 加载文件并校验 repr 表头使用树模型元数据列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        rendered = repr(tsdf)
        assert "_col_1" in rendered
        assert "field" in rendered
        assert "count" in rendered

        # 3. 校验 show 输出同样保留树模型元数据列
        tsdf.show(max_rows=6)
        df_show = capsys.readouterr().out
        assert "_col_1" in df_show

def test_tree_subset_repr_header_reports_subset_size(tmp_path):
    """用例 104：验证视图 repr 头部显示 subset of N。"""
    # 1. 生成 30 个设备的树模型TsFile文件
    path = tmp_path / "many.tsfile"
    devices = {f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(30)}
    _write_tree_rows(path, devices, t_count=2)

    # 2. 加载文件并构造子集视图，校验 repr 头部标注子集大小
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        subset = tsdf[0:2]
        assert (
            "TsFileDataFrame(tree model, 2 time series, subset of 30)"
            in repr(subset)
        )

def test_tree_aligned_repr_prints_readable_table(tmp_path):
    """用例 106：验证 print(data) 输出可读的对齐表格并自动截断。"""
    # 1. 生成稀疏时间轴文件，用于校验展示中的 NaN 和可读表格
    sparse_path = tmp_path / "display_sparse.tsfile"
    _write_tree_points(
        sparse_path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        [
            ("root.a.d1", 1, {"m1": 10.0}),
            ("root.a.d2", 2, {"m1": 20.0}),
        ],
    )

    # 2. 加载稀疏文件并校验 AlignedTimeseries repr 表格内容
    with TsFileDataFrame(str(sparse_path), show_progress=False) as tsdf:
        rendered_sparse = repr(tsdf.loc[1:2, ["root.a.d1.m1", "root.a.d2.m1"]])
        assert "AlignedTimeseries(2 rows, 2 series)" in rendered_sparse
        assert "time" in rendered_sparse
        assert "root.a.d1.m1" in rendered_sparse
        assert "root.a.d2.m1" in rendered_sparse
        assert "NaN" in rendered_sparse

    # 3. 生成 30 个时间点的对齐结果，校验 repr 超出 max_rows 时自动截断
    large_path = tmp_path / "display_large.tsfile"
    _write_tree_rows(
        large_path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        t_count=30,
    )
    with TsFileDataFrame(str(large_path), show_progress=False) as tsdf:
        rendered_large = repr(tsdf.loc[:, ["root.a.d1.m1", "root.a.d2.m1"]])
        assert "AlignedTimeseries(30 rows, 2 series)" in rendered_large
        assert "..." in rendered_large

def test_tree_aligned_show_limits_rendered_rows(tmp_path, capsys):
    """用例 107：验证 data.show(max_rows) 控制对齐结果输出行数。"""
    # 1. 生成 30 个设备的树模型TsFile文件
    path = tmp_path / "many.tsfile"
    devices = {f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(30)}
    _write_tree_rows(path, devices, t_count=2)

    # 2. 执行对齐查询并调用 show(max_rows=1)，校验输出表头和行数截断
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[0:1, [0, 1]]
        aligned.show(max_rows=1)
        aligned_show = capsys.readouterr().out
        assert "AlignedTimeseries" in aligned_show
        assert "time" in aligned_show
        assert "root.db.d00.m1" in aligned_show
        assert "root.db.d01.m1" in aligned_show
        assert "..." in aligned_show

def test_tree_display_timestamp_format_keeps_milliseconds(tmp_path, capsys):
    """用例 133：验证展示层时间戳格式保留毫秒且整秒不追加多余小数。"""
    # 1. 生成 30 个设备的树模型TsFile文件
    path = tmp_path / "many.tsfile"
    devices = {f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)] for i in range(30)}
    _write_tree_rows(path, devices, t_count=2)

    # 2. 加载文件并校验 DataFrame repr 的时间戳格式
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        rendered = repr(tsdf)
        assert "1970-01-01 08:00:00.001" in rendered
        assert ".000" not in rendered

        # 3. 校验 DataFrame.show 的时间戳格式
        tsdf.show(max_rows=6)
        df_show = capsys.readouterr().out
        assert "1970-01-01 08:00:00.001" in df_show

        # 4. 校验对齐结果 show 的时间戳格式
        aligned = tsdf.loc[0:1, [0, 1]]
        aligned.show(max_rows=1)
        aligned_show = capsys.readouterr().out
        assert "1970-01-01 08:00:00.001" in aligned_show

    # 5. 生成稀疏时间轴文件，校验对齐结果 repr 的毫秒时间戳
    sparse_path = tmp_path / "display_sparse.tsfile"
    _write_tree_points(
        sparse_path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        [
            ("root.a.d1", 1, {"m1": 10.0}),
            ("root.a.d2", 2, {"m1": 20.0}),
        ],
    )
    with TsFileDataFrame(str(sparse_path), show_progress=False) as tsdf:
        rendered_sparse = repr(tsdf.loc[1:2, ["root.a.d1.m1", "root.a.d2.m1"]])
        assert "1970-01-01 08:00:00.001" in rendered_sparse
        assert "1970-01-01 08:00:00.002" in rendered_sparse

def test_tree_aligned_display_truncates_large_results(tmp_path):
    """用例 135：验证对齐结果展示仅渲染展示行并在大结果集下正确截断。"""
    # 1. 生成 30 个时间点、2 条序列的树模型TsFile文件
    path = tmp_path / "aligned_large.tsfile"
    _write_tree_rows(
        path,
        {
            "root.a.d1": [("m1", TSDataType.DOUBLE)],
            "root.a.d2": [("m1", TSDataType.DOUBLE)],
        },
        t_count=30,
    )

    # 2. 加载文件并执行覆盖全部时间点的对齐查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        aligned = tsdf.loc[:, ["root.a.d1.m1", "root.a.d2.m1"]]
        assert aligned.shape == (30, 2)

        # 3. 校验 repr 仅为展示行渲染，超出 max_rows 时按头尾截断
        rendered = repr(aligned)
        assert "AlignedTimeseries(30 rows, 2 series)" in rendered
        assert "..." in rendered
        rendered_rows = [
            line for line in rendered.splitlines() if line.strip().startswith("1970-")
        ]
        assert len(rendered_rows) == 20
        assert "29.50" in rendered
        assert "10.50" not in rendered


def test_tree_repr_only_builds_preview_rows(tmp_path, monkeypatch):
    """用例 105：验证 repr 仅为预览行构建元信息，避免为所有序列做不必要的展示计算。"""
    # 1. 生成 30 条序列的树模型TsFile文件
    path = tmp_path / "many.tsfile"
    devices = {
        f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)]
        for i in range(30)
    }
    _write_tree_rows(path, devices, t_count=1)

    # 2. 加载文件并给预览行构建方法打 spy
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        original_build_series_info = tsdf._build_series_info
        built = []

        def build_series_info_spy(series_ref):
            built.append(series_ref)
            return original_build_series_info(series_ref)

        monkeypatch.setattr(tsdf, "_build_series_info", build_series_info_spy)

        # 3. 调用 repr，确认只构建头尾预览行而不是全量序列
        rendered = repr(tsdf)

        assert "..." in rendered
        assert len(built) == 20


# 用例 PYTHON-TREE-091：DataFrame关闭幂等
# ── 由 test_tree_dataframe_and_series_lifecycle 拆分而来（5 个）──
def test_tree_dataframe_close_releases_resources(tmp_path):
    """用例 109：验证可手动调用 df.close() 释放资源。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 创建根 DataFrame 并手动关闭
    tsdf = TsFileDataFrame(str(path), show_progress=False)
    tsdf.close()

    # 3. 校验关闭后已释放 reader 资源
    assert tsdf._closed
    assert tsdf._readers == {}

def test_tree_dataframe_access_after_close_raises(tmp_path):
    """用例 110：验证 close 后再访问 df 抛 RuntimeError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 创建根 DataFrame 并关闭
    tsdf = TsFileDataFrame(str(path), show_progress=False)
    tsdf.close()

    # 3. 关闭后访问 DataFrame 均报错
    with pytest.raises(RuntimeError, match="closed"):
        tsdf[0]
    with pytest.raises(RuntimeError, match="closed"):
        tsdf.loc[0:1, [0]]

def test_tree_series_access_after_dataframe_close_raises(tmp_path):
    """用例 111：验证 close 后访问已取得的 Timeseries 抛 RuntimeError。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 创建根 DataFrame 并取得序列对象后关闭
    tsdf = TsFileDataFrame(str(path), show_progress=False)
    series = tsdf[0]
    tsdf.close()

    # 3. 关闭后访问既有 Timeseries 报错
    with pytest.raises(RuntimeError, match="closed"):
        series[0]

def test_tree_subset_view_close_is_noop(tmp_path):
    """用例 112：验证对子集视图调用 close() 为静默 no-op、不影响父。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 创建根 DataFrame 和子集视图
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        subset = tsdf[:1]

        # 3. 关闭子集视图，确认是静默 no-op 且不影响根 DataFrame
        subset.close()
        assert len(tsdf) == 3

def test_tree_dataframe_repeated_close_is_idempotent(tmp_path):
    """用例 113：验证重复 close 幂等、不报错。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 连续关闭根 DataFrame，确认 close 幂等
    tsdf = TsFileDataFrame(str(path), show_progress=False)
    tsdf.close()
    tsdf.close()

    # 3. 校验对象仍处于已关闭状态
    assert tsdf._closed


# [去 skip 裸跑] 已知缺陷：复用 reader 跨设备名读取第二条序列为空（失败即缺陷证据；验证期再按 AGENTS.md 挂 xfail(strict=True)）
# 用例 PYTHON-TREE-092：Reader残留列隔离
def test_tree_reader_handles_stale_path_columns_after_reused_queries(tmp_path):
    """用例 130：回归验证复用 reader 后，前一次查询遗留的 col_i 列不会误导后续设备读取。"""
    # 1. 生成树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    writer = TsFileWriter(str(path))
    writer.register_timeseries(
        "root.ln.wf01.wt01", TimeseriesSchema("status", TSDataType.INT32)
    )
    writer.register_timeseries(
        "root.ln.wf01.wt01", TimeseriesSchema("temperature", TSDataType.DOUBLE)
    )
    writer.register_timeseries(
        "root.ln.wf02.wt02", TimeseriesSchema("status", TSDataType.INT32)
    )
    for t in range(5):
        writer.write_row_record(
            RowRecord(
                "root.ln.wf01.wt01",
                t,
                [
                    Field("status", t, TSDataType.INT32),
                    Field("temperature", float(t) + 0.5, TSDataType.DOUBLE),
                ],
            )
        )
        writer.write_row_record(
            RowRecord(
                "root.ln.wf02.wt02",
                t,
                [Field("status", t * 2, TSDataType.INT32)],
            )
        )
    writer.close()

    # 2. 加载文件并读取待校验数据
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 读取第一个序列
        first_values = tsdf["root.ln.wf01.wt01.temperature"][:]
        # 4. 读取第二个序列
        second_values = tsdf["root.ln.wf02.wt02.status"][:]

        # 5. 使用 .loc 跨设备对齐读取，确认完整数据可一次性获取
        aligned = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf02.wt02.status"],
        ]

    # 6. 退出 context 后统一断言，避免失败时干扰 reader 关闭流程
    np.testing.assert_array_equal(first_values, np.array([0.5, 1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_array_equal(
        second_values,
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
    )
    np.testing.assert_array_equal(
        aligned.values,
        np.array([[0.5, 0.0], [1.5, 2.0], [2.5, 4.0], [3.5, 6.0], [4.5, 8.0]]),
    )


# 用例 PYTHON-TREE-093：多文件并行加载
def test_tree_loading_uses_thread_pool_for_multiple_files(tmp_path, monkeypatch):
    """用例 136：验证多文件加载会走 ThreadPoolExecutor，worker 数为 min(文件数, CPU 核数)。"""
    # 1. 生成 4 个树模型TsFile文件，用于触发多文件并行加载分支
    paths = []
    for idx in range(4):
        path = tmp_path / f"part{idx}.tsfile"
        _write_tree_rows(path, {f"root.p{idx}.d": [("m1", TSDataType.DOUBLE)]})
        paths.append(str(path))

    # 2. 准备 RecordingExecutor，记录 max_workers 并同步执行任务
    recorded_workers = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            recorded_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    # 3. 替换 ThreadPoolExecutor，观察 TsFileDataFrame 的并行加载参数
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", RecordingExecutor)

    # 4. 加载多个文件，确认所有序列均被加载
    with TsFileDataFrame(paths, show_progress=False) as tsdf:
        assert len(tsdf) == 4

    # 5. 校验 worker 数为 min(文件数, CPU 核数)
    assert recorded_workers == [min(len(paths), os.cpu_count() or 4)]


# 用例 PYTHON-TREE-094：单文件串行加载
def test_tree_single_file_uses_serial_loader(tmp_path, monkeypatch):
    """用例 137：验证单文件加载只走串行元数据扫描分支，不误触发并行加载逻辑。"""
    # 1. 生成单个树模型TsFile文件
    path = tmp_path / "single.tsfile"
    _write_tree_rows(path, {"root.single.d": [("m1", TSDataType.DOUBLE)]})
    calls = []

    # 2. 保存串行和并行加载方法，准备 spy 记录调用分支
    original_serial = TsFileDataFrame._load_metadata_serial
    original_parallel = TsFileDataFrame._load_metadata_parallel

    def serial_spy(self, reader_class):
        calls.append("serial")
        return original_serial(self, reader_class)

    def parallel_spy(self, reader_class):
        calls.append("parallel")
        return original_parallel(self, reader_class)

    monkeypatch.setattr(TsFileDataFrame, "_load_metadata_serial", serial_spy)
    monkeypatch.setattr(TsFileDataFrame, "_load_metadata_parallel", parallel_spy)

    # 3. 加载单文件，确认序列可正常读取
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.single.d.m1"]

    # 4. 校验只调用串行加载分支
    assert calls == ["serial"]


# 用例 PYTHON-TREE-095：加载进度输出
# ── 由 test_tree_show_progress_reports_to_stderr 拆分而来（2 个）──
def test_tree_dataframe_load_reports_shard_progress(tmp_path, capsys):
    """用例 138：验证 show_progress=True 时向 stderr 输出加载进度。"""
    # 1. 生成两个树模型TsFile文件，用于验证 DataFrame 多分片加载进度
    path1 = tmp_path / "part1.tsfile"
    path2 = tmp_path / "part2.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.c": [("m1", TSDataType.DOUBLE)]}, t_start=10)

    # 2. 以 show_progress=True 加载文件列表并捕获 stderr
    with TsFileDataFrame([str(path1), str(path2)], show_progress=True):
        pass

    # 3. 校验 DataFrame 加载阶段输出分片进度
    stderr = capsys.readouterr().err
    assert "Loading TsFile shards: 0/2" in stderr
    assert "Loading TsFile shards: 2/2 (2 series) ... done" in stderr

def test_tree_reader_metadata_scan_reports_device_and_series_counts(tmp_path, capsys):
    """用例 139：验证 show_progress=True 时树模型元数据扫描输出设备名数量和序列数量。"""
    # 1. 生成基础树模型TsFile文件，用于验证树模型元数据扫描进度
    reader_path = tmp_path / "reader.tsfile"
    _write_tree_file(reader_path)
    reader = TsFileSeriesReader(str(reader_path), show_progress=True)
    try:
        # 2. 校验 reader 元数据扫描输出设备数和序列数
        stderr = capsys.readouterr().err
        assert "Reading TsFile metadata: 0/2 devices" in stderr
        assert "Reading TsFile metadata (tree): 2 device(s), 3 series ... done" in stderr
    finally:
        reader.close()


# ── 由 test_tree_device_path_rebuild_reports_unsupported_segments 拆分而来（2 个）──
def test_tree_device_path_rebuild_rejects_dotted_segment():
    """用例 131：验证树模型设备名分段本身含点号时读取路径重建给出明确错误。"""
    # 1. 构造 root 表入口，模拟树模型根段
    table_entry = SimpleNamespace(table_name="root")

    # 2. 设备路径段包含点号时，应提示底层 cwrapper 路径 API 不支持
    with pytest.raises(NotImplementedError, match="Tree device segment with"):
        TsFileSeriesReader._build_tree_device_path(
            table_entry, SimpleNamespace(tag_values=("a.b",))
        )

def test_tree_device_path_rebuild_rejects_null_segment():
    """用例 132：验证树模型设备名分段值出现中间 None 时不被当成尾部浅层补齐处理。"""
    # 1. 构造 root 表入口，模拟树模型根段
    table_entry = SimpleNamespace(table_name="root")

    # 2. 设备路径中间出现 None 时，应提示非法空路径段
    with pytest.raises(ValueError, match="Tree device path cannot include a null segment"):
        TsFileSeriesReader._build_tree_device_path(
            table_entry, SimpleNamespace(tag_values=("a", None, "c"))
        )


def test_empty_table_shard_mixed_with_tree_file_raises_model_conflict(tmp_path):
    """用例 12：验证空表模型分片与树模型文件同组加载时仍按树表混用报错。"""
    # 1. 生成空表模型TsFile分片
    empty_table = tmp_path / "empty_table.tsfile"
    # 2. 生成有效树模型TsFile文件
    tree_path = tmp_path / "tree.tsfile"
    _write_empty_table(empty_table)
    _write_tree_rows(tree_path, {"root.a.b": [("m1", TSDataType.DOUBLE)]})

    # 3. 同时加载空表分片和树模型文件
    with pytest.raises(ValueError, match="Mixed table-model and tree-model"):
        # 4. 校验空表模型分片仍按表模型参与树表一致性校验
        TsFileDataFrame([str(empty_table), str(tree_path)], show_progress=False)
