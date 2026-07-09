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


def test_tree_loads_single_file_and_model_metadata(tmp_path):
    """用例 1、9、15：验证单个树模型文件可加载为 tree，且序列清单和展示表头符合树模型元数据规则。"""
    # 1. 生成包含 2 个设备、3 条数值序列的树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载单个树模型文件
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        # 3. 校验模型类型、序列数量和序列清单
        assert tsdf.model == "tree"
        assert len(tsdf) == 3
        assert sorted(tsdf.list_timeseries()) == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
            "root.ln.wf02.wt02.status",
        ]

        # 4. 校验展示内容使用树模型元数据列，不出现表模型 table 列
        rendered = repr(tsdf)
        assert "TsFileDataFrame(tree model, 3 time series, 1 files)" in rendered
        assert "_col_1" in rendered and "_col_2" in rendered and "_col_3" in rendered
        assert "table" not in rendered.splitlines()[1]


def test_tree_loads_file_list_and_recursive_directory(tmp_path):
    """用例 2-4、7：验证文件列表和目录递归两种入口都能加载树模型文件，并忽略非 .tsfile 文件。"""
    # 1. 构造包含子目录的测试目录和两个树模型TsFile文件
    root_dir = tmp_path / "tree_data"
    sub_dir = root_dir / "sub"
    sub_dir.mkdir(parents=True)
    first = root_dir / "b.tsfile"
    second = sub_dir / "a.tsfile"
    _write_tree_rows(first, {"root.b.device": [("m1", TSDataType.DOUBLE)]}, 10)
    _write_tree_rows(second, {"root.a.device": [("m1", TSDataType.DOUBLE)]}, 0)
    (root_dir / "ignore.txt").write_text("not a tsfile", encoding="utf-8")

    # 2. 按文件列表加载，确认两个文件均参与合并
    with TsFileDataFrame([str(first), str(second)], show_progress=False) as listed:
        assert len(listed) == 2

    # 3. 按目录递归加载，确认只加载 .tsfile 且路径被归一化
    with TsFileDataFrame(str(root_dir), show_progress=False) as from_dir:
        assert sorted(from_dir.list_timeseries()) == [
            "root.a.device.m1",
            "root.b.device.m1",
        ]
        assert all(Path(path).is_absolute() for path in from_dir._paths)


def test_tree_load_path_errors_and_relative_path(tmp_path, monkeypatch):
    """用例 5、6、8：验证空目录、缺失文件的错误信息，以及相对路径会归一化为绝对路径后加载。"""
    # 1. 构造空目录并校验无 .tsfile 文件时报错
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .tsfile files found"):
        TsFileDataFrame(str(empty_dir), show_progress=False)

    # 2. 校验缺失文件路径时报错
    with pytest.raises(FileNotFoundError, match="TsFile not found"):
        TsFileDataFrame(str(tmp_path / "missing.tsfile"), show_progress=False)

    # 3. 生成树模型TsFile文件并切换到临时目录
    path = tmp_path / "relative.tsfile"
    _write_tree_rows(path, {"root.rel.device": [("m1", TSDataType.DOUBLE)]})
    monkeypatch.chdir(tmp_path)

    # 4. 使用相对路径加载，确认内部路径归一化为绝对路径
    with TsFileDataFrame("relative.tsfile", show_progress=False) as tsdf:
        assert tsdf._paths == [os.path.abspath("relative.tsfile")]
        assert tsdf.list_timeseries() == ["root.rel.device.m1"]


def test_tree_rejects_mixed_table_and_tree_models(tmp_path):
    """用例 10、11：验证同一加载集合中同时存在表模型和树模型时会稳定抛出混用错误。"""
    # 1. 分别生成树模型和表模型TsFile文件
    tree_path = tmp_path / "tree.tsfile"
    table_path = tmp_path / "table.tsfile"
    _write_tree_file(tree_path)
    _write_weather_table(table_path)

    # 2. 表模型在前、树模型在后时，应抛出树表混用错误
    with pytest.raises(ValueError, match="Mixed table-model and tree-model"):
        TsFileDataFrame([str(table_path), str(tree_path)], show_progress=False)

    # 3. 树模型在前、表模型在后时，也应抛出树表混用错误
    with pytest.raises(ValueError, match="Mixed table-model and tree-model"):
        TsFileDataFrame([str(tree_path), str(table_path)], show_progress=False)


def test_tree_empty_and_no_numeric_inputs_report_stable_errors(tmp_path):
    """用例 13、115、116：验证空树文件和仅包含非数值测点的树文件会给出明确错误，而不是生成空视图。"""
    # 1. 生成空树模型TsFile文件
    empty_tree = tmp_path / "empty_tree.tsfile"
    writer = TsFileWriter(str(empty_tree))
    writer.close()

    # 2. 加载空树文件，校验无设备元数据时报错
    with pytest.raises(ValueError, match="No devices found in tree-model TsFile"):
        TsFileDataFrame(str(empty_tree), show_progress=False)

    # 3. 生成仅包含非数值测点的树模型TsFile文件
    string_only = tmp_path / "string_only.tsfile"
    _write_tree_rows(
        string_only,
        {"root.a.b": [("status", TSDataType.STRING)]},
    )

    # 4. 加载非数值测点文件，校验无数值测点时报错
    with pytest.raises(ValueError, match="No numeric measurements found"):
        TsFileDataFrame(str(string_only), show_progress=False)


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


def test_tree_list_timeseries_prefix_is_segment_aware(tmp_path):
    """用例 23-28：验证 list_timeseries 的前缀匹配按设备名分段生效，不把 root.dbx 误判为 root.db。"""
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

    # 2. 加载文件并按 root.db 前缀筛选
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries("root.db")) == [
            "root.db.d1.s1",
            "root.db.d1.s2",
            "root.db.d2.s1",
        ]

        # 3. 按设备名前缀筛选，只返回对应设备名下序列
        assert sorted(tsdf.list_timeseries("root.db.d1")) == [
            "root.db.d1.s1",
            "root.db.d1.s2",
        ]

        # 4. 校验缺失前缀和非法转义前缀不会误匹配
        assert tsdf.list_timeseries("root.none") == []
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


def test_tree_list_timeseries_handles_different_series_name_depths(tmp_path):
    """用例 16、143：验证 list_timeseries 可处理不同序列名深度，并按中间设备名前缀筛选。"""
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
        # 3. 校验不同深度序列名全部返回
        assert sorted(tsdf.list_timeseries("root.depth")) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
            "root.depth.m1",
        ]

        # 4. 校验中间设备名前缀命中自身及更深设备名下序列
        assert sorted(tsdf.list_timeseries("root.depth.area")) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
        ]

        # 5. 校验完整序列名前缀只返回单条序列
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


def test_tree_list_timeseries_metadata_handles_different_series_name_depths(tmp_path):
    """用例 17、148：验证 list_timeseries_metadata 可处理不同序列名深度和浅层设备名补 NaN。"""
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
        # 3. 校验不同深度序列元数据全部返回，且全局设备名分段列扩展到最深
        meta = tsdf.list_timeseries_metadata("root.depth")
        assert sorted(meta.index.tolist()) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
            "root.depth.m1",
        ]
        assert "_col_4" in meta.columns
        assert pd.isna(meta.loc["root.depth.m1", "_col_2"])
        assert meta.loc["root.depth.area.line.unit.m1", "_col_4"] == "unit"

        # 4. 校验中间设备名前缀命中自身及更深设备名下序列
        area = tsdf.list_timeseries_metadata("root.depth.area")
        assert sorted(area.index.tolist()) == [
            "root.depth.area.line.m1",
            "root.depth.area.line.unit.m1",
            "root.depth.area.m1",
        ]

        # 5. 校验完整序列名前缀只返回单条元数据
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


def test_tree_metadata_column_projection_and_boolean_filter(tmp_path):
    """用例 38-45：验证元数据列投影、树模型 table 列不可用、未知列报错，以及布尔过滤返回子视图。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并按元数据列名投影
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert list(tsdf["field"]) == ["status", "temperature", "status"]
        assert list(tsdf["_col_1"]) == ["ln", "ln", "ln"]
        assert list(tsdf["_col_3"]) == ["wt01", "wt01", "wt02"]
        assert list(tsdf["count"]) == [5, 5, 5]

        # 3. 校验树模型不暴露 table 元数据列，未知列按序列查询失败
        with pytest.raises(KeyError):
            tsdf["table"]
        with pytest.raises(KeyError, match="Series not found"):
            tsdf["unknown_column"]

        # 4. 使用布尔条件筛选子集视图
        subset = tsdf[tsdf["_col_2"] == "wf01"]
        assert isinstance(subset, TsFileDataFrame)
        assert len(subset) == 2
        assert subset.list_timeseries() == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf01.wt01.temperature",
        ]

        # 5. 使用无命中布尔条件筛选，返回空子集视图
        empty_subset = tsdf[tsdf["_col_2"] == "missing"]
        assert len(empty_subset) == 0
        assert empty_subset.list_timeseries() == []


def test_tree_lazy_metadata_projection_view_and_context_close(tmp_path, monkeypatch):
    """用例 14、56、108：验证初始化和元数据投影不触发数据读取，子集视图复用 reader，with 退出会关闭根对象。"""
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

        # 5. 构造子集视图，确认复用父对象 reader
        subset = tsdf[0:2]
        assert subset._readers is tsdf._readers
        assert subset._root is tsdf

        # 6. 读取子集序列，确认子集视图结果正确
        np.testing.assert_array_equal(
            subset["root.ln.wf01.wt01.temperature"][:2],
            np.array([0.5, 1.5]),
        )

    # 7. 退出 with 后确认根对象已关闭并释放 reader
    assert tsdf._closed
    assert tsdf._readers == {}


def test_tree_getitem_series_and_subset_validation(tmp_path):
    """用例 46-56：验证按名称/索引取单序列、切片/列表取子集，以及越界和非法 key 的异常路径。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并按名称/索引获取单序列
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        by_name = tsdf["root.ln.wf01.wt01.temperature"]
        assert isinstance(by_name, Timeseries)
        assert by_name.name == "root.ln.wf01.wt01.temperature"
        assert isinstance(tsdf[0], Timeseries)
        assert tsdf[-1].name == "root.ln.wf02.wt02.status"

        # 3. 按切片和列表获取子集视图
        subset = tsdf[0:2]
        assert len(subset) == 2
        assert "subset of 3" in repr(subset)
        assert [ts.name for ts in tsdf[[0, 2]]] == [
            "root.ln.wf01.wt01.status",
            "root.ln.wf02.wt02.status",
        ]

        # 4. 校验越界、缺失序列和非法 key 的错误类型
        with pytest.raises(IndexError, match="out of range"):
            tsdf[99]
        with pytest.raises(KeyError, match="Series not found"):
            tsdf["root.ln.missing"]
        with pytest.raises(TypeError, match="List index must contain integers"):
            tsdf[[0, "bad"]]
        with pytest.raises(IndexError, match="out of range"):
            tsdf[[0, 99]]
        with pytest.raises(TypeError, match="Unsupported key type"):
            tsdf[{"bad": "key"}]


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


def test_tree_getitem_list_variants_duplicates_and_errors(tmp_path):
    """用例 52-55、154、155：验证列表子集的顺序、负索引、重复索引，以及越界和非法元素报错。"""
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

        # 5. 列表全为同一个索引时，应保留重复序列视图
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

        # 6. 校验非法索引和非法列表元素的错误类型
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


def test_tree_loc_alignment_forms_and_validation(tmp_path):
    """用例 73、75、78-82、84-90：验证 .loc 基本对齐、单时间戳、负序列索引，以及参数校验错误类型。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并执行基本 .loc 对齐查询
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        data = tsdf.loc[
            0:4,
            ["root.ln.wf01.wt01.temperature", "root.ln.wf01.wt01.status"],
        ]

        # 3. 校验对齐结果的类型、形状、时间戳和值矩阵
        assert isinstance(data, AlignedTimeseries)
        assert data.shape == (5, 2)
        np.testing.assert_array_equal(data.timestamps, np.arange(5, dtype=np.int64))
        np.testing.assert_array_equal(
            data.values,
            np.array(
                [[0.5, 0.0], [1.5, 1.0], [2.5, 2.0], [3.5, 3.0], [4.5, 4.0]]
            ),
        )

        # 4. 校验单时间戳查询和负序列索引查询
        single = tsdf.loc[1, [0, "root.ln.wf01.wt01.temperature"]]
        assert single.shape == (1, 2)
        np.testing.assert_array_equal(single.timestamps, np.array([1], dtype=np.int64))

        negative = tsdf.loc[:, [-1]]
        assert negative.series_names == ["root.ln.wf02.wt02.status"]

        # 5. 校验 .loc 参数格式、时间索引、序列参数和越界异常
        with pytest.raises(ValueError, match="requires exactly 2"):
            tsdf.loc[0]
        with pytest.raises(TypeError, match="Time index must be slice or int"):
            tsdf.loc["bad", [0]]
        with pytest.raises(TypeError, match="Series specifier must be int or str"):
            tsdf.loc[0:1, [object()]]
        with pytest.raises(IndexError, match="out of range"):
            tsdf.loc[0:1, [999]]
        with pytest.raises(KeyError, match="Series not found"):
            tsdf.loc[0:1, ["root.no.such"]]

        # 6. 校验空时间范围返回空 AlignedTimeseries 且形状稳定
        empty = tsdf.loc[100:200, ["root.ln.wf01.wt01.temperature"]]
        assert isinstance(empty, AlignedTimeseries)
        assert empty.shape == (0, 1)
        assert empty.series_names == ["root.ln.wf01.wt01.temperature"]
        np.testing.assert_array_equal(empty.timestamps, np.array([], dtype=np.int64))


def test_tree_loc_sparse_union_open_ranges_and_requested_bounds(tmp_path):
    """用例 74、77、80、83：验证稀疏时间轴按时间戳并集对齐，缺失值填 NaN，且开区间不会越过请求边界。"""
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

        # 3. 校验时间戳取并集，缺失值用 NaN 补齐
        np.testing.assert_array_equal(aligned.timestamps, np.array([0, 1, 2]))
        assert aligned.values[0, 0] == 10.0
        assert np.isnan(aligned.values[0, 1])
        assert aligned.values[1, 0] == 11.0
        assert aligned.values[1, 1] == 21.0
        assert np.isnan(aligned.values[2, 0])
        assert aligned.values[2, 1] == 22.0

        # 4. 校验左开/右开/全开时间范围不会越过请求边界
        np.testing.assert_array_equal(
            tsdf.loc[:1, ["root.a.d1.m1"]].timestamps, np.array([0, 1])
        )
        np.testing.assert_array_equal(
            tsdf.loc[1:, ["root.a.d2.m1"]].timestamps, np.array([1, 2])
        )
        np.testing.assert_array_equal(
            tsdf.loc[:, ["root.a.d1.m1"]].timestamps, np.array([0, 1])
        )

        # 5. 校验显式闭区间裁剪后的时间戳和值
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


@pytest.mark.xfail(
    reason="产品待审查：跨设备且物理量名不同时 .loc 对齐查询第二列返回 NaN",
    strict=True,
    raises=AssertionError,
)
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


def test_tree_loc_dedups_repeated_series_specifiers(tmp_path):
    """用例 82、126-128：验证 .loc 内重复指定同一序列时，内部可去重但输出仍保留请求中的重复列位置。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 加载文件并获取目标序列名称和索引
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        name = "root.ln.wf01.wt01.temperature"
        idx = tsdf.list_timeseries().index(name)

        # 3. 使用名称和索引重复指定同一序列，确认输出保留两列
        repeated = tsdf.loc[0:2, [name, idx]]
        assert repeated.shape == (3, 2)
        np.testing.assert_array_equal(
            repeated.values, np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        )

        # 4. 将重复序列夹在其它序列中，确认输出列顺序稳定
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


def test_tree_merges_identical_series_across_files(tmp_path):
    """用例 91、92、94：验证多个文件中同名序列按时间分片合并为一条逻辑序列，并累加 count。"""
    # 1. 生成两个包含同名序列、时间范围不重叠的树模型TsFile文件
    path1 = tmp_path / "t1.tsfile"
    path2 = tmp_path / "t2.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=10)

    # 2. 按文件列表加载并确认同名序列合并为一条逻辑序列
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.a.b.m1"]
        ts = tsdf["root.a.b.m1"]

        # 3. 校验合并后的长度、时间戳和 metadata count
        assert len(ts) == 6
        np.testing.assert_array_equal(
            ts.timestamps, np.array([0, 1, 2, 10, 11, 12], dtype=np.int64)
        )
        assert tsdf.list_timeseries_metadata().loc["root.a.b.m1", "count"] == 6


def test_tree_duplicate_timestamps_across_shards_raise(tmp_path):
    """用例 93：验证跨分片出现重复时间戳时，读取时间轴和值都会抛出明确错误。"""
    # 1. 生成两个同名序列且时间戳重叠的树模型TsFile文件
    path1 = tmp_path / "first.tsfile"
    path2 = tmp_path / "second.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=0)
    _write_tree_rows(path2, {"root.a.b": [("m1", TSDataType.DOUBLE)]}, t_start=2)

    # 2. 加载文件并校验读取时间戳和值时均提示重复时间戳
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        with pytest.raises(ValueError, match="Duplicate timestamp 2.*across shards"):
            _ = tsdf["root.a.b.m1"].timestamps
        with pytest.raises(ValueError, match="Duplicate timestamp 2.*across shards"):
            _ = tsdf["root.a.b.m1"][:]


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


def test_tree_unions_field_subsets_and_preserves_local_field_mapping(tmp_path):
    """用例 95、117-119、125：验证多文件字段并集按首次出现顺序扩展，且 reader-local 字段索引不会串列。"""
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

    # 5. 加载两个文件，校验全局字段并集顺序和序列清单
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert tsdf._index.table_entries["root"].field_columns == ("m1", "m3", "m2")
        assert sorted(tsdf.list_timeseries()) == [
            "root.a.b.m1",
            "root.a.b.m2",
            "root.a.b.m3",
        ]

        # 6. 分别读取 m1/m2/m3，确认 reader-local 字段索引映射正确
        np.testing.assert_array_equal(
            tsdf["root.a.b.m1"][:],
            np.array([100.0, 101.0, 102.0, 110.0, 111.0, 112.0]),
        )
        np.testing.assert_array_equal(
            tsdf["root.a.b.m2"][:], np.array([210.0, 211.0, 212.0])
        )
        np.testing.assert_array_equal(
            tsdf["root.a.b.m3"][:], np.array([300.0, 301.0, 302.0])
        )


def test_tree_unions_different_depths_and_short_paths_stay_resolvable(tmp_path):
    """用例 96、120、121：验证不同设备深度 union 后全局 _col_i 扩展，浅层路径补 NaN 且仍可按短路径读取。"""
    # 1. 生成浅层设备和深层设备两个树模型TsFile文件
    path1 = tmp_path / "shallow.tsfile"
    path2 = tmp_path / "deep.tsfile"
    _write_tree_rows(path1, {"root.a.b": [("m1", TSDataType.DOUBLE)]})
    _write_tree_rows(path2, {"root.a.b.c": [("m1", TSDataType.DOUBLE)]})

    # 2. 加载两个文件并校验不同深度序列都可识别
    with TsFileDataFrame([str(path1), str(path2)], show_progress=False) as tsdf:
        assert sorted(tsdf.list_timeseries()) == ["root.a.b.c.m1", "root.a.b.m1"]

        # 3. 校验全局 _col_i 扩展，短路径末尾补 NaN
        meta = tsdf.list_timeseries_metadata()
        assert "_col_3" in meta.columns
        assert meta.loc["root.a.b.c.m1", "_col_3"] == "c"
        assert pd.isna(meta.loc["root.a.b.m1", "_col_3"])

        # 4. 校验短路径仍可按原路径读取值
        np.testing.assert_array_equal(tsdf["root.a.b.m1"][:], np.array([0.5, 1.5, 2.5]))

        # 5. 校验前缀 root.a.b 能同时命中浅层和深层序列
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


def test_tree_omits_non_numeric_measurements_and_keeps_nan_values(tmp_path):
    """用例 98-100：验证非数值测点不暴露为序列，数值测点中的 NaN 保留且不影响其它值读取。"""
    # 1. 手动生成同时包含数值测点和字符串测点的树模型TsFile文件
    path = tmp_path / "mixed.tsfile"
    writer = TsFileWriter(str(path))
    writer.register_timeseries("root.a.b", TimeseriesSchema("temp", TSDataType.DOUBLE))
    writer.register_timeseries("root.a.b", TimeseriesSchema("status", TSDataType.STRING))

    # 2. 写入数值测点，其中包含一个 NaN 值
    for t, value in enumerate([0.5, float("nan"), 2.5]):
        writer.write_row_record(
            RowRecord(
                "root.a.b",
                t,
                [
                    Field("temp", value, TSDataType.DOUBLE),
                    Field("status", "ok", TSDataType.STRING),
                ],
            )
        )
    writer.close()

    # 3. 加载文件并确认只暴露数值测点
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        assert tsdf.list_timeseries() == ["root.a.b.temp"]
        with pytest.raises(KeyError):
            tsdf["root.a.b.status"]

        # 4. 读取数值测点，确认 NaN 原样保留
        values = tsdf["root.a.b.temp"][:]
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


@pytest.mark.xfail(
    reason="产品待审查：CSV 用例 21 中大写物理量 root.case.d1.Temperature 按完整路径读取为空",
    strict=True,
    raises=AssertionError,
)
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


@pytest.mark.xfail(
    reason="产品待审查：大写物理量 root.case.d1.Temperature 参与 .loc 对齐查询返回空结果",
    strict=True,
    raises=AssertionError,
)
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


def test_tree_repr_show_and_aligned_display_are_truncated(tmp_path, capsys):
    """用例 101-104、106、107、133、135：验证 DataFrame 和 AlignedTimeseries 展示会按 max_rows 截断，避免大结果全量输出。"""
    # 1. 生成 30 个设备的树模型TsFile文件，用于触发展示截断
    path = tmp_path / "many.tsfile"
    devices = {
        f"root.db.d{i:02d}": [("m1", TSDataType.DOUBLE)]
        for i in range(30)
    }
    _write_tree_rows(path, devices, t_count=2)

    # 2. 加载文件并校验 DataFrame repr 的截断展示
    with TsFileDataFrame(str(path), show_progress=False) as tsdf:
        rendered = repr(tsdf)
        assert "30 time series" in rendered
        assert "_col_1" in rendered
        assert "field" in rendered
        assert "count" in rendered
        assert "1970-01-01 08:00:00.001" in rendered
        assert ".000" not in rendered
        assert "..." in rendered

        # 3. 调用 DataFrame.show(max_rows)，校验输出表头、截断和毫秒时间格式
        tsdf.show(max_rows=6)
        df_show = capsys.readouterr().out
        assert "TsFileDataFrame(tree model" in df_show
        assert "_col_1" in df_show
        assert "..." in df_show
        assert "1970-01-01 08:00:00.001" in df_show

        # 4. 执行对齐查询并校验 AlignedTimeseries.show 的截断展示
        aligned = tsdf.loc[0:1, [0, 1]]
        aligned.show(max_rows=1)
        aligned_show = capsys.readouterr().out
        assert "AlignedTimeseries" in aligned_show
        assert "time" in aligned_show
        assert "root.db.d00.m1" in aligned_show
        assert "root.db.d01.m1" in aligned_show
        assert "..." in aligned_show
        assert "1970-01-01 08:00:00.001" in aligned_show

    # 5. 生成稀疏时间轴文件，用于校验展示中的 NaN 和毫秒格式
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

    # 6. 加载稀疏文件并校验 AlignedTimeseries repr 展示内容
    with TsFileDataFrame(str(sparse_path), show_progress=False) as tsdf:
        sparse_aligned = tsdf.loc[1:2, ["root.a.d1.m1", "root.a.d2.m1"]]
        rendered_sparse = repr(sparse_aligned)
        assert "AlignedTimeseries(2 rows, 2 series)" in rendered_sparse
        assert "time" in rendered_sparse
        assert "root.a.d1.m1" in rendered_sparse
        assert "root.a.d2.m1" in rendered_sparse
        assert "NaN" in rendered_sparse
        assert "1970-01-01 08:00:00.001" in rendered_sparse
        assert "1970-01-01 08:00:00.002" in rendered_sparse


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


def test_tree_dataframe_and_series_lifecycle(tmp_path):
    """用例 109-113：验证子集 close 是 no-op，根 DataFrame close 幂等，关闭后访问 df/series 均报错。"""
    # 1. 生成基础树模型TsFile文件
    path = tmp_path / "tree.tsfile"
    _write_tree_file(path)

    # 2. 创建根 DataFrame、单序列对象和子集视图
    tsdf = TsFileDataFrame(str(path), show_progress=False)
    series = tsdf[0]
    subset = tsdf[:1]

    # 3. 关闭子集视图，确认是 no-op 且不影响根 DataFrame
    with pytest.warns(RuntimeWarning, match="no-op"):
        subset.close()
    assert len(tsdf) == 3

    # 4. 连续关闭根 DataFrame，确认 close 幂等
    tsdf.close()
    tsdf.close()

    # 5. 关闭后访问 DataFrame 和既有 Timeseries 均应报错
    with pytest.raises(RuntimeError, match="closed"):
        tsdf[0]
    with pytest.raises(RuntimeError, match="closed"):
        series[0]
    with pytest.raises(RuntimeError, match="closed"):
        tsdf.loc[0:1, [0]]


@pytest.mark.xfail(
    reason="产品待审查：CSV 用例 130 中复用 reader 跨设备名读取第二条序列为空",
    strict=True,
    raises=AssertionError,
)
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


def test_tree_show_progress_reports_to_stderr(tmp_path, capsys):
    """用例 138、139：验证 show_progress=True 时，DataFrame 加载和树 reader 元数据扫描都会向 stderr 输出进度。"""
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

    # 4. 生成单独 reader 文件，用于验证树模型元数据扫描进度
    reader_path = tmp_path / "reader.tsfile"
    _write_tree_file(reader_path)
    reader = TsFileSeriesReader(str(reader_path), show_progress=True)
    try:
        # 5. 校验 reader 元数据扫描输出设备数和序列数
        stderr = capsys.readouterr().err
        assert "Reading TsFile metadata: 0/2 devices" in stderr
        assert "Reading TsFile metadata (tree): 2 device(s), 3 series ... done" in stderr
    finally:
        reader.close()


def test_tree_device_path_rebuild_reports_unsupported_segments():
    """用例 131、132：验证树设备路径重建时，对含点号路径段和中间 None 段给出明确异常。"""
    # 1. 构造 root 表入口，模拟树模型根段
    table_entry = SimpleNamespace(table_name="root")

    # 2. 设备路径段包含点号时，应提示底层 cwrapper 路径 API 不支持
    with pytest.raises(NotImplementedError, match="Tree device segment with"):
        TsFileSeriesReader._build_tree_device_path(
            table_entry, SimpleNamespace(tag_values=("a.b",))
        )

    # 3. 设备路径中间出现 None 时，应提示非法空路径段
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
