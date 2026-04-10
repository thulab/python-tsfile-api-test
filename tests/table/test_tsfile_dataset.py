import os

import pandas as pd
import pytest
import numpy as np
from datetime import date
import random

from tsfile import (
    TsFileReader, TsFileTableWriter, TableSchema, ColumnSchema, Tablet,
    TSDataType, ColumnCategory, TsFileDataFrame, Timeseries, to_dataframe
)

"""
标题：TsFileDataFrame 功能测试
日期：2026/04
"""

# 测试用的常量列名和类型
TABLE_NAME = "Test_Table"
FIELD_COLUMNS = ["Boolean_Field", "Int32_Field", "Int64_Field", "Float_Field", "Double_Field", "Timestamp_Field", ]
FIELD_TYPES = [TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE,
               TSDataType.TIMESTAMP]
TAG_COLUMNS = ["Device_id1", "Device_id2"]
TAG_TYPES = [TSDataType.STRING, TSDataType.STRING]


# ==================== Fixtures ====================

@pytest.fixture
def tsfile_path(tmp_path):
    """生成临时tsfile文件路径"""
    return str(tmp_path / "test.tsfile")


@pytest.fixture
def tsfile_path2(tmp_path):
    """生成第二个临时tsfile文件路径"""
    return str(tmp_path / "test2.tsfile")


@pytest.fixture
def test_dir(tmp_path):
    """生成临时测试目录"""
    return str(tmp_path / "test_dir")


def add_value_by_type(tablet, column_name, data_type, row_idx, is_contains_null_values):
    """根据数据类型添加值"""
    # 判断是否为需要包含空值
    if is_contains_null_values:
        # 偶数行设为空值
        if row_idx % 2 == 0:
            return

    if data_type == TSDataType.BOOLEAN:
        tablet.add_value_by_name(column_name, row_idx, bool(row_idx % 2 == 0))
    elif data_type == TSDataType.INT32:
        tablet.add_value_by_name(column_name, row_idx, int(random.randint(-2147483648, 2147483647)))
    elif data_type == TSDataType.INT64:
        tablet.add_value_by_name(column_name, row_idx, int(random.randint(-9223372036854775808, 9223372036854775807)))
    elif data_type == TSDataType.FLOAT:
        tablet.add_value_by_name(column_name, row_idx,
                                 float(123456789.123456789) if row_idx % 2 == 0 else float(-123456789.123456789))
    elif data_type == TSDataType.DOUBLE:
        tablet.add_value_by_name(column_name, row_idx,
                                 float(123456789.123456789) if row_idx % 2 == 0 else float(-123456789.123456789))
    elif data_type == TSDataType.TEXT:
        tablet.add_value_by_name(column_name, row_idx, str(''.join(random.choices(''.join(chr(i) for i in range(0x4e00,
                                                                                                                0x9fa6)) + 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' + r"!@#$%^&*()_+-=[]{}|;':,./<>?",
                                                                                  k=10))))
    elif data_type == TSDataType.TIMESTAMP:
        tablet.add_value_by_name(column_name, row_idx, int(random.randint(-9223372036854775808, 9223372036854775807)))
    elif data_type == TSDataType.DATE:
        tablet.add_value_by_name(column_name, row_idx,
                                 date(random.randint(1970, 9999), random.randint(1, 12), random.randint(1, 20)))
    elif data_type == TSDataType.BLOB:
        tablet.add_value_by_name(column_name, row_idx, ''.join(random.choices(''.join(chr(i) for i in range(0x4e00,
                                                                                                            0x9fa6)) + 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' + r"!@#$%^&*()_+-=[]{}|;':,./<>?",
                                                                              k=10)).encode('utf-8'))
    elif data_type == TSDataType.STRING:
        tablet.add_value_by_name(column_name, row_idx, str(''.join(random.choices(''.join(chr(i) for i in range(0x4e00,
                                                                                                                0x9fa6)) + 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' + r"!@#$%^&*()_+-=[]{}|;':,./<>?",
                                                                                  k=10))))


def create_tsfile1(file_path, table_name, tag_columns, tag_types, field_columns, field_types, row_num=100,
                   start_timestamp=0, is_spans_time_partitions=False, is_contains_null_values=True,
                   is_same_device_name=True):
    """根据提供的要求创建需要的测试TsFile文件

    Args:
        file_path：文件或目录路径
        table_name：表名
        tag_columns：TAG列列名
        tag_types：TAG列数据类型
        field_columns：FIELD列列名
        field_types：FIELD列数据类型
        row_num：数据行，默认为100
        start_timestamp：开始时间戳，默认为0
        is_same_device_name：是否设备名相同，默认为True
        is_contains_null_values：是否包含空值，默认为True
        is_spans_time_partitions：是否时间戳跨时间分区，默认为False
    """
    # 1. 创建文件
    if os.path.exists(file_path):
        os.remove(file_path)

    # 2. 创建表结构
    columns = []
    for tag_col, tag_type in zip(tag_columns, tag_types):
        columns.append(ColumnSchema(tag_col, tag_type, ColumnCategory.TAG))
    for col_name, col_type in zip(field_columns, field_types):
        columns.append(ColumnSchema(col_name, col_type, ColumnCategory.FIELD))
    table_schema = TableSchema(table_name, columns)

    # 3. 写入数据
    with TsFileTableWriter(file_path, table_schema) as writer:
        tablet = Tablet(tag_columns + field_columns, tag_types + field_types, row_num)
        for i in range(row_num):
            # 添加时间戳
            if is_spans_time_partitions:
                tablet.add_timestamp(i, start_timestamp + 604800000)
            else:
                tablet.add_timestamp(i, start_timestamp + i)
            # 为TAG列添加值
            for tag_column in tag_columns:
                # 判断是否需要相同设备名
                if is_same_device_name:
                    tablet.add_value_by_name(tag_column, i, "Device1_中文_")
                else:
                    tablet.add_value_by_name(tag_column, i, "Device1_中文_" + str(i))
            # 为FIELD列添加值
            for col_name, col_type in zip(field_columns, field_types):
                add_value_by_type(tablet, col_name, col_type, i, is_contains_null_values)
        writer.write_table(tablet)
        writer.flush()
        writer.close()

    return file_path


def create_tsfile2(file_path, table_name, num_rows=100):
    # 1. 创建文件
    if os.path.exists(file_path):
        os.remove(file_path)

    # 2. 创建表结构
    schema = TableSchema(
        table_name,
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("tag1", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("tag2", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("bool_", TSDataType.BOOLEAN, ColumnCategory.FIELD),
            ColumnSchema("int32_", TSDataType.INT32, ColumnCategory.FIELD),
            ColumnSchema("int64_", TSDataType.INT64, ColumnCategory.FIELD),
            ColumnSchema("float_", TSDataType.FLOAT, ColumnCategory.FIELD),
            ColumnSchema("double_", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("timestamp_", TSDataType.TIMESTAMP, ColumnCategory.FIELD),
        ],
    )

    # 3. 创建数据
    data = {
        'time': range(num_rows),
        'tag1': ['tag1_' + str(i % 10) if i % 3 != 0 else None for i in range(num_rows)],
        'tag2': ['tag2_' + str(i % 5) if i % 5 != 0 else None for i in range(num_rows)],
        'bool_': [i % 2 == 0 if i % 3 != 0 else None for i in range(num_rows)],
        'int32_': [i * 2 if i % 4 != 0 else None for i in range(num_rows)],
        'int64_': [i * 3 if i % 5 != 0 else None for i in range(num_rows)],
        'float_': [i * 1.5 if i % 3 != 0 else None for i in range(num_rows)],
        'double_': [i * 2.5 if i % 4 != 0 else None for i in range(num_rows)],
        'timestamp_': [i * 1000 if i % 4 != 0 else None for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    df.int32_ = df.int32_.astype('Int32')
    df.int64_ = df.int64_.astype('Int64')
    df.bool_ = df.bool_.astype('boolean')
    df.float_ = df.float_.astype(np.float32)
    df.timestamp_ = df.timestamp_.astype('Int64')

    # 4. 写入数据
    with TsFileTableWriter(str(file_path), schema) as writer:
        writer.write_dataframe(df)


# ============================================
# 1. TsFileDataFrame - 加载多个TsFile文件测试
# ============================================

class TestTsFileDataFramePaths:
    """测试TsFileDataFrame - 加载文件/目录"""

    def test_single_file_path(self, tsfile_path):
        """测试单条文件路径"""
        # 1. 创建文件
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        # 2. 加载文件
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            # 3. 验证结果
            assert len(tsdf) == 6, f"Expected 6 series, got {len(tsdf)}"

    def test_multiple_file_paths_different_timeseries_same_timestamp(self, tsfile_path, tsfile_path2):
        """测试多条文件路径 - 不同时间序列重复时间戳"""
        # 1. 创建文件
        create_tsfile1(tsfile_path, "t1", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        create_tsfile1(tsfile_path2, "t2", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        # 2. 加载文件
        with TsFileDataFrame([tsfile_path, tsfile_path2], show_progress=False) as tsdf:
            # 3. 验证结果
            assert len(tsdf) == 12, f"Expected 12 series, got {len(tsdf)}"

    def test_multiple_file_paths_same_timeseries_same_timestamp(self, tsfile_path, tsfile_path2):
        """测试多条文件路径 - 相同时间序列重复时间戳"""
        # 1. 创建文件
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        create_tsfile1(tsfile_path2, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        # 2. 加载文件，验证相同设备重复时间戳异常
        with TsFileDataFrame([tsfile_path, tsfile_path2], show_progress=False) as tsdf:
            with pytest.raises(ValueError):
                print(tsdf[0].timestamps)

    def test_multiple_file_paths_same_timeseries_different_timestamp(self, tsfile_path, tsfile_path2):
        """测试多条文件路径 - 相同时间序列不重复时间戳"""
        # 1. 创建文件
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20,
                       start_timestamp=0)
        create_tsfile1(tsfile_path2, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20,
                       start_timestamp=100)
        # 2. 加载文件
        with TsFileDataFrame([tsfile_path, tsfile_path2], show_progress=False) as tsdf:
            # 3. 验证结果
            assert len(tsdf) == 6, f"Expected 6 series, got {len(tsdf)}"
            assert len(tsdf[0][0:40]) == 40

    def test_nonexistent_file_path(self, tmp_path):
        """测试不存在的文件路径"""
        nonexistent_path = str(tmp_path / "nonexistent.tsfile")
        with pytest.raises(FileNotFoundError):
            TsFileDataFrame(nonexistent_path, show_progress=False)

    def test_directory_path_same_timeseries(self, test_dir, tsfile_path):
        """测试目录路径，每个文件序列名不一样"""
        # 1. 创建文件
        os.makedirs(test_dir, exist_ok=True)
        for idx in range(2):
            file_path = os.path.join(test_dir, f"file_{idx}.tsfile")
            create_tsfile1(file_path, f"table_{idx}", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)

        # 2. 加载文件
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            # 3. 验证结果
            assert len(tsdf) == 12, f"Expected 12 series, got {len(tsdf)}"

    def test_directory_path_different_timeseries(self, test_dir, tsfile_path):
        """测试目录路径，每个文件序列名一样"""
        # 1. 创建文件
        os.makedirs(test_dir, exist_ok=True)
        for idx in range(2):
            file_path = os.path.join(test_dir, f"file_{idx}.tsfile")
            create_tsfile1(file_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)

        # 2. 加载文件
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            with pytest.raises(ValueError):
                print(tsdf[0][10])

    def test_directory_path_contains_non_tsfile(self, test_dir, tsfile_path):
        """测试目录路径，包含非TsFile文件"""
        # 1. 创建TsFile文件
        os.makedirs(test_dir, exist_ok=True)
        for idx in range(2):
            file_path = os.path.join(test_dir, f"file_{idx}.tsfile")
            create_tsfile1(file_path, f"table_{idx}", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)

        # 2. 创建一些非 .tsfile 文件，验证会被正确过滤
        with open(os.path.join(test_dir, "readme.txt"), "w") as f:
            f.write("这是一个说明文件")
        with open(os.path.join(test_dir, "data.csv"), "w") as f:
            f.write("col1,col2\n1,2\n")
        with open(os.path.join(test_dir, "config.json"), "w") as f:
            f.write('{"key": "value"}')
        os.makedirs(os.path.join(test_dir, "subdir"), exist_ok=True)
        with open(os.path.join(test_dir, "subdir", "notes.md"), "w") as f:
            f.write("# 笔记")

        # 3. 加载包含非TsFile目录
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            # 4. 验证结果
            assert len(tsdf) == 12, f"Expected 12 series, got {len(tsdf)}"

    def test_nested_directory_path_contains_non_tsfile(self, test_dir, tsfile_path):
        """测试嵌套目录路径，包含非 TsFile 文件"""
        # 1. 创建嵌套目录结构
        nested_dir = os.path.join(test_dir, "d1", "d2", "d3")
        os.makedirs(nested_dir, exist_ok=True)

        # 2. 在嵌套目录中创建 TsFile 文件
        for idx in range(3):
            file_path = os.path.join(nested_dir, f"file_{idx}.tsfile")
            create_tsfile1(file_path, f"table_{idx}", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)

        # 3. 在各层目录中创建一些非 .tsfile 文件，验证会被正确过滤
        with open(os.path.join(test_dir, "readme.txt"), "w") as f:
            f.write("根目录说明文件")
        with open(os.path.join(test_dir, "d1", "data.csv"), "w") as f:
            f.write("col1,col2\n1,2\n")
        with open(os.path.join(test_dir, "d1", "d2", "config.json"), "w") as f:
            f.write('{"key": "value"}')
        with open(os.path.join(nested_dir, "notes.md"), "w") as f:
            f.write("# 嵌套目录笔记")
        with open(os.path.join(nested_dir, "test.log"), "w") as f:
            f.write("日志文件")

        # 4. 创建额外的子目录和文件
        os.makedirs(os.path.join(test_dir, "d1", "d2", "subdir"), exist_ok=True)
        with open(os.path.join(test_dir, "d1", "d2", "subdir", "extra.txt"), "w") as f:
            f.write("额外文件")

        # 5. 加载根目录，验证能正确递归查找所有嵌套目录中的 .tsfile 文件
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            # 6. 验证结果：3 个文件，每个文件 6 个序列 = 18 个序列
            assert len(tsdf) == 18, f"Expected 18 series, got {len(tsdf)}"
            # 7. 验证能找到所有表
            series_list = tsdf.list_timeseries()
            assert len(series_list) == 18
            assert "table_0" in series_list[0]
            assert "table_1" in series_list[6]
            assert "table_2" in series_list[12]

    def test_directory_without_tsfile(self, tmp_path):
        """测试目录内无tsfile文件"""
        empty_dir = str(tmp_path / "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="No .tsfile files found"):
            TsFileDataFrame(empty_dir, show_progress=False)


# ============================================
# 2. TsFileDataFrame - 内部函数方法测试
# ============================================

class TestTsFileDataFrameBasic:
    """测试TsFileDataFrame - 内部函数方法（）"""

    def test_all_types(self, tsfile_path):
        """测试单设备所有类型"""
        field_columns = ["boolean_field", "int32_field", "int64_field", "float_field", "double_field",
                         "timestamp_field", "string_field", "blob_field", "text_field", "date_field"]
        field_type = [TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE,
                      TSDataType.TIMESTAMP, TSDataType.TEXT, TSDataType.BLOB, TSDataType.TEXT, TSDataType.DATE]
        tag_columns = ["device_id1", "device_id2"]
        tag_type = [TSDataType.STRING, TSDataType.STRING]
        create_tsfile1(tsfile_path, TABLE_NAME, tag_columns, tag_type, field_columns, field_type, row_num=20,
                       is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            assert len(tsdf) == 6
            assert len(tsdf.list_timeseries()) == 6
            assert tsdf.list_timeseries(TABLE_NAME.lower())[0] == "test_table.Device1_中文_.Device1_中文_.boolean_field"
            assert tsdf.list_timeseries(TABLE_NAME.lower())[5] == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf.list_timeseries("test_table.Device1_中文_")[
                       0] == "test_table.Device1_中文_.Device1_中文_.boolean_field"
            assert tsdf.list_timeseries("test_table.Device1_中文_.Device1_中文_")[
                       5] == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf.list_timeseries("test_table.Device1_中文_.Device1_中文_.timestamp_field")[
                       0] == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf[
                       "test_table.Device1_中文_.Device1_中文_.timestamp_field"].name == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf[5].name == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf[-1].name == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf[0:6][5].name == "test_table.Device1_中文_.Device1_中文_.timestamp_field"
            assert tsdf[2:4][0].name == "test_table.Device1_中文_.Device1_中文_.int64_field"
            assert tsdf[4:100][0].name == "test_table.Device1_中文_.Device1_中文_.double_field"

    def test_len_no_tag_table(self, tsfile_path):
        """测试没有TAG列的表"""
        create_tsfile1(tsfile_path, TABLE_NAME, [], [], FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            assert len(tsdf) == 6

    def test_len_no_field_table(self, tsfile_path):
        """测试没有FIELD列的表，没有有效序列会报错"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, [], [], row_num=20)
        with  pytest.raises(ValueError, match="No valid numeric series found in TsFile"):
            TsFileDataFrame(tsfile_path, show_progress=False)

    def test_len_multiple_devices(self, tsfile_path):
        """测试多设备 - 不同 tag 组合会创建不同的设备"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20,
                       is_same_device_name=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            assert len(tsdf) > 6, f"Expected more than 6 series due to multiple devices, got {len(tsdf)}"

    def test_list_timeseries_table_prefix_uppercase(self, tsfile_path):
        """测试Table前缀 - 大写英文"""
        create_tsfile1(tsfile_path, "TABLE1", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_list = tsdf.list_timeseries("TABLE1".lower())
            assert len(series_list) == 6

    def test_list_timeseries_table_prefix_lowercase(self, tsfile_path):
        """测试Table前缀 - 小写英文"""
        create_tsfile1(tsfile_path, "lowercase_table", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_list = tsdf.list_timeseries("lowercase_table")
            assert len(series_list) == 6

    def test_list_timeseries_table_prefix_chinese(self, tsfile_path):
        """测试Table前缀 - 中文"""
        create_tsfile1(tsfile_path, "测试表", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_list = tsdf.list_timeseries("测试表")
            assert len(series_list) == 6

    def test_list_timeseries_table_prefix_numbers(self, tsfile_path):
        """测试Table前缀 - 数字"""
        create_tsfile1(tsfile_path, "123456", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_list = tsdf.list_timeseries("123456")
            assert len(series_list) == 6

    def test_list_timeseries_table_prefix_nonexistent(self, tsfile_path):
        """测试Table前缀 - 不存在的表"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_list = tsdf.list_timeseries("nonexistent_table")
            assert len(series_list) == 0

    def test_special_column_name(self, tsfile_path):
        table_name = "table."
        create_tsfile1(tsfile_path, table_name, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            ts = tsdf.list_timeseries("table\\.")
            assert len(ts) == 6


# ============================================
# 3. Timeseries 序列访问测试
# ============================================

class TestTimeseries:
    """测试Timeseries - 序列访问"""

    def test_timeseries_index_valid(self, tsfile_path):
        """测试索引访问 - 存在的索引"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20,
                       start_timestamp=10, is_contains_null_values=True)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            for i in range(6):
                ts = tsdf[i]
                assert ts.name == tsdf.list_timeseries()[i]
                assert len(ts) == 20
                assert ts.stats == {'start_time': 10, 'end_time': 29, 'count': 20}
                # assert ts[0] is None # 待确认，nan != None
                assert len(ts[0:len(ts)]) == 20
                assert ts.timestamps[0] == 10

    def test_timeseries_index_negative(self, tsfile_path):
        """测试索引访问 - 负索引从后往前取"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            ts = tsdf[-2]
            assert ts.name == "test_table.Device1_中文_.Device1_中文_.double_field"
            assert len(ts) == 20

    def test_timeseries_index_out_of_range(self, tsfile_path):
        """测试索引访问 - 超出实际数量的索引 预期报错索引越界"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            with pytest.raises(IndexError):
                print(tsdf[100])

    def test_timeseries_name(self, tsfile_path):
        """测试序列名访问"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            ts = tsdf[tsdf.list_timeseries()[0]]
            assert ts.name == tsdf.list_timeseries()[0]
            assert len(ts) == 20

    def test_timeseries_name_nonexistent(self, tsfile_path):
        """测试序列名访问 - 不存在的序列名"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            with pytest.raises(KeyError):
                print(tsdf["nonexistent_table.nonexistent_tag.nonexistent_field"])


# ============================================
# 4. AlignedTimeseries 按时间戳对齐序列测试
# ============================================

class TestTsFileDataFrameLoc:
    """测试TsFileDataFrame.loc - 高性能对齐查询"""

    def test_AlignedTimeseries_Basic(self, tsfile_path):
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=40,
                       is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()
            data = tsdf.loc[0:20, series_names]
            assert len(data) == 21
            assert len(data.timestamps) == 21
            assert data.timestamps[0] == np.int64(0)
            assert data.timestamps[20] == np.int64(20)
            assert len(data.values) == 21
            assert data.shape == (21, 6)
            assert len(data[0]) == 6
            assert len(data[0:10]) == 10
            # data.show(40)
            # print(data)


    @pytest.mark.skip(reason="loc时间切片功能存在bug，含空值时，返回超出指定时间范围的数据")
    def test_loc_time_slice_within_range_contain_null_value(self, tsfile_path):
        """测试时间切片 - 数据含空值"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=40, is_contains_null_values=True)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()
            data = tsdf.loc[0:20, series_names]
            assert len(data) <= 21

    def test_loc_time_slice_within_range(self, tsfile_path):
        """测试时间切片 - 不超出范围"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=40, is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()
            data = tsdf.loc[0:20, series_names]
            assert len(data) == 21

    def test_loc_time_slice_left_exceed(self, tsfile_path):
        """测试时间切片 - 左切片超出"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20, is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()
            data = tsdf.loc[-100:10, series_names]
            assert len(data) == 11

    def test_loc_time_slice_right_exceed(self, tsfile_path):
        """测试时间切片 - 右切片超出"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20, is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()
            data = tsdf.loc[10:100, series_names]
            assert len(data) == 10

    def test_loc_time_slice_all_exceed(self, tsfile_path):
        """测试时间切片 - 全超出"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20, is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()
            data = tsdf.loc[100:200, series_names]
            assert len(data) == 0

    def test_loc_index1(self, tsfile_path):
        """测试使用索引"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20, is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            data = tsdf.loc[6:15, [0, 1]]
            assert len(data) == 10

    def test_loc_index2(self, tsfile_path):
        """测试使用索引"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20, is_contains_null_values=False)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            data = tsdf.loc[0:10, [0, 1, 2, 3, 4, 5]]
            assert len(data) == 11

    def test_loc_series_names_nonexistent_table_prefix(self, tsfile_path):
        """测试序列名 - 不存在的Table前缀"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            with pytest.raises(KeyError):
                print(tsdf.loc[0:10, ["nonexistent_table.device1.s1"]])


# ============================================
# 5. 特殊测试
# ============================================

class TestBoundaryCases:
    """测试特殊情况"""

    def test_close_dataframe_twice(self, tsfile_path):
        """测试多次关闭DataFrame"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        df.close()
        df.close()

    def test_access_after_close(self, tsfile_path):
        """测试关闭后访问"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES, row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        df.close()
        with pytest.raises(RuntimeError, match="closed"):
            print(df[0])

    @pytest.mark.skip(reason="所有值都为空，实际查询会出现问题")
    def test_all_null_values(self, tsfile_path):
        """测试所有值都为空"""
        columns = [
            ColumnSchema("device_id", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("s1", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("s2", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("s3", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("s4", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ]
        table_schema = TableSchema(TABLE_NAME, columns)
        with TsFileTableWriter(tsfile_path, table_schema) as writer:
            tablet = Tablet(["device_id", "s1", "s2", "s3", "s4",], [TSDataType.STRING, TSDataType.DOUBLE, TSDataType.DOUBLE, TSDataType.DOUBLE, TSDataType.DOUBLE], 10)
            for i in range(10):
                tablet.add_timestamp(i, i)
            writer.write_table(tablet)

        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            print(len(tsdf))
            print(tsdf[0][0])

# ============================================
# 6. 数据类型专项测试
# ============================================

class TestAllDataTypesValue:
    """测试所有数据类型"""

    def test_boolean_type(self, tsfile_path):
        """测试BOOLEAN类型"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, ["bool_field"], [TSDataType.BOOLEAN],
                       row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        series = df[0]
        assert len(series) == 20
        df.close()

    def test_int32_type(self, tsfile_path):
        """测试INT32类型"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, ["int32_field"], [TSDataType.INT32],
                       row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        series = df[0]
        assert len(series) == 20
        df.close()

    def test_int64_type(self, tsfile_path):
        """测试INT64类型"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, ["int64_field"], [TSDataType.INT64],
                       row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        series = df[0]
        assert len(series) == 20
        df.close()

    def test_float_type(self, tsfile_path):
        """测试FLOAT类型"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, ["float_field"], [TSDataType.FLOAT],
                       row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        series = df[0]
        assert len(series) == 20
        df.close()

    def test_double_type(self, tsfile_path):
        """测试DOUBLE类型"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, ["double_field"], [TSDataType.DOUBLE],
                       row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        series = df[0]
        assert len(series) == 20
        df.close()

    def test_timestamp_type(self, tsfile_path):
        """测试TIMESTAMP类型"""
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, ["timestamp_field"], [TSDataType.TIMESTAMP],
                       row_num=20)
        df = TsFileDataFrame(tsfile_path, show_progress=False)
        series = df[0]
        assert len(series) == 20
        df.close()


# ============================================
# 7. 性能测试
# ============================================

class TestPerformance:
    """测试性能 - 并行加载和对齐查询瓶颈"""

    # ==================== 并行加载测试 ====================

    def test_parallel_load_multiple_files(self, test_dir):
        """测试并行加载多个文件 - 自动使用线程池并行扫描元数据"""
        import time
        import multiprocessing

        # 1. 创建多个文件
        os.makedirs(test_dir, exist_ok=True)
        file_count = multiprocessing.cpu_count()
        for idx in range(file_count):
            file_path = os.path.join(test_dir, f"perf_file_{idx}.tsfile")
            create_tsfile1(file_path, f"table_{idx}", TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES,
                           row_num=10000, is_contains_null_values=False)

        # 2. 记录加载时间
        start_time = time.time()
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            load_time = time.time() - start_time
            # 3. 验证结果
            assert len(tsdf) == file_count * 6, f"Expected {file_count * 6} series, got {len(tsdf)}"
            assert len(tsdf[0].timestamps) == 10000
            assert load_time < 10.0, f"Parallel load took too long: {load_time}s for {file_count} files"

    def test_parallel_load_nested_directory(self, test_dir):
        """测试并行加载嵌套目录中的多个文件"""
        import time

        # 1. 创建嵌套目录结构
        nested_dirs = [test_dir]
        for i in range(3):
            nested_dir = os.path.join(nested_dirs[-1], f"level_{i}")
            os.makedirs(nested_dir, exist_ok=True)
            nested_dirs.append(nested_dir)

        # 2. 在每个目录中创建文件
        file_count = 0
        for idx, dir_path in enumerate(nested_dirs[:3]):  # 在前3层各创建文件
            for sub_idx in range(2):
                file_path = os.path.join(dir_path, f"nested_file_{idx}_{sub_idx}.tsfile")
                create_tsfile1(file_path, f"nested_table_{idx}_{sub_idx}", TAG_COLUMNS, TAG_TYPES,
                               FIELD_COLUMNS, FIELD_TYPES, row_num=50, is_contains_null_values=False)
                file_count += 1

        # 3. 加载根目录，测试递归并行扫描
        start_time = time.time()
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            load_time = time.time() - start_time

            # 4. 验证结果 - 所有嵌套文件都被正确加载
            assert len(tsdf) == file_count * 6, f"Expected {file_count * 6} series, got {len(tsdf)}"
            assert load_time < 15.0, f"Nested directory load took too long: {load_time}s"

    # ==================== 对齐查询瓶颈测试 ====================

    def test_loc_alignment_small_scale(self, tsfile_path):
        """测试loc对齐查询 - 小规模数据（数千时间戳）"""
        import time

        # 1. 创建小规模测试数据
        row_num = 1000
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES,
                       row_num=row_num, is_contains_null_values=False)

        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()

            # 2. 测试小规模对齐查询
            start_time = time.time()
            data = tsdf.loc[0:500, series_names]
            query_time = time.time() - start_time

            # 3. 验证结果
            assert len(data) == 501
            assert data.shape == (501, 6)
            # 小规模查询应该很快
            assert query_time < 1.0, f"Small scale loc query took too long: {query_time}s"

    def test_loc_alignment_medium_scale(self, tsfile_path):
        """测试loc对齐查询 - 中规模数据（数万时间戳）"""
        import time

        # 1. 创建中规模测试数据
        row_num = 10000
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES,
                       row_num=row_num, is_contains_null_values=False)

        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()

            # 2. 测试中规模对齐查询
            start_time = time.time()
            data = tsdf.loc[0:5000, series_names]
            query_time = time.time() - start_time

            # 3. 验证结果
            assert len(data) == 5001
            assert data.shape == (5001, 6)
            # 中规模查询时间记录（用于基准对比）
            print(f"Medium scale loc query time: {query_time}s for {5001} timestamps")
            assert query_time < 5.0, f"Medium scale loc query took too long: {query_time}s"

    @pytest.mark.xfail(reason="loc时间切片功能存在bug，含空值时返回超出指定时间范围的数据")
    def test_loc_alignment_with_null_values(self, tsfile_path):
        """测试loc对齐查询 - 含空值时的合并去重和NaN填充性能"""
        import time

        # 1. 创建含空值的测试数据
        row_num = 5000
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES,
                       row_num=row_num, is_contains_null_values=True)

        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()

            # 2. 测试含空值的对齐查询（需要NaN填充）
            start_time = time.time()
            data = tsdf.loc[0:2500, series_names]
            query_time = time.time() - start_time

            # 3. 验证结果 - 空值处理会增加一定开销
            # 注意：由于已知bug，含空值时可能返回超出指定范围的数据
            assert len(data) == 2501
            # 含空值的查询时间应该合理
            assert query_time < 3.0, f"Loc query with null values took too long: {query_time}s"

    def test_loc_alignment_multiple_files_different_timestamps(self, test_dir):
        """测试loc对齐查询 - 多文件不同时间戳范围的合并对齐"""
        import time

        # 1. 创建多个文件，每个文件有不同的时间戳范围
        os.makedirs(test_dir, exist_ok=True)
        file_count = 4

        for idx in range(file_count):
            file_path = os.path.join(test_dir, f"align_file_{idx}.tsfile")
            # 每个文件的时间戳范围不同，测试合并对齐
            create_tsfile1(file_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES,
                           row_num=1000, start_timestamp=idx * 1000, is_contains_null_values=False)

        # 2. 加载多文件
        with TsFileDataFrame(test_dir, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()

            # 3. 测试跨文件的对齐查询
            start_time = time.time()
            data = tsdf.loc[0:2000, series_names]  # 跨越前2个文件的时间戳范围
            query_time = time.time() - start_time

            # 4. 验证结果 - 多文件合并对齐
            assert len(data) == 2001
            assert data.shape == (2001, 6)
            assert query_time < 5.0, f"Multi-file loc alignment took too long: {query_time}s"

    def test_loc_alignment_large_series_count(self, tsfile_path):
        """测试loc对齐查询 - 大量序列的对齐性能"""
        import time

        # 1. 创建有大量FIELD列的表（增加序列数量）
        field_columns_large = [f"field_{i}" for i in range(20)]
        field_types_large = [TSDataType.DOUBLE for _ in range(20)]
        row_num = 2000

        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, field_columns_large, field_types_large,
                       row_num=row_num, is_contains_null_values=False)

        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()

            # 2. 验证序列数量
            assert len(series_names) == 20, f"Expected 20 series, got {len(series_names)}"

            # 3. 测试大量序列的对齐查询
            start_time = time.time()
            data = tsdf.loc[0:1000, series_names]
            query_time = time.time() - start_time

            # 4. 验证结果 - 大量序列对齐
            assert len(data) == 1001
            assert data.shape == (1001, 20)
            assert query_time < 3.0, f"Large series count loc alignment took too long: {query_time}s"

    def test_loc_alignment_full_range(self, tsfile_path):
        """测试loc对齐查询 - 全范围查询性能"""
        import time

        # 1. 创建测试数据
        row_num = 5000
        create_tsfile1(tsfile_path, TABLE_NAME, TAG_COLUMNS, TAG_TYPES, FIELD_COLUMNS, FIELD_TYPES,
                       row_num=row_num, is_contains_null_values=False)

        with TsFileDataFrame(tsfile_path, show_progress=False) as tsdf:
            series_names = tsdf.list_timeseries()

            # 2. 测试全范围对齐查询
            start_time = time.time()
            data = tsdf.loc[0:row_num - 1, series_names]
            query_time = time.time() - start_time

            # 3. 验证结果
            assert len(data) == row_num
            assert data.shape == (row_num, 6)
            print(f"Full range loc query time: {query_time}s for {row_num} timestamps")
            assert query_time < 5.0, f"Full range loc query took too long: {query_time}s"
