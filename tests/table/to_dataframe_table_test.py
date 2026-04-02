import os
from datetime import date
import pytest
import numpy as np
from tsfile import to_dataframe, TableNotExistError, TableSchema, ColumnSchema, TSDataType, ColumnCategory, \
    TsFileTableWriter, Tablet, ColumnNotExistError

"""
标题：表模型 to_dataframe 接口功能测试
日期：2025/12
优化：2026/04 - 使用fixture减少重复代码，拆分测试函数，使用pytest参数化
"""

# ==================== Fixtures ====================

@pytest.fixture
def tsfile_path(tmp_path):
    """生成临时tsfile文件路径"""
    return str(tmp_path / "to_dataframe_test.tsfile")

@pytest.fixture
def max_row_num():
    """默认测试行数"""
    return 100

@pytest.fixture
def test_table_schema():
    """创建测试表结构"""
    return TableSchema("test_table", [
        ColumnSchema("Device1", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("Device2", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("Value1", TSDataType.BOOLEAN, ColumnCategory.FIELD),
        ColumnSchema("Value2", TSDataType.INT32, ColumnCategory.FIELD),
        ColumnSchema("Value3", TSDataType.INT64, ColumnCategory.FIELD),
        ColumnSchema("Value4", TSDataType.FLOAT, ColumnCategory.FIELD),
        ColumnSchema("Value5", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ColumnSchema("Value6", TSDataType.TEXT, ColumnCategory.FIELD),
        ColumnSchema("Value7", TSDataType.STRING, ColumnCategory.FIELD),
        ColumnSchema("Value8", TSDataType.BLOB, ColumnCategory.FIELD),
        ColumnSchema("Value9", TSDataType.TIMESTAMP, ColumnCategory.FIELD),
        ColumnSchema("Value10", TSDataType.DATE, ColumnCategory.FIELD),
        ColumnSchema("Value11", TSDataType.BOOLEAN, ColumnCategory.FIELD),
        ColumnSchema("Value12", TSDataType.INT32, ColumnCategory.FIELD),
        ColumnSchema("Value13", TSDataType.INT64, ColumnCategory.FIELD),
        ColumnSchema("Value14", TSDataType.FLOAT, ColumnCategory.FIELD),
        ColumnSchema("Value15", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ColumnSchema("Value16", TSDataType.TEXT, ColumnCategory.FIELD),
        ColumnSchema("Value17", TSDataType.STRING, ColumnCategory.FIELD),
        ColumnSchema("Value18", TSDataType.BLOB, ColumnCategory.FIELD),
        ColumnSchema("Value19", TSDataType.TIMESTAMP, ColumnCategory.FIELD),
        ColumnSchema("Value20", TSDataType.DATE, ColumnCategory.FIELD)
    ])

@pytest.fixture
def tsfile_with_data(tsfile_path, test_table_schema, max_row_num):
    """创建包含测试数据的TsFile"""
    column_names = [
        "Device1", "Device2",
        "Value1", "Value2", "Value3", "Value4", "Value5", "Value6", "Value7", "Value8", "Value9", "Value10",
        "Value11", "Value12", "Value13", "Value14", "Value15", "Value16", "Value17", "Value18", "Value19", "Value20"
    ]
    data_types = [
        TSDataType.STRING, TSDataType.STRING,
        TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.TEXT, TSDataType.STRING, TSDataType.BLOB, TSDataType.TIMESTAMP, TSDataType.DATE,
        TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.TEXT, TSDataType.STRING, TSDataType.BLOB, TSDataType.TIMESTAMP, TSDataType.DATE
    ]

    with TsFileTableWriter(tsfile_path, test_table_schema) as writer:
        tablet = Tablet(column_names, data_types, max_row_num)
        for i in range(max_row_num):
            tablet.add_timestamp(i, i)
            tablet.add_value_by_index(0, i, "Device1_" + str(i))
            tablet.add_value_by_name("Device2", i, "Device2_" + str(i))
            tablet.add_value_by_index(2, i, i % 2 == 0)
            tablet.add_value_by_index(3, i, i * 3)
            tablet.add_value_by_index(4, i, i * 4)
            tablet.add_value_by_index(5, i, i * 5.5)
            tablet.add_value_by_index(6, i, i * 6.6)
            tablet.add_value_by_index(7, i, f"string_value_{i}")
            tablet.add_value_by_index(8, i, f"text_value_{i}")
            tablet.add_value_by_index(9, i, f"blob_data_{i}".encode('utf-8'))
            tablet.add_value_by_index(10, i, i * 9)
            tablet.add_value_by_index(11, i, date(2025, 1, i % 20 + 1))
            tablet.add_value_by_name("Value11", i, i % 2 == 0)
            tablet.add_value_by_name("Value12", i, i * 12)
            tablet.add_value_by_name("Value13", i, i * 13)
            tablet.add_value_by_name("Value14", i, i * 14.14)
            tablet.add_value_by_name("Value15", i, i * 15.15)
            tablet.add_value_by_name("Value16", i, f"string_value_{i}")
            tablet.add_value_by_name("Value17", i, f"text_value_{i}")
            tablet.add_value_by_name("Value18", i, f"blob_data_{i}".encode('utf-8'))
            tablet.add_value_by_name("Value19", i, i * 9)
            tablet.add_value_by_name("Value20", i, date(2025, 1, i % 20 + 1))
        writer.write_table(tablet)

    return tsfile_path

# ==================== 辅助验证函数 ====================

def verify_column_value(df, column_name, value_func):
    """验证列值是否正确"""
    time_col = df.iloc[:, 0]
    # tsfile返回的列名为小写，需要转换
    value_col = df[column_name.lower()]
    for i in range(len(df)):
        expected = value_func(time_col.iloc[i])
        assert value_col.iloc[i] == expected

# ==================== 测试默认查询 ====================

def test_default_query(tsfile_with_data, max_row_num):
    """测试默认查询"""
    df = to_dataframe(tsfile_with_data)
    assert df.shape[0] == max_row_num
    assert df.iloc[0, 0] == 0

# ==================== 测试指定列名 ====================

@pytest.mark.parametrize("column_name,value_func", [
    ("Device1", lambda t: "Device1_" + str(t)),
    ("Value1", lambda t: np.bool_(t % 2 == 0)),
    ("Value2", lambda t: np.int32(t * 3)),
    ("Value3", lambda t: np.int64(t * 4)),
    ("Value4", lambda t: np.float32(t * 5.5)),
    ("Value5", lambda t: np.float64(t * 6.6)),
    ("Value6", lambda t: f"string_value_{t}"),
    ("Value7", lambda t: f"text_value_{t}"),
])
def test_single_column_query(tsfile_with_data, max_row_num, column_name, value_func):
    """测试单列查询"""
    df = to_dataframe(tsfile_with_data, column_names=[column_name])
    assert df.shape[0] == max_row_num
    verify_column_value(df, column_name, value_func)

def test_two_columns_query(tsfile_with_data, max_row_num):
    """测试双列查询"""
    df = to_dataframe(tsfile_with_data, column_names=["Device1", "Value1"])
    assert df.shape[0] == max_row_num
    verify_column_value(df, "Device1", lambda t: "Device1_" + str(t))
    verify_column_value(df, "Value1", lambda t: np.bool_(t % 2 == 0))

def test_all_columns_query(tsfile_with_data, max_row_num):
    """测试全部列查询"""
    all_columns = [
        "Device1", "Device2", "Value1", "Value2", "Value3", "Value4", "Value5", "Value6", "Value7", "Value8", "Value9", "Value10"
    ]
    df = to_dataframe(tsfile_with_data, column_names=all_columns)
    assert df.shape[0] == max_row_num
    verify_column_value(df, "Device1", lambda t: "Device1_" + str(t))
    verify_column_value(df, "Device2", lambda t: "Device2_" + str(t))
    verify_column_value(df, "Value1", lambda t: np.bool_(t % 2 == 0))
    verify_column_value(df, "Value2", lambda t: np.int32(t * 3))

# ==================== 测试指定表名 ====================

def test_table_name_lowercase(tsfile_with_data, max_row_num):
    """测试小写表名"""
    df = to_dataframe(tsfile_with_data, table_name="test_table")
    assert df.shape[0] == max_row_num
    assert df.iloc[0, 0] == 0

def test_table_name_uppercase(tsfile_with_data, max_row_num):
    """测试大写表名"""
    df = to_dataframe(tsfile_with_data, table_name="TEST_TABLE")
    assert df.shape[0] == max_row_num
    assert df.iloc[0, 0] == 0

# ==================== 测试时间段 ====================

@pytest.mark.parametrize("start_time,end_time,expected_rows", [
    (10, None, 90),       # start_time=10
    (-10, None, 100),     # start_time=-10（无效值）
    (None, 5, 6),         # end_time=5（包含0-5，共6行）
    (None, -5, 0),        # end_time=-5（无效值）
    (5, 5, 1),            # start_time=5, end_time=5
    (-5, -5, 0),          # 无效时间段
    (10, -10, 0),         # start > end（无效）
    (-10, 10, 11),        # 包含0-10，共11行
    (-50, 50, 51),        # 包含0-50，共51行
])
def test_time_range(tsfile_with_data, start_time, end_time, expected_rows):
    """测试时间段查询"""
    df = to_dataframe(tsfile_with_data, start_time=start_time, end_time=end_time)
    assert df.shape[0] == expected_rows

# ==================== 测试最大行数 ====================

@pytest.mark.parametrize("max_rows,expected_rows", [
    (1, 1),
    (50, 50),
    (100, 100),
    (1000, 100),  # 超出实际行数
    (0, 0),
    (-10, 0),     # 无效值
])
def test_max_row_num(tsfile_with_data, max_rows, expected_rows):
    """测试最大行数限制"""
    df = to_dataframe(tsfile_with_data, max_row_num=max_rows)
    assert df.shape[0] == expected_rows

# ==================== 测试迭代式查询 ====================

@pytest.mark.parametrize("max_rows,chunk_size", [
    (20, 20),
    (1000, 100),  # 单次迭代返回全部数据
])
def test_iterator_query(tsfile_with_data, max_rows, chunk_size):
    """测试迭代式查询"""
    total_rows = 0
    for df in to_dataframe(tsfile_with_data, max_row_num=max_rows, as_iterator=True):
        assert df.shape[0] == chunk_size
        total_rows += df.shape[0]
    assert total_rows <= 100

def test_iterator_with_all_params(tsfile_with_data):
    """测试迭代式查询带全部参数"""
    for df in to_dataframe(
        tsfile_with_data,
        table_name="test_table",
        column_names=["Device1", "Value1"],
        start_time=21,
        end_time=50,
        max_row_num=10,
        as_iterator=True
    ):
        assert df.shape[0] == 10
        verify_column_value(df, "Device1", lambda t: "Device1_" + str(t))
        verify_column_value(df, "Value1", lambda t: np.bool_(t % 2 == 0))

# ==================== 测试异常情况 ====================

def test_table_not_exist(tsfile_with_data):
    """测试表不存在异常"""
    with pytest.raises(TableNotExistError, match="Requested table does not exist"):
        to_dataframe(tsfile_with_data, table_name="non_existent_table")

def test_column_not_exist(tsfile_with_data):
    """测试列不存在异常"""
    with pytest.raises(ColumnNotExistError, match="Column does not exist"):
        to_dataframe(tsfile_with_data, column_names=["non_existent_column"])