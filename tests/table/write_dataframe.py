import os
import pandas as pd
import numpy as np
import pytest
from datetime import date
from tsfile import to_dataframe, TableSchema, ColumnSchema, TSDataType, ColumnCategory, TsFileTableWriter, Tablet

"""
标题：表模型 write_dataframe 接口功能测试
日期：2026/03
优化：2026/04 - 使用fixture减少重复代码，改进命名，增加异常测试
"""

# ==================== Fixtures ====================

@pytest.fixture
def tsfile_path(tmp_path):
    """生成临时tsfile文件路径"""
    return str(tmp_path / "write_dataframe_test.tsfile")

@pytest.fixture
def num_rows():
    """默认测试行数"""
    return 100

@pytest.fixture
def full_table_schema():
    """创建完整表结构（包含所有数据类型）"""
    return TableSchema("test_table", [
        ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
        ColumnSchema("tag1", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("tag2", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("bool_", TSDataType.BOOLEAN, ColumnCategory.FIELD),
        ColumnSchema("int32_", TSDataType.INT32, ColumnCategory.FIELD),
        ColumnSchema("int64_", TSDataType.INT64, ColumnCategory.FIELD),
        ColumnSchema("float_", TSDataType.FLOAT, ColumnCategory.FIELD),
        ColumnSchema("double_", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ColumnSchema("string_", TSDataType.STRING, ColumnCategory.FIELD),
        ColumnSchema("text_", TSDataType.TEXT, ColumnCategory.FIELD),
        ColumnSchema("blob_", TSDataType.BLOB, ColumnCategory.FIELD),
        ColumnSchema("timestamp_", TSDataType.TIMESTAMP, ColumnCategory.FIELD),
        ColumnSchema("date_", TSDataType.DATE, ColumnCategory.FIELD),
    ])

@pytest.fixture
def partial_null_dataframe(num_rows):
    """创建部分值为空的DataFrame"""
    data = {
        'time': range(num_rows),
        'tag1': ['tag1_' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
        'tag2': ['tag2_' + str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'bool_': [i % 2 == 0 if i % 3 != 0 else None for i in range(num_rows)],
        'int32_': [i * 2 if i % 4 != 0 else None for i in range(num_rows)],
        'int64_': [i * 3 if i % 5 != 0 else None for i in range(num_rows)],
        'float_': [i * 1.5 if i % 3 != 0 else None for i in range(num_rows)],
        'double_': [i * 2.5 if i % 4 != 0 else None for i in range(num_rows)],
        'string_': [str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'text_': [str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'blob_': [i.to_bytes(4, byteorder='big') if i % 3 != 0 else None for i in range(num_rows)],
        'timestamp_': [i * 1000 if i % 4 != 0 else None for i in range(num_rows)],
        'date_': [pd.Timestamp(f'2023-01-01 {i % 24}:00:00').date() if i % 5 != 0 else None for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    df.int32_ = df.int32_.astype('Int32')
    df.int64_ = df.int64_.astype('Int64')
    df.bool_ = df.bool_.astype('boolean')
    df.float_ = df.float_.astype(np.float32)
    df.timestamp_ = df.timestamp_.astype('Int64')
    return df

@pytest.fixture
def simple_table_schema():
    """创建简单表结构"""
    return TableSchema("source_table", [
        ColumnSchema("tag1", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("value1", TSDataType.BOOLEAN, ColumnCategory.FIELD),
        ColumnSchema("value2", TSDataType.INT32, ColumnCategory.FIELD),
        ColumnSchema("value3", TSDataType.INT64, ColumnCategory.FIELD),
        ColumnSchema("value4", TSDataType.FLOAT, ColumnCategory.FIELD),
        ColumnSchema("value5", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ColumnSchema("value6", TSDataType.STRING, ColumnCategory.FIELD),
        ColumnSchema("value7", TSDataType.TEXT, ColumnCategory.FIELD),
        ColumnSchema("value8", TSDataType.BLOB, ColumnCategory.FIELD),
        ColumnSchema("value9", TSDataType.DATE, ColumnCategory.FIELD),
        ColumnSchema("value10", TSDataType.TIMESTAMP, ColumnCategory.FIELD)
    ])

# ==================== 辅助验证函数 ====================

def verify_dataframe_content(df, table_schema, num_rows):
    """验证DataFrame内容"""
    assert df.shape[0] == num_rows
    expected_columns = table_schema.get_column_names()
    assert list(df.columns) == expected_columns

# ==================== 测试正常写入 ====================

def test_write_partial_null_dataframe(tsfile_path, full_table_schema, partial_null_dataframe, num_rows):
    """测试写入部分值为空的DataFrame"""
    with TsFileTableWriter(tsfile_path, full_table_schema) as writer:
        writer.write_dataframe(partial_null_dataframe)

    df_read = to_dataframe(tsfile_path)
    verify_dataframe_content(df_read, full_table_schema, num_rows)

def test_write_dataframe_from_to_dataframe(tsfile_path, simple_table_schema, num_rows):
    """测试写入由to_dataframe生成的DataFrame"""
    # 先创建源数据
    column_names = ["tag1", "value1", "value2", "value3", "value4", "value5", "value6", "value7", "value8", "value9", "value10"]
    data_types = [TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.STRING, TSDataType.TEXT, TSDataType.BLOB, TSDataType.DATE, TSDataType.TIMESTAMP]

    with TsFileTableWriter(tsfile_path, simple_table_schema) as writer:
        tablet = Tablet(column_names, data_types, num_rows)
        for i in range(num_rows):
            tablet.add_timestamp(i, i)
            if i % 2 == 0:
                tablet.add_value_by_index(0, i, f"tag1_{i}")
                tablet.add_value_by_index(1, i, i % 2 == 0)
                tablet.add_value_by_index(2, i, i * 2)
                tablet.add_value_by_index(3, i, i * 3)
                tablet.add_value_by_index(4, i, i * 4.4)
                tablet.add_value_by_index(5, i, i * 5.5)
                tablet.add_value_by_index(6, i, f"string_{i}")
                tablet.add_value_by_index(7, i, f"text_{i}")
                tablet.add_value_by_index(8, i, i.to_bytes(4, byteorder='big'))
                tablet.add_value_by_index(9, i, pd.Timestamp(f'2023-01-01 {i % 24}:00:00').date())
                tablet.add_value_by_index(10, i, i * 1000)
        writer.write_table(tablet)

    # 读取并重新写入
    df = to_dataframe(tsfile_path, table_name="source_table")

    # 清理后重新写入
    os.remove(tsfile_path)

    with TsFileTableWriter(tsfile_path, simple_table_schema) as writer:
        writer.write_dataframe(df)

    df_read = to_dataframe(tsfile_path)
    expected_columns = ['time', 'tag1', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7', 'value8', 'value9', 'value10']
    assert df_read.shape[0] == num_rows
    assert list(df_read.columns) == expected_columns

# ==================== 测试不同数据类型 ====================

@pytest.mark.parametrize("column_name,data_type,test_values", [
    ("bool_", TSDataType.BOOLEAN, [True, False, None]),
    ("int32_", TSDataType.INT32, [1, 2, None]),
    ("int64_", TSDataType.INT64, [100, 200, None]),
    ("float_", TSDataType.FLOAT, [1.5, 2.5, None]),
    ("double_", TSDataType.DOUBLE, [3.14, 6.28, None]),
    ("string_", TSDataType.STRING, ["a", "b", None]),
    ("text_", TSDataType.TEXT, ["text1", "text2", None]),
])
def test_write_single_type_column(tsfile_path, column_name, data_type, test_values):
    """测试写入单个类型的列"""
    table = TableSchema("test_table", [
        ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
        ColumnSchema(column_name, data_type, ColumnCategory.FIELD),
    ])

    data = {
        'time': range(len(test_values)),
        column_name: test_values,
    }
    df = pd.DataFrame(data)

    # 类型转换
    if column_name == "bool_":
        df[column_name] = df[column_name].astype('boolean')
    elif column_name in ["int32_", "int64_"]:
        df[column_name] = df[column_name].astype(column_name.replace("_", "").capitalize())
    elif column_name == "float_":
        df[column_name] = df[column_name].astype(np.float32)

    with TsFileTableWriter(tsfile_path, table) as writer:
        writer.write_dataframe(df)

    df_read = to_dataframe(tsfile_path)
    assert df_read.shape[0] == len(test_values)

# ==================== 测试异常情况 ====================

def test_write_empty_dataframe(tsfile_path, full_table_schema):
    """测试写入空DataFrame"""
    df = pd.DataFrame()
    assert df.empty

    with pytest.raises(Exception):  # 具体异常类型根据API确定
        with TsFileTableWriter(tsfile_path, full_table_schema) as writer:
            writer.write_dataframe(df)

def test_write_dataframe_column_mismatch(tsfile_path, full_table_schema):
    """测试DataFrame列与TableSchema不匹配"""
    # DataFrame缺少部分列
    data = {
        'time': range(10),
        'tag1': ['tag_' + str(i) for i in range(10)],
        # 缺少其他列
    }
    df = pd.DataFrame(data)

    with pytest.raises(Exception):
        with TsFileTableWriter(tsfile_path, full_table_schema) as writer:
            writer.write_dataframe(df)

def test_write_dataframe_extra_column(tsfile_path, full_table_schema):
    """测试DataFrame包含多余列"""
    data = {
        'time': range(10),
        'tag1': ['tag_' + str(i) for i in range(10)],
        'tag2': ['tag2_' + str(i) for i in range(10)],
        'bool_': [True for i in range(10)],
        'extra_column': [i for i in range(10)],  # 多余列
    }
    df = pd.DataFrame(data)

    with pytest.raises(Exception):
        with TsFileTableWriter(tsfile_path, full_table_schema) as writer:
            writer.write_dataframe(df)

def test_write_dataframe_type_mismatch(tsfile_path):
    """测试DataFrame类型与TableSchema不匹配"""
    table = TableSchema("test_table", [
        ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
        ColumnSchema("value", TSDataType.INT32, ColumnCategory.FIELD),
    ])

    # 类型不匹配：写入字符串到INT32列
    data = {
        'time': range(10),
        'value': ['string_' + str(i) for i in range(10)],  # 应该是int
    }
    df = pd.DataFrame(data)

    with pytest.raises(Exception):
        with TsFileTableWriter(tsfile_path, table) as writer:
            writer.write_dataframe(df)

def test_write_dataframe_without_time_column(tsfile_path, full_table_schema):
    """测试DataFrame缺少时间列"""
    data = {
        'tag1': ['tag_' + str(i) for i in range(10)],
        'bool_': [True for i in range(10)],
    }
    df = pd.DataFrame(data)

    with pytest.raises(Exception):
        with TsFileTableWriter(tsfile_path, full_table_schema) as writer:
            writer.write_dataframe(df)