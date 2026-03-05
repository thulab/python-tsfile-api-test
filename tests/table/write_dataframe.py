import os
import pandas as pd
import numpy as np
import pytest
from tsfile import dataframe_to_tsfile, to_dataframe, TableSchema, ColumnSchema, TSDataType, ColumnCategory, TsFileReader, TsFileTableWriter, Tablet

"""
标题：表模型 dataframe_to_tsfile 接口功能测试
日期：2026/03
"""

# tsfile文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
tsfile_path = os.path.join(current_dir, "../../data/tsfile/dataframe_to_tsfile_test.tsfile")
tsfile_path = os.path.normpath(tsfile_path)

# 清理tsfile文件
@pytest.fixture(autouse=True, scope="function")
def cleanup_tsfile():
    # 测试前清理
    if os.path.exists(tsfile_path):
            os.remove(tsfile_path)
    yield
    # 测试后清理
    if os.path.exists(tsfile_path):
            os.remove(tsfile_path)

def test_write_dataframe_valid_dataframe1():
    """
    测试 1: 功能测试 - write_dataframe 接口 - 参数 - 正确的DataFrame - 手动生成，部分为空
    """
    # 创建TableSchema
    table = TableSchema("test_table",
        [ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
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
    # 创建手动生成的DataFrame
    num_rows = 100
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
    df.int32_ = df.int32_.astype('Int32')  # 使用可空整数类型
    df.int64_ = df.int64_.astype('Int64')  # 使用可空整数类型
    df.bool_ = df.bool_.astype('boolean')  # 使用可空布尔类型
    df.float_ = df.float_.astype(np.float32)
    df.timestamp_ = df.timestamp_.astype('Int64') # 使用可空整数类型
    # 写入TsFile
    with TsFileTableWriter(tsfile_path, table) as writer:
        writer.write_dataframe(df)
    # 验证写入成功
    df_read = to_dataframe(tsfile_path)
    assert df_read.shape[0] == num_rows
    assert list(df_read.columns) == table.get_column_names()

def test_write_dataframe_valid_dataframe2():
    """
    测试 2: 功能测试 - write_dataframe 接口 - 参数 - 正确的DataFrame - 使用to_dataframe生成的DataFrame
    """
    # 先创建一个TsFile并写入数据
    table = TableSchema("source_table", [
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
    
    num_rows = 100
    with TsFileTableWriter(tsfile_path, table) as writer:
        tablet = Tablet(
            ["tag1", "value1", "value2", "value3", "value4", "value5", "value6", "value7", "value8", "value9", "value10"], 
            [TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.STRING, TSDataType.TEXT, TSDataType.BLOB, TSDataType.DATE, TSDataType.TIMESTAMP], 
            num_rows)
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
    
    # 使用to_dataframe读取
    df = to_dataframe(tsfile_path, table_name="source_table")

    # 清理生成的TsFile
    os.remove(tsfile_path)

    # 写入TsFile
    with TsFileTableWriter(tsfile_path, table) as writer:
        writer.write_dataframe(df)
    # 验证写入成功
    df_read = to_dataframe(tsfile_path)
    assert df_read.shape[0] == num_rows
    assert list(df_read.columns) == ['time', 'tag1', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7', 'value8', 'value9', 'value10']

    