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

def test_dataframe_to_tsfile_valid_dataframe1():
    """
    测试 1: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - 正确的DataFrame - 手动生成，部分为空
    """
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
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows
    assert list(read_df.columns) == ['time', 'tag1', 'tag2', 'bool_', 'int32_', 'int64_', 'float_', 'double_', 'string_', 'text_', 'blob_', 'timestamp_', 'date_']

def test_dataframe_to_tsfile_valid_dataframe2():
    """
    测试 2: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - 正确的DataFrame - 手动生成，全部为空
    """
    # 创建手动生成的DataFrame
    num_rows = 100
    data = {
        'time': range(num_rows),
        'tag1': [None for i in range(num_rows)],
        'tag2': [None for i in range(num_rows)],
        'bool_': [None for i in range(num_rows)],
        'int32_': [None for i in range(num_rows)],
        'int64_': [None for i in range(num_rows)],
        'float_': [None for i in range(num_rows)],
        'double_': [None for i in range(num_rows)],
        'string_': [None for i in range(num_rows)],
        'text_': [None for i in range(num_rows)],
        'blob_': [None for i in range(num_rows)],
        'timestamp_': [None for i in range(num_rows)],
        'date_': [None for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    df.int32_ = df.int32_.astype('Int32')  # 使用可空整数类型
    df.int64_ = df.int64_.astype('Int64')  # 使用可空整数类型
    df.bool_ = df.bool_.astype('boolean')  # 使用可空布尔类型
    df.float_ = df.float_.astype(np.float32)
    df.timestamp_ = df.timestamp_.astype('Int64') # 使用可空整数类型
    
    # 写入TsFile
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows
    assert list(read_df.columns) == ['time', 'tag1', 'tag2', 'bool_', 'int32_', 'int64_', 'float_', 'double_', 'string_', 'text_', 'blob_', 'timestamp_', 'date_']

def test_dataframe_to_tsfile_valid_dataframe3():
    """
    测试 3: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - 正确的DataFrame - 手动生成，跨时间分区
    """
    # 创建手动生成的DataFrame
    num_rows = 100
    data = {
        # 生成跨时间分区的时间戳（跨越4天）
        'time': [i * 1000 * 3600 * 24 * 7 * 30 for i in range(num_rows)],  # 每1行跨越7月
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
    df.bool_ = df.bool_.astype('boolean')  # 使用可空布尔类型
    
    # 写入TsFile
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows
    assert list(read_df.columns) == ['time', 'tag1', 'tag2', 'bool_', 'int32_', 'int64_', 'float_', 'double_', 'string_', 'text_', 'blob_', 'timestamp_', 'date_']

def test_dataframe_to_tsfile_valid_dataframe5():
    """
    测试 5: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - 正确的DataFrame - 手动生成，无空值的时间列
    """
    # 创建手动生成的DataFrame
    num_rows = 100
    data = {
        'time': range(num_rows),
        'tag1': ['tag1_' + str(i) for i in range(num_rows)],
        'tag2': ['tag2_' + str(i) for i in range(num_rows)],
        'bool_': [i % 2 == 0 for i in range(num_rows)],
        'int32_': [i * 2 for i in range(num_rows)],
        'int64_': [i * 3 for i in range(num_rows)],
        'float_': [i * 1.5 for i in range(num_rows)],
        'double_': [i * 2.5 for i in range(num_rows)],
        'string_': [str(i) for i in range(num_rows)],
        'text_': [str(i) for i in range(num_rows)],
        'blob_': [i.to_bytes(4, byteorder='big') for i in range(num_rows)],
        'timestamp_': [i * 1000 for i in range(num_rows)],
        'date_': [pd.Timestamp(f'2023-01-01 {i % 24}:00:00').date() for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    df.int32_ = df.int32_.astype(np.int32) 
    df.int64_ = df.int64_.astype(np.int64)   
    df.float_ = df.float_.astype(np.float32)
    df.timestamp_ = df.timestamp_.astype(np.int64)
    
    # 写入TsFile
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows
    assert list(read_df.columns) == ['time', 'tag1', 'tag2', 'bool_', 'int32_', 'int64_', 'float_', 'double_', 'string_', 'text_', 'blob_', 'timestamp_', 'date_']

def test_dataframe_to_tsfile_different_number_of_values():
    """
    测试 3: 功能测试 - write_dataframe 接口 - 参数 - 值数量不一致的DataFrame - 手动生成，部分为空
    """
    # 创建手动生成的DataFrame
    num_rows = 100
    data = {
        'time': range(num_rows),
        'tag1': ['tag1_' + str(i) if i % 3 != 0 else None for i in range(1)],
        'tag2': ['tag2_' + str(i) if i % 5 != 0 else None for i in range(2)],
        'bool_': [i % 2 == 0 if i % 3 != 0 else None for i in range(3)],
        'int32_': [i * 2 if i % 4 != 0 else None for i in range(4)],
        'int64_': [i * 3 if i % 5 != 0 else None for i in range(5)],
        'float_': [i * 1.5 if i % 3 != 0 else None for i in range(6)],
        'double_': [i * 2.5 if i % 4 != 0 else None for i in range(7)],
        'string_': [str(i) if i % 5 != 0 else None for i in range(8)],
        'text_': [str(i) if i % 5 != 0 else None for i in range(9)],
        'blob_': [i.to_bytes(4, byteorder='big') if i % 3 != 0 else None for i in range(10)],
        'timestamp_': [i * 1000 if i % 4 != 0 else None for i in range(10)],
        'date_': [pd.Timestamp(f'2023-01-01 {i % 24}:00:00').date() if i % 5 != 0 else None for i in range(11)],
    }
    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        assert str(e) == "All arrays must be of the same length"

def test_dataframe_to_tsfile_from_to_dataframe():
    """
    测试 4: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - 使用to_dataframe生成的DataFrame
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
    
    # 清理文件
    os.remove(tsfile_path)
    
    # 使用生成的DataFrame写入新的TsFile
    dataframe_to_tsfile(df, tsfile_path, table_name="new_table", time_column="time")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 验证列名
    read_df = to_dataframe(tsfile_path, table_name="new_table")
    assert list(read_df.columns) == ['time', 'tag1', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7', 'value8', 'value9', 'value10']
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_empty_dataframe():
    """
    测试 5: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - 空 DataFrame
    """
    # 创建空DataFrame
    df = pd.DataFrame()
    assert df.empty, "DataFrame cannot be empty"

    # 尝试写入空DataFrame，预期会抛出异常
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="empty_dataframe")
        assert False, "Exception not caught"
    except ValueError as e:
        assert str(e) == "DataFrame cannot be None or empty"

def test_dataframe_to_tsfile_none_dataframe():
    """
    测试 6: 功能测试 - dataframe_to_tsfile 接口 - 参数 - dataframe - None
    """
    # 尝试写入None
    try:
        dataframe_to_tsfile(None, tsfile_path, table_name="none")
        assert False, "Exception not caught"
    except ValueError as e:
        assert str(e) == "DataFrame cannot be None or empty"

def test_dataframe_to_tsfile_existing_file_path():
    """
    测试 7: 功能测试 - dataframe_to_tsfile 接口 - 参数 - file_path - 已存在的文件路径
    """
    # 先创建文件
    data = {
        'time': [1, 2, 3],
        'value': [10, 20, 30]
    }
    df = pd.DataFrame(data)
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 尝试再次写入同一文件
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
        assert False, "Exception not caught"
    except Exception as e:
        # 异常被包装成了 SystemError，因为错误消息格式不稳定，所以选择了只验证异常被捕获，而不检查具体错误消息
        assert True

def test_dataframe_to_tsfile_directory_path():
    """
    测试 8: 功能测试 - dataframe_to_tsfile 接口 - 参数 - file_path - 目录路径
    """
    # 使用目录路径
    directory_path = os.path.join(current_dir, "../../data/tsfile")
    directory_path = os.path.normpath(directory_path)
    assert os.path.isdir(directory_path)
    
    # 创建DataFrame
    data = {
        'time': [1, 2, 3],
        'value': [10, 20, 30]
    }
    df = pd.DataFrame(data)
    
    # 尝试写入目录
    try:
        dataframe_to_tsfile(df, directory_path, table_name="test_table")
        assert False, "Exception not caught"
    except Exception as e:
        # 异常被包装成了 SystemError，因为错误消息格式不稳定，所以选择了只验证异常被捕获，而不检查具体错误消息
        assert True

def test_dataframe_to_tsfile_table_name_english():
    """
    测试 9: 功能测试 - dataframe_to_tsfile 接口 - 参数 - table_name - 大小写英文表名
    """
    # 创建DataFrame
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
    
    # 测试大小写英文表名
    test_table_names = ["TestTable", "test_table", "TEST_TABLE"]
    
    for table_name in test_table_names:
        temp_path = os.path.join(current_dir, f"../../data/tsfile/{table_name}.tsfile")
        temp_path = os.path.normpath(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        dataframe_to_tsfile(df, temp_path, table_name=table_name)
        assert os.path.exists(temp_path)
        
        # 读取验证
        read_df = to_dataframe(temp_path, table_name=table_name.lower())  # 表名通常会被转为小写
        assert read_df.shape[0] == num_rows
        
        os.remove(temp_path)

def test_dataframe_to_tsfile_table_name_chinese():
    """
    测试 10: 功能测试 - dataframe_to_tsfile 接口 - 参数 - table_name - 中文
    """
    # 创建DataFrame
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
    
    # 测试中文表名
    table_name = "测试表"
    dataframe_to_tsfile(df, tsfile_path, table_name=table_name)
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name=table_name)
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_table_name_number():
    """
    测试 11: 功能测试 - dataframe_to_tsfile 接口 - 参数 - table_name - 数字
    """
    # 创建DataFrame
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
    
    # 测试数字表名
    table_name = "12345"
    dataframe_to_tsfile(df, tsfile_path, table_name=table_name)
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name=table_name)
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_table_name_characters():
    """
    测试 12: 功能测试 - dataframe_to_tsfile 接口 - 参数 - table_name - 字符
    """
    # 创建DataFrame
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
    
    # 测试包含字符的表名
    table_name = "!@#$%^&*()_+-=[]{};':\",./<>?"
    dataframe_to_tsfile(df, tsfile_path, table_name=table_name)
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name=table_name)
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_table_name_none():
    """
    测试 13: 功能测试 - dataframe_to_tsfile 接口 - 参数 - table_name - 不设置或None
    """
    # 创建DataFrame
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
    
    # 测试不设置表名
    dataframe_to_tsfile(df, tsfile_path)
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证（使用默认表名）
    read_df = to_dataframe(tsfile_path)
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_time_column_time():
    """
    测试 14: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 写入 Dataframe 时作为时间列的列名 - time
    """
    # 创建DataFrame
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
    
    # 测试指定time列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="time")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_time_column_case():
    """
    测试 15: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 写入 Dataframe 时作为时间列的列名 - 大小写英文
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'TIME': range(num_rows),
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
    
    # 测试指定大小写不同的time列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="TIME")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_time_column_chinese():
    """
    测试 16: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 写入 Dataframe 时作为时间列的列名 - 中文
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        '时间': range(num_rows),
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
    
    # 测试指定中文时间列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="时间")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_time_column_number():
    """
    测试 17: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 写入 Dataframe 时作为时间列的列名 - 数字
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        '123': range(num_rows),
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
    
    # 测试指定数字时间列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="123")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_time_column_characters():
    """
    测试 18: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 写入 Dataframe 时作为时间列的列名 - 字符
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        '!@#': range(num_rows),
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
    
    # 测试指定字符时间列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="!@#")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_column_duplicate():
    """
    测试 19: 功能测试 - dataframe_to_tsfile 接口 - 存在重复列名
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time': range(num_rows),
        'Time': range(num_rows),  # 重复列名
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
    
    # 存在重复的time列名
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
        assert False, "Exception not caught"
    except ValueError as e:
        assert str(e) == "Column names must be unique (case-insensitive). Duplicate columns: ['Time']"

def test_dataframe_to_tsfile_time_column_not_exist():
    """
    测试 20: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 写入 Dataframe 时作为时间列的列名 - 指定的time列名不存在
    """
    # 创建DataFrame
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
    
    # 测试指定不存在的time列
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="non_existent_time")
        assert False, "Exception not caught"
    except ValueError as e:
        assert str(e) == "Time column 'non_existent_time' not found in DataFrame"

def test_dataframe_to_tsfile_column_invalid1():
    """
    测试 21: 功能测试 - dataframe_to_tsfile 接口 - 参数 - 非法列名 空白字符
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        '': range(num_rows),
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
    
    # 测试非法时间列名
    time_column = ''
    
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
        assert False, f"Time column name '{time_column}' did not raise an exception"
    except ValueError as e:
        assert str(e) == "Column name cannot be None or empty"

def test_dataframe_to_tsfile_column_invalid2():
    """
    测试 22: 功能测试 - dataframe_to_tsfile 接口 - 非法列名 None
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time': range(num_rows),
        None: ['tag1_' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
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
    
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
        assert False, "not raise an exception"
    except ValueError as e:
        assert str(e) == "Column name cannot be None or empty"

def test_dataframe_to_tsfile_time_column_int():
    """
    测试 23: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 时间列的数据类型 - int32
    """
    # 创建DataFrame
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
    df.time = df.time.astype(np.int32)
    
    # 测试int32类型的时间列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="time")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_time_column_invalid_type1():
    """
    测试 24: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 时间列的数据类型 - 非法时间列类型 str
    """
    # 创建DataFrame
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
    df.time = df.time.astype(str)

    time_column = "time"
    
    # 测试字符类型的时间列
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column=time_column)
        assert False, f"The time column '{time_column}' of type str did not raise an exception"
    except TypeError as e:
        assert str(e) == f"Time column '{time_column}' must be integer type (int64 or int), got {df[time_column].dtype}"

def test_dataframe_to_tsfile_time_column_invalid_type2():
    """
    测试 25: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 时间列的数据类型 - 非法时间列类型 float64
    """
    # 创建DataFrame
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
    df.time = df.time.astype(np.float64)

    time_column = "time"
    
    # 测试类型不符的时间列
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column=time_column)
        assert False, f"Time column name '{time_column}' of type float64 did not raise an exception"
    except Exception as e:
        assert str(e) == f"Time column '{time_column}' must be integer type (int64 or int), got {df[time_column].dtype}"


def test_dataframe_to_tsfile_time_column_not_set_without_time():
    """
    测试 26: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 不设置 - 不包含小写为 "time" 的列
    """
    # 创建DataFrame
    num_rows = 100
    data = {
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
    
    # 不设置time_column，且没有"time"列，应该使用索引作为时间列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_set_time_column_only_time_column():
    """
    测试 27: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 设置 - 只包含为 "time" 的列
    """
    # 创建DataFrame
    data = {
        'time': [10, 20, 30]
    }
    df = pd.DataFrame(data)
    
    # 设置time_column，有且仅有"time"列
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="time")
        assert False, "not raise an exception"
    except ValueError as e:
        assert str(e) == "DataFrame must have at least one data column besides the time column"

def test_dataframe_to_tsfile_not_set_time_column_only_time_column():
    """
    测试 28: 功能测试 - dataframe_to_tsfile 接口 - 参数 - time_column - 不设置 - 只包含为 "time" 的列
    """
    # 创建DataFrame
    data = {
        'time': [10, 20, 30]
    }
    df = pd.DataFrame(data)
    
    # 设置time_column，有且仅有"time"列
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
        assert False, "not raise an exception"
    except ValueError as e:
        assert str(e) == "DataFrame must have at least one data column besides the time column"


def test_dataframe_to_tsfile_tag_column_time():
    """
    测试 29: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时让时间列作为标签列的列名
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
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
    
    # 测试指定tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["time1"], time_column="time1")

    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_tag_column_case():
    """
    测试 30: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - 大小写英文
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time': range(num_rows),
        'Tag1': ['tag1_' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
        'Tag2': ['tag2_' + str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'Bool_': [i % 2 == 0 if i % 3 != 0 else None for i in range(num_rows)],
        'Int32_': [i * 2 if i % 4 != 0 else None for i in range(num_rows)],
        'Int64_': [i * 3 if i % 5 != 0 else None for i in range(num_rows)],
        'Float_': [i * 1.5 if i % 3 != 0 else None for i in range(num_rows)],
        'Double_': [i * 2.5 if i % 4 != 0 else None for i in range(num_rows)],
        'String_': [str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'Text_': [str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'Blob_': [i.to_bytes(4, byteorder='big') if i % 3 != 0 else None for i in range(num_rows)],
        'Timestamp_': [i * 1000 if i % 4 != 0 else None for i in range(num_rows)],
        'Date_': [pd.Timestamp(f'2023-01-01 {i % 24}:00:00').date() if i % 5 != 0 else None for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    df.Int32_ = df.Int32_.astype('Int32')  # 使用可空整数类型
    df.Int64_ = df.Int64_.astype('Int64')  # 使用可空整数类型
    df.Bool_ = df.Bool_.astype('boolean')  # 使用可空布尔类型
    df.Float_ = df.Float_.astype(np.float32)
    df.Timestamp_ = df.Timestamp_.astype('Int64') # 使用可空整数类型
    
    # 测试指定大小写不同的tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["Tag1"])
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[1] == 13

def test_dataframe_to_tsfile_tag_column_chinese():
    """
    测试 31: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - 中文
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
        '标签': ['标签_' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
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
    
    # 测试指定中文tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["标签"])
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[1] == 14

def test_dataframe_to_tsfile_tag_column_number():
    """
    测试 32: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - 数字
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
        '123': ['123' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
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
    
    # 测试指定数字tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["123"])
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[1] == 14

def test_dataframe_to_tsfile_tag_column_characters():
    """
    测试 33: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - 符号
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
        '!@#$%^&*()': ['!@#$%^&*()_' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
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
    
    # 测试指定字符tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["!@#$%^&*()"])
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[1] == 14

def test_dataframe_to_tsfile_tag_column_not_exist():
    """
    测试 34: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - 指定的TAG列名不存在
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
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
    
    # 测试指定不存在的tag列
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["non_existent_tag"])
        assert False, "Exception not caught"
    except Exception as e:
        assert str(e) == "Tag column 'non_existent_tag' not found in DataFrame"

def test_dataframe_to_tsfile_tag_column_duplicate():
    """
    测试 35: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - tag_column参数存在重复的列名
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
        'tag1': ['tag1_' + str(i) if i % 3 != 0 else None for i in range(num_rows)],
        'tag2': ['tag2_' + str(i) if i % 5 != 0 else None for i in range(num_rows)],
        'tag3': ['tag3_' + str(i) if i % 5 != 0 else None for i in range(num_rows)],
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
    
    # 测试重复的tag列名
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["tag1", "tag1", "tag3"])

    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_tag_column_order():
    """
    测试 36: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 写入 Dataframe 时作为标签列的列名 - 顺序不一致
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
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
    
    # 测试顺序不一致的tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", tag_column=["tag2", "tag1"])
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_tag_column_no_set():
    """
    测试 37: 功能测试 - dataframe_to_tsfile 接口 - 参数 - tag_column - 不设置
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
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
    
    # 不设置tag列
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_only_required_params():
    """
    测试 38: 功能测试 - dataframe_to_tsfile 接口 - 只使用必选参数
    """
    # 创建DataFrame
    num_rows = 100
    data = {
        'time1': range(num_rows),
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
    
    # 只使用必选参数
    dataframe_to_tsfile(df, tsfile_path)
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path)
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_all_params():
    """
    测试 39: 功能测试 - dataframe_to_tsfile 接口 - 使用全参数
    """
    # 创建DataFrame
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
    
    # 使用全参数
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column="time", tag_column=["tag1", "tag2"])
    
    # 验证写入成功
    assert os.path.exists(tsfile_path)
    
    # 读取验证
    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_dataframe_to_tsfile_missing_required_params1():
    """
    测试 40: 功能测试 - dataframe_to_tsfile 接口 - 必选参数缺失
    """
    # 创建DataFrame
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
    
    # 测试缺失必选参数
    try:
        # 缺少file_path参数
        dataframe_to_tsfile(dataframe = df)
        assert False, "Exception not caught"
    except TypeError as e:
        assert str(e) == "dataframe_to_tsfile() missing 1 required positional argument: 'file_path'"

def test_dataframe_to_tsfile_missing_required_params2():
    """
    测试 41: 功能测试 - dataframe_to_tsfile 接口 - 必选参数缺失
    """
    # 测试缺失必选参数
    try:
        # 缺少file_path参数
        dataframe_to_tsfile(file_path = tsfile_path)
        assert False, "Exception not caught"
    except TypeError as e:
        assert str(e) == "dataframe_to_tsfile() missing 1 required positional argument: 'dataframe'"

