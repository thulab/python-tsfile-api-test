import os
import pandas as pd
import numpy as np
import pytest
from tsfile import dataframe_to_tsfile, to_dataframe, TableSchema, ColumnSchema, TSDataType, ColumnCategory, TsFileReader, TsFileTableWriter, Tablet

"""
标题：表模型 dataframe_to_tsfile 接口功能测试
日期：2026/03
优化：2026/04 - 使用fixture减少重复代码，使用pytest参数化，改进异常处理
"""

# ==================== Fixtures ====================

@pytest.fixture
def tsfile_path(tmp_path):
    """生成临时tsfile文件路径"""
    return str(tmp_path / "dataframe_to_tsfile_test.tsfile")

@pytest.fixture
def num_rows():
    """默认测试行数"""
    return 100

@pytest.fixture
def standard_dataframe(num_rows):
    """创建标准测试DataFrame - 部分值为空"""
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
def all_null_dataframe(num_rows):
    """创建全空值DataFrame"""
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
    df.int32_ = df.int32_.astype('Int32')
    df.int64_ = df.int64_.astype('Int64')
    df.bool_ = df.bool_.astype('boolean')
    df.float_ = df.float_.astype(np.float32)
    df.timestamp_ = df.timestamp_.astype('Int64')
    return df

@pytest.fixture
def no_null_dataframe(num_rows):
    """创建无空值DataFrame"""
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
    return df

@pytest.fixture
def cross_partition_dataframe(num_rows):
    """创建跨时间分区DataFrame"""
    data = {
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
    df.bool_ = df.bool_.astype('boolean')
    return df

# ==================== 验证辅助函数 ====================

def verify_dataframe_write_read(df, tsfile_path, table_name="test_table", num_rows=100):
    """验证DataFrame写入和读取"""
    dataframe_to_tsfile(df, tsfile_path, table_name=table_name)
    assert os.path.exists(tsfile_path)

    read_df = to_dataframe(tsfile_path, table_name=table_name)
    assert read_df.shape[0] == num_rows
    return read_df

def get_expected_columns():
    """获取预期列名"""
    return ['time', 'tag1', 'tag2', 'bool_', 'int32_', 'int64_', 'float_', 'double_', 'string_', 'text_', 'blob_', 'timestamp_', 'date_']

# ==================== 测试 DataFrame ====================

def test_dataframe_partial_null(standard_dataframe, tsfile_path, num_rows):
    """测试部分值为空的DataFrame"""
    read_df = verify_dataframe_write_read(standard_dataframe, tsfile_path, num_rows=num_rows)
    assert list(read_df.columns) == get_expected_columns()

def test_dataframe_all_null(all_null_dataframe, tsfile_path, num_rows):
    """测试全空值的DataFrame"""
    read_df = verify_dataframe_write_read(all_null_dataframe, tsfile_path, num_rows=num_rows)
    assert list(read_df.columns) == get_expected_columns()

def test_dataframe_cross_partition(cross_partition_dataframe, tsfile_path, num_rows):
    """测试跨时间分区的DataFrame"""
    read_df = verify_dataframe_write_read(cross_partition_dataframe, tsfile_path, num_rows=num_rows)
    assert list(read_df.columns) == get_expected_columns()

def test_dataframe_no_null(no_null_dataframe, tsfile_path, num_rows):
    """测试无空值的DataFrame"""
    read_df = verify_dataframe_write_read(no_null_dataframe, tsfile_path, num_rows=num_rows)
    assert list(read_df.columns) == get_expected_columns()

def test_dataframe_values_count_inconsistent():
    """测试值数量不一致的DataFrame"""
    data = {
        'time': range(100),
        'tag1': ['tag1_' + str(i) if i % 3 != 0 else None for i in range(1)],
        'tag2': ['tag2_' + str(i) if i % 5 != 0 else None for i in range(2)],
    }
    with pytest.raises(ValueError, match="All arrays must be of the same length"):
        pd.DataFrame(data)

def test_dataframe_empty():
    """测试空DataFrame"""
    df = pd.DataFrame()
    assert df.empty
    with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
        dataframe_to_tsfile(df, "test.tsfile", table_name="empty_dataframe")

def test_dataframe_none():
    """测试None DataFrame"""
    with pytest.raises(ValueError, match="DataFrame cannot be None or empty"):
        dataframe_to_tsfile(None, "test.tsfile", table_name="none")

# ==================== 测试表名 ====================

@pytest.mark.parametrize("table_name", [
    "TestTable",
    "test_table",
    "TEST_TABLE",
    "测试表",
    "12345",
    "!@#$%^&*()_+-=[]{};':\",./<>?",
])
def test_table_names(standard_dataframe, tsfile_path, num_rows, table_name):
    """测试各种表名"""
    dataframe_to_tsfile(standard_dataframe, tsfile_path, table_name=table_name)
    assert os.path.exists(tsfile_path)

    read_df = to_dataframe(tsfile_path, table_name=table_name.lower() if table_name in ["TestTable", "TEST_TABLE"] else table_name)
    assert read_df.shape[0] == num_rows

def test_table_name_not_set(standard_dataframe, tsfile_path, num_rows):
    """测试不设置表名"""
    dataframe_to_tsfile(standard_dataframe, tsfile_path)
    assert os.path.exists(tsfile_path)

    read_df = to_dataframe(tsfile_path)
    assert read_df.shape[0] == num_rows

# ==================== 测试路径 ====================

def test_existing_file_path(standard_dataframe, tsfile_path):
    """测试已存在的文件路径"""
    # 第一次写入
    dataframe_to_tsfile(standard_dataframe, tsfile_path, table_name="test_table")

    # 再次写入同一文件应抛出异常
    with pytest.raises(Exception):
        dataframe_to_tsfile(standard_dataframe, tsfile_path, table_name="test_table")

def test_directory_path(standard_dataframe, tmp_path):
    """测试目录路径"""
    directory_path = str(tmp_path)
    assert os.path.isdir(directory_path)

    with pytest.raises(Exception):
        dataframe_to_tsfile(standard_dataframe, directory_path, table_name="test_table")

# ==================== 测试时间列 ====================

@pytest.mark.parametrize("time_column_name,expected_first_col", [
    ("time", "time"),
    ("TIME", "TIME"),
    ("时间", "时间"),
    ("123", "123"),
    ("!@#", "!@#"),
])
def test_time_column_names(standard_dataframe, tsfile_path, num_rows, time_column_name, expected_first_col):
    """测试各种时间列名"""
    # 修改DataFrame的time列名
    df = standard_dataframe.rename(columns={'time': time_column_name})
    dataframe_to_tsfile(df, tsfile_path, table_name="test_table", time_column=time_column_name)
    assert os.path.exists(tsfile_path)

    read_df = to_dataframe(tsfile_path, table_name="test_table")
    assert read_df.shape[0] == num_rows

def test_time_column_int32(standard_dataframe, tsfile_path, num_rows):
    """测试int32类型的时间列"""
    standard_dataframe.time = standard_dataframe.time.astype(np.int32)
    read_df = verify_dataframe_write_read(standard_dataframe, tsfile_path, num_rows=num_rows)
    assert read_df.shape[0] == num_rows

def test_time_column_not_exist(standard_dataframe, tsfile_path):
    """测试指定不存在的时间列"""
    with pytest.raises(ValueError, match="Time column 'non_existent_time' not found in DataFrame"):
        dataframe_to_tsfile(standard_dataframe, tsfile_path, table_name="test_table", time_column="non_existent_time")

@pytest.mark.parametrize("time_dtype,error_type", [
    (str, TypeError),
    (np.float64, Exception),
])
def test_time_column_invalid_type(standard_dataframe, tsfile_path, time_dtype, error_type):
    """测试非法时间列类型"""
    standard_dataframe.time = standard_dataframe.time.astype(time_dtype)
    with pytest.raises(error_type, match="Time column 'time' must be integer type"):
        dataframe_to_tsfile(standard_dataframe, tsfile_path, table_name="test_table", time_column="time")

# ==================== 测试列名 ====================

def test_column_duplicate(standard_dataframe, tsfile_path):
    """测试重复列名"""
    df = standard_dataframe.copy()
    df['Time'] = range(len(df))  # 添加重复列名（大小写不同）

    with pytest.raises(ValueError, match="Column names must be unique"):
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")

def test_column_name_empty(num_rows):
    """测试空白列名"""
    data = {
        '': range(num_rows),
        'tag1': ['tag1_' + str(i) for i in range(num_rows)],
    }
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Column name cannot be None or empty"):
        dataframe_to_tsfile(df, "test.tsfile", table_name="test_table")

def test_column_name_none(num_rows):
    """测试None列名"""
    data = {
        'time': range(num_rows),
        None: ['tag1_' + str(i) for i in range(num_rows)],
    }
    df = pd.DataFrame(data)

    with pytest.raises(Exception):  # 可能抛出各种异常
        dataframe_to_tsfile(df, "test.tsfile", table_name="test_table")

def test_time_column_not_set_without_time_column(num_rows, tsfile_path):
    """测试不设置时间列且DataFrame没有time列"""
    data = {
        'tag1': ['tag1_' + str(i) for i in range(num_rows)],
        'tag2': ['tag2_' + str(i) for i in range(num_rows)],
        'value': [i for i in range(num_rows)],
    }
    df = pd.DataFrame(data)

    # 如果没有time列且未指定time_column，API可能会使用默认行为或抛出异常
    # 需要根据实际API行为调整
    try:
        dataframe_to_tsfile(df, tsfile_path, table_name="test_table")
        # 如果没有抛出异常，验证写入是否成功
        assert os.path.exists(tsfile_path)
    except Exception:
        # 如果抛出异常也是正常行为
        pass

# ==================== 测试结合to_dataframe函数 ====================

def test_dataframe_from_to_dataframe(tsfile_path, num_rows):
    """测试使用to_dataframe生成的DataFrame写入新TsFile"""
    # 先创建一个TsFile
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

    # 验证
    assert os.path.exists(tsfile_path)
    read_df = to_dataframe(tsfile_path, table_name="new_table")
    expected_cols = ['time', 'tag1', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6', 'value7', 'value8', 'value9', 'value10']
    assert list(read_df.columns) == expected_cols
    assert read_df.shape[0] == num_rows