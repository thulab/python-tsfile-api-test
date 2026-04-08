import os
import csv
import pytest
from numpy.ma.testutils import assert_equal

from tsfile import TsFileReader, TsFileTableWriter, TableSchema, ColumnSchema, Tablet
from tsfile import TSDataType, ColumnCategory
from tsfile.exceptions import TableNotExistError, ColumnNotExistError


"""
标题：表模型TsFileReader query_table_by_row接口功能测试
日期：2025/4

测试query_table_by_row接口：
- table_name: 表名（各种字符类型、空值、空白字符）
- column_names: 列名列表（单列/多列、存在/不存在的测点、各种列类型）
- offset: 偏移量（负数、正常范围、超出范围）
- limit: 限制行数（负数无限制、正常范围、超出范围）
- result_set: 空结果集、有数据结果集
"""

# tsfile文件路径
tsfile_path = "test_query_table_by_row.tsfile"


def parse_csv(file_path):
    """解析CSV文件，跳过以#开头的行"""
    parsed_data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0].startswith('#'):
                continue
            if row:
                parsed_data.append(row)
    return parsed_data


def create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=10):
    """创建测试用的TsFile文件"""
    table_data_dir = os.path.join(os.path.dirname(__file__), tsfile_path)
    if os.path.exists(table_data_dir):
        os.remove(table_data_dir)

    columns = []
    for i in range(len(column_name_list)):
        columns.append(ColumnSchema(column_name_list[i], data_type_list[i], column_category_list[i]))
    table_schema = TableSchema(table_name, columns)

    with TsFileTableWriter(table_data_dir, table_schema) as writer:
        tablet = Tablet(column_name_list, data_type_list, row_num)
        for i in range(row_num):
            tablet.add_timestamp(i, i)
            for j in range(len(column_name_list)):
                if data_type_list[j] == TSDataType.STRING:
                    tablet.add_value_by_name(column_name_list[j], i, f"value_{i}")
                elif data_type_list[j] == TSDataType.BOOLEAN:
                    tablet.add_value_by_name(column_name_list[j], i, i % 2 == 0)
                elif data_type_list[j] == TSDataType.INT32:
                    tablet.add_value_by_name(column_name_list[j], i, i * 10)
                elif data_type_list[j] == TSDataType.INT64:
                    tablet.add_value_by_name(column_name_list[j], i, i * 100)
                elif data_type_list[j] == TSDataType.FLOAT:
                    tablet.add_value_by_name(column_name_list[j], i, float(i * 1.5))
                elif data_type_list[j] == TSDataType.DOUBLE:
                    tablet.add_value_by_name(column_name_list[j], i, float(i * 2.5))
        writer.write_table(tablet)

    return table_data_dir


def get_table_data_dir():
    """获取测试文件路径"""
    return os.path.join(os.path.dirname(__file__), tsfile_path)


# ============================================
# 1. 测试table_name参数 - 各种表名
# ============================================

def test_table_name_lowercase():
    """测试表名：小写英文"""
    try:
        table_name = "lowercase_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_table_name_uppercase():
    """测试表名：大写英文"""
    try:
        table_name = "UPPERCASE_TABLE"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            # 注意：表名会被转换为小写存储
            result = reader.query_table_by_row(table_name.lower(), column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_table_name_numbers():
    """测试表名：数字"""
    try:
        table_name = "1234567890"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_table_name_underscore():
    """测试表名：下划线"""
    try:
        table_name = "test_table_name"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_table_name_chinese():
    """测试表名：UNICODE中文字符"""
    try:
        table_name = "测试表名"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_table_name_special_chars():
    """测试表名：特殊字符（空格等）"""
    try:
        table_name = "   "
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            # 表名包含空格时，需要使用原始表名查询
            schemas = reader.get_all_table_schemas()
            actual_table_name = list(schemas.keys())[0]
            result = reader.query_table_by_row(actual_table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_table_name_not_exist():
    """测试表名：不存在的表"""
    try:
        table_name = "existing_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            try:
                reader.query_table_by_row("non_existent_table", column_name_list)
                assert False, "Non-existent table should raise TableNotExistError"
            except TableNotExistError as e:
                assert e._default_message == "Requested table does not exist"

    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


# ============================================
# 2. 测试column_names参数 - 各种列名和列类型
# ============================================

def test_column_names_single_exist():
    """测试列名：单列存在的测点"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1", "s2"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64, TSDataType.DOUBLE]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            # 查询单个存在的列
            result = reader.query_table_by_row(table_name, ["s1"])
            count = 0
            while result.next():
                count += 1
                # 验证能获取到值
                val = result.get_value_by_name("s1")
                assert val is not None
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_single_not_exist():
    """测试列名：单列不存在的测点"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            try:
                reader.query_table_by_row(table_name, ["non_existent_column"])
                assert False, "Non-existent column should raise ColumnNotExistError"
            except ColumnNotExistError as e:
                assert e._default_message == "Column does not exist"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_multi_all_exist():
    """测试列名：多列全存在的测点"""
    try:
        table_name = "test_table"
        column_name_list = ["Tag1", "S1", "S2", "S3"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64, TSDataType.DOUBLE, TSDataType.INT32]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD, ColumnCategory.FIELD, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())

@pytest.mark.skip(reason="预期只输出存在的列，实际会报错列不存在")
def test_column_names_multi_partial_exist():
    """测试列名：多列部分测点不存在"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1", "s2"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64, TSDataType.DOUBLE]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, ["tag1", "s1", "s2", "non_existent_column"])
            while result.next():
                print(result.read_data_frame)
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_multi_device_query():
    """测试多设备查询：查询全部设备测点"""
    try:
        table_name = "test_table"
        column_name_list = ["device_id", "s1", "s2"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64, TSDataType.DOUBLE]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD, ColumnCategory.FIELD]

        table_data_dir = get_table_data_dir()
        if os.path.exists(table_data_dir):
            os.remove(table_data_dir)

        columns = []
        for i in range(len(column_name_list)):
            columns.append(ColumnSchema(column_name_list[i], data_type_list[i], column_category_list[i]))
        table_schema = TableSchema(table_name, columns)

        # 写入多个设备的数据
        devices = ["device1", "device2", "device3"]
        row_num_per_device = 5

        with TsFileTableWriter(table_data_dir, table_schema) as writer:
            tablet = Tablet(column_name_list, data_type_list, row_num_per_device * len(devices))
            row_idx = 0
            for device in devices:
                for i in range(row_num_per_device):
                    tablet.add_timestamp(row_idx, i)
                    tablet.add_value_by_name("device_id", row_idx, device)
                    tablet.add_value_by_name("s1", row_idx, i * 100)
                    tablet.add_value_by_name("s2", row_idx, i * 1.5)
                    row_idx += 1
            writer.write_table(tablet)

        with TsFileReader(table_data_dir) as reader:
            # 查询全部设备的测点数据
            result = reader.query_table_by_row(table_name, ["device_id", "s1", "s2"])
            count = 0
            while result.next():
                count += 1
            # 期望返回所有设备的所有行（3设备 * 5行 = 15行）
            assert count == row_num_per_device * len(devices), f"Expected {row_num_per_device * len(devices)} rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())

@pytest.mark.skip(reason="待确认不支持只查询TAG列？")
def test_column_type_tag():
    """测试列类型：只查询TAG列"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, ["tag1", "s1"])
            count = 0
            while result.next():
                count += 1
            # 仅查询TAG列返回0行
            assert count == 10, f"Expected 10 rows when querying only TAG column, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_type_field():
    """测试列类型：只查询FIELD列"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1", "s2"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64, TSDataType.DOUBLE]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, ["s1", "s2"])
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_lowercase():
    """测试列名：小写英文"""
    try:
        table_name = "test_table"
        column_name_list = ["lowercase_col", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_uppercase():
    """测试列名：大写英文"""
    try:
        table_name = "test_table"
        column_name_list = ["UPPERCASE_COL", "S1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())

def test_column_names_numbers():
    """测试列名：数字"""
    try:
        table_name = "test_table"
        column_name_list = ["123456", "1234567"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())

def test_column_names_underscore():
    """测试列名：符号"""
    try:
        table_name = "test_table"
        column_name_list = ["_!@#", "_!@#%"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())

def test_column_names_chinese():
    """测试列名：UNICODE中文字符"""
    try:
        table_name = "test_table"
        column_name_list = ["中文列名1", "中文列名2"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_special_chars():
    """测试列名：特殊字符（空格）"""
    try:
        table_name = "test_table"
        column_name_list = ["    ", " "]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_empty():
    """测试列名：空白值"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            try:
                reader.query_table_by_row(table_name, [""])
                assert False, "Empty column name should raise an exception"
            except ColumnNotExistError as e:
                assert e._default_message == "Column does not exist"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_column_names_whitespace():
    """测试列名：空值"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list)

        with TsFileReader(table_data_dir) as reader:
            try:
                reader.query_table_by_row(table_name, None)
                assert False, "Whitespace column name should raise an exception"
            except TypeError as e:
                assert str(e) == "Argument 'column_names' has incorrect type (expected list, got NoneType)"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


# ============================================
# 3. 测试offset参数
# ============================================

def test_offset_negative():
    """测试offset：小于0"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=20)

        with TsFileReader(table_data_dir) as reader:
            # offset为负数时的行为测试
            result = reader.query_table_by_row(table_name, column_name_list, offset=-10)
            count = 0
            while result.next():
                count += 1
            assert count == 20, f"Expected 20 rows with offset=-10, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_offset_zero():
    """测试offset：等于0"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=10)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, offset=0)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows with offset=0, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_offset_within_range():
    """测试offset：大于0且不超过实际行数"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=20)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, offset=5)
            count = 0
            while result.next():
                count += 1
            assert count == 15, f"Expected 15 rows with offset=5 (20-5), got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_offset_exceeds_rows():
    """测试offset：超过实际行数"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=10)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, offset=100)
            count = 0
            while result.next():
                count += 1
            assert count == 0, f"Expected 0 rows with offset exceeding total, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


# ============================================
# 4. 测试limit参数
# ============================================

def test_limit_negative():
    """测试limit：小于0（代表无限制）"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=20000)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, limit=-1)
            count = 0
            while result.next():
                count += 1
            assert count == 20000, f"Expected 200000 rows with limit=-1 (unlimited), got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_limit_zero():
    """测试limit：等于0"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=10)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, limit=0)
            count = 0
            while result.next():
                count += 1
            assert count == 0, f"Expected 0 rows with limit=0, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_limit_within_range():
    """测试limit：大于0且不超过实际行数"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=20)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, limit=5)
            count = 0
            while result.next():
                count += 1
            assert count == 5, f"Expected 5 rows with limit=5, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_limit_exceeds_rows():
    """测试limit：超过实际行数"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=10)

        with TsFileReader(table_data_dir) as reader:
            result = reader.query_table_by_row(table_name, column_name_list, limit=100)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows (actual count) with limit=100, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())


def test_offset_and_limit_combination():
    """测试offset和limit组合"""
    try:
        table_name = "test_table"
        column_name_list = ["tag1", "s1"]
        data_type_list = [TSDataType.STRING, TSDataType.INT64]
        column_category_list = [ColumnCategory.TAG, ColumnCategory.FIELD]

        table_data_dir = create_test_tsfile(table_name, column_name_list, data_type_list, column_category_list, row_num=30)

        with TsFileReader(table_data_dir) as reader:
            # offset=10, limit=10, 应该返回第11-20行的10条数据
            result = reader.query_table_by_row(table_name, column_name_list, offset=10, limit=10)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows with offset=10, limit=10, got {count}"
    finally:
        if os.path.exists(get_table_data_dir()):
            os.remove(get_table_data_dir())



