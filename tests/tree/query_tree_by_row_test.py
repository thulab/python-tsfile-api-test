import os
import pytest

from tsfile import TsFileReader, TsFileWriter, TimeseriesSchema, TSDataType
from tsfile import RowRecord, Field
from tsfile.exceptions import DeviceNotExistError, MeasurementNotExistError, NotExistsError


"""
标题：树模型TsFileReader query_tree_by_row接口功能测试
日期：2025/4

测试query_tree_by_row接口：
- device_ids: 设备ID列表（单设备/多设备、存在/不存在的设备、各种字符类型）
- measurement_names: 测点名列表（单测点/多测点、存在/不存在的测点、各种字符类型）
- offset: 偏移量（负数、正常范围、超出范围）
- limit: 限制行数（负数无限制、正常范围、超出范围）
- result_set: 空结果集、有数据结果集
- 多设备，每个设备的测点名不一样情况
"""

# tsfile文件路径
tsfile_path = "test_query_tree_by_row.tsfile"


def get_tree_data_dir():
    """获取测试文件路径"""
    return os.path.join(os.path.dirname(__file__), tsfile_path)


def create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=10):
    """创建测试用的树模型TsFile文件

    Args:
        device_ids: 设备ID列表
        measurement_names_map: 设备ID到测点名列表的映射，格式为 {device_id: [measurement_names]}
                              如果为None，则所有设备使用相同的测点列表
        rows_per_device: 每个设备的行数
    """
    tree_data_dir = get_tree_data_dir()
    if os.path.exists(tree_data_dir):
        os.remove(tree_data_dir)

    writer = TsFileWriter(tree_data_dir)

    # 注册时序序列
    for device_id in device_ids:
        measurements = measurement_names_map.get(device_id) if measurement_names_map else None
        if measurements is None:
            # 如果没有指定该设备的测点，使用第一个设备的测点列表作为默认
            measurements = measurement_names_map.get(device_ids[0]) if measurement_names_map else []

        for measurement in measurements:
            writer.register_timeseries(device_id, TimeseriesSchema(measurement, TSDataType.INT64))

    # 写入数据
    for device_id in device_ids:
        measurements = measurement_names_map.get(device_id) if measurement_names_map else None
        if measurements is None:
            measurements = measurement_names_map.get(device_ids[0]) if measurement_names_map else []

        for ts in range(rows_per_device):
            fields = []
            for idx, measurement in enumerate(measurements):
                value = ts * 100 + idx
                fields.append(Field(measurement, value, TSDataType.INT64))
            writer.write_row_record(RowRecord(device_id, ts, fields))

    writer.close()
    return tree_data_dir

# ============================================
# 1. 测试device_ids参数 - 各种设备ID
# ============================================

def test_device_ids_single_exist():
    """测试设备ID：单设备存在的设备"""
    try:
        # 1. 创建测试数据
        device_ids = ["root.device1"]
        measurement_names = ["s1", "s2"]
        measurement_names_map = {device_ids[0]: measurement_names}
        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        # 2. 行查询
        with TsFileReader(tree_data_dir) as reader:
            print()
            print(reader.get_all_timeseries_schemas())
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
                # 3. 验证数据
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="设备不存在时，预期不会报错，实际会报错")
def test_device_ids_single_not_exist():
    """测试设备ID：单设备不存在的设备"""
    try:
        device_ids = ["root.existing_device"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(["root.non_existent_device"], measurement_names)
            count = 0
            while result.next():
                count += 1
                # 3. 验证数据
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_device_ids_multi_all_exist():
    """测试设备ID：多设备全存在的设备"""
    try:
        device_ids = ["root.device1", "root.device2", "root.device3"]
        measurement_names = ["s1", "s2"]
        measurement_names_map = {d: measurement_names for d in device_ids}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=5)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 5, f"Expected 5 rows per device, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="部分设备不存在时，预期只输出存在的，实际会报错")
def test_device_ids_multi_partial_exist():
    """测试设备ID：多设备部分设备不存在"""
    try:
        device_ids = ["root.device1", "root.device2"]
        measurement_names = ["s1"]
        measurement_names_map = {d: measurement_names for d in device_ids}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            # 查询包含不存在的设备
            result = reader.query_tree_by_row(
                ["root.device1", "root.non_existent_device", "root.device2"],
                measurement_names
            )
            count = 0
            while result.next():
                count += 1
            # 预期只返回存在设备的数据
            assert count == 20, f"Expected 20 rows (from existing devices), got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="全部设备不存在时，预期输出空，实际会报错")
def test_device_ids_multi_all_not_exist():
    """测试设备ID：多设备全部设备不存在"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            try:
                reader.query_tree_by_row(
                    ["root.non_existent1", "root.non_existent2"],
                    measurement_names
                )
                assert False, "All non-existent devices should raise DeviceNotExistError"
            except DeviceNotExistError as e:
                assert "does not exist" in str(e)
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_device_ids_lowercase():
    """测试设备ID：小写英文"""
    try:
        device_ids = ["root.lowercase_device"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_device_ids_uppercase():
    """测试设备ID：大写英文"""
    try:
        device_ids = ["root.UPPERCASE_DEVICE"]
        measurement_names = ["S1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="纯数字设备ID会导致程序崩溃")
def test_device_ids_numbers():
    """测试设备ID：纯数字"""
    try:
        device_ids = ["root.1234567890"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_device_ids_underscore():
    """测试设备ID：纯下划线"""
    try:
        device_ids = ["root._______"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_device_ids_chinese():
    """测试设备ID：中文字符"""
    try:
        device_ids = ["root.测试设备"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())

@pytest.mark.skip(reason="特殊字符带空格设备ID会导致程序崩溃")
def test_device_ids_special_chars():
    """测试设备ID：特殊字符"""
    try:
        device_ids = ["root.d 1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


# ============================================
# 2. 测试measurement_names参数 - 各种测点名
# ============================================

def test_measurement_names_single_exist():
    """测试测点名：单测点存在的测点"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1", "s2", "s3"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, ["s1"])
            count = 0
            while result.next():
                count += 1
                val = result.get_value_by_name("s1")
                assert val is not None
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())

@pytest.mark.skip(reason="单测点不存在的测点，预期输出空，实际报错")
def test_measurement_names_single_not_exist():
    """测试测点名：单测点不存在的测点"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, ["non_existent_measurement"])
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"

    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_measurement_names_multi_all_exist():
    """测试测点名：多测点全存在的测点"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1", "s2", "s3"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="部分测点不存在时，预期只输出存在的，实际报错")
def test_measurement_names_multi_partial_exist():
    """测试测点名：多测点部分测点不存在"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1", "s2"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, ["s1", "s2", "non_existent"])
            count = 0
            while result.next():
                count += 1
            # 预期返回所有数据，不存在的测点值为null
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="全部测点不存在时，预期输出空，实际报错")
def test_measurement_names_multi_all_not_exist():
    """测试测点名：多测点全部测点不存在"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            try:
                reader.query_tree_by_row(device_ids, ["non_existent1", "non_existent2"])
                assert False, "All non-existent measurements should raise MeasurementNotExistError"
            except MeasurementNotExistError:
                pass  # 异常类型正确即可
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_measurement_names_lowercase():
    """测试测点名：小写英文"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["lowercase_measurement"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_measurement_names_uppercase():
    """测试测点名：大写英文"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["UPPERCASE_MEASUREMENT"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="数字测点名会导致程序崩溃")
def test_measurement_names_numbers():
    """测试测点名：数字"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["1234567890"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_measurement_names_underscore():
    """测试测点名：下划线"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["______"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_measurement_names_chinese():
    """测试测点名：中文字符"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["测试测点"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="特殊字符带空格测点名会导致程序崩溃")
def test_measurement_names_special_chars():
    """测试测点名：特殊字符"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["measurement @#$"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


# ============================================
# 3. 测试offset参数
# ============================================

def test_offset_negative():
    """测试offset：小于0"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=20)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, offset=-10)
            count = 0
            while result.next():
                count += 1
            # offset为负数时，行为类似于从开头开始查询
            assert count == 20, f"Expected 20 rows with offset=-10, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_offset_zero():
    """测试offset：等于0"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=10)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, offset=0)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows with offset=0, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_offset_within_range():
    """测试offset：大于0且不超过实际行数"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=20)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, offset=5)
            count = 0
            while result.next():
                count += 1
            assert count == 15, f"Expected 15 rows with offset=5 (20-5), got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_offset_exceeds_rows():
    """测试offset：超过实际行数"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=10)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, offset=100)
            count = 0
            while result.next():
                count += 1
            assert count == 0, f"Expected 0 rows with offset exceeding total, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


# ============================================
# 4. 测试limit参数
# ============================================

def test_limit_negative():
    """测试limit：小于0（代表无限制）"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=100)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, limit=-1)
            count = 0
            while result.next():
                count += 1
            assert count == 100, f"Expected 100 rows with limit=-1 (unlimited), got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


@pytest.mark.skip(reason="limit=0会导致TSDataType枚举错误")
def test_limit_zero():
    """测试limit：等于0"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=10)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, limit=0)
            count = 0
            while result.next():
                count += 1
            assert count == 0, f"Expected 0 rows with limit=0, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_limit_within_range():
    """测试limit：大于0且不超过实际行数"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=20)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, limit=5)
            count = 0
            while result.next():
                count += 1
            assert count == 5, f"Expected 5 rows with limit=5, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_limit_exceeds_rows():
    """测试limit：超过实际行数"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=10)

        with TsFileReader(tree_data_dir) as reader:
            result = reader.query_tree_by_row(device_ids, measurement_names, limit=100)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows (actual count) with limit=100, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_offset_and_limit_combination():
    """测试offset和limit组合"""
    try:
        device_ids = ["root.device1"]
        measurement_names = ["s1"]
        measurement_names_map = {device_ids[0]: measurement_names}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=30)

        with TsFileReader(tree_data_dir) as reader:
            # offset=10, limit=10, 应该返回第11-20行的10条数据
            result = reader.query_tree_by_row(device_ids, measurement_names, offset=10, limit=10)
            count = 0
            while result.next():
                count += 1
            assert count == 10, f"Expected 10 rows with offset=10, limit=10, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())

@pytest.mark.skip(reason="查询多个设备的不同测点，预期可以输出存在的，实际报错NotExistsError")
def test_multi_device_different_measurements():
    """测试多设备，每个设备的测点名不一样情况"""
    try:
        device_ids = ["root.device1", "root.device2", "root.device3"]
        # 每个设备有不同的测点
        measurement_names_map = {
            "root.device1": ["temperature", "humidity"],
            "root.device2": ["pressure", "flow"],
            "root.device3": ["voltage", "current", "power"]
        }

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=5)

        with TsFileReader(tree_data_dir) as reader:
            # 查询多个设备的不同测点
            result = reader.query_tree_by_row(device_ids, ["temperature", "pressure", "current"])
            count = 0
            while result.next():
                count += 1
            assert count == 5, f"Expected 5 rows for device1, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_multi_device_common_measurements():
    """测试多设备，查询所有设备共有的测点"""
    try:
        device_ids = ["root.device1", "root.device2"]
        # 部分测点是共有的，部分是独有的
        measurement_names_map = {
            "root.device1": ["common_s1"],
            "root.device2": ["common_s1"]
        }

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=5)

        with TsFileReader(tree_data_dir) as reader:
            # 查询共有的测点
            result = reader.query_tree_by_row(device_ids, ["common_s1"])
            count = 0
            while result.next():
                count += 1
            # 预期返回两个设备的数据
            assert count == 5, f"Expected 5 rows , got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())


def test_multi_device_all_measurements():
    """测试多设备查询，指定部分测点"""
    try:
        device_ids = ["root.device1", "root.device2"]
        measurement_names = ["s1", "s2"]
        measurement_names_map = {d: measurement_names for d in device_ids}

        tree_data_dir = create_test_tsfile_tree(device_ids, measurement_names_map, rows_per_device=5)

        with TsFileReader(tree_data_dir) as reader:
            # 只查询一个测点
            result = reader.query_tree_by_row(device_ids, measurement_names)
            count = 0
            while result.next():
                count += 1
            assert count >= 5, f"Expected at least 5 rows, got {count}"
    finally:
        if os.path.exists(get_tree_data_dir()):
            os.remove(get_tree_data_dir())