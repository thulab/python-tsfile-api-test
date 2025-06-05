import os
import random
from typing import List

import numpy
import csv
from tsfile import ColumnSchema, TableSchema
from tsfile import Tablet
from tsfile import TsFileTableWriter, TsFileReader, TSDataType, ColumnCategory

"""
标题：表模型写入测试
作者：肖林捷
日期：2025/5
"""

# tsfile文件路径
tsfile_path = "../../data/tsfile/python_table.tsfile"


def read(table_data_dir: str, table_name: str, column_name_list: List[str], column_type_list: List[TSDataType],
         expect_num_rows: int):
    """
    读取数据，统计实际数量，验证测试有效性

    参数:
    table_data_dir (str): tsfile 文件的路径。
    table_name (str): 需要查询验证的表名。
    column_name_list (List[str]): 需要查询验证的列名。
    column_type_list: List[TSDataType]: 需要查询验证的数据类型。

    """
    with TsFileReader(table_data_dir) as reader:
        # 用于记录实际查询的结果行数
        actual_num_rows = 0
        start_time = numpy.iinfo(numpy.int64).min
        end_time = numpy.iinfo(numpy.int64).max
        with reader.query_table(table_name, column_name_list, start_time, end_time) as result:
            # 获取元数据
            metadata = result.get_metadata()
            # 验证元数据信息匹配
            for i in range(1, metadata.get_column_num()):
                # 验证列名
                assert metadata.get_column_name(i + 1) == column_name_list[
                    i - 1], "Inconsistent column name,actual_column_name: {}, expect_column_name: {}".format(
                    metadata.get_column_name(i + 1), column_name_list[i - 1])
                # 验证数据类型
                assert metadata.get_data_type(i + 1) == column_type_list[
                    i - 1], "Inconsistent data type,actual_data_type: {}, expect_data_type: {}".format(
                    metadata.get_data_type(i + 1), column_type_list[i - 1])

            while result.next():
                for i in range(0, len(column_name_list)):
                    # 验证值
                    assert result.get_value_by_index(i + 2) == result.get_value_by_name(column_name_list[
                                                                                            i]), "The values obtained by the get-value-by_index method and the get-value-by_name method are inconsistent"
                actual_num_rows += 1
            # 验证行数
            assert actual_num_rows == expect_num_rows, "actual_num_rows: {}, expect_num_rows: {}".format(
                actual_num_rows, expect_num_rows)


def parse_csv(file_path):
    """
    解析 CSV 文件，跳过以 # 开头的行。

    参数:
    file_path (str): CSV 文件的路径。

    返回:
    list of lists: 解析后的数据，每一行是一个列表。
    """
    # 用于存储解析后的数据
    parsed_data = []

    # 解析 CSV 文件
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        # 创建 CSV 读取器
        reader = csv.reader(file)

        for row in reader:
            # 检查行是否以 # 开头
            if row and row[0].startswith('#'):
                continue  # 跳过该行

            # 如果行不为空，添加到解析数据中
            if row:
                parsed_data.append(row)

    return parsed_data


##  1、测试写入：各种表名
def test_write1():
    # 解析各种表名数据文件
    table_names = parse_csv("../data/csv/table_name_normal.csv")
    for name in table_names:
        # 构造元数据
        table_data_dir = os.path.join(os.path.dirname(__file__), tsfile_path)
        if os.path.exists(table_data_dir):
            os.remove(table_data_dir)
        table_name = name[0]
        column_name_list = [
            "tag1", "tag2", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"
        ]
        data_type_list = [
            TSDataType.STRING, TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
            TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
            TSDataType.FLOAT, TSDataType.DOUBLE
        ]
        columnCategory_list = [
            ColumnCategory.TAG, ColumnCategory.TAG,
            ColumnCategory.FIELD, ColumnCategory.FIELD, ColumnCategory.FIELD, ColumnCategory.FIELD,
            ColumnCategory.FIELD,
            ColumnCategory.FIELD, ColumnCategory.FIELD, ColumnCategory.FIELD, ColumnCategory.FIELD, ColumnCategory.FIELD
        ]
        columns = []
        for i in range(0, len(column_name_list)):
            columns.append(ColumnSchema(column_name_list[i], data_type_list[i], columnCategory_list[i]))
        table_schema = TableSchema(table_name, columns)

        # 解析各种数据类型和各种范围值数据文件
        datas = parse_csv("../data/csv/data1.csv")
        # 写入数据
        with TsFileTableWriter(table_data_dir, table_schema) as writer:
            tablet_row_num = len(datas)
            tablet = Tablet(
                column_name_list,
                data_type_list,
                tablet_row_num)

            expect_num_rows = 0

            row_index = 0
            for data in datas:
                tablet.add_timestamp(row_index, int(data[0]))
                for i in range(len(column_name_list)):
                    if data_type_list[i] == TSDataType.STRING:
                        tablet.add_value_by_name(column_name_list[i], row_index, data[i + 1])
                    elif data_type_list[i] == TSDataType.BOOLEAN:
                        tablet.add_value_by_name(column_name_list[i], row_index, bool(data[i + 1]))
                    elif data_type_list[i] == TSDataType.INT64:
                        tablet.add_value_by_name(column_name_list[i], row_index, int(data[i + 1]))
                    elif data_type_list[i] == TSDataType.INT32:
                        tablet.add_value_by_name(column_name_list[i], row_index, int(data[i + 1]))
                    elif data_type_list[i] == TSDataType.FLOAT:
                        tablet.add_value_by_name(column_name_list[i], row_index, float(data[i + 1]))
                    elif data_type_list[i] == TSDataType.DOUBLE:
                        tablet.add_value_by_name(column_name_list[i], row_index, float(data[i + 1]))
                row_index += 1
                expect_num_rows += 1

            writer.write_table(tablet)
            writer.close()

            # 验证有效性
            read(table_data_dir, table_name, column_name_list, data_type_list, expect_num_rows)


##  2、测试写入：列的元数据，各种列名、各种数据类型和各种列类别、含空值（暂且不含）
# def test_write2():
#     table_data_dir = os.path.join(os.path.dirname(__file__), tsfile_path)
#     if os.path.exists(table_data_dir):
#         os.remove(table_data_dir)
#     # 构造元数据
#     table_name = "t2"
#     column_name_list = ["1234567890", "QWERTYUIOPASDFGHJKLZXCVBNM", "qwertyuiopasdfghjklzxcvbnm123", "没问题", "     ",
#                         "   s4   ", "!@#$%^&*()", "insert", "for",
#                         "   12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890  ",
#                         "s9", "s10"]
#     data_type_list = [TSDataType.STRING, TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
#                  TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
#                  TSDataType.FLOAT, TSDataType.DOUBLE]
#     column1 = ColumnSchema(column_name_list[0], data_type_list[0], ColumnCategory.TAG)
#     column2 = ColumnSchema(column_name_list[1], data_type_list[1], ColumnCategory.TAG)
#     column3 = ColumnSchema(column_name_list[2], data_type_list[2], ColumnCategory.FIELD)
#     column4 = ColumnSchema(column_name_list[3], data_type_list[3], ColumnCategory.FIELD)
#     column5 = ColumnSchema(column_name_list[4], data_type_list[4], ColumnCategory.FIELD)
#     column6 = ColumnSchema(column_name_list[5], data_type_list[5], ColumnCategory.FIELD)
#     column7 = ColumnSchema(column_name_list[6], data_type_list[6], ColumnCategory.FIELD)
#     column8 = ColumnSchema(column_name_list[7], data_type_list[7])
#     column9 = ColumnSchema(column_name_list[8], data_type_list[8])
#     column10 = ColumnSchema(column_name_list[9], data_type_list[9])
#     column11 = ColumnSchema(column_name_list[10], data_type_list[10])
#     column12 = ColumnSchema(column_name_list[11], data_type_list[11])
#     columns = [column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, column11,
#                column12]
#     table_schema = TableSchema(table_name, columns)
#     assert table_name.lower() == table_schema.get_table_name()
#
#     # 写入数据
#     with TsFileTableWriter(table_data_dir, table_schema) as writer:
#         tablet_row_num = 100
#         tablet = Tablet(
#             column_name_list,
#             data_type_list,
#             tablet_row_num)
#
#         datas = parse_csv("../../data/csv/data1.csv")
#
#         row_index = 0
#         for data in datas:
#             tablet.add_timestamp(row_index, int(data[0]))
#             for i in range(len(column_name_list)):
#                 if "null" == data[i + 1]:
#                     continue
#                 elif data_type_list[i] == TSDataType.STRING:
#                     tablet.add_value_by_name(column_name_list[i], row_index, data[i + 1])
#                 elif data_type_list[i] == TSDataType.BOOLEAN:
#                     tablet.add_value_by_name(column_name_list[i], row_index, bool(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.INT64:
#                     tablet.add_value_by_name(column_name_list[i], row_index, int(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.INT32:
#                     tablet.add_value_by_name(column_name_list[i], row_index, int(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.FLOAT:
#                     tablet.add_value_by_name(column_name_list[i], row_index, float(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.DOUBLE:
#                     tablet.add_value_by_name(column_name_list[i], row_index, float(data[i + 1]))
#             row_index += 1
#
#         writer.write_table(tablet)
#         writer.close()
#
#         read(table_data_dir, table_name, column_name_list, data_type_list, 10)


##  3、测试写入：乱序时间戳、不同时间分区、add_value_by_index写入、含空值
# def test_write3():
#     table_data_dir = os.path.join(os.path.dirname(__file__), tsfile_path)
#     if os.path.exists(table_data_dir):
#         os.remove(table_data_dir)
#     # 构造元数据
#     table_name = "t3"
#     column_name_list = ["tag1", "tag2", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
#     data_type_list = [TSDataType.STRING, TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
#                  TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
#                  TSDataType.FLOAT, TSDataType.DOUBLE]
#     column1 = ColumnSchema(column_name_list[0], data_type_list[0], ColumnCategory.TAG)
#     column2 = ColumnSchema(column_name_list[1], data_type_list[1], ColumnCategory.TAG)
#     column3 = ColumnSchema(column_name_list[2], data_type_list[2], ColumnCategory.FIELD)
#     column4 = ColumnSchema(column_name_list[3], data_type_list[3], ColumnCategory.FIELD)
#     column5 = ColumnSchema(column_name_list[4], data_type_list[4], ColumnCategory.FIELD)
#     column6 = ColumnSchema(column_name_list[5], data_type_list[5], ColumnCategory.FIELD)
#     column7 = ColumnSchema(column_name_list[6], data_type_list[6], ColumnCategory.FIELD)
#     column8 = ColumnSchema(column_name_list[7], data_type_list[7])
#     column9 = ColumnSchema(column_name_list[8], data_type_list[8])
#     column10 = ColumnSchema(column_name_list[9], data_type_list[9])
#     column11 = ColumnSchema(column_name_list[10], data_type_list[10])
#     column12 = ColumnSchema(column_name_list[11], data_type_list[11])
#     columns = [column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, column11,
#                column12]
#     table_schema = TableSchema(table_name, columns)
#     assert table_name.lower() == table_schema.get_table_name()
#
#     # 写入数据
#     with TsFileTableWriter(table_data_dir, table_schema) as writer:
#         tablet_row_num = 100
#         tablet = Tablet(
#             column_name_list,
#             data_type_list,
#             tablet_row_num)
#
#         datas = parse_csv("../../data/csv/data2.csv")
#
#         row_index = 0
#         for data in datas:
#             tablet.add_timestamp(row_index, int(data[0]))
#             for i in range(len(column_name_list)):
#                 if "null" == data[i + 1]:
#                     continue
#                 elif data_type_list[i] == TSDataType.STRING:
#                     tablet.add_value_by_index(i, row_index, data[i + 1])
#                 elif data_type_list[i] == TSDataType.BOOLEAN:
#                     tablet.add_value_by_index(i, row_index, bool(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.INT64:
#                     tablet.add_value_by_index(i, row_index, int(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.INT32:
#                     tablet.add_value_by_index(i, row_index, int(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.FLOAT:
#                     tablet.add_value_by_index(i, row_index, float(data[i + 1]))
#                 elif data_type_list[i] == TSDataType.DOUBLE:
#                     tablet.add_value_by_index(i, row_index, float(data[i + 1]))
#             row_index += 1
#
#         writer.write_table(tablet)
#         writer.close()
#
#         read(table_data_dir, table_name, column_name_list, data_type_list, 10)

#  4、测试1万行、1TAG列和FIELD列 【TIMECHODB-0278】
def test_write4():
    table_data_dir = os.path.join(os.path.dirname(__file__), tsfile_path)
    if os.path.exists(table_data_dir):
        os.remove(table_data_dir)
    # 构造元数据
    table_name = "t4"
    column_name_list = []
    data_type_list = []
    columns = []
    for i in range(2):
        column_name_list.append("tag" + str(i))
        data_type_list.append(TSDataType.STRING)
        columns.append(ColumnSchema("tag" + str(i), TSDataType.STRING, ColumnCategory.TAG))
    for i in range(1):
        column_name_list.append("f1_" + str(i))
        data_type_list.append(TSDataType.BOOLEAN)
        columns.append(ColumnSchema("f1_" + str(i), TSDataType.BOOLEAN, ColumnCategory.FIELD))
        column_name_list.append("f2_" + str(i))
        data_type_list.append(TSDataType.INT32)
        columns.append(ColumnSchema("f2_" + str(i), TSDataType.INT32, ColumnCategory.FIELD))
        column_name_list.append("f3_" + str(i))
        data_type_list.append(TSDataType.INT64)
        columns.append(ColumnSchema("f3_" + str(i), TSDataType.INT64, ColumnCategory.FIELD))
        column_name_list.append("f4_" + str(i))
        data_type_list.append(TSDataType.FLOAT)
        columns.append(ColumnSchema("f4_" + str(i), TSDataType.FLOAT, ColumnCategory.FIELD))
        column_name_list.append("f5_" + str(i))
        data_type_list.append(TSDataType.DOUBLE)
        columns.append(ColumnSchema("f5_" + str(i), TSDataType.DOUBLE, ColumnCategory.FIELD))
    table_schema = TableSchema(table_name, columns)

    # 写入数据
    with TsFileTableWriter(table_data_dir, table_schema) as writer:
        tablet_row_num = 10
        tablet = Tablet(
            column_name_list,
            data_type_list,
            tablet_row_num)

        for row_num in range(tablet_row_num):
            tablet.add_timestamp(row_num, row_num)
            for i in range(len(column_name_list)):
                if data_type_list[i] == TSDataType.STRING:
                    tablet.add_value_by_name(column_name_list[i], row_num, "tag" + str(random.randint(1, 10)))
                elif data_type_list[i] == TSDataType.BOOLEAN:
                    tablet.add_value_by_name(column_name_list[i], row_num, True)
                elif data_type_list[i] == TSDataType.INT64:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.INT32:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.FLOAT:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))
                elif data_type_list[i] == TSDataType.DOUBLE:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))

        writer.write_table(tablet)
        writer.close()

        read(table_data_dir, table_name, column_name_list, data_type_list, tablet_row_num)


##  5、多次写入：重复写入和不重复写入
def test_write5():
    # 构造元数据
    table_data_dir = os.path.join(os.path.dirname(__file__), tsfile_path)
    if os.path.exists(table_data_dir):
        os.remove(table_data_dir)
    # 构造元数据
    table_name = "t5"
    column_name_list = ["tag1", "tag2", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
    data_type_list = [TSDataType.STRING, TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
                      TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
                      TSDataType.FLOAT, TSDataType.DOUBLE]
    column1 = ColumnSchema(column_name_list[0], data_type_list[0], ColumnCategory.TAG)
    column2 = ColumnSchema(column_name_list[1], data_type_list[1], ColumnCategory.TAG)
    column3 = ColumnSchema(column_name_list[2], data_type_list[2], ColumnCategory.FIELD)
    column4 = ColumnSchema(column_name_list[3], data_type_list[3], ColumnCategory.FIELD)
    column5 = ColumnSchema(column_name_list[4], data_type_list[4], ColumnCategory.FIELD)
    column6 = ColumnSchema(column_name_list[5], data_type_list[5], ColumnCategory.FIELD)
    column7 = ColumnSchema(column_name_list[6], data_type_list[6], ColumnCategory.FIELD)
    column8 = ColumnSchema(column_name_list[7], data_type_list[7], ColumnCategory.FIELD)
    column9 = ColumnSchema(column_name_list[8], data_type_list[8], ColumnCategory.FIELD)
    column10 = ColumnSchema(column_name_list[9], data_type_list[9], ColumnCategory.FIELD)
    column11 = ColumnSchema(column_name_list[10], data_type_list[10], ColumnCategory.FIELD)
    column12 = ColumnSchema(column_name_list[11], data_type_list[11], ColumnCategory.FIELD)
    columns = [column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, column11,
               column12]
    table_schema = TableSchema(table_name, columns)

    # 写入数据
    with TsFileTableWriter(table_data_dir, table_schema) as writer:
        tablet_row_num = 1000
        tablet = Tablet(
            column_name_list,
            data_type_list,
            tablet_row_num)

        for row_num in range(0, 100):
            tablet.add_timestamp(row_num, row_num)
            for i in range(len(column_name_list)):
                if data_type_list[i] == TSDataType.STRING:
                    tablet.add_value_by_name(column_name_list[i], row_num, "tag" + str(random.randint(1, 10)))
                elif data_type_list[i] == TSDataType.BOOLEAN:
                    tablet.add_value_by_name(column_name_list[i], row_num, True)
                elif data_type_list[i] == TSDataType.INT64:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.INT32:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.FLOAT:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))
                elif data_type_list[i] == TSDataType.DOUBLE:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))

        writer.write_table(tablet)

        for row_num in range(100, 200):
            tablet.add_timestamp(row_num, row_num - 100)
            for i in range(len(column_name_list)):
                if data_type_list[i] == TSDataType.STRING:
                    tablet.add_value_by_name(column_name_list[i], row_num, "tag" + str(random.randint(10, 20)))
                elif data_type_list[i] == TSDataType.BOOLEAN:
                    tablet.add_value_by_name(column_name_list[i], row_num, False)
                elif data_type_list[i] == TSDataType.INT64:
                    tablet.add_value_by_name(column_name_list[i], row_num, i + 100)
                elif data_type_list[i] == TSDataType.INT32:
                    tablet.add_value_by_name(column_name_list[i], row_num, i + 100)
                elif data_type_list[i] == TSDataType.FLOAT:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i + 100))
                elif data_type_list[i] == TSDataType.DOUBLE:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i + 100))

        writer.write_table(tablet)

        for row_num in range(200, 300):
            tablet.add_timestamp(row_num, row_num + 100)
            for i in range(len(column_name_list)):
                if data_type_list[i] == TSDataType.STRING:
                    tablet.add_value_by_name(column_name_list[i], row_num, "tag" + str(random.randint(1, 10)))
                elif data_type_list[i] == TSDataType.BOOLEAN:
                    tablet.add_value_by_name(column_name_list[i], row_num, True)
                elif data_type_list[i] == TSDataType.INT64:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.INT32:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.FLOAT:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))
                elif data_type_list[i] == TSDataType.DOUBLE:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))

        writer.write_table(tablet)

        for row_num in range(300, 400):
            tablet.add_timestamp(row_num, row_num + 100)
            for i in range(len(column_name_list)):
                if data_type_list[i] == TSDataType.STRING:
                    tablet.add_value_by_name(column_name_list[i], row_num, "tag" + str(random.randint(1, 10)))
                elif data_type_list[i] == TSDataType.BOOLEAN:
                    tablet.add_value_by_name(column_name_list[i], row_num, True)
                elif data_type_list[i] == TSDataType.INT64:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.INT32:
                    tablet.add_value_by_name(column_name_list[i], row_num, i)
                elif data_type_list[i] == TSDataType.FLOAT:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))
                elif data_type_list[i] == TSDataType.DOUBLE:
                    tablet.add_value_by_name(column_name_list[i], row_num, float(i))

        for i in range(10):
            writer.write_table(tablet)
        writer.close()

        read(table_data_dir, table_name, column_name_list, data_type_list, 4600)
