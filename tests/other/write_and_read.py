# import os
# from typing import List
#
# import numpy
# import pytest
# from tsfile import ColumnSchema, TableSchema
# from tsfile import Tablet
# from tsfile import TsFileTableWriter, TsFileReader, TSDataType, ColumnCategory
#
#
# def read(table_data_dir: str, table_name: str, column_name_list: List[str], tablet_row_num: int):
#     with TsFileReader(table_data_dir) as reader:
#         print(reader.get_all_table_schemas())
#         start_time = numpy.iinfo(numpy.int64).min
#         end_time = numpy.iinfo(numpy.int64).max
#         with reader.query_table(table_name, column_name_list, start_time, end_time) as result:
#             metadata = result.get_metadata()
#             for i in range(1, len(column_name_list) - 1):
#                 print(metadata.get_column_name(i), end=" ")
#             print()
#             for i in range(1, len(column_name_list) - 1):
#                 type = metadata.get_data_type(i)
#                 if type == TSDataType.INT32:
#                     print("INT32", end=" ")
#                 elif type == TSDataType.INT64:
#                     print("INT64", end=" ")
#                 elif type == TSDataType.FLOAT:
#                     print("FLOAT", end=" ")
#                 elif type == TSDataType.DOUBLE:
#                     print("DOUBLE", end=" ")
#                 elif type == TSDataType.STRING:
#                     print("STRING", end=" ")
#                 elif type == TSDataType.BOOLEAN:
#                     print("BOOLEAN", end=" ")
#                 else:
#                     print(f"UNKNOWN: {type}", end=" ")
#             print()
#             while result.next():
#                 for i in range(1, len(column_name_list)):
#                     print(result.get_value_by_index(i), end=" ")
#                 print(result.get_value_by_name("tag1"), end=" ")
#                 print(result.get_value_by_name("tag2"), end=" ")
#                 print(result.get_value_by_name("s1"), end=" ")
#                 print(result.get_value_by_name("s2"), end=" ")
#                 print(result.get_value_by_name("s3"), end=" ")
#                 print(result.get_value_by_name("s4"), end=" ")
#                 print(result.get_value_by_name("s5"))
#                 # print(result.read_data_frame())
#
#
# ##  测试写入
# def test_write():
#     table_data_dir = os.path.join(os.path.dirname(__file__), "../../data/table_data.tsfile")
#     if os.path.exists(table_data_dir):
#         os.remove(table_data_dir)
#     # 构造元数据
#     table_name = "t1"
#     column_name_list = ["tag1", "tag2", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
#     type_list = [TSDataType.STRING, TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
#                  TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
#                  TSDataType.FLOAT, TSDataType.DOUBLE]
#     column1 = ColumnSchema(column_name_list[0], type_list[0], ColumnCategory.TAG)
#     print(column1.get_column_name(), end=" ")
#     print(column1.get_data_type(), end=" ")
#     print(column1.get_category())
#     column2 = ColumnSchema(column_name_list[1], type_list[1], ColumnCategory.TAG)
#     column3 = ColumnSchema(column_name_list[2], type_list[2], ColumnCategory.FIELD)
#     print(column3.get_column_name(), end=" ")
#     print(column3.get_data_type(), end=" ")
#     print(column3.get_category())
#     column4 = ColumnSchema(column_name_list[3], type_list[3], ColumnCategory.FIELD)
#     print(column4.get_column_name(), end=" ")
#     print(column4.get_data_type(), end=" ")
#     print(column4.get_category())
#     column5 = ColumnSchema(column_name_list[4], type_list[4], ColumnCategory.FIELD)
#     print(column5.get_column_name(), end=" ")
#     print(column5.get_data_type(), end=" ")
#     print(column5.get_category())
#     column6 = ColumnSchema(column_name_list[5], type_list[5], ColumnCategory.FIELD)
#     print(column6.get_column_name(), end=" ")
#     print(column6.get_data_type(), end=" ")
#     print(column6.get_category())
#     column7 = ColumnSchema(column_name_list[6], type_list[6], ColumnCategory.FIELD)
#     print(column7.get_column_name(), end=" ")
#     print(column7.get_data_type(), end=" ")
#     print(column7.get_category())
#     column8 = ColumnSchema(column_name_list[7], type_list[7], ColumnCategory.FIELD)
#     column9 = ColumnSchema(column_name_list[8], type_list[8], ColumnCategory.FIELD)
#     column10 = ColumnSchema(column_name_list[9], type_list[9], ColumnCategory.FIELD)
#     column11 = ColumnSchema(column_name_list[10], type_list[10], ColumnCategory.FIELD)
#     column12 = ColumnSchema(column_name_list[11], type_list[11], ColumnCategory.FIELD)
#     columns = [column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, column11,
#                column12]
#     table_schema = TableSchema(table_name, columns)
#     print(table_schema.get_table_name(), end=" ")
#     print(table_schema.get_columns())
#
#     # 写入数据
#     with TsFileTableWriter(table_data_dir, table_schema) as writer:
#         tablet_row_num = 100
#         tablet = Tablet(
#             column_name_list,
#             type_list,
#             tablet_row_num)
#
#         # time_stamp = numpy.iinfo(numpy.int64).max
#         for i in range(tablet_row_num):
#             tablet.add_timestamp(i, i)
#             tablet.add_value_by_name("tag1", i, "tag1")
#             tablet.add_value_by_name("tag2", i, "tag2_" + str(100 - i))
#             tablet.add_value_by_name("s1", i, True)
#             tablet.add_value_by_name("s2", i, 100)
#             tablet.add_value_by_name("s3", i, 100)
#             tablet.add_value_by_name("s4", i, 100.001)
#             tablet.add_value_by_name("s5", i, 100.001)
#             tablet.add_value_by_name("s6", i, False)
#             tablet.add_value_by_index(8, i, 100)
#             tablet.add_value_by_index(9, i, 100)
#             tablet.add_value_by_index(10, i, 100.001)
#             tablet.add_value_by_index(11, i, 100.001)
#             with pytest.raises(OverflowError):
#                 tablet.add_value_by_name("s2", i, 2 ** 32)
#             with pytest.raises(ValueError):
#                 tablet.add_value_by_name("no", i, "error")
#             with pytest.raises(IndexError):
#                 tablet.add_value_by_name("s3", -1, "error")
#             with pytest.raises(IndexError):
#                 tablet.add_value_by_index(1000, i, "error")
#             with pytest.raises(IndexError):
#                 tablet.add_value_by_index(11, -1, "error")
#             with pytest.raises(TypeError):
#                 tablet.add_value_by_name("s4", i, 100)
#             with pytest.raises(TypeError):
#                 tablet.add_value_by_index(8, i, 100.001)
#
#         writer.write_table(tablet)
#         writer.close()
#
#         tablet.add_column("s11", TSDataType.BOOLEAN)
#         tablet.set_timestamp_list([0, 1, 2])
#         tablet.remove_column("s11")
#         tablet.get_value_by_index(1, 1)
#         tablet.get_value_by_name("s1", 1)
#         tablet.get_value_list_by_name("s1")
#         with pytest.raises(ValueError):
#             tablet.get_value_by_name("no", 1)
#         with pytest.raises(IndexError):
#             tablet.get_value_by_name("s1", -1)
#         with pytest.raises(ValueError):
#             tablet.get_value_list_by_name("no")
#
#         read(table_data_dir, table_name, column_name_list)
# # table_data_dir = os.path.join(os.path.dirname(__file__), "../../data/table_data.tsfile")
# # if os.path.exists(table_data_dir):
# #     os.remove(table_data_dir)
# # # 构造元数据
# # table_name = "t1"
# # column_name_list = ["tag1", "tag2", "s1", "s2", "s3", "s4", "s5"]
# # type_list = [TSDataType.STRING, TSDataType.STRING, TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64,
# #              TSDataType.FLOAT, TSDataType.DOUBLE]
# # column1 = ColumnSchema(column_name_list[0], type_list[0], ColumnCategory.TAG)
# # print(column1.get_column_name(), end=" ")
# # print(column1.get_data_type(), end=" ")
# # print(column1.get_category())
# # column2 = ColumnSchema(column_name_list[1], type_list[1], ColumnCategory.TAG)
# # column3 = ColumnSchema(column_name_list[2], type_list[2], ColumnCategory.FIELD)
# # print(column3.get_column_name(), end=" ")
# # print(column3.get_data_type(), end=" ")
# # print(column3.get_category())
# # column4 = ColumnSchema(column_name_list[3], type_list[3], ColumnCategory.FIELD)
# # print(column4.get_column_name(), end=" ")
# # print(column4.get_data_type(), end=" ")
# # print(column4.get_category())
# # column5 = ColumnSchema(column_name_list[4], type_list[4], ColumnCategory.FIELD)
# # print(column5.get_column_name(), end=" ")
# # print(column5.get_data_type(), end=" ")
# # print(column5.get_category())
# # column6 = ColumnSchema(column_name_list[5], type_list[5], ColumnCategory.FIELD)
# # print(column6.get_column_name(), end=" ")
# # print(column6.get_data_type(), end=" ")
# # print(column6.get_category())
# # column7 = ColumnSchema(column_name_list[6], type_list[6], ColumnCategory.FIELD)
# # print(column7.get_column_name(), end=" ")
# # print(column7.get_data_type(), end=" ")
# # print(column7.get_category())
# # columns = [column1, column2, column3, column4, column5, column6, column7]
# # table_schema = TableSchema(table_name, columns)
# # print(table_schema.get_table_name(), end=" ")
# # print(table_schema.get_columns())
# #
# # # 写入数据
# # with TsFileTableWriter(table_data_dir, table_schema) as writer:
# #     tablet_row_num = 1024
# #     tablet = Tablet(
# #         column_name_list,
# #         type_list,
# #         tablet_row_num)
# #
# #     # time_stamp = numpy.iinfo(numpy.int64).max
# #     time_stamp = 0
# #     for i in range(tablet_row_num):
# #         tablet.add_timestamp(i, time_stamp)
# #         tablet.add_value_by_name("tag1", i, "test1")
# #         tablet.add_value_by_name("tag2", i, "test" + str(i))
# #         tablet.add_value_by_name("s1", i, True)
# #         if i % 2 == 0:
# #             tablet.add_value_by_name("s2", i, 100)
# #             tablet.add_value_by_name("s3", i, 100)
# #             tablet.add_value_by_name("s4", i, 100.001)
# #             tablet.add_value_by_name("s5", i, 100.001)
# #         time_stamp += 1
# #     writer.write_table(tablet)
# #     writer.close()
# #
# #     read(table_data_dir, table_name, column_name_list, tablet_row_num)
