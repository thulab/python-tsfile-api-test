# import os
# from typing import List
#
# import numpy
# from tsfile import TsFileReader, TSDataType
#
# tsfile_path = "../../data/tsfile/python_table"
#
# table_data_dir = os.path.join(tsfile_path)
#
# table_name = "t1"
# column_name_list = ["tag1",  "s1"]
#
# with TsFileReader(table_data_dir) as reader:
#     # 打印全部的schema
#     print(reader.get_all_table_schemas())
#     start_time = numpy.iinfo(numpy.int64).min
#     end_time = numpy.iinfo(numpy.int64).max
#     with reader.query_table(table_name, column_name_list, start_time, end_time) as result:
#         metadata = result.get_metadata()
#         # 查询列名
#         for i in range(1, metadata.get_column_num() + 1):
#             print(metadata.get_column_name(i), end=" ")
#         print()
#         # 查询列对应的数据类型
#         for i in range(1, metadata.get_column_num() + 1):
#             type = metadata.get_data_type(i)
#             if type == TSDataType.INT32:
#                 print("INT32", end=" ")
#             elif type == TSDataType.INT64:
#                 print("INT64", end=" ")
#             elif type == TSDataType.FLOAT:
#                 print("FLOAT", end=" ")
#             elif type == TSDataType.DOUBLE:
#                 print("DOUBLE", end=" ")
#             elif type == TSDataType.STRING:
#                 print("STRING", end=" ")
#             elif type == TSDataType.BOOLEAN:
#                 print("BOOLEAN", end=" ")
#             else:
#                 print(f"UNKNOWN: {type}", end=" ")
#         print()
#         # 查询数据
#         while result.next():
#             for i in range(1, len(column_name_list)):
#                 if result.is_null_by_index(i) == 1:
#                     print("is null", end=" ")
#                 else:
#                     print(result.get_value_by_index(i), end=" ")
#             for i in range(1, len(column_name_list)):
#                 if result.is_null_by_name(column_name_list[i]) == 1:
#                     print("is null", end=" ")
#                 else:
#                     print(result.get_value_by_name(column_name_list[i]), end=" ")
#             print()
#             print(result.read_data_frame())
