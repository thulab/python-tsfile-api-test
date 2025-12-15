import os
from datetime import date

import numpy as np
from tsfile import to_dataframe, TableNotExistError, TableSchema, ColumnSchema, TSDataType, ColumnCategory, \
    TsFileTableWriter, Tablet, ColumnNotExistError

"""
标题：表模型 to_dataframe 接口功能测试
作者：肖林捷
日期：2025/12
"""


# tsfile文件路径
# tsfile_path = "../../data/tsfile/table_data.tsfile"
# tsfile_path = "C:\\IoTDBProjects\\iotdb-all\\timechodb-2.0.6-SNAPSHOT-bin-all\\timechodb-2.0.6.4-bin-rc2-alone-ainode\\data\\datanode\\data\\sequence\\test_g_0\\12\\-102790\\1764743152066-1-0-0.tsfile"

def test_table_all_datatype_query_to_dataframe_variants():
    """
    测试 to_dataframe 函数的正常功能
    """
    tsfile_path = "test_table.tsfile"
    table = TableSchema("test_table",
                        [ColumnSchema("Device1", TSDataType.STRING, ColumnCategory.TAG),
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

    dateSet = set()

    try:
        if os.path.exists(tsfile_path):
            os.remove(tsfile_path)
        max_row_num = 100
        with TsFileTableWriter(tsfile_path, table) as writer:
            tablet = Tablet(
                ["Device1", "Device2",
                 "Value1", "Value2", "Value3", "Value4", "Value5", "Value6", "Value7", "Value8", "Value9", "Value10",
                 "Value11", "Value12", "Value13", "Value14", "Value15", "Value16", "Value17", "Value18", "Value19", "Value20"],
                [TSDataType.STRING, TSDataType.STRING,
                 TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.TEXT, TSDataType.STRING, TSDataType.BLOB, TSDataType.TIMESTAMP, TSDataType.DATE,
                 TSDataType.BOOLEAN, TSDataType.INT32, TSDataType.INT64, TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.TEXT, TSDataType.STRING, TSDataType.BLOB, TSDataType.TIMESTAMP, TSDataType.DATE],
                max_row_num)
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
                dateSet.add(date(2025, 1, i % 20 + 1))
            writer.write_table(tablet)

        print()
        # 1. 默认
        df1_1 = to_dataframe(tsfile_path)
        assert df1_1.shape[0] == max_row_num
        assert df1_1.iloc[0, 0] == 0

        # 2. 指定列名
        # df2_0 = to_dataframe(tsfile_path, column_names=["Device1"]) # TODO：目前返回空，下个版本优化后也要单独列出TAG列数据
        # for i in range(max_row_num):
        #     assert df2_0.iloc[i, 1] == "Device1_" + str(df2_0.iloc[i, 0])
        df2_1 = to_dataframe(tsfile_path, column_names=["Value1"])
        for i in range(max_row_num):
            assert df2_1.iloc[i, 1] == np.bool_(df2_1.iloc[i, 0] % 2 == 0)
        df2_2 = to_dataframe(tsfile_path, column_names=["Value2"])
        for i in range(max_row_num):
            assert df2_2.iloc[i, 1] == np.int32(df2_2.iloc[i, 0] * 3)
        df2_3 = to_dataframe(tsfile_path, column_names=["Value3"])
        for i in range(max_row_num):
            assert df2_3.iloc[i, 1] == np.int64(df2_3.iloc[i, 0] * 4)
        df2_4 = to_dataframe(tsfile_path, column_names=["Value4"])
        for i in range(max_row_num):
            assert df2_4.iloc[i, 1] == np.float32(df2_4.iloc[i, 0] * 5.5)
        df2_5 = to_dataframe(tsfile_path, column_names=["Value5"])
        for i in range(max_row_num):
            assert df2_5.iloc[i, 1] == np.float64(df2_5.iloc[i, 0] * 6.6)
        df2_6 = to_dataframe(tsfile_path, column_names=["Value6"])
        for i in range(max_row_num):
            assert df2_6.iloc[i, 1] == f"string_value_{df2_6.iloc[i, 0]}"
        df2_7 = to_dataframe(tsfile_path, column_names=["Value7"])
        for i in range(max_row_num):
            assert df2_7.iloc[i, 1] == f"text_value_{df2_7.iloc[i, 0]}"
        df2_8 = to_dataframe(tsfile_path, column_names=["Value8"])
        for i in range(max_row_num):
            assert df2_8.iloc[i, 1] == f"blob_data_{df2_8.iloc[i, 0]}".encode('utf-8')
        df2_9 = to_dataframe(tsfile_path, column_names=["Value9"])
        for i in range(max_row_num):
            assert df2_9.iloc[i, 1] == np.int64(df2_9.iloc[i, 0] * 9)
        df2_10 = to_dataframe(tsfile_path, column_names=["Value10"])
        for i in range(max_row_num):
            assert df2_10.iloc[i, 1] in dateSet
        df2_11 = to_dataframe(tsfile_path, column_names=["Device1", "Value1"])
        for i in range(max_row_num):
            assert df2_11.iloc[i, 1] == "Device1_" + str(df2_11.iloc[i, 0])
            assert df2_11.iloc[i, 2] == np.bool_(df2_11.iloc[i, 0] % 2 == 0)
        df2_12 = to_dataframe(tsfile_path, column_names=["Device1", "Device2", "Value1", "Value2", "Value3", "Value4", "Value5", "Value6", "Value7",  "Value8", "Value9", "Value10"])
        for i in range(max_row_num):
            assert df2_12.iloc[i, 1] == "Device1_" + str(df2_12.iloc[i, 0])
            assert df2_12.iloc[i, 2] == "Device2_" + str(df2_12.iloc[i, 0])
            assert df2_12.iloc[i, 3] == np.bool_(df2_12.iloc[i, 0] % 2 == 0)
            assert df2_12.iloc[i, 4] == np.int32(df2_12.iloc[i, 0] * 3)
            assert df2_12.iloc[i, 5] == np.int64(df2_12.iloc[i, 0] * 4)
            assert df2_12.iloc[i, 6] == np.float32(df2_12.iloc[i, 0] * 5.5)
            assert df2_12.iloc[i, 7] == np.float64(df2_12.iloc[i, 0] * 6.6)
            assert df2_12.iloc[i, 8] == f"string_value_{df2_12.iloc[i, 0]}"
            assert df2_12.iloc[i, 9] == f"text_value_{df2_12.iloc[i, 0]}"
            assert df2_12.iloc[i, 10] == f"blob_data_{df2_12.iloc[i, 0]}".encode('utf-8')
            assert df2_12.iloc[i, 11] == np.int64(df2_12.iloc[i, 0] * 9)
            assert df2_12.iloc[i, 12] in dateSet
        df2_13 = to_dataframe(tsfile_path, column_names=["Device1", "Device2", "Value1"]) # TODO：目前未自动转小写
        for i in range(max_row_num):
            assert df2_13.iloc[i, 1] == "Device1_" + str(df2_13.iloc[i, 0])
            assert df2_13.iloc[i, 2] == "Device2_" + str(df2_13.iloc[i, 0])
            assert df2_13.iloc[i, 3] == np.bool_(df2_13.iloc[i, 0] % 2 == 0)

        # 3. 指定表名
        df3_1 = to_dataframe(tsfile_path, table_name="test_table")
        assert df3_1.shape[0] == max_row_num
        assert df3_1.iloc[0, 0] == 0
        df3_2 = to_dataframe(tsfile_path, table_name="TEST_TABLE") # TODO：目前未自动转小写
        assert df3_2.shape[0] == max_row_num
        assert df3_2.iloc[0, 0] == 0

        # 4. 指定时间段
        df4_1 = to_dataframe(tsfile_path, start_time=10)
        assert df4_1.shape[0] == 90
        df4_2 = to_dataframe(tsfile_path, start_time=-10)
        assert df4_2.shape[0] == max_row_num
        df4_3 = to_dataframe(tsfile_path, end_time=5)
        assert df4_3.shape[0] == 6
        df4_4 = to_dataframe(tsfile_path, end_time=-5)
        assert df4_4.shape[0] == 0
        df4_5 = to_dataframe(tsfile_path, start_time=5, end_time=5)
        assert df4_5.shape[0] == 1
        df4_6 = to_dataframe(tsfile_path, start_time=-5, end_time=-5)
        assert df4_6.shape[0] == 0
        df4_7 = to_dataframe(tsfile_path, start_time=10, end_time=-10)
        assert df4_7.shape[0] == 0
        df4_8 = to_dataframe(tsfile_path, start_time=-10, end_time=10)
        assert df4_8.shape[0] == 11
        df4_8 = to_dataframe(tsfile_path, start_time=-50, end_time=50)
        assert df4_8.shape[0] == 51

        # 5. 指定最大行数
        df5_1 = to_dataframe(tsfile_path, max_row_num=1)
        assert df5_1.shape[0] == 1
        df5_2 = to_dataframe(tsfile_path, max_row_num=50)
        assert df5_2.shape[0] == 50
        df5_3 = to_dataframe(tsfile_path, max_row_num=100)
        assert df5_3.shape[0] == 100
        df5_4 = to_dataframe(tsfile_path, max_row_num=1000)
        assert df5_4.shape[0] == 100
        df5_5 = to_dataframe(tsfile_path, max_row_num=0)
        assert df5_5.shape[0] == 0
        df5_6 = to_dataframe(tsfile_path, max_row_num=-10)
        assert df5_6.shape[0] == 0
        # 6. 迭代式
        for df6_1 in to_dataframe(tsfile_path, max_row_num=20, as_iterator=True):
            assert df6_1.shape[0] == 20
        for df6_2 in to_dataframe(tsfile_path, max_row_num=1000, as_iterator=True):
            assert df6_2.shape[0] == 100
        # 7. 全量参数
        for df7_1 in to_dataframe(tsfile_path, table_name="Test_Table", column_names=["Device1", "Value1"],
                                  start_time=21, end_time=50, max_row_num=10, as_iterator=True):
            assert df7_1.shape[0] == 10
            for i in range(30):
                assert df2_11.iloc[i, 1] == "Device1_" + str(df2_11.iloc[i, 0])
                assert df2_11.iloc[i, 2] == np.bool_(df2_11.iloc[i, 0] % 2 == 0)

        # 8. 获取表模型 TsFile 指定不存在的表
        try:
            to_dataframe(tsfile_path, table_name="non_existent_table")
            assert False, "获取表模型 TsFile 获取不存在的表未抛出异常"
        except TableNotExistError as e:
            assert e.args[0] == "[non_existent_table] Requested table does not exist"

        # # 9.获取路径不存在的异常 TODO：目前未处理
        # try:
        #     to_dataframe("non_existent_path.tsfile")
        #     assert False, "获取表模型 TsFile 获取不存在的路径未抛出异常"
        # except Exception as e:
        #     print(f"路径不存在的异常: {e}")

        # 10. 获取表模型 TsFile 获取不存在的列的异常
        try:
            to_dataframe(tsfile_path, column_names=["non_existent_column"])
            assert False, "获取表模型 TsFile 获取不存在的列未抛出异常"
        except ColumnNotExistError as e:
            assert e.args[0] == "[non_existent_column] Column does not exist"

    finally:
        if os.path.exists(tsfile_path):
            os.remove(tsfile_path)