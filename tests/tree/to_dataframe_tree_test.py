import os
from datetime import date

import numpy as np
import pandas as pd
from tsfile import to_dataframe, TimeseriesSchema, TSDataType, RowRecord, Field, ColumnNotExistError, \
    TableNotExistError, TableSchema, ColumnSchema, TsFileTableWriter, Tablet
from tsfile import TsFileWriter, TsFileReader, ColumnCategory

"""
标题：树模型 to_dataframe 接口功能测试
作者：肖林捷
日期：2025/12
"""


# tsfile文件路径
# tsfile_path = "../../data/tsfile/table_data.tsfile"
# tsfile_path = "C:\\IoTDBProjects\\iotdb-all\\timechodb-2.0.6-SNAPSHOT-bin-all\\timechodb-2.0.6.4-bin-rc2-alone-ainode\\data\\datanode\\data\\sequence\\root.test.g_0\\13\\0\\1764744753824-1-0-0.tsfile"

def test_tree_all_datatype_query_to_dataframe_variants1():
    tsfile_path = "record_write_and_read.tsfile"
    try:
        if os.path.exists(tsfile_path):
            os.remove(tsfile_path)
        writer = TsFileWriter(tsfile_path)
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL1", TSDataType.INT64))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL2", TSDataType.DOUBLE))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL3", TSDataType.INT32))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL4", TSDataType.STRING))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL5", TSDataType.TEXT))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL6", TSDataType.BLOB))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL7", TSDataType.DATE))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL8", TSDataType.TIMESTAMP))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL9", TSDataType.BOOLEAN))
        writer.register_timeseries("root.Device1", TimeseriesSchema("LeveL10", TSDataType.FLOAT))

        max_row_num = 100

        for i in range(max_row_num):
            row = RowRecord("root.Device1", i - int(max_row_num / 2),
                            [Field("LeveL1", i * 1, TSDataType.INT64),
                             Field("LeveL2", i * 2.2, TSDataType.DOUBLE),
                             Field("LeveL3", i * 3, TSDataType.INT32),
                             Field("LeveL4", f"string_value_{i}", TSDataType.STRING),
                             Field("LeveL5", f"text_value_{i}", TSDataType.TEXT),
                             Field("LeveL6", f"blob_data_{i}".encode('utf-8'), TSDataType.BLOB),
                             Field("LeveL7", date(2025, 1, i % 20 + 1), TSDataType.DATE),
                             Field("LeveL8", i * 8, TSDataType.TIMESTAMP),
                             Field("LeveL9", i % 2 == 0, TSDataType.BOOLEAN),
                             Field("LeveL10", i * 10.1, TSDataType.FLOAT)])
            writer.write_row_record(row)

        writer.close()

        # 1. 默认情况
        df1_1 = to_dataframe(tsfile_path)
        assert df1_1.shape[0] == max_row_num
        for i in range(max_row_num):
            assert df1_1.iloc[i, 0] == i - int(max_row_num / 2)
            assert df1_1.iloc[i, 1] == "root"
            assert df1_1.iloc[i, 2] == "Device1"

        # 2. 指定测点
        df2_1 = to_dataframe(tsfile_path, column_names=["LeveL1"])
        for i in range(max_row_num):
            assert df2_1.iloc[i, 3] == np.int64(i * 1)
        df2_2 = to_dataframe(tsfile_path, column_names=["LeveL2"])
        for i in range(max_row_num):
            assert df2_2.iloc[i, 3] == np.float64(i * 2.2)
        df2_3 = to_dataframe(tsfile_path, column_names=["LeveL3"])
        for i in range(max_row_num):
            assert df2_3.iloc[i, 3] == np.int32(i * 3)
        df2_4 = to_dataframe(tsfile_path, column_names=["LeveL4"])
        for i in range(max_row_num):
            assert df2_4.iloc[i, 3] == f"string_value_{i}"
        df2_5 = to_dataframe(tsfile_path, column_names=["LeveL5"])
        for i in range(max_row_num):
            assert df2_5.iloc[i, 3] == f"text_value_{i}"
        df2_6 = to_dataframe(tsfile_path, column_names=["LeveL6"])
        for i in range(max_row_num):
            assert df2_6.iloc[i, 3] == f"blob_data_{i}".encode('utf-8')
        df2_7 = to_dataframe(tsfile_path, column_names=["LeveL7"])
        for i in range(max_row_num):
            assert df2_7.iloc[i, 3] == (date(2025, 1, i % 20 + 1))
        df2_8 = to_dataframe(tsfile_path, column_names=["LeveL8"])
        for i in range(max_row_num):
            assert df2_8.iloc[i, 3] == np.int64(i * 8)
        df2_9 = to_dataframe(tsfile_path, column_names=["LeveL9"])
        for i in range(max_row_num):
            assert df2_9.iloc[i, 3] == (i % 2 == 0)
        df2_10 = to_dataframe(tsfile_path, column_names=["LeveL10"])
        for i in range(max_row_num):
            assert df2_10.iloc[i, 3] == np.float32(i * 10.1)
        df2_11 = to_dataframe(tsfile_path, column_names=["LeveL1", "LeveL2", "LeveL3", "LeveL4", "LeveL5", "LeveL6", "LeveL7", "LeveL8", "LeveL9", "LeveL10"])
        for i in range(max_row_num):
            assert df2_11.iloc[i, 3] == np.int64(i * 1)
            assert df2_11.iloc[i, 4] == np.float64(i * 2.2)
            assert df2_11.iloc[i, 5] == np.int32(i * 3)
            assert df2_11.iloc[i, 6] == f"string_value_{i}"
            assert df2_11.iloc[i, 7] == f"text_value_{i}"
            assert df2_11.iloc[i, 8] == f"blob_data_{i}".encode('utf-8')
            assert df2_11.iloc[i, 9] == date(2025, 1, i % 20 + 1)
            assert df2_11.iloc[i, 10] == np.int64(i * 8)
            assert df2_11.iloc[i, 11] == (i % 2 == 0)
            assert df2_11.iloc[i, 12] == np.float32(i * 10.1)

        # 3. 指定时间
        df3_1 = to_dataframe(tsfile_path, start_time=10)
        assert df3_1.shape[0] == 40
        df3_2 = to_dataframe(tsfile_path, start_time=-10)
        assert df3_2.shape[0] == 60
        df3_3 = to_dataframe(tsfile_path, end_time=5)
        assert df3_3.shape[0] == 56
        df3_4 = to_dataframe(tsfile_path, end_time=-5)
        assert df3_4.shape[0] == 46
        df3_5 = to_dataframe(tsfile_path, start_time=5, end_time=5)
        assert df3_5.shape[0] == 1
        df3_6 = to_dataframe(tsfile_path, start_time=-5, end_time=-5)
        assert df3_6.shape[0] == 1
        df3_7 = to_dataframe(tsfile_path, start_time=10, end_time=-10)
        assert df3_7.shape[0] == 0
        df3_8 = to_dataframe(tsfile_path, start_time=-10, end_time=10)
        assert df3_8.shape[0] == 21
        df3_8 = to_dataframe(tsfile_path, start_time=-50, end_time=50)
        assert df3_8.shape[0] == max_row_num

        # 4. 指定最大行数
        df4_1 = to_dataframe(tsfile_path, max_row_num=1)
        assert df4_1.shape[0] == 1
        df4_2 = to_dataframe(tsfile_path, max_row_num=10)
        assert df4_2.shape[0] == 10
        df4_3 = to_dataframe(tsfile_path, max_row_num=100)
        assert df4_3.shape[0] == 100
        df4_4 = to_dataframe(tsfile_path, max_row_num=1000)
        assert df4_4.shape[0] == 100
        df4_5 = to_dataframe(tsfile_path, max_row_num=0)
        assert df4_5.shape[0] == 0
        df4_6 = to_dataframe(tsfile_path, max_row_num=-10)
        assert df4_6.shape[0] == 0

        # 5. 获取树模型 TsFile 迭代式（需要配合max_row_num使用，否则直接输出全部）
        for df5_1 in to_dataframe(tsfile_path, max_row_num=10, as_iterator=True):
            assert df5_1.shape[0] == 10
        for df5_2 in to_dataframe(tsfile_path, max_row_num=-10, as_iterator=True):
            assert df5_2.shape[0] == 1
        for df5_3 in to_dataframe(tsfile_path, max_row_num=1000, as_iterator=True):
            assert df5_3.shape[0] == 100
        for df5_4 in to_dataframe(tsfile_path, max_row_num=3, as_iterator=True):
            if df5_4.iloc[0, 0] <= 48:
                assert df5_4.shape[0] == 3
            else:
                assert df5_4.shape[0] == 1

        # 6. 全部的参数
        row_num = 0
        for df6_1 in to_dataframe(tsfile_path, column_names=["LeveL1", "LeveL2"], start_time=-50, end_time=10,
                                  max_row_num=1, as_iterator=True):
            assert df6_1.shape[0] == 1
            assert df6_1.iloc[0, 0] == -50 + row_num
            assert df6_1.iloc[0, 3] == row_num
            assert df6_1.iloc[0, 4] == row_num * 2.2
            row_num += 1

        # 7. 树模型 TsFile 指定表
        df7_1 = to_dataframe(tsfile_path, table_name="test")
        assert df7_1.shape[0] == max_row_num
        assert df7_1.iloc[0, 0] == -int(max_row_num / 2)

        # 8. 获取树模型 TsFile 获取不存在的测点的异常
        try:
            to_dataframe(tsfile_path, column_names=["non_existent_column"])
            assert False, "获取不存在的列名异常未抛出"
        except ColumnNotExistError:
            pass

        # # 9.获取路径不存在的异常 TODO：暂未实现
        # try:
        #     to_dataframe("non_existent_path.tsfile")
        #     assert False, "路径不存在的异常未抛出"
        # except Exception as e:
        #     print(f"路径不存在的异常: {e}")

    finally:
        if os.path.exists(tsfile_path):
            os.remove(tsfile_path)