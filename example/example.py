import os

from tsfile import ColumnSchema, TableSchema
from tsfile import Tablet
from tsfile import TsFileTableWriter, TsFileReader, TSDataType, ColumnCategory

##  Write
table_data_dir = os.path.join("C:\\IoTDBProjects\\iotdb-all\\timechodb-2.0.4-SNAPSHOT-bin-all\\timechodb-2.0.4.1-bin-rc1\\data\\datanode\\data\\sequence\\test\\1\\0\\1747704088443-1-0-0.tsfile")

### Free resource automatically
with TsFileReader(table_data_dir) as reader:
    print(reader.get_all_table_schemas())
    with reader.query_table("t1", ["tag1","s1"], 0, 50) as result:
        metadata = result.get_metadata()
        # 查询列名
        for i in range(1, metadata.get_column_num() + 1):
            print(metadata.get_column_name(i), end=" ")
        print()
        # 查询列对应的数据类型
        for i in range(1, metadata.get_column_num() + 1):
            type = metadata.get_data_type(i)
            if type == TSDataType.INT32:
                print("INT32", end=" ")
            elif type == TSDataType.INT64:
                print("INT64", end=" ")
            elif type == TSDataType.FLOAT:
                print("FLOAT", end=" ")
            elif type == TSDataType.DOUBLE:
                print("DOUBLE", end=" ")
            elif type == TSDataType.STRING:
                print("STRING", end=" ")
            elif type == TSDataType.BOOLEAN:
                print("BOOLEAN", end=" ")
            else:
                print(f"UNKNOWN: {type}", end=" ")
        print()
        # 查询数据
        while result.next():
            print(result.get_value_by_name("tag1"))
            print(result.get_value_by_name("s1"))
            print(result.read_data_frame())
