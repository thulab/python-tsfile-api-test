import os

from tsfile import ColumnSchema, TableSchema
from tsfile import Tablet
from tsfile import TsFileTableWriter, TsFileReader, TSDataType, ColumnCategory

##  Write
table_data_dir = os.path.join(os.path.dirname(__file__), "../data/tsfile/table_data.tsfile")
if os.path.exists(table_data_dir):
    os.remove(table_data_dir)

column_name_list = ["tag1", "f1"]
type_list = [TSDataType.STRING, TSDataType.DOUBLE]
column1 = ColumnSchema("tag1", TSDataType.STRING, ColumnCategory.TAG)
column2 = ColumnSchema("f1", TSDataType.DOUBLE, ColumnCategory.FIELD)
columns = [column1, column2]
# for i in range(2):
#     column_name_list.append("tag" + str(i))
#     type_list.append(TSDataType.STRING)
#     columns.append(ColumnSchema("tag" + str(i), TSDataType.STRING, ColumnCategory.TAG))
# for i in range(10):
#     column_name_list.append("f" + str(i))
#     type_list.append(TSDataType.DOUBLE)
#     columns.append(ColumnSchema("f" + str(i), TSDataType.DOUBLE, ColumnCategory.FIELD))

table_schema = TableSchema("test_table", columns)

### Free resource automatically
with TsFileTableWriter(table_data_dir, table_schema) as writer:
    tablet_row_num = 10
    tablet = Tablet(
        column_name_list,
        type_list,
        tablet_row_num)

    for row_num in range(tablet_row_num):
        tablet.add_timestamp(row_num, row_num)
        for i in range(len(column_name_list)):
            tablet.add_value_by_name("tag1", row_num, "tag1")
            tablet.add_value_by_name("f1", row_num, 1.1)
            # if type_list[i] == TSDataType.STRING:
            #     tablet.add_value_by_name(column_name_list[i], row_num, "tag1")
            # elif type_list[i] == TSDataType.DOUBLE:
            #     tablet.add_value_by_name(column_name_list[i], row_num, 1.1)

    writer.write_table(tablet)

##  Read

### Free resource automatically
with TsFileReader(table_data_dir) as reader:
    print(reader.get_all_table_schemas())
    with reader.query_table("test_table", column_name_list, 0, 10) as result:
        print(result.get_metadata().get_data_type(2))
        print(result.get_metadata().get_column_name(2))
        print(result.get_metadata().get_column_num())
        while result.next():
            print(result.get_value_by_name("tag1"))
            print(result.get_value_by_name("f1"))
            print(result.read_data_frame())
