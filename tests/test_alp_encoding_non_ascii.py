"""TsFile-168 case 31: non-ASCII names round-trip with ALP."""

import struct

import numpy as np
import pandas as pd

from tsfile import (
    ColumnCategory,
    ColumnSchema,
    Compressor,
    Field,
    RowRecord,
    TableSchema,
    TimeseriesSchema,
    TSDataType,
    TSEncoding,
    TsFileReader,
    TsFileTableWriter,
    TsFileWriter,
)

ROWS = 1000


def f32_bits(value):
    return struct.pack("<f", value)


def f64_bits(value):
    return struct.pack("<d", value)


def test_non_ascii_names(tmp_path):
    device = "root.设备_α_温度"
    tree_float = "温度_01"
    tree_double = "湿度_β"
    tree_path = str(tmp_path / "ts168_non_ascii_tree.tsfile")

    writer = TsFileWriter(tree_path)
    try:
        writer.register_timeseries(
            device,
            TimeseriesSchema(
                tree_float,
                TSDataType.FLOAT,
                TSEncoding.ALP,
                Compressor.UNCOMPRESSED,
            ),
        )
        writer.register_timeseries(
            device,
            TimeseriesSchema(
                tree_double,
                TSDataType.DOUBLE,
                TSEncoding.ALP,
                Compressor.UNCOMPRESSED,
            ),
        )
        for r in range(ROWS):
            writer.write_row_record(
                RowRecord(
                    device,
                    r,
                    [
                        Field(
                            tree_float,
                            struct.unpack(
                                "<f",
                                struct.pack("<f", (r % 1000) * 0.25 - 125.0),
                            )[0],
                            TSDataType.FLOAT,
                        ),
                        Field(
                            tree_double,
                            (r % 2000) * 0.5 - 500.0,
                            TSDataType.DOUBLE,
                        ),
                    ],
                )
            )
    finally:
        writer.close()

    tree_mismatches = []
    reader = TsFileReader(tree_path)
    try:
        result = reader.query_timeseries(
            device, [tree_float, tree_double], 0, ROWS - 1
        )
        try:
            metadata = result.get_metadata()
            assert metadata.get_column_num() == 3, "tree metadata column count"
            assert metadata.get_column_name(1) == "time"
            assert metadata.get_column_name(2) == device + "." + tree_float
            assert metadata.get_column_name(3) == device + "." + tree_double
            row = 0
            while result.next():
                expected_f = (row % 1000) * 0.25 - 125.0
                expected_d = (row % 2000) * 0.5 - 500.0
                actual_f = result.get_value_by_name(tree_float)
                actual_d = result.get_value_by_name(tree_double)
                if f32_bits(actual_f) != f32_bits(expected_f):
                    tree_mismatches.append("float row %d" % row)
                if f64_bits(actual_d) != f64_bits(expected_d):
                    tree_mismatches.append("double row %d" % row)
                row += 1
            assert row == ROWS, "tree rows=%d expected %d" % (row, ROWS)
        finally:
            result.close()
    finally:
        reader.close()
    assert not tree_mismatches, "tree mismatches: %s" % tree_mismatches[:5]

    # Table model with non-ASCII table, tag and field names.
    table = "表_α_传感器"
    tag_name = "设备编号"
    float_name = "测量值_温度"
    double_name = "测量值_湿度"
    table_path = str(tmp_path / "ts168_non_ascii_table.tsfile")
    schema = TableSchema(
        table,
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema(tag_name, TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema(float_name, TSDataType.FLOAT, ColumnCategory.FIELD),
            ColumnSchema(double_name, TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    rows = np.arange(ROWS)
    float_column = ((rows % 1000) * 0.25 - 125.0).astype(np.float32)
    double_column = (rows % 2000) * 0.5 - 500.0
    null_mask = (rows % 11) == 0
    float_column[null_mask] = np.nan
    double_column[null_mask] = np.nan
    dataframe = pd.DataFrame(
        {
            "time": rows.astype(np.int64),
            tag_name: ["设备A"] * ROWS,
            float_name: float_column,
            double_name: double_column,
        }
    )
    with TsFileTableWriter(table_path, schema) as writer:
        writer.write_dataframe(dataframe)

    table_mismatches = []
    reader = TsFileReader(table_path)
    try:
        result = reader.query_table(
            table, [tag_name, float_name, double_name], 0, ROWS - 1
        )
        try:
            metadata = result.get_metadata()
            assert metadata.get_column_num() == 4, "table metadata column count"
            assert metadata.get_column_name(1) == "time"
            assert metadata.get_column_name(2) == tag_name
            assert metadata.get_column_name(3) == float_name
            assert metadata.get_column_name(4) == double_name
            row = 0
            while result.next():
                if result.get_value_by_name(tag_name) != "设备A":
                    table_mismatches.append("tag row %d" % row)
                if row % 11 == 0:
                    if not result.is_null_by_name(float_name):
                        table_mismatches.append("float null row %d" % row)
                    if not result.is_null_by_name(double_name):
                        table_mismatches.append("double null row %d" % row)
                else:
                    expected_f = (row % 1000) * 0.25 - 125.0
                    expected_d = (row % 2000) * 0.5 - 500.0
                    if f32_bits(result.get_value_by_name(float_name)) != f32_bits(
                        expected_f
                    ):
                        table_mismatches.append("float row %d" % row)
                    if f64_bits(result.get_value_by_name(double_name)) != f64_bits(
                        expected_d
                    ):
                        table_mismatches.append("double row %d" % row)
                row += 1
            assert row == ROWS, "table rows=%d expected %d" % (row, ROWS)
        finally:
            result.close()
    finally:
        reader.close()
    assert not table_mismatches, "table mismatches: %s" % table_mismatches[:5]
