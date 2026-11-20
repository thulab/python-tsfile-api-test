"""TsFile-168 case 15: wide table ALP round trip through the global config.

The Python table model has no per-column encoding argument, so ALP is selected
through set_tsfile_config.  The Python writer supports one table per file, so
the two 1010-column tables are written to two files (documented in the case
record); the schema shape and scale stay as specified.
"""

import os

import numpy as np
import pandas as pd

from tsfile import (
    ColumnCategory,
    ColumnSchema,
    TableSchema,
    TSDataType,
    TSEncoding,
    TsFileReader,
    TsFileTableWriter,
    get_tsfile_config,
    set_tsfile_config,
    to_dataframe,
)

ROWS = 10000
TAGS = 10
FLOAT_FIELDS = 500
DOUBLE_FIELDS = 500

TAG_NAMES = ["tag_%02d" % i for i in range(TAGS)]
FLOAT_NAMES = ["f_%03d" % i for i in range(FLOAT_FIELDS)]
DOUBLE_NAMES = ["d_%03d" % i for i in range(DOUBLE_FIELDS)]


def build_table_schema(table_name):
    schemas = [ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME)]
    for name in TAG_NAMES:
        schemas.append(ColumnSchema(name, TSDataType.STRING, ColumnCategory.TAG))
    for name in FLOAT_NAMES:
        schemas.append(ColumnSchema(name, TSDataType.FLOAT, ColumnCategory.FIELD))
    for name in DOUBLE_NAMES:
        schemas.append(ColumnSchema(name, TSDataType.DOUBLE, ColumnCategory.FIELD))
    return TableSchema(table_name, schemas)


def build_dataframe(table_index):
    data = {"time": np.arange(ROWS, dtype=np.int64)}
    for c, name in enumerate(TAG_NAMES):
        data[name] = ["g%02d_%d" % (c, table_index)] * ROWS
    rows = np.arange(ROWS)
    for c, name in enumerate(FLOAT_NAMES):
        column = ((rows + c) % 2000) * 0.25 - 250.0
        column = column.astype(np.float32)
        column[::7] = np.nan
        data[name] = column
    for c, name in enumerate(DOUBLE_NAMES):
        column = ((rows * 3 + c) % 4000) * 0.5 - 1000.0
        column = column.astype(np.float64)
        column[::7] = np.nan
        data[name] = column
    return pd.DataFrame(data)


def expected_float(column_index):
    rows = np.arange(ROWS)
    column = ((rows + column_index) % 2000) * 0.25 - 250.0
    column = column.astype(np.float32)
    column[::7] = np.nan
    return column


def expected_double(column_index):
    rows = np.arange(ROWS)
    column = ((rows * 3 + column_index) % 4000) * 0.5 - 1000.0
    column[::7] = np.nan
    return column


def test_table_wide_large_scale(tmp_path):
    """Case 15: 2 tables x (10 tags + 1000 fields) x 10000 rows with NULLs."""
    table_names = ["t168_wide_a", "t168_wide_b"]
    paths = [str(tmp_path / (name + ".tsfile")) for name in table_names]
    base_config = dict(get_tsfile_config())
    set_tsfile_config(
        {
            "float_encoding_type_": TSEncoding.ALP,
            "double_encoding_type_": TSEncoding.ALP,
        }
    )
    try:
        assert get_tsfile_config()["float_encoding_type_"] == TSEncoding.ALP
        assert get_tsfile_config()["double_encoding_type_"] == TSEncoding.ALP
        for index, (table_name, path) in enumerate(zip(table_names, paths)):
            dataframe = build_dataframe(index)
            schema = build_table_schema(table_name)
            with TsFileTableWriter(path, schema) as writer:
                writer.write_dataframe(dataframe)
            del dataframe
    finally:
        set_tsfile_config(
            {
                "float_encoding_type_": base_config["float_encoding_type_"],
                "double_encoding_type_": base_config["double_encoding_type_"],
            }
        )

    assert get_tsfile_config()["float_encoding_type_"] == base_config[
        "float_encoding_type_"
    ]
    assert get_tsfile_config()["double_encoding_type_"] == base_config[
        "double_encoding_type_"
    ]

    total_values = 0
    for index, (table_name, path) in enumerate(zip(table_names, paths)):
        assert os.path.getsize(path) > 0
        df = to_dataframe(
            path,
            table_name,
            column_names=TAG_NAMES + FLOAT_NAMES + DOUBLE_NAMES,
        )
        try:
            assert len(df) == ROWS, "%s rows=%d expected %d" % (
                table_name,
                len(df),
                ROWS,
            )
            for c, name in enumerate(TAG_NAMES):
                expected_tag = "g%02d_%d" % (c, index)
                assert (df[name] == expected_tag).all(), "%s %s tag" % (
                    table_name,
                    name,
                )
            for c, name in enumerate(FLOAT_NAMES):
                actual = df[name].to_numpy(dtype=np.float32)
                expected = expected_float(c)
                expected_nan = np.isnan(expected)
                assert np.array_equal(np.isnan(actual), expected_nan), (
                    "%s %s null positions" % (table_name, name)
                )
                valid = ~expected_nan
                assert np.array_equal(
                    actual[valid].view(np.uint32), expected[valid].view(np.uint32)
                ), "%s %s values" % (table_name, name)
                total_values += int(valid.sum())
            for c, name in enumerate(DOUBLE_NAMES):
                actual = df[name].to_numpy(dtype=np.float64)
                expected = expected_double(c)
                expected_nan = np.isnan(expected)
                assert np.array_equal(np.isnan(actual), expected_nan), (
                    "%s %s null positions" % (table_name, name)
                )
                valid = ~expected_nan
                assert np.array_equal(
                    actual[valid].view(np.uint64), expected[valid].view(np.uint64)
                ), "%s %s values" % (table_name, name)
                total_values += int(valid.sum())
        finally:
            del df

    print(
        "[TSFILE168][case15] tables=%d rows=%d compared_values=%d sizes=%s"
        % (
            len(table_names),
            ROWS,
            total_values,
            [os.path.getsize(p) for p in paths],
        )
    )

    reader = TsFileReader(paths[0])
    try:
        schemas = reader.get_all_table_schemas()
        assert table_names[0] in schemas
    finally:
        reader.close()
