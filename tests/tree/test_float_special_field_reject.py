"""TsFile-168 supplemental coverage: Python FLOAT NaN/Inf at the Field layer.

Observed 2026-11-19 (problem report 2026-11-20): the tree-model Field path
rejects FLOAT NaN/+Inf/-Inf with OverflowError because
Field.get_float_value validates min_float32 <= v <= max_float32 (NaN compares
false; +/-Inf exceeds the finite range). The table-model path and the DOUBLE
path accept the same values.

Tree-model FLOAT tests assert the EXPECTED behaviour and are marked
xfail(strict=True): they xfail today and will XPASS once the binding is fixed,
which is the signal to drop the marker. Table-model and DOUBLE tests are
positive coverage.
"""

import struct

import numpy as np
import pandas as pd
import pytest

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
    get_tsfile_config,
    set_tsfile_config,
    to_dataframe,
)

DEVICE = "root.float_special.dev"
TABLE = "t_float_special"

FLOAT_ENCODINGS = [
    ("PLAIN", TSEncoding.PLAIN),
    ("RLE", TSEncoding.RLE),
    ("TS_2DIFF", TSEncoding.TS_2DIFF),
    ("GORILLA_V1", TSEncoding.GORILLA_V1),
    ("GORILLA", TSEncoding.GORILLA),
    ("CHIMP", TSEncoding.CHIMP),
    ("SPRINTZ", TSEncoding.SPRINTZ),
    ("RLBE", TSEncoding.RLBE),
    ("ALP", TSEncoding.ALP),
]

SPECIAL_VALUES = [
    ("nan", float("nan")),
    ("posinf", float("inf")),
    ("neginf", float("-inf")),
]

FLOAT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def f32_bits(value):
    return struct.pack("<f", value)


def f64_bits(value):
    return struct.pack("<d", value)


def write_tree(path, encoding, dtype, value):
    writer = TsFileWriter(str(path))
    try:
        writer.register_timeseries(
            DEVICE,
            TimeseriesSchema("v", dtype, encoding, Compressor.UNCOMPRESSED),
        )
        writer.write_row_record(RowRecord(DEVICE, 0, [Field("v", value, dtype)]))
    finally:
        writer.close()


def read_tree(path):
    reader = TsFileReader(str(path))
    try:
        result = reader.query_timeseries(DEVICE, ["v"], 0, 0)
        try:
            values = []
            while result.next():
                values.append(result.get_value_by_name("v"))
            return values
        finally:
            result.close()
    finally:
        reader.close()


@pytest.mark.parametrize("enc_name,encoding", FLOAT_ENCODINGS)
@pytest.mark.parametrize("label,value", SPECIAL_VALUES)
@pytest.mark.xfail(
    strict=True,
    reason="Python Field.get_float_value rejects FLOAT NaN/Inf; issue reported 2026-11-20",
)
def test_tree_float_special_writable(tmp_path, enc_name, encoding, label, value):
    path = tmp_path / ("tree_%s_%s.tsfile" % (enc_name, label))
    write_tree(path, encoding, TSDataType.FLOAT, value)
    values = read_tree(path)
    assert len(values) == 1
    assert f32_bits(values[0]) == f32_bits(value)


@pytest.mark.parametrize("enc_name,encoding", FLOAT_ENCODINGS)
@pytest.mark.parametrize("label,value", SPECIAL_VALUES)
def test_tree_double_special_writable(tmp_path, enc_name, encoding, label, value):
    path = tmp_path / ("tree_double_%s_%s.tsfile" % (enc_name, label))
    write_tree(path, encoding, TSDataType.DOUBLE, value)
    values = read_tree(path)
    assert len(values) == 1
    assert f64_bits(values[0]) == f64_bits(value)


@pytest.mark.parametrize("enc_name,encoding", FLOAT_ENCODINGS)
def test_table_float_special_writable(tmp_path, enc_name, encoding):
    base = dict(get_tsfile_config())
    path = tmp_path / ("table_%s.tsfile" % enc_name)
    set_tsfile_config({"float_encoding_type_": encoding})
    try:
        schema = TableSchema(
            TABLE,
            [
                ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
                ColumnSchema("v", TSDataType.FLOAT, ColumnCategory.FIELD),
            ],
        )
        data = np.array([np.nan, np.inf, -np.inf, np.float32(FLOAT_MAX)], dtype=np.float32)
        dataframe = pd.DataFrame({"time": np.arange(4, dtype=np.int64), "v": data})
        with TsFileTableWriter(str(path), schema) as writer:
            writer.write_dataframe(dataframe)
        actual = to_dataframe(str(path), TABLE, column_names=["v"])["v"].to_numpy(dtype=np.float32)
        actual = actual[:4]
        assert np.array_equal(np.isnan(actual), np.isnan(data))
        valid = ~np.isnan(data)
        assert np.array_equal(actual[valid].view(np.uint32), data[valid].view(np.uint32))
    finally:
        set_tsfile_config({"float_encoding_type_": base["float_encoding_type_"]})
