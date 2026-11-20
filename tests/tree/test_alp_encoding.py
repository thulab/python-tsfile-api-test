"""TsFile-168 cases 14 and 16: tree-model ALP round trip and special values."""

import math
import struct

import pytest

from tsfile import (
    Compressor,
    Field,
    RowRecord,
    TimeseriesSchema,
    TSDataType,
    TSEncoding,
    TsFileReader,
    TsFileWriter,
)


def f32(bits):
    """Build a Python float from the exact float32 bit pattern."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value):
    return struct.pack("<f", value)


def f64_bits(value):
    return struct.pack("<d", value)


FLOAT_NAN = float("nan")
FLOAT_NEG_NAN = -float("nan")
FLOAT_INF = float("inf")
FLOAT_NEG_INF = float("-inf")
FLOAT_MAX = f32(0x7F7FFFFF)
FLOAT_MIN_NORMAL = f32(0x00800000)
FLOAT_DENORM_MIN = f32(0x00000001)

DOUBLE_NAN = float("nan")
DOUBLE_NEG_NAN = -float("nan")
DOUBLE_INF = float("inf")
DOUBLE_NEG_INF = float("-inf")
DOUBLE_MAX = struct.unpack("<d", struct.pack("<Q", 0x7FEFFFFFFFFFFFFF))[0]
DOUBLE_MIN_NORMAL = struct.unpack("<d", struct.pack("<Q", 0x0010000000000000))[0]
DOUBLE_DENORM_MIN = struct.unpack("<d", struct.pack("<Q", 0x0000000000000001))[0]


def tree_float_value(index):
    # The Python FLOAT Field validation rejects NaN/Inf (see
    # test_special_values_and_enum); the reachable float specials stay here.
    m = index % 64
    if m == 1:
        return FLOAT_MAX
    if m == 2:
        return -FLOAT_MAX
    if m == 3:
        return FLOAT_DENORM_MIN
    if m == 4:
        return -FLOAT_DENORM_MIN
    return float(m - 32) * 0.25


def tree_double_value(index):
    m = index % 64
    if m == 1:
        return DOUBLE_MAX
    if m == 2:
        return -DOUBLE_MAX
    if m == 3:
        return DOUBLE_DENORM_MIN
    if m == 4:
        return -DOUBLE_DENORM_MIN
    if m == 5:
        return DOUBLE_NAN
    if m == 6:
        return DOUBLE_INF
    if m == 7:
        return DOUBLE_NEG_INF
    return float(m - 32) * 0.5


def measurement_schema(name, data_type):
    return TimeseriesSchema(
        name, data_type, TSEncoding.ALP, Compressor.UNCOMPRESSED
    )


def test_tree_large_scale(tmp_path):
    """Case 14: 10 devices x 20 measurements x 10000 rows, bit-exact read-back."""
    devices = ["root.alp_py.dev%02d" % d for d in range(10)]
    float_names = ["f_%02d" % m for m in range(10)]
    double_names = ["d_%02d" % m for m in range(10)]
    rows = 10000
    path = str(tmp_path / "ts168_tree_large.tsfile")

    writer = TsFileWriter(path)
    try:
        for device in devices:
            for name in float_names:
                writer.register_timeseries(
                    device, measurement_schema(name, TSDataType.FLOAT)
                )
            for name in double_names:
                writer.register_timeseries(
                    device, measurement_schema(name, TSDataType.DOUBLE)
                )
        for device in devices:
            for r in range(rows):
                fields = []
                for m, name in enumerate(float_names):
                    value = tree_float_value(r * 20 + m)
                    fields.append(Field(name, value, TSDataType.FLOAT))
                for m, name in enumerate(double_names):
                    value = tree_double_value(r * 20 + m + 10)
                    fields.append(Field(name, value, TSDataType.DOUBLE))
                writer.write_row_record(RowRecord(device, r, fields))
    finally:
        writer.close()

    mismatches = 0
    first_mismatch = None
    reader = TsFileReader(path)
    try:
        for device in devices:
            result = reader.query_timeseries(
                device, float_names + double_names, 0, rows - 1
            )
            try:
                row = 0
                while result.next():
                    for m, name in enumerate(float_names):
                        expected = tree_float_value(row * 20 + m)
                        actual = result.get_value_by_name(name)
                        if f32_bits(actual) != f32_bits(expected):
                            mismatches += 1
                            if first_mismatch is None:
                                first_mismatch = (
                                    "%s row %d %s expected %r actual %r"
                                    % (device, row, name, expected, actual)
                                )
                    for m, name in enumerate(double_names):
                        expected = tree_double_value(row * 20 + m + 10)
                        actual = result.get_value_by_name(name)
                        if f64_bits(actual) != f64_bits(expected):
                            mismatches += 1
                            if first_mismatch is None:
                                first_mismatch = (
                                    "%s row %d %s expected %r actual %r"
                                    % (device, row, name, expected, actual)
                                )
                    row += 1
                assert row == rows, "%s rows=%d expected %d" % (
                    device,
                    row,
                    rows,
                )
            finally:
                result.close()
    finally:
        reader.close()

    assert mismatches == 0, (
        "values=%d mismatches=%d first=%s"
        % (len(devices) * rows * 20, mismatches, first_mismatch)
    )


def test_special_values_and_enum(tmp_path):
    """Case 16: TSEncoding.ALP == 15 and reachable special values are bit-exact.

    The Python FLOAT Field path validates ``min_float32 <= value <= max_float32``
    and therefore rejects NaN and +/-Inf at write time (OverflowError); the
    DOUBLE path has no such check.  The rejection is asserted here and reported
    as an observation; FLOAT NaN/Inf coverage stays on the C++ side (case 7).
    """
    assert int(TSEncoding.ALP) == 15

    # FLOAT NaN/Inf are not writable through the Python public API.
    reject_device = "root.alp_py.special_reject"
    reject_path = str(tmp_path / "ts168_tree_special_reject.tsfile")
    reject_writer = TsFileWriter(reject_path)
    try:
        reject_writer.register_timeseries(
            reject_device, measurement_schema("f", TSDataType.FLOAT)
        )
        for bad_value in (FLOAT_NAN, FLOAT_INF, FLOAT_NEG_INF):
            with pytest.raises(OverflowError):
                reject_writer.write_row_record(
                    RowRecord(
                        reject_device,
                        0,
                        [Field("f", bad_value, TSDataType.FLOAT)],
                    )
                )
    finally:
        reject_writer.close()

    device = "root.alp_py.special"
    path = str(tmp_path / "ts168_tree_special.tsfile")
    float_values = [
        0.0,
        -0.0,
        FLOAT_DENORM_MIN,
        -FLOAT_DENORM_MIN,
        FLOAT_MIN_NORMAL,
        -FLOAT_MIN_NORMAL,
        FLOAT_MAX,
        -FLOAT_MAX,
        f32(0x7F7FFFF0),
        f32(0x80000010),
    ]
    double_values = [
        DOUBLE_NAN,
        DOUBLE_NEG_NAN,
        DOUBLE_INF,
        DOUBLE_NEG_INF,
        0.0,
        -0.0,
        DOUBLE_DENORM_MIN,
        -DOUBLE_DENORM_MIN,
        DOUBLE_MIN_NORMAL,
        -DOUBLE_MIN_NORMAL,
        DOUBLE_MAX,
        -DOUBLE_MAX,
        struct.unpack("<d", struct.pack("<Q", 0x7FF8123456789ABC))[0],
        struct.unpack("<d", struct.pack("<Q", 0xFFF8123456789ABC))[0],
    ]
    for i in range(50):
        float_values.append(float(i) * 0.25 - 6.25)
        double_values.append(float(i) * 0.5 - 12.5)

    writer = TsFileWriter(path)
    try:
        writer.register_timeseries(
            device, measurement_schema("f", TSDataType.FLOAT)
        )
        writer.register_timeseries(
            device, measurement_schema("d", TSDataType.DOUBLE)
        )
        for r in range(len(float_values)):
            writer.write_row_record(
                RowRecord(
                    device,
                    r,
                    [
                        Field("f", float_values[r], TSDataType.FLOAT),
                        Field("d", double_values[r], TSDataType.DOUBLE),
                    ],
                )
            )
    finally:
        writer.close()

    mismatches = []
    reader = TsFileReader(path)
    try:
        result = reader.query_timeseries(device, ["f", "d"], 0, len(float_values) - 1)
        try:
            row = 0
            while result.next():
                actual_f = result.get_value_by_name("f")
                actual_d = result.get_value_by_name("d")
                if f32_bits(actual_f) != f32_bits(float_values[row]):
                    mismatches.append(
                        "float row %d expected %s actual %s"
                        % (
                            row,
                            f32_bits(float_values[row]).hex(),
                            f32_bits(actual_f).hex(),
                        )
                    )
                if f64_bits(actual_d) != f64_bits(double_values[row]):
                    mismatches.append(
                        "double row %d expected %s actual %s"
                        % (
                            row,
                            f64_bits(double_values[row]).hex(),
                            f64_bits(actual_d).hex(),
                        )
                    )
                row += 1
            assert row == len(float_values), "rows=%d expected %d" % (
                row,
                len(float_values),
            )
        finally:
            result.close()
    finally:
        reader.close()

    assert not mismatches, "mismatches: %s" % "; ".join(mismatches[:5])
