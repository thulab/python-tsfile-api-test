"""TsFile-168 case 21: Python misuse paths must raise clear exceptions."""

import pytest

from tsfile import (
    Compressor,
    Field,
    NotSupportedError,
    InvalidArgumentError,
    RowRecord,
    TimeseriesSchema,
    TSDataType,
    TSEncoding,
    TsFileWriter,
    get_tsfile_config,
    set_tsfile_config,
)


def test_misuse_reports_clear_exception(tmp_path):
    base_config = dict(get_tsfile_config())
    try:
        # 1) INT64 series registered with ALP: registration is accepted, the
        #    write must fail with a readable error instead of corrupting data.
        path = str(tmp_path / "int64_alp.tsfile")
        writer = TsFileWriter(path)
        try:
            writer.register_timeseries(
                "root.alp_err",
                TimeseriesSchema(
                    "int_ts",
                    TSDataType.INT64,
                    TSEncoding.ALP,
                    Compressor.UNCOMPRESSED,
                ),
            )
            with pytest.raises(InvalidArgumentError) as excinfo:
                writer.write_row_record(
                    RowRecord(
                        "root.alp_err",
                        0,
                        [Field("int_ts", 1, TSDataType.INT64)],
                    )
                )
            assert str(excinfo.value).strip(), "empty exception message"
        finally:
            writer.close()

        # 2) An unsupported global encoding for FLOAT must be rejected and must
        #    not pollute the configuration.
        with pytest.raises(NotSupportedError) as excinfo:
            set_tsfile_config({"float_encoding_type_": TSEncoding.BITMAP})
        assert str(excinfo.value).strip(), "empty exception message"
        assert (
            get_tsfile_config()["float_encoding_type_"] != TSEncoding.BITMAP
        ), "float encoding was polluted with BITMAP after a rejected set"

        # 3) A non-TSEncoding object must be rejected with TypeError.
        for bad_value in (999, "ALP"):
            with pytest.raises(TypeError):
                set_tsfile_config({"float_encoding_type_": bad_value})
    finally:
        set_tsfile_config(
            {
                "float_encoding_type_": base_config["float_encoding_type_"],
                "double_encoding_type_": base_config["double_encoding_type_"],
            }
        )

    assert (
        get_tsfile_config()["float_encoding_type_"]
        == base_config["float_encoding_type_"]
    )
