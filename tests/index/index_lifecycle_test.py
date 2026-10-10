# -*- coding: utf-8 -*-
"""RN-142 用例 1–14、16、49：索引生命周期/失效/损坏/边界。

设计：全部在 tmp_path 内对 D1/table 的副本操作（夹具由 session fixture 现造）。
记录型用例（8 指纹边界 / 12 section 与 locator 篡改）不判产品失败，只登记实测。
"""
import glob
import hashlib
import inspect
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import fixtures as fx
from tsfile import (
    ColumnCategory,
    ColumnSchema,
    TableSchema,
    TSDataType,
    TsFileDataFrame,
    TsFileTableWriter,
)
from tsfile.dataset.index import crc32c

HEADER = struct.Struct("<8sHHIQIIQI20s")
DIRENT = struct.Struct("<IIQQII")


def work_dir(d1_root, tmp_path, name="data"):
    dst = tmp_path / name
    shutil.copytree(str(Path(d1_root) / "table"), str(dst))
    return dst


def copy_file(d1_root, name):
    return str(Path(d1_root) / "table" / name)


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def index_files(directory):
    return sorted(glob.glob(os.path.join(str(directory), ".tsfile_dataframe_index.*.tsidx")))


def other_leftovers(directory):
    names = os.listdir(str(directory))
    return sorted(name for name in names
                  if name.startswith(".tsfile_dataframe_index") and not name.endswith(".tsidx"))


def header_of(path):
    with open(path, "rb") as handle:
        raw = handle.read(64)
    (magic, major, minor, header_size, directory_offset, section_count, entry_size,
     file_length, crc, reserved) = HEADER.unpack(raw)
    return dict(magic=magic, major=major, minor=minor, header_size=header_size,
                directory_offset=directory_offset, section_count=section_count,
                entry_size=entry_size, file_length=file_length, header_crc=crc,
                reserved_zero=reserved == b"\0" * 20, actual_length=os.path.getsize(path))


def sections_of(path):
    info = header_of(path)
    result = []
    with open(path, "rb") as handle:
        handle.seek(info["directory_offset"])
        for _ in range(info["section_count"]):
            entry = DIRENT.unpack(handle.read(32))
            result.append(dict(type=entry[0], record_size=entry[1], offset=entry[2],
                               length=entry[3], count=entry[4], crc=entry[5]))
    return result


def build(data):
    return TsFileDataFrame(str(data), show_progress=False, use_index=True)


def dev_a_series(frame):
    for item in frame.list_timeseries():
        text = str(item)
        if text.startswith("weather.dev_a") and text.endswith(".temperature"):
            return text
    raise AssertionError("dev_a temperature series not found")


def rows_of(frame):
    return len(frame[dev_a_series(frame)][:])


def rows_of_any(frame):
    for item in frame.list_timeseries():
        text = str(item)
        if text.endswith(".temperature"):
            return len(frame[text][:])
    raise AssertionError("no temperature series found")


def rewrite(path, start, stop, devices):
    if os.path.exists(path):
        os.remove(path)
    fx._write_table_shard(path, start, stop, devices)


def test_case01_default_off_creates_no_index(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    default = inspect.signature(TsFileDataFrame.__init__).parameters["use_index"].default
    frame = TsFileDataFrame(str(data), show_progress=False, use_index=False)
    rows = rows_of(frame)
    frame.close()
    assert default is False
    assert index_files(data) == []
    assert other_leftovers(data) == []
    assert rows == 300, rows
    print("[case 1] pass default_use_index=%s rows=%d" % (default, rows))


def test_case02_default_off_ignores_existing_index(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    assert rows_of(frame) == 300
    frame.close()
    index = index_files(data)[0]
    before = (sha256(index), os.stat(index).st_mtime_ns)
    rewrite(os.path.join(str(data), "weather_c.tsfile"), 200, 349, {"dev_a": ("cn", 5)})
    frame = TsFileDataFrame(str(data), show_progress=False, use_index=False)
    rows = rows_of(frame)
    frame.close()
    after = (sha256(index), os.stat(index).st_mtime_ns)
    assert rows == 350, rows
    assert after == before, "use_index=False 不应触碰既有索引"
    print("[case 2] pass rows=%d index_unchanged=True" % rows)


def test_case03_cold_build_publishes_valid_index(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    frame.close()
    indexes = index_files(data)
    assert len(indexes) == 1, indexes
    info = header_of(indexes[0])
    assert info["magic"] == b"TSIDX\0\0\0"
    assert (info["major"], info["minor"]) == (1, 0)
    assert info["header_size"] == 64 and info["entry_size"] == 32
    assert info["section_count"] == 13
    assert info["file_length"] == info["actual_length"]
    assert info["reserved_zero"] is True
    leftovers = other_leftovers(data)
    assert leftovers == [os.path.basename(indexes[0]) + ".lock"], leftovers
    print("[case 3] pass index=%s bytes=%d sections=%d"
          % (os.path.basename(indexes[0]), info["actual_length"], info["section_count"]))


def test_case04_hot_construct_opens_no_data_file(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    frame.close()
    index = index_files(data)[0]
    before = (sha256(index), os.stat(index).st_mtime_ns)
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_hot_open.py")
    child = subprocess.run([sys.executable, probe, str(data)],
                           capture_output=True, text=True, check=True)
    payload = json.loads(child.stdout.strip().splitlines()[-1])
    assert payload["tsfile_fds"] == 0, payload
    assert payload["index_fds"] >= 1, payload
    assert (sha256(index), os.stat(index).st_mtime_ns) == before
    print("[case 4] pass %s" % payload)


def test_case05_index_scoped_to_expanded_file_set(d1_root, tmp_path):
    three = work_dir(d1_root, tmp_path, "three")
    two = tmp_path / "two"
    two.mkdir()
    shutil.copy(copy_file(d1_root, "weather_a.tsfile"), str(two))
    shutil.copy(copy_file(d1_root, "weather_b.tsfile"), str(two))
    frame = build(three)
    frame.close()
    index_three = index_files(three)[0]
    three_before = sha256(index_three)
    frame = build(two)
    frame.close()
    index_two = index_files(two)[0]
    assert os.path.basename(index_three) != os.path.basename(index_two)
    assert sha256(index_three) == three_before, "构造 two 的索引不应改写 three 的索引"
    assert index_files(three) == [index_three]
    print("[case 5] pass three=%s two=%s"
          % (os.path.basename(index_three), os.path.basename(index_two)))


def test_case06_size_change_triggers_rebuild(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    frame.close()
    index = index_files(data)[0]
    before = sha256(index)
    rewrite(os.path.join(str(data), "weather_c.tsfile"), 200, 249, {"dev_a": ("cn", 5)})
    frame = build(data)
    rows = rows_of(frame)
    frame.close()
    assert sha256(index_files(data)[0]) != before
    assert rows == 250, rows
    print("[case 6] pass rows=%d index_rebuilt=True" % rows)


def test_case07_mtime_change_triggers_rebuild(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    frame.close()
    index = index_files(data)[0]
    before = sha256(index)
    target = os.path.join(str(data), "weather_a.tsfile")
    stat = os.stat(target)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    frame = build(data)
    rows = rows_of(frame)
    frame.close()
    assert sha256(index_files(data)[0]) != before
    assert rows == 300, rows
    print("[case 7] pass rows=%d index_rebuilt=True" % rows)


def _write_sized_shard(path, device, start, stop, temperature_base):
    """写一个与其它分片等长（同行数/同名长）但内容不同的分片。"""
    rows = [
        {"time": moment, "device": device, "region": "cn",
         "temperature": temperature_base + (moment - start),
         "humidity": 50.0 + (moment - start) % 7}
        for moment in range(start, stop + 1)
    ]
    with TsFileTableWriter(str(path), fx._weather_schema()) as writer:
        writer.write_dataframe(pd.DataFrame(rows))


def test_case08_fingerprint_boundary_record(d1_root, tmp_path):
    """记录型：fingerprint=size+mtime，同 size+同 mtime 的内容替换不检出（背景文档已声明）。"""
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(copy_file(d1_root, "weather_a.tsfile"), str(data / "weather_a.tsfile"))
    victim = str(data / "weather_victim.tsfile")
    donor = str(tmp_path / "donor.tsfile")          # 供体在数据集之外，避免同设备重复时间戳
    _write_sized_shard(victim, "dev_v", 200, 299, 20.0)
    _write_sized_shard(donor, "dev_v", 700, 799, 20.0)    # 同设备、同行数、不同时间区间
    same_size = os.path.getsize(victim) == os.path.getsize(donor)

    frame = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    name = [str(item) for item in frame.list_timeseries()
            if str(item).startswith("weather.dev_v") and str(item).endswith(".temperature")][0]
    baseline = [int(x) for x in list(frame[name].timestamps)]
    baseline_rows = len(baseline)
    frame.close()
    index = index_files(data)[0]
    index_before = sha256(index)
    stat = os.stat(victim)

    if not same_size:
        print("[case 8] record same_size=False（等长对照样本未构造出）")
        return

    shutil.copyfile(donor, victim)
    os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    frame = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    rows, error, times = None, None, None
    try:
        series = frame[name]
        rows = int(len(series))
        times = [int(x) for x in list(series.timestamps)[:1] + list(series.timestamps)[-1:]]
    except Exception as exc:  # noqa: BLE001
        error = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    finally:
        frame.close()
    rebuilt = sha256(index) != index_before
    print("[case 8] record same_size=True index_rebuilt=%s rows=%s baseline_rows=%d error=%s"
          % (rebuilt, rows, baseline_rows, error))


def _corrupt(path, kind):
    if kind == "magic":
        with open(path, "r+b") as handle:
            handle.write(b"XXXX\0\0\0\0")
    elif kind == "version":
        with open(path, "r+b") as handle:
            raw = bytearray(handle.read())
            struct.pack_into("<H", raw, 8, 2)
            struct.pack_into("<I", raw, 40, 0)
            struct.pack_into("<I", raw, 40, crc32c(bytes(raw[:64])))
            handle.seek(0)
            handle.write(bytes(raw))
    elif kind == "header_crc":
        with open(path, "r+b") as handle:
            raw = bytearray(handle.read())
            offset = struct.unpack_from("<Q", raw, 16)[0]
            struct.pack_into("<Q", raw, 16, offset + 64)
            handle.seek(0)
            handle.write(bytes(raw))
    elif kind == "truncate":
        os.truncate(path, 1024)
    elif kind == "empty":
        open(path, "wb").close()
    elif kind == "random":
        with open(path, "wb") as handle:
            handle.write(os.urandom(4096))
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("kind,case", [
    ("magic", 9), ("version", 10), ("header_crc", 11),
    ("truncate", 13), ("empty", 14), ("random", 14),
])
def test_case09_14_corruption_triggers_rebuild(d1_root, tmp_path, kind, case):
    data = work_dir(d1_root, tmp_path, "data-%s" % kind)
    frame = build(data)
    frame.close()
    index = index_files(data)[0]
    before = sha256(index)
    _corrupt(index, kind)
    corrupted = sha256(index)
    frame = build(data)
    rows = rows_of(frame)
    frame.close()
    current = index_files(data)
    after = sha256(current[0])
    # 判据：损坏态 sha 与处理后 sha 必须不同（重建）；内容应回到原始确定形态
    rebuilt = after != corrupted
    restored = after == before
    info = header_of(current[0])
    assert rebuilt, "%s: 索引未重建（corrupted==after）" % kind
    assert info["magic"] == b"TSIDX\0\0\0" and info["file_length"] == info["actual_length"], (kind, info)
    assert rows == 300, (kind, rows)
    leftovers = other_leftovers(data)
    assert leftovers == [os.path.basename(current[0]) + ".lock"], leftovers
    print("[case %d] pass kind=%s rows=%d restored=%s" % (case, kind, rows, restored))


def test_case12_section_tamper_detection(d1_root, tmp_path):
    """记录型：section 12 单字节篡改后打开的行为登记（Python 打开期 verify_sections=False）。"""
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    baseline_rows = rows_of(frame)
    frame.close()
    index = index_files(data)[0]
    before = sha256(index)
    target = next(item for item in sections_of(index) if item["type"] == 12)
    with open(index, "r+b") as handle:
        handle.seek(target["offset"] + 16)
        raw = handle.read(1)
        handle.seek(target["offset"] + 16)
        handle.write(bytes([raw[0] ^ 0x3F]))
    corrupted = sha256(index)
    frame = build(data)
    rows, error = None, None
    try:
        rows = rows_of(frame)
    except Exception as exc:  # noqa: BLE001
        error = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    finally:
        frame.close()
    after = sha256(index_files(data)[0])
    rebuilt = after != corrupted
    restored = after == before
    wrong = (error is not None) or (rows != baseline_rows)
    verdict = "detected(rebuild)" if rebuilt else ("wrong_result" if wrong else "accepted_no_effect")
    print("[case 12] record verdict=%s rebuilt=%s restored=%s rows=%s baseline=%d error=%s"
          % (verdict, rebuilt, restored, rows, baseline_rows, error))


def test_case12b_locator_offset_tamper(d1_root, tmp_path):
    """记录型：locator 段 offset 篡改后打开的行为登记（编号 12 加强变体）。"""
    data = tmp_path / "data"
    shutil.copytree(str(Path(d1_root) / "table"), str(data))
    frame = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    name = [str(item) for item in frame.list_timeseries()
            if str(item).startswith("weather.dev_a") and str(item).endswith(".temperature")][0]
    baseline_rows = int(len(frame[name]))
    frame.close()
    index = index_files(data)[0]
    index_before = sha256(index)

    with open(index, "rb") as handle:
        header = struct.unpack("<8sHHIQIIQI20s", handle.read(64))
        handle.seek(header[4])
        entries = {}
        for _ in range(header[5]):
            entry = struct.unpack("<IIQQII", handle.read(32))
            entries[entry[0]] = entry
        locator = entries[13]
        handle.seek(locator[2])
        offset_bytes = handle.read(24)
    meta_offset = struct.unpack_from("<Q", offset_bytes, 8)[0]
    with open(index, "r+b") as handle:
        handle.seek(locator[2] + 8)
        handle.write(struct.pack("<Q", meta_offset ^ 0x7F))
    corrupted = sha256(index)

    frame = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    rows, error = None, None
    try:
        rows = int(len(frame[name]))
    except Exception as exc:  # noqa: BLE001
        error = "%s: %s" % (type(exc).__name__, str(exc)[:200])
    finally:
        frame.close()
    after_sha = sha256(index_files(data)[0])
    rebuilt = after_sha != corrupted
    restored = after_sha == index_before
    wrong = (error is not None) or (rows != baseline_rows)
    verdict = "detected(rebuild)" if rebuilt else ("wrong_result" if wrong else "accepted_no_effect")
    print("[case 12b] record variant=locator_offset verdict=%s rebuilt=%s restored=%s rows=%s error=%s"
          % (verdict, rebuilt, restored, rows, error))


def test_case16_no_temp_residue_and_lock_reusable(d1_root, tmp_path):
    data = work_dir(d1_root, tmp_path)
    frame = build(data)
    frame.close()
    index = index_files(data)[0]
    assert other_leftovers(data) == [os.path.basename(index) + ".lock"]
    _corrupt(index, "truncate")
    frame = build(data)
    frame.close()
    leftovers = other_leftovers(data)
    assert leftovers == [os.path.basename(index) + ".lock"], leftovers
    assert index_files(data) == [index]
    print("[case 16] pass leftovers=%s lock_reused=True" % leftovers)


def test_case49_same_table_schema_conflict_aborts(d1_root, tmp_path):
    data = tmp_path / "conflict"
    data.mkdir()
    shutil.copy(copy_file(d1_root, "weather_a.tsfile"), str(data))
    schema = TableSchema("weather", [
        ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("region", TSDataType.STRING, ColumnCategory.TAG),
        ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ColumnSchema("pressure", TSDataType.DOUBLE, ColumnCategory.FIELD),
    ])
    frame = pd.DataFrame({"time": [1, 2], "device": ["dev_z", "dev_z"], "region": ["cn", "cn"],
                          "temperature": [1.0, 2.0], "pressure": [3.0, 4.0]})
    with TsFileTableWriter(str(data / "weather_conflict.tsfile"), schema) as writer:
        writer.write_dataframe(frame)
    errors = {}
    for use_index in (False, True):
        try:
            probe = TsFileDataFrame(str(data), show_progress=False, use_index=use_index)
            probe.close()
            errors[use_index] = None
        except Exception as exc:  # noqa: BLE001
            errors[use_index] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    assert errors[False] is not None and errors[True] is not None, errors
    assert "Incompatible schema" in (errors[False] or "") or \
        "Incompatible schema" in (errors[True] or "")
    assert index_files(data) == [], "失败路径不应发布索引"
    print("[case 49] pass errors=%s index_published=False" % errors)
