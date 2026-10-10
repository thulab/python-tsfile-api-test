# -*- coding: utf-8 -*-
"""RN-142 夹具生成器：D1 数据集在 tmp 目录现造，用例自包含。

D1 结构（与验证期一致）：
  <root>/table/{weather_a,weather_b,weather_c}.tsfile   表模型，三段分片
  <root>/tree/tree_ln.tsfile                            树模型
  <root>/mixed/{weather_a,tree_ln}.tsfile               混合模型（用于拒绝语义）
dev_a 覆盖三段（跨分片合并）；dev_b 仅 A 且 humidity 全空；dev_c 在 A/B 且 region 为空。
"""
import os
import shutil
import time

import pandas as pd

from tsfile import (ColumnCategory, ColumnSchema, Field, RowRecord, TableSchema,
                    TimeseriesSchema, TSDataType, TsFileTableWriter, TsFileWriter)


def _weather_schema():
    return TableSchema(
        "weather",
        [
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("region", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )


def _write_table_shard(path, start, stop, devices):
    """写一个分片：devices = {device: (region, humidity_nan_every)}；时间 [start, stop]。"""
    times = list(range(start, stop + 1))
    rows = []
    for device, (region, nan_every) in devices.items():
        for offset, timestamp in enumerate(times):
            humidity = float("nan") if nan_every and offset % nan_every == 0 else 50.0 + offset % 7
            rows.append(
                {
                    "time": timestamp,
                    "device": device,
                    "region": region,
                    "temperature": 20.0 + (offset % 13) / 2.0,
                    "humidity": humidity,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["device", "time"]).reset_index(drop=True)
    with TsFileTableWriter(str(path), _weather_schema()) as writer:
        writer.write_dataframe(frame)


def _write_tree_file(path):
    writer = TsFileWriter(str(path))
    writer.register_timeseries("root.ln.wf01.wt01", TimeseriesSchema("status", TSDataType.INT32))
    writer.register_timeseries("root.ln.wf01.wt01", TimeseriesSchema("temperature", TSDataType.DOUBLE))
    writer.register_timeseries("root.ln.wf02.wt02", TimeseriesSchema("status", TSDataType.INT32))
    for moment in range(5):
        writer.write_row_record(
            RowRecord(
                "root.ln.wf01.wt01",
                moment,
                [
                    Field("status", moment, TSDataType.INT32),
                    Field("temperature", float(moment) + 0.5, TSDataType.DOUBLE),
                ],
            )
        )
        writer.write_row_record(
            RowRecord("root.ln.wf02.wt02", moment, [Field("status", moment * 2, TSDataType.INT32)])
        )
    writer.close()


def build_d1(root):
    """在 root（即 D1 数据集根）下生成 table/tree/mixed 三套夹具。"""
    d1 = str(root)
    table = os.path.join(d1, "table")
    tree = os.path.join(d1, "tree")
    mixed = os.path.join(d1, "mixed")
    for path in (table, tree, mixed):
        os.makedirs(path, exist_ok=True)
    started = time.time()
    # dev_a 覆盖三段（跨分片合并）；dev_b 仅 A 且 humidity 全空；dev_c 在 A/B 且 region 为空
    _write_table_shard(
        os.path.join(table, "weather_a.tsfile"),
        0, 99,
        {"dev_a": ("cn", 0), "dev_b": ("us", 1), "dev_c": ("", 0)},
    )
    _write_table_shard(
        os.path.join(table, "weather_b.tsfile"),
        100, 199,
        {"dev_a": ("cn", 0), "dev_c": ("", 0)},
    )
    _write_table_shard(
        os.path.join(table, "weather_c.tsfile"),
        200, 299,
        {"dev_a": ("cn", 5)},
    )
    _write_tree_file(os.path.join(tree, "tree_ln.tsfile"))
    shutil.copy(os.path.join(table, "weather_a.tsfile"), os.path.join(mixed, "weather_a.tsfile"))
    shutil.copy(os.path.join(tree, "tree_ln.tsfile"), os.path.join(mixed, "tree_ln.tsfile"))
    print("D1 built in %.1fs -> %s" % (time.time() - started, d1))
    return d1
