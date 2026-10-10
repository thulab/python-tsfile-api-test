# -*- coding: utf-8 -*-
"""子进程探针：热开索引构造 DataFrame（不查询），统计是否打开数据文件与耗时。

用法：python probe_hot_open.py <数据集目录>
输出：单行 JSON
"""
import json
import os
import sys
import time

from tsfile import TsFileDataFrame


def main():
    directory = sys.argv[1]
    started = time.perf_counter()
    frame = TsFileDataFrame(directory, show_progress=False, use_index=True)
    seconds = time.perf_counter() - started
    tsfile_fds, index_fds = [], []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink("/proc/self/fd/" + fd)
        except OSError:
            continue
        if target.endswith(".tsfile"):
            tsfile_fds.append(target)
        if ".tsidx" in target:
            index_fds.append(target)
    payload = {
        "seconds": round(seconds, 4),
        "tsfile_fds": len(tsfile_fds),
        "index_fds": len(index_fds),
        "series": len(frame.list_timeseries()),
    }
    frame.close()
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
