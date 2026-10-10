# -*- coding: utf-8 -*-
"""子进程探针：mmap 映射观测（/proc/self/maps 中 .tsidx 存在 / close 后消失）。

用法：python probe_index_maps.py --data <数据集目录>
输出：单行 JSON
"""
import argparse
import json
import os


def _tsidx_maps():
    found = []
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if ".tsidx" in line:
                found.append(line.strip())
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    from tsfile import TsFileDataFrame

    frame = TsFileDataFrame(args.data, show_progress=False, use_index=True)
    before = _tsidx_maps()
    frame.close()
    after = _tsidx_maps()
    print(json.dumps({"during": before, "after_close": after}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
