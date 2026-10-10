# -*- coding: utf-8 -*-
"""RN-142 用例 37：Linux 映射观测（索引页映射在构造期存在、close 后消失）。

在子进程探针内查 /proc/self/maps（74 单平台 Linux），D1 级夹具即可。
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tsfile import TsFileDataFrame

PROBE_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_index_maps.py")


def test_case37_maps_observation(d1_root, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(str(Path(d1_root) / "table"), str(data))
    frame = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    frame.close()
    proc = subprocess.run([sys.executable, PROBE_MAPS, "--data", str(data)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-500:])
    text = proc.stdout.strip()
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        report = json.loads(text.splitlines()[-1])
    print("[case 37] pass mappings_during=%d mappings_after=%d during_sample=%s"
          % (len(report["during"]), len(report["after_close"]), report["during"][:2]))
    assert report["during"], "构造后应存在 .tsidx 映射"
    assert report["after_close"] == [], "close 后映射应消失"
