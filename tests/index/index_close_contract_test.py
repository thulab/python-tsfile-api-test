# -*- coding: utf-8 -*-
"""RN-142 编号 29：关闭语义契约（关闭后的访问必须抛 RuntimeError）。

契约要求 ``RuntimeError``（旧实测为 ``TypeError``，即 TsFile-166）；
2026-10-10 复测 XPASS 确认已修复，xfail(strict) 标记摘除。
问题报告：``records/TsFile-142-问题报告-关闭后访问异常类型违约-20260929.md``。
"""
import shutil
import traceback
from pathlib import Path

import numpy as np
import pytest

from tsfile import TsFileDataFrame


def test_case29_close_contract_violation(d1_root, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(str(Path(d1_root) / "table"), str(data))
    first = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    second = TsFileDataFrame(str(data), show_progress=False, use_index=True)
    subset = first[:1]
    name = str(subset.list_timeseries()[0])
    try:
        independent = first._runtime is not second._runtime
        before = len(np.asarray(subset[name][:]))
        first.close()
        after = len(np.asarray(subset[name][:]))
        runtime = subset._runtime
        subset.close()
        torn_down = bool(getattr(runtime, "_torn_down", False))
        leases = int(getattr(runtime, "_object_leases", -1))
        observed = None
        trace_tail = None
        try:
            subset[name][:]
        except Exception as exc:  # noqa: BLE001
            observed = {"type": type(exc).__name__, "message": str(exc)}
            trace_tail = traceback.format_exc().strip().splitlines()[-6:]
        print("[case 29] runtime_independent=%s rows_after_root_close=%d torn_down=%s leases=%d "
              "observed=%s contract_error_type=RuntimeError trace_tail=%s"
              % (independent, after, torn_down, leases, observed, trace_tail))
        assert independent and before == after and after > 0
        assert torn_down and leases == 0
        assert observed is not None and observed["type"] == "RuntimeError", (
            "contract requires RuntimeError, observed %s" % (observed,)
        )
    finally:
        first.close()
        subset.close()
        second.close()
