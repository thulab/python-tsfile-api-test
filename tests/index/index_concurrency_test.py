# -*- coding: utf-8 -*-
"""RN-142 用例 39–43：并发、资源上限、缓存复用与 teardown 租约（增强版）。

批次 5 增强重写版（正式版）；每个用例在 tmp_path 内对 D1/table 副本操作。
"""
from __future__ import annotations

import gc
import glob
import hashlib
import multiprocessing as mp
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

import fixtures as fx
from tsfile import TsFileDataFrame
from tsfile.dataset.index import index_path_for


def _snapshot(frame, names, start, end):
    result = frame.loc[start:end, names]
    return list(result.series_names), np.asarray(result.timestamps).copy(), np.asarray(result.values).copy()


def _fd_snapshot():
    tsfiles = []
    for entry in glob.glob("/proc/self/fd/*"):
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.endswith(".tsfile"):
            tsfiles.append(target)
    with open("/proc/self/maps", encoding="utf-8") as handle:
        tsidx = [line.strip() for line in handle if ".tsidx" in line]
    return {"tsfile": sorted(tsfiles), "tsidx": tsidx}


def test_case39_concurrent_tagged_series_queries_match_unindexed(d1_root, tmp_path):
    table_dir = tmp_path / "data39"
    shutil.copytree(str(Path(d1_root) / "table"), str(table_dir))
    table = str(table_dir)
    plain = TsFileDataFrame(table, show_progress=False, use_index=False)
    indexed = TsFileDataFrame(table, show_progress=False, use_index=True)
    try:
        all_names = [str(item) for item in indexed.list_timeseries()]
        # The public series path includes its TAG values (device and region).
        requested = [name for name in all_names if ".dev_a.cn." in name and name.endswith((".temperature", ".humidity"))]
        assert len(requested) == 2, all_names
        ranges = ((0, 120), (80, 220), (200, 299))
        expected = {span: _snapshot(plain, requested, *span) for span in ranges}
        barrier = threading.Barrier(8)
        failures = []
        checks = 0
        lock = threading.Lock()

        def worker(worker_id):
            nonlocal checks
            try:
                barrier.wait(timeout=20)
                for start, end in ranges:
                    actual = _snapshot(indexed, requested, start, end)
                    want = expected[(start, end)]
                    assert actual[0] == want[0]
                    np.testing.assert_array_equal(actual[1], want[1])
                    np.testing.assert_equal(actual[2], want[2])
                    with lock:
                        checks += 1
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    failures.append(f"worker-{worker_id}: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=90)
        assert all(not thread.is_alive() for thread in threads), "query worker timeout"
        assert not failures, failures
        assert checks == 24
    finally:
        plain.close()
        indexed.close()
        gc.collect()
    after = _fd_snapshot()
    assert not after["tsfile"], after
    print("[case 39] pass workers=8 ranges=3 queries=24 tagged_series=%s tsfile_fd_after_close=0"
          % requested)


def _cold_worker(directory, barrier, queue):
    frame = None
    try:
        barrier.wait(timeout=120)
        frame = TsFileDataFrame(str(directory), show_progress=False, use_index=True)
        names = [str(item) for item in frame.list_timeseries()]
        queried = frame.loc[0:299, names]
        queue.put({"ok": True, "names": list(queried.series_names),
                   "timestamps": np.asarray(queried.timestamps).tolist(),
                   "values": np.asarray(queried.values).tolist()})
    except BaseException as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if frame is not None:
            frame.close()


def test_case40_barrier_synchronized_cold_build_returns_equal_queries(d1_root, tmp_path):
    work = tmp_path / "cold-build"
    work.mkdir()
    for source in (Path(d1_root) / "table").glob("*.tsfile"):
        shutil.copy2(source, work / source.name)
    index_path = Path(index_path_for([str(p) for p in sorted(work.glob("*.tsfile"))]))
    for path in (index_path, Path(str(index_path) + ".lock")):
        if path.exists():
            path.unlink()
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(4), ctx.Queue()
    processes = [ctx.Process(target=_cold_worker, args=(work, barrier, queue)) for _ in range(4)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=240) for _ in processes]
    for process in processes:
        process.join(timeout=30)
    assert all(process.exitcode == 0 for process in processes), [p.exitcode for p in processes]
    assert all(result.get("ok") for result in results), results
    reference = results[0]
    for result in results[1:]:
        assert result["names"] == reference["names"]
        assert result["timestamps"] == reference["timestamps"]
        np.testing.assert_equal(np.asarray(result["values"]), np.asarray(reference["values"]))
    indexes = sorted(work.glob("*.tsidx"))
    assert len(indexes) == 1, indexes
    assert not list(work.glob("*.tmp"))
    print("[case 40] pass processes=4 rows=%d index_count=1 temp_files=0 index_sha256=%s"
          % (len(reference["timestamps"]), hashlib.sha256(indexes[0].read_bytes()).hexdigest()[:16]))


def test_case41_reader_limit_blocks_then_wakes_waiter(d1_root, tmp_path, monkeypatch):
    work = tmp_path / "reader-pool"
    work.mkdir()
    fx._write_table_shard(str(work / "a.tsfile"), 0, 9, {"a": ("cn", 0)})
    fx._write_table_shard(str(work / "b.tsfile"), 10, 19, {"b": ("us", 0)})
    monkeypatch.setenv("TSFILE_DATAFRAME_MAX_OPEN_FILES", "1")
    frame = TsFileDataFrame(str(work), show_progress=False, use_index=True)
    pool = frame._runtime.readers
    entered, release, acquired = threading.Event(), threading.Event(), threading.Event()
    error = []
    elapsed = []

    def hold_first():
        try:
            with pool.acquire(0):
                entered.set()
                if not release.wait(30):
                    raise TimeoutError("holder release timeout")
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    def wait_second():
        try:
            started = time.monotonic()
            with pool.acquire(1):
                elapsed.append(time.monotonic() - started)
                acquired.set()
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    first = threading.Thread(target=hold_first)
    second = threading.Thread(target=wait_second)
    try:
        first.start()
        assert entered.wait(10)
        second.start()
        time.sleep(0.2)
        assert not acquired.is_set(), "second reader bypassed occupied single slot"
        assert pool.open_count == 1
        release.set()
        assert acquired.wait(15), "waiting reader was not notified after slot release"
        first.join(timeout=10)
        second.join(timeout=10)
        assert not error, error
        assert elapsed and elapsed[0] >= 0.15, elapsed
        assert pool.open_count <= 1
    finally:
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)
        frame.close()
    print("[case 41] pass max_open_files=1 waiter_blocked_and_woke=True wait_seconds=%.4f"
          % round(elapsed[0], 4))


def test_case42_prepared_series_cache_returns_same_object_after_eviction(d1_root, tmp_path, monkeypatch):
    table_dir = tmp_path / "data42"
    shutil.copytree(str(Path(d1_root) / "table"), str(table_dir))
    table = str(table_dir)
    monkeypatch.setenv("TSFILE_DATAFRAME_MAX_OPEN_FILES", "1")
    frame = TsFileDataFrame(table, show_progress=False, use_index=True)
    cache = frame._runtime.prepared
    original_get = cache.get
    identities = []

    def observed_get(file_id, locator_id, reader, time_owner=None):
        value = original_get(file_id, locator_id, reader, time_owner=time_owner)
        identities.append((int(file_id), int(locator_id), id(value)))
        return value

    cache.get = observed_get
    try:
        names = [str(item) for item in frame.list_timeseries() if str(item).endswith(".temperature")]
        assert len(names) >= 2
        for name in names[:2]:
            np.asarray(frame[name][0:100]).copy()
        a_calls_before = len(identities)
        first_name = names[0]
        first_values = np.asarray(frame[first_name][0:100]).copy()
        np.asarray(frame[names[1]][0:100]).copy()
        replay = np.asarray(frame[first_name][0:100]).copy()
        np.testing.assert_equal(first_values, replay)
        key_ids = [entry[2] for entry in identities if entry[:2] == identities[0][:2]]
        assert len(key_ids) >= 2, identities
        assert len(set(key_ids)) == 1, identities
        assert cache.size >= 2
        final_cache_size = cache.size
    finally:
        frame.close()
    print("[case 42] pass access_sequence=A-B-A cache_size=%d same_prepared_object_reused=True"
          % final_cache_size)


def test_case43_teardown_waits_for_all_object_and_query_leases(d1_root, tmp_path):
    table_dir = tmp_path / "data43"
    shutil.copytree(str(Path(d1_root) / "table"), str(table_dir))
    table = str(table_dir)
    rounds = []
    for iteration in range(10):
        root = TsFileDataFrame(table, show_progress=False, use_index=True)
        view = root[[0, 1]]
        series = root[str(root.list_timeseries()[0])]
        runtime = root._runtime
        query_lease = runtime.query_lease()
        query_lease.__enter__()
        root.close()
        mid = {"object_leases": runtime._object_leases, "query_leases": runtime._query_leases,
               "tsfile_fd": len(_fd_snapshot()["tsfile"]), "tsidx_maps": len(_fd_snapshot()["tsidx"])}
        assert not runtime._torn_down and runtime._query_leases == 1, mid
        series.close()
        close_result = []
        closer = threading.Thread(target=lambda: close_result.append(view.close()))
        closer.start()
        time.sleep(0.1)
        assert closer.is_alive(), "last object close should wait for active query lease"
        assert not runtime._torn_down
        query_lease.__exit__(None, None, None)
        closer.join(timeout=10)
        assert not closer.is_alive(), "object close did not resume after query lease release"
        gc.collect()
        after = _fd_snapshot()
        assert runtime._torn_down
        assert not after["tsfile"] and not after["tsidx"], after
        rounds.append({"iteration": iteration + 1, "mid_teardown": mid,
                       "released_tsfile_fd": len(after["tsfile"]), "released_tsidx_maps": len(after["tsidx"])})
    print("[case 43] pass rounds=%d teardown_deferred_for_query_lease=True" % len(rounds))
