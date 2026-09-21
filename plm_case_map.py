#!/usr/bin/env python3
"""Validate the PLM CASE_ID contract and emit the executable inventory."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class MappingContractError(RuntimeError):
    pass


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MappingContractError(f"cannot read {path.name}: {exc}") from exc


def collect_test_nodes(root: Path = ROOT) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "--color=no",
            "tests/",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise MappingContractError(
            f"pytest collection failed (exitCode={result.returncode}): {detail}"
        )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith("=")
    ]


def reverse_key(node_id: str) -> str:
    parts = node_id.split("::")
    if len(parts) < 2:
        raise MappingContractError(f"invalid pytest nodeId: {node_id}")
    file_stem = str(Path(parts[0]).with_suffix("")).replace("\\", ".").replace("/", ".")
    if len(parts) == 2:
        return f"{file_stem}::{parts[1]}"
    return f"{file_stem}.{'.'.join(parts[1:-1])}::{parts[-1]}"


def load_and_validate(root: Path = ROOT, collected_nodes: list[str] | None = None):
    mapping = load_json(root / ".plm-case-id-map.json")
    info = load_json(root / ".plm-case-info.json")
    names = load_json(root / ".plm-cn-names.json")
    nmap = mapping.get("nmap")
    revmap = mapping.get("revmap")
    if not isinstance(nmap, dict) or not isinstance(revmap, dict):
        raise MappingContractError(".plm-case-id-map.json must contain nmap and revmap objects")

    key_set = set(nmap)
    if key_set != set(info) or key_set != set(names):
        raise MappingContractError("nmap, case-info and cn-names CASE_ID sets differ")

    mapped_nodes: list[str] = []
    normalized_ids: set[str] = set()
    for case_id, nodes in nmap.items():
        normalized = case_id.upper()
        if not CASE_ID_PATTERN.fullmatch(case_id) or normalized in normalized_ids:
            raise MappingContractError(f"invalid or duplicate CASE_ID: {case_id}")
        normalized_ids.add(normalized)
        if not isinstance(nodes, list) or len(nodes) != 1 or not isinstance(nodes[0], str):
            raise MappingContractError(f"CASE_ID must map to exactly one nodeId: {case_id}")
        node_id = nodes[0]
        mapped_nodes.append(node_id)
        if revmap.get(reverse_key(node_id), "").upper() != normalized:
            raise MappingContractError(f"revmap mismatch: {case_id} -> {node_id}")

    if len(mapped_nodes) != len(set(mapped_nodes)):
        raise MappingContractError("multiple CASE_IDs map to the same pytest nodeId")

    if collected_nodes is not None:
        if len(collected_nodes) != len(set(collected_nodes)):
            raise MappingContractError("pytest collection contains duplicate nodeIds")
        missing = sorted(set(collected_nodes) - set(mapped_nodes))
        stale = sorted(set(mapped_nodes) - set(collected_nodes))
        if missing or stale:
            raise MappingContractError(
                "collection/map mismatch: "
                f"missing={missing[:10]}, stale={stale[:10]}"
            )

    return nmap, info, names


def build_inventory(root: Path = ROOT) -> list[dict[str, object]]:
    collected = collect_test_nodes(root)
    nmap, info, names = load_and_validate(root, collected)
    cases = []
    for case_id, nodes in nmap.items():
        node_id = nodes[0]
        metadata = info[case_id]
        cases.append(
            {
                "caseId": case_id,
                "automationType": "ts-py",
                "nodeId": node_id,
                "sourceFile": node_id.split("::", 1)[0],
                "name": names[case_id],
                "description": metadata.get("name") or "",
            }
        )
    return cases


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] != "list-cases":
        print("Usage: python3 plm_case_map.py list-cases", file=sys.stderr)
        return 2
    try:
        print(json.dumps(build_inventory(), ensure_ascii=False, indent=2))
        return 0
    except (MappingContractError, subprocess.TimeoutExpired) as exc:
        print(f"[plm-map] {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
