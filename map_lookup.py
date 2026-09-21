#!/usr/bin/env python3
"""Lookup test node IDs from PLM CASE_ID (case-insensitive)."""
import json, sys, os
cid = sys.argv[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(script_dir, ".plm-case-id-map.json")
try:
    with open(map_path) as f:
        d = json.load(f)
    nmap = d.get("nmap", d)
    # 1. exact match
    v = nmap.get(cid)
    # 2. case-insensitive fallback (PLM sends UPPERCASE, keys are PascalCase)
    if not v:
        uc = cid.upper()
        for k, kv in nmap.items():
            if isinstance(kv, list) and k.upper() == uc:
                v = kv; break
    # 3. 旧编号别名兼容（存量用例的 execution_case_id）
    if not v:
        al = d.get("aliases") or {}
        v = al.get(cid)
        if not v:
            for k, kv in al.items():
                if k.upper() == uc:
                    v = kv; break
    if v and isinstance(v, list):
        print(" ".join(v))
    else:
        print(cid)  # fallback
except Exception as e:
    print(cid)
