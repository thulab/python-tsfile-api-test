from pathlib import Path

from plm_case_map import collect_test_nodes, load_and_validate


ROOT = Path(__file__).resolve().parents[1]


def test_plm_inventory_matches_pytest_collection_one_to_one():
    collected = collect_test_nodes(ROOT)
    nmap, info, names = load_and_validate(ROOT, collected)

    assert len(nmap) == len(collected)
    assert set(nmap) == set(info) == set(names)


def test_parameterized_case_has_exact_execution_mapping():
    nmap, _, _ = load_and_validate(ROOT)

    parameterized = [(cid, nodes[0]) for cid, nodes in nmap.items() if "[" in nodes[0]]
    assert parameterized, "至少应存在一个参数化用例节点"
    case_id, node_id = parameterized[0]
    assert nmap[case_id] == [node_id]
