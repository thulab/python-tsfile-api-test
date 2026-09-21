from pathlib import Path

from plm_case_map import collect_test_nodes, load_and_validate


ROOT = Path(__file__).resolve().parents[1]
TARGET_CASE_ID = "Python-Table-008-P46B83AFB"
TARGET_NODE_ID = "tests/table/dataframe_to_tsfile_test.py::test_table_names[test_table]"


def test_plm_inventory_matches_pytest_collection_one_to_one():
    collected = collect_test_nodes(ROOT)
    nmap, info, names = load_and_validate(ROOT, collected)

    assert len(nmap) == len(collected) == 276
    assert set(nmap) == set(info) == set(names)


def test_parameterized_case_has_exact_execution_mapping():
    nmap, _, _ = load_and_validate(ROOT)

    assert nmap[TARGET_CASE_ID] == [TARGET_NODE_ID]
