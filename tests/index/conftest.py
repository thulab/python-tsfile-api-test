"""RN-142 tests/index 共享夹具：D1 数据集在 tmp 目录现造（session 级，仅生成一次）。

用例一律先用 ``work_dir()`` 把 D1/table 复制到各自 tmp_path 再操作，
避免互相污染；不读写任何仓库外的固定路径。
"""
import shutil
from pathlib import Path

import pytest

import fixtures


@pytest.fixture(scope="session")
def d1_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rn142-d1-root")
    fixtures.build_d1(str(root))
    return root


def copy_table(d1_root, dst):
    """把 D1/table 复制到 dst（返回 Path），供各用例独立工作。"""
    dst = Path(dst)
    shutil.copytree(str(Path(d1_root) / "table"), str(dst))
    return dst
