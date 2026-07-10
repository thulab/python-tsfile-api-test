# Python TsFile 全量回归测试报告

报告时间：2026-07-10 15:14:23 +08:00

## 测试范围

- 测试程序：`D:\TestProgram\python\python-tsfile-api-test`
- 用例文档：`C:\Users\timecho\Desktop\TimechoDB测试\测试用例\TsFile 测试用例 - TsFileDataFrame 支持 2.0 树模型.csv`
- 当前约束：测试人员仅修改测试代码、测试文档和测试报告；产品源码问题只记录，不修改产品源码。

## 本轮维护

本轮继续保持“只改测试不改源码”，对树模型 `AlignedTimeseries` / `.loc` 覆盖做专项检查，并补齐 3 条跨场景自动化覆盖：

- `test_tree_loc_named_series_variants_align_by_tree_path_rules`：覆盖普通、中文、反引号、点号转义、大小写设备名等命名规则序列参与 `.loc` 对齐查询。
- `test_tree_loc_metadata_filtered_subset_aligns_series`：覆盖元数据布尔过滤得到的子集视图继续执行 `.loc`，并校验名称和索引混用。
- `test_tree_loc_cross_file_field_union_aligns_missing_values`：覆盖跨文件合并、同设备物理量并集、缺失物理量在对齐矩阵中填 `NaN`。
- 在生命周期用例中补充 `close()` 后访问 `tsdf.loc[...]` 抛 `RuntimeError`。
- 针对已知产品缺陷保留专项断言，并按本次提交要求暂时标记为 skip，避免影响仓库自动化；包括大写物理量读取、跨设备不同物理量对齐、复用 reader 跨设备读取和 `limit=0`。
- 新增表模型 null tag 回归用例：`test_nullable_tag_values_list_timeseries_paths_do_not_crash_loc` 直接在主进程验证 `list_timeseries()` 返回结果继续传给 `loc[:, series]` 可正常运行。
- 扩展相似 null tag 场景共 7 组：单 tag 全 null、单 tag 混合 null、双 tag 首列 null、双 tag 尾列 null、双 tag 全 null、三 tag 中间 null、三 tag 稀疏混合 null；当前均未出现崩溃或异常。
- 补强 null tag 场景断言：同步校验 `list_timeseries_metadata()` 可读取、metadata 行数和 `count` 正确、每条 `Timeseries` 可按 `[:]` 读取数据、`.loc` 对齐矩阵中的实际值与写入值一致。
- 优化测试函数注释：`tests/tree/test_tsfile_dataframe_tree_model.py` 中 63 个 `test_*` 函数 docstring 均以 CSV 用例序号开头，复合场景使用序号组合或区间；表模型 null tag 用例标注为补充回归，不新增 CSV 编号。
- 新增序列名专项回归覆盖 11 条：树模型 2 条、表模型参数化 9 条，覆盖 `list_timeseries` 返回路径继续用于 metadata、`df[name]`、子集视图、`.loc` 和展示时不因转义、大小写、中文、空格、反引号或点号路径出错。
- 完成历史 skip 整改：依据 Java 契约修正过期断言并解除 12 条 skip；全空 TAG/FIELD 用例当前测试体可安全通过，且没有已确认的 Java 行为差异，因此一并解除 skip。
- 纯数字设备段和测点名按合法路径规则改为反引号包裹：root.\`1234567890\`、\`1234567890\`；Python 与 Java 均可成功写入并查询 10 行，两条 skip 已解除。
- `limit=0` 已确认属于产品缺陷，测试仍按 Java 语义断言返回全部 10 行；当前按本次提交要求暂时标记为 skip，避免影响仓库自动化。

CSV 用例总数仍为 155 条；用例结论仍为 153 条结果一致、2 条结果不一致、0 条未执行。新增自动化属于对既有场景的对齐序列和序列名交叉覆盖，不新增 CSV 编号。

## 对齐序列覆盖检查

专项结论：已有专门的 CSV 用例 73-90 覆盖 `多序列对齐查询(.loc)` 主路径，本轮又补充了跨场景 `.loc` 交叉覆盖。当前对齐序列覆盖矩阵如下：

| 场景 | 覆盖情况 |
|---|---|
| 多序列对齐查询(.loc) 73-90 | 已覆盖：返回 `AlignedTimeseries`、时间戳并集、列顺序、同设备多物理量、跨设备填 `NaN`、名称/索引混用、单时间戳、开区间、负索引、重复序列、时间范围不扩展、结果属性和参数校验。 |
| 序列命名规则 | 本轮补齐：普通、中文、反引号、点号转义、大小写设备名路径均参与 `.loc`。大写物理量读取及其 `.loc` 对齐读取当前暂时 skip，并保留已提交 issue 跟踪。 |
| 元数据过滤与子集视图 | 本轮补齐：布尔过滤后的 `TsFileDataFrame` 子集可继续 `.loc`，且子集内索引和名称混用顺序稳定。 |
| 跨文件合并与树模型 union | 本轮补齐：跨文件同名序列按时间分片合并，物理量并集在 `.loc` 中缺失位置填 `NaN`。 |
| 展示与格式化 | 已覆盖：`AlignedTimeseries.__repr__`、`show(max_rows)`、大结果截断、毫秒时间格式。 |
| 源码共享 .loc 回归 | 已覆盖：重复序列名、名称加索引重复、重复列位置保留、底层读取去重。 |
| 树读路径防护 | 已覆盖并暂时 skip：复用 reader 跨设备名读取第二条序列为空；跨设备不同物理量 `.loc` 对齐查询第二列返回 `NaN`。 |
| 资源与生命周期 | 本轮补齐：根对象关闭后访问 `.loc` 抛 `RuntimeError`。 |
| 加载错误、纯元数据浏览、单条 Timeseries 行号读取等场景 | 不属于对齐序列职责范围；相关功能已有各自正/异常覆盖。 |

## 序列名覆盖检查

专项结论：本轮补充后，`list_timeseries()` 暴露出的常见特殊序列名均已回放到元数据、单序列读取、子集视图和 `.loc` 路径中验证。当前未发现新的非预期失败；大写物理量读取/对齐问题当前暂时 skip 并继续跟踪，设备路径段含点号时底层 cwrapper 不支持的明确异常仍由正常异常断言覆盖。

新增覆盖如下：

| 场景 | 覆盖情况 |
|---|---|
| 树模型可闭环序列名 | 已覆盖：普通小写、设备段大写、中文设备/物理量、物理量点号转义、反引号空格设备段、反引号特殊符号设备段、反引号数字设备/物理量、反引号空格物理量。 |
| 树模型相似前缀 | 已覆盖：`root.prefix.a` 不误命中 `root.prefix.ab`，metadata 和读取结果保持一致。 |
| 树模型不支持路径段 | 已覆盖：反引号设备路径段包含点号时，`list_timeseries` 和 metadata 可暴露序列，后续 `df[name][:]` / `.loc` 给出明确 `NotImplementedError`。 |
| 表模型可闭环序列名 | 已覆盖：TAG 值含点号、空格、反引号点号、中文、大写，FIELD 名含点号/空格，多 TAG 混合点号与空格，表名/字段名大小写归一后路径可读取。 |

## 树模型目标文件

执行命令：

```powershell
python -m pytest tests\tree\test_tsfile_dataframe_tree_model.py -q -rs --tb=short
```

执行结果：

- 通过：59
- 跳过：4
- 失败：0
- 退出码：0

skip 项：

- `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_iotdb_uppercase_measurement_can_be_read_by_full_path`
- `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_loc_uppercase_measurement_can_align_by_full_path`
- `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_loc_cross_device_different_measurements_preserves_values`
- `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_reader_handles_stale_path_columns_after_reused_queries`

## 全量测试

执行命令：

```powershell
python -m pytest tests -q -rs --tb=short
```

执行结果：

- 通过：400
- 跳过：5
- 预期失败 xfail：0
- 失败：0
- 退出码：0

skip 用例 1：

- 测试：`tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_iotdb_uppercase_measurement_can_be_read_by_full_path`
- 关联 CSV 用例：21
- 预期结果：`root.case.d1.Temperature` 按完整路径读取返回 `[40.0, 41.0, 42.0]`。
- 当前实际：`list_timeseries` 与 metadata 均能识别 `Temperature`，但读取返回空数组 `array([], dtype=float64)`。
- 跟踪方式：暂时 skip 以避免影响仓库自动化；产品修复后移除 skip，并将 CSV 用例 21 结论改为结果一致。

skip 用例 2：

- 测试：`tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_loc_uppercase_measurement_can_align_by_full_path`
- 关联 CSV 用例：21
- 预期结果：`root.case.d1.temperature` 与 `root.case.d1.Temperature` 参与 `.loc` 对齐查询时返回时间戳 `[0, 1, 2]` 和值矩阵 `[[30.0, 40.0], [31.0, 41.0], [32.0, 42.0]]`。
- 当前实际：对齐结果为空，`timestamps` 和 `values` 均无数据。
- 跟踪方式：暂时 skip 以避免影响仓库自动化；与大写物理量读取 issue 关联。

skip 用例 3：

- 测试：`tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_loc_cross_device_different_measurements_preserves_values`
- 关联 CSV 用例：130
- 预期结果：跨设备读取 `root.ln.wf01.wt01.temperature` 与 `root.ln.wf02.wt02.status` 时，第二列保留 `[0.0, 2.0, 4.0, 6.0, 8.0]`。
- 当前实际：`.loc` 对齐查询第一列可取到值，第二列为 `NaN`。
- 跟踪方式：暂时 skip 以避免影响仓库自动化；与跨设备读取 issue 关联。

skip 用例 4：

- 测试：`tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_reader_handles_stale_path_columns_after_reused_queries`
- 关联 CSV 用例：130
- 预期结果：复用 reader 先读取 `root.ln.wf01.wt01.temperature` 后，再读取 `root.ln.wf02.wt02.status`，第二个序列仍返回 `[0.0, 2.0, 4.0, 6.0, 8.0]`。
- 当前实际：第二个序列返回空数组 `array([], dtype=float64)`；同场景 `.loc` 跨设备名对齐中第二条序列也无法得到有效值。
- 跟踪方式：暂时 skip 以避免影响仓库自动化；产品修复后移除 skip，并将 CSV 用例 130 结论改为结果一致。

skip 用例 5：

- 测试：`tests/tree/query_tree_by_row_test.py::test_limit_zero`
- Java 参考：底层 `QueryDataSet.setRowLimit(0)` 表示不限制，10 行输入应返回 10 行。
- 预期结果：`reader.query_tree_by_row(..., limit=0)` 返回全部 10 行。
- 当前实际：创建结果对象时抛 `ValueError: 254 is not a valid TSDataType`。
- 跟踪方式：暂时 skip 以避免影响仓库自动化；产品修复后移除 skip 标记并恢复正常执行。

## 覆盖率

执行目录：`D:\TestProgram\python\python-tsfile-api-test\tests`

执行命令：

```powershell
python -m pytest --cov=tsfile --cov-report=html:htmlcov --cov-report=json:coverage.json --cov-report=term --cov-config=coveragerc --cov-branch -q -rs --tb=short
```

执行结果：

- 通过：400
- 跳过：5
- 预期失败 xfail：0
- 失败：0
- 总覆盖率：88%
- 语句覆盖率：90%
- 分支覆盖率：84%
- 覆盖行数：1836 / 2042
- 缺失行数：206
- 分支覆盖：640 / 764
- 部分覆盖分支：84
- 缺失分支：124
- 退出码：0

覆盖率报告：

- `tests\htmlcov\index.html`
- `tests\coverage.json`

## Java 端参考结论

参考工程：`D:\TestProgram\java\java-tsfile-api-test`

依赖基线：已按 README 编译安装 `D:\iotdb-test\tsfile-rc-2.4.0`，该源码当前 Maven 版本为 `2.3.2-SNAPSHOT`；Java 测试工程依赖 `org.apache.tsfile:tsfile:2.3.2-SNAPSHOT`。

执行结果：

- `mvn '-Dtest=org.apache.tsfile.regression.TestPythonApiIssueBehavior' test`：9 passed
- `mvn test`：59 passed

Java API 行为作为本轮 Python 侧 skip 语义判定基准：

| 场景 | Java 当前效果 | Python 侧判定 |
|---|---|---|
| 表模型混合存在列和不存在列 | 抛 `NoMeasurementException`，不会只返回存在列 | Python 抛 `ColumnNotExistError` 与 Java 契约一致；已改为异常断言并解除 skip |
| 单个不存在设备 | 不抛错，返回 0 行 | Python 返回 0 行与 Java 契约一致；已改为 0 行断言并解除 skip |
| 部分设备不存在 | 不抛错，返回 10 行，按时间戳合并已有设备结果 | Python 返回 10 行与 Java 契约一致；已改为 10 行断言并解除 skip |
| 全部设备不存在 | 不抛错，返回 0 行 | Python 返回 0 行与 Java 契约一致；已改为空结果断言并解除 skip |
| 反引号纯数字设备 root.\`1234567890\` | 成功写入并查询 10 行 | Python 同样成功写入并查询 10 行；已解除 skip |
| 单个不存在测点 | 不抛错，返回 0 行 | Python 返回 0 行与 Java 契约一致；已改为 0 行断言并解除 skip |
| 全部测点不存在 | 不抛错，返回 0 行 | Python 返回 0 行与 Java 契约一致；已改为空结果断言并解除 skip |
| 反引号纯数字测点 \`1234567890\` | 成功写入并查询 10 行 | Python 同样成功写入并查询 10 行；已解除 skip |
| `limit=0` | Java v4 `TsFileTreeReader.query` 无 `limit` 参数；底层 `QueryDataSet.setRowLimit(0)` 表示不限制，返回 10 行 | Python 抛 `ValueError: 254 is not a valid TSDataType`；测试保留 10 行目标断言，当前暂时 skip 以避免影响仓库自动化 |

## 跳过用例

当前共有 5 条 skip，均为已知产品缺陷。本次按提交要求暂时跳过，避免影响仓库自动化；缺陷目标断言和问题说明仍保留在测试体及本报告中。

| 测试 | 跳过原因 |
|---|---|
| `tests/tree/query_tree_by_row_test.py::test_limit_zero` | Java 语义应返回全部 10 行，Python 当前抛 `ValueError: 254 is not a valid TSDataType` |
| `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_iotdb_uppercase_measurement_can_be_read_by_full_path` | CSV 用例 21：大写物理量按完整路径读取为空 |
| `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_loc_uppercase_measurement_can_align_by_full_path` | CSV 用例 21：大写物理量参与 `.loc` 返回空结果 |
| `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_loc_cross_device_different_measurements_preserves_values` | CSV 用例 130：跨设备不同物理量对齐查询第二列返回 `NaN` |
| `tests/tree/test_tsfile_dataframe_tree_model.py::test_tree_reader_handles_stale_path_columns_after_reused_queries` | CSV 用例 130：复用 reader 跨设备读取第二条序列为空 |

## 历史 skip 整改结果

原 16 条 skip 已逐个使用独立 Python 子进程隔离复测，避免历史崩溃用例影响整轮执行；随后按 Java 契约和当前安全复测结果完成整改。

原始记录：

- `test_run\rerun_skipped_cases_20260709.json`
- `test_run\rerun_skipped_extra_probes_20260709.json`
- `test_run\rerun_skipped_cases_report.md`

整改汇总：

- 恢复正常执行并通过：15 / 16
- 保留为临时 skip：1 / 16，即 `limit=0`
- 当前 skip：5，其中另 4 条为树模型已知缺陷专项用例
- 全量结果：`400 passed, 5 skipped`
- 全空 TAG/FIELD 用例当前测试体可安全通过，且未确认存在 Java 行为差异，因此不再保留 skip。补充探针中的 `TsFileDataFrame` 空文件加载语义作为独立观察记录，不影响当前用例执行。

已解除的历史 skip：

| 用例 | 解除依据 |
|---|---|
| `tests/table/query_table_by_row_test.py::test_column_names_multi_partial_exist` | 改为断言 `ColumnNotExistError`，与 Java 抛 `NoMeasurementException` 的契约一致 |
| `tests/table/query_table_by_row_test.py::test_column_type_tag` | TAG-only 查询返回 10 行，复测通过 |
| `tests/table/test_tsfile_dataset.py::TestBoundaryCases::test_all_null_values` | 当前测试体写入全空值并读取 schema 可安全通过，未确认存在 Java 行为差异 |
| `tests/table/test_tsfile_dataset.py::TestBugFixValidation::test_sparse_tags_with_none_values_current_behavior` | 稀疏 TAG 为 `None` 后继续 `.loc` 未崩溃，复测通过 |
| `tests/tree/query_tree_by_row_test.py::test_device_ids_single_not_exist` | 改为断言 0 行，与 Java 契约一致 |
| `tests/tree/query_tree_by_row_test.py::test_device_ids_multi_partial_exist` | 改为断言合并后的 10 行，与 Java 契约一致 |
| `tests/tree/query_tree_by_row_test.py::test_device_ids_multi_all_not_exist` | 改为断言空结果且不抛错，与 Java 契约一致 |
| `tests/tree/query_tree_by_row_test.py::test_device_ids_numbers` | 纯数字设备段改为反引号包裹，Python 与 Java 均成功写入并查询 10 行 |
| `tests/tree/query_tree_by_row_test.py::test_device_ids_special_chars` | `root.d 1` 未再崩溃，复测通过 |
| `tests/tree/query_tree_by_row_test.py::test_measurement_names_single_not_exist` | 改为断言 0 行，与 Java 契约一致 |
| `tests/tree/query_tree_by_row_test.py::test_measurement_names_multi_partial_exist` | 部分测点不存在场景复测通过 |
| `tests/tree/query_tree_by_row_test.py::test_measurement_names_multi_all_not_exist` | 改为断言空结果且不抛错，与 Java 契约一致 |
| `tests/tree/query_tree_by_row_test.py::test_measurement_names_numbers` | 纯数字测点名改为反引号包裹，Python 与 Java 均成功写入并查询 10 行 |
| `tests/tree/query_tree_by_row_test.py::test_measurement_names_special_chars` | `measurement @#$` 未再崩溃，复测通过 |
| `tests/tree/query_tree_by_row_test.py::test_multi_device_different_measurements` | 多设备不同测点集合查询复测通过 |

## 结论

本轮继续保持“只改测试不改源码”。反引号纯数字设备段和测点名在 Python 与 Java 中均能成功写入并查询 10 行，不再发生进程级崩溃。当前 5 条已知缺陷用例按本次提交要求暂时标记为 skip，避免影响仓库自动化；全量回归结果为 `400 passed, 5 skipped`，非预期失败和 xfail 均为 0，退出码为 0。覆盖率为 88%，覆盖行数为 `1836 / 2042`。5 条 skip 分别跟踪 `limit=0` 元数据类型异常，以及 CSV 用例 21、130 的大写物理量读取和跨设备读取问题；产品修复后应移除对应 skip 并恢复正常断言执行。
