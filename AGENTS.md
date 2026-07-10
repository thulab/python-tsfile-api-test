# Python TsFile Test Maintenance Rules

## Startup

- At the start of each task, state that this `AGENTS.md` was read and that the
  work will follow the tester-only, no-product-source-change rule.
- Run `git status` before editing and preserve all existing user changes.
- Read `test_run/tsfile_dataframe_tree_model_regression_report.md` before
  changing test expectations or result classifications.

## Role And Scope

- Always work as a tester in this repository.
- Only modify test code, test data, test documentation, and test reports.
- Do not modify product source code, installed package source code, or wheel
  contents.
- Keep changes strictly within the current testing task.

## Product Source Boundary

The following locations are read-only and may only be inspected for diagnosis:

- This repository's `tsfile/` directory.
- The active Python environment's `site-packages/tsfile` directory.
- `D:\iotdb-test\tsfile-rc-2.4.0`.
- Product code in wheel extraction or build directories.

When a product issue is found, reproduce, classify, and document it through
tests. Do not fix it in product source code.

## Java Reference

Use current Java behavior as the reference when the API contract or expected
result is unclear:

- Java test repository: `D:\TestProgram\java\java-tsfile-api-test`.
- Java comparison test:
  `src/test/java/org/apache/tsfile/regression/TestPythonApiIssueBehavior.java`.
- When a task explicitly requires Java-side changes, only modify Java tests;
  do not modify Java product source code.

## Result Classification

- A newly observed failure must remain an ordinary failed test until a human
  reviews it.
- Do not automatically add skip or xfail merely to make the suite pass.
- Record an unreviewed failure as `pending human review` in the test report,
  including the command, expected behavior, actual behavior, and traceback or
  process exit code.
- After human review, classify the result as follows:
  - Confirmed product defect: use `pytest.mark.xfail(strict=True)`.
  - Incorrect or outdated test expectation: fix the test and run it normally.
  - Missing external environment: use `pytest.mark.skip` with a precise reason.
  - Process-level crash: reproduce it in an isolated subprocess and prefer a
    strict xfail; use skip only when safe isolation is impossible.
- Every xfail or skip must include a specific reason and a related issue ID
  when one is available.
- If Python behavior agrees with the contract and Java behavior, run the test
  normally and assert the result directly.
- Track a confirmed product defect with `pytest.mark.xfail(strict=True)`.
- A strict xfail test must assert the correct target behavior, not the current
  defective behavior.
- When a product fix causes XPASS, remove the xfail marker and restore the test
  as a normal passing case.
- Use `skip` only when the case cannot currently execute safely, such as a
  process-level crash or a missing external environment.
- Prefer an isolated subprocess when reproducing a process-level crash, and
  record its exit code.
- Do not use skip, broad exception handling, or weakened assertions to hide a
  confirmed defect.
- Correct inputs that violate the API contract instead of treating them as
  product defects. Pure numeric path nodes must be enclosed in backticks.

## Test Workflow

Use this order for test maintenance:

1. Inspect the worktree and the latest report.
2. Run the directly affected tests.
3. Run the full Python suite.
4. Re-run coverage when result totals or the report are updated.
5. Run the Java comparison test when Java behavior is used as the contract.
6. Run the full Java suite when Java test code changes.
7. Update the report from actual command output; never infer result totals.

Run the Python full suite from the repository root:

```powershell
python -m pytest tests -q -rs --tb=short
```

Run coverage from the `tests` directory:

```powershell
python -m pytest --cov=tsfile --cov-report=html:htmlcov `
  --cov-report=json:coverage.json --cov-report=term `
  --cov-config=coveragerc --cov-branch -q -rs --tb=short
```

Run the Java comparison test from the Java test repository:

```powershell
mvn '-Dtest=org.apache.tsfile.regression.TestPythonApiIssueBehavior' test
```

Run the Java full suite with:

```powershell
mvn test
```

## Reporting

The current report is:

`test_run/tsfile_dataframe_tree_model_regression_report.md`

Keep the report synchronized with the latest verified run. Include:

- Commands that were actually run.
- Passed, failed, skipped, and xfailed totals.
- Process exit codes.
- Statement and branch coverage.
- Expected behavior, actual behavior, and tracking method for every failed or
  xfailed case.
- Python and Java behavior comparisons when Java is the reference.
- Whether any product source was modified.

Do not present historical totals as current totals. Avoid putting dynamic test
counts in this `AGENTS.md`; dynamic state belongs in the report.

## Commit Gate

- Do not commit any change while a required targeted, full-suite, or coverage
  command reports `FAILED`, `ERROR`, or strict `XPASS`, or otherwise exits with
  a non-zero status because of a test result.
- A newly observed failure that is pending human review must remain
  uncommitted. Record it in the report and wait for it to be fixed or formally
  classified.
- A human-confirmed strict `XFAIL` is not an automation failure when pytest
  exits with status 0. It may be committed only after the required suite passes
  its commit gate and the user explicitly requests a commit.
- A skipped test does not bypass this gate. It must still satisfy the skip
  rules above and include a precise reason.
- Before an explicitly requested commit, re-run the required automation
  command from the current worktree and verify its exit code. Do not rely on an
  older run or inferred totals.
- If the user requests a commit while the commit gate is failing, do not
  commit. Report the blocking test and its latest output instead.

## Git And File Safety

- Preserve all pre-existing and concurrent user changes.
- Do not clean or roll back unrelated files.
- Do not run destructive commands such as `git reset --hard` or
  `git checkout --`.
- Do not delete existing test artifacts unless the user explicitly requests it.
- Do not commit changes unless the user explicitly requests a commit.
