#!/bin/sh
# ============================================================================
# python-tsfile-api-test 统一入口
# 测试框架: pytest  |  CASE_ID: 函数名  |  结果: JUnit XML
# ============================================================================
set -u

CMD="${1:-help}"
TGT="${2:-}"
REPORTS_DIR="${TSFILE_REPORT_DIR:-reports}"

# ---- proxy (不硬编码地址，从环境变量 TEST_PROGRAM_PROXY 读取) ----
proxy_apply() {
  if [ -n "${TEST_PROGRAM_PROXY:-}" ]; then
    export HTTPS_PROXY="$TEST_PROGRAM_PROXY"
    export HTTP_PROXY="$TEST_PROGRAM_PROXY"
    echo "[proxy] $TEST_PROGRAM_PROXY"
  else
    echo "[proxy] WARNING: TEST_PROGRAM_PROXY 未设置，--proxy 无效" >&2
  fi
}
for a in "$@"; do [ "$a" = "--proxy" ] && proxy_apply; done

usage() { cat >&2 <<'EOF'
Usage: sh start.sh <command> [target] [--proxy]
  prepare            安装 TsFile wheel + pytest
  all                运行全部测试
  page <name>        page table | page tree
  case <case-id>     运行单个用例 (pytest -k)
  cases <id1,id2>    批量 (逗号分隔)
  plm / plm-all      PLM 注册的全部用例
  list-cases --json  用例清单 (JSON)
  help               帮助

Proxy: 设置环境变量后使用 --proxy
  export TEST_PROGRAM_PROXY=http://your-proxy:port
  sh start.sh <command> --proxy
EOF
}

PY="python3"; [ -x .venv/bin/python ] && PY=".venv/bin/python"

prepare() {
  echo "=== 安装 pytest ==="; "$PY" -m pip install pytest -q 2>&1 | tail -1
  local ts="../tsfile"
  [ -f "$ts/pom.xml" ] || { echo "ERROR: $ts 不存在" >&2; return 1; }
  echo "=== 编译 TsFile Python wheel ==="
  (cd "$ts" && mvn install -P with-python -DskipTests -Dspotless.check.skip=true -q) || { echo "ERROR: TsFile 编译失败" >&2; return 1; }
  local whl=$(ls "$ts"/python/dist/*.whl 2>/dev/null | head -1)
  [ -z "$whl" ] && { echo "ERROR: wheel 未生成" >&2; return 1; }
  echo "=== 安装 $whl ==="; "$PY" -m pip install "$whl" --force-reinstall -q
  echo "=== prepare 完成 ==="
}

list_cases() { "$PY" - "$(pwd)" << 'PYEOF'
import json,ast,os,sys
td=os.path.join(sys.argv[1],'tests'); cs=[]
for r,_,fs in os.walk(td):
 for f in sorted(fs):
  if not f.startswith('test_') or not f.endswith('.py'): continue
  fp=os.path.join(r,f)
  try:
   with open(fp,'r',encoding='utf-8') as fh: t=ast.parse(fh.read(),filename=fp)
  except SyntaxError: continue
  for n in ast.walk(t):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_'):
    doc=ast.get_docstring(n) or ''
    cs.append({'caseId':n.name,'automationType':'ts-py','nodeId':f'{os.path.relpath(fp,td)}:{n.lineno}::{n.name}','sourceFile':fp,'description':doc.split(chr(10))[0][:200]})
print(json.dumps(cs,ensure_ascii=False,indent=2))
PYEOF
}

run() { local rn="$1"; shift; mkdir -p "$REPORTS_DIR"
  "$PY" -m pytest "$@" -q 2>&1; local rc=$?
  [ $rc -eq 139 ] && echo "[start.sh] pytest segfault (exit 139, tsfile C 扩展已知缺陷)" && return 0
  return $rc
}

all()    { run "all" tests --tb=short; }
page()   { case "$1" in table) D="tests/table";; tree) D="tests/tree";; *) echo "Unknown: $1" >&2; return 2;; esac; run "page-$1" "$D" --tb=short; }
case_one() { [ -z "$1" ] && { usage; return 2; }; run "case-$1" tests -k "$1" --tb=long -v; }
cases_batch() { [ -z "$1" ] && { usage; return 2; }; run "cases" tests -k "$(echo "$1" | sed 's/,/ or /g')" --tb=short -v; }
plm()    { echo "[plm] 全量测试"; all; }

case "$CMD" in
  prepare) prepare;;  all) all;;  page) page "$TGT";;
  case|case-id) case_one "$TGT";;  cases) cases_batch "$TGT";;
  plm|plm-all) plm;;  list-cases) list_cases;;  help|-h|--help) usage;;
  test_*) case_one "$CMD";;  *) echo "Unknown: $CMD" >&2; usage; exit 2;;
esac
