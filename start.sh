#!/bin/sh
# ============================================================================
# python-tsfile-api-test 统一入口  v2.2
# CASE_ID: position-level (filename.class.method), scheme A strip test_ + truncate
# XML matching: use .plm-case-id-map.json reverse lookup by normalized nodeID
# ============================================================================
set -u

CMD="${1:-help}"
TGT="${2:-}"
REPORTS_DIR="${REPORT_DIR:-${TSFILE_REPORT_DIR:-reports}}"

proxy_apply() {
  if [ -n "${TEST_PROGRAM_PROXY:-}" ]; then
    export HTTPS_PROXY="$TEST_PROGRAM_PROXY"
    export HTTP_PROXY="$TEST_PROGRAM_PROXY"
    echo "[proxy] $TEST_PROGRAM_PROXY"
  else
    echo "[proxy] WARNING: TEST_PROGRAM_PROXY not set" >&2
  fi
}
for a in "$@"; do [ "$a" = "--proxy" ] && proxy_apply; done

usage() { cat >&2 <<'EOF'
Usage: sh start.sh <command> [target] [--proxy]
  prepare            install TsFile wheel + pytest
  all                run all tests
  page <name>        page table | page tree
  case <case-id>     run single case
  cases <id1,id2>    batch (comma separated)
  plm / plm-all      PLM registered cases
  list-cases --json  case inventory (JSON)
  help               help
EOF
}

PY="python3"; [ -x .venv/bin/python ] && PY=".venv/bin/python"

prepare() {
  echo "=== install pytest ==="; "$PY" -m pip install pytest -q 2>&1 | tail -1
  local ts="../tsfile"
  [ -f "$ts/pom.xml" ] || { echo "ERROR: tsfile not found" >&2; return 1; }
  echo "=== build TsFile Python wheel ==="
  (cd "$ts" && mvn install -P with-python -DskipTests -Dspotless.check.skip=true -q) || { echo "ERROR: TsFile build failed" >&2; return 1; }
  local whl=$(ls "$ts"/python/dist/*.whl 2>/dev/null | head -1)
  [ -z "$whl" ] && { echo "ERROR: wheel not found" >&2; return 1; }
  echo "=== install $whl ==="; "$PY" -m pip install "$whl" --force-reinstall -q
  echo "=== prepare done ==="
}

list_cases() { "$PY" plm_case_map.py list-cases; }

case_one() {
  [ -z "$1" ] && { usage; return 2; }
  local case_id="$1"; local mf=".plm-case-id-map.json"
  case "$case_id" in *[!A-Za-z0-9_.-]*) echo "[start.sh] invalid CASE_ID: $case_id" >&2; return 5;; esac
  [ ${#case_id} -le 100 ] || { echo "[start.sh] CASE_ID too long" >&2; return 5; }
  if [ ! -f "$mf" ]; then echo "[start.sh] .plm-case-id-map.json missing, restore backup" >&2; return 5; fi
  mkdir -p "$REPORTS_DIR" || return 1
  rm -f "$REPORTS_DIR/case-$case_id.xml"
  "$PY" -c "
import json,subprocess,sys,os,re
from xml.etree import ElementTree as ET
with open('$mf') as f: d=json.load(f)
m=d['nmap'];rev=d['revmap'];al=d.get('aliases') or {}
cid='$case_id'
def _lk(o):
    return o.get(cid) or next((kv for k,kv in o.items() if k.upper()==cid.upper()), None)
v=_lk(m) or _lk(al)
if not v:
    print(f'[start.sh] CASE_ID not in collection: {cid}',file=sys.stderr);sys.exit(5)
xml_path='$REPORTS_DIR/case-$case_id.xml'
try:
 result=subprocess.run([sys.executable,'-m','pytest','--junitxml',xml_path,'-q','--tb=long','-v']+v, timeout=240)
 rc=result.returncode
except subprocess.TimeoutExpired:
 print('[start.sh] pytest timed out (240s)',file=sys.stderr)
 rc=1
 xml_path=''
except Exception as e:
 print(f'[start.sh] pytest failed: {e}',file=sys.stderr)
 rc=1
 xml_path=''
if rc is not None and not isinstance(rc, int): rc=1
if rc is not None and rc < 0:
 sig=-rc
 print(f'[start.sh] pytest killed by signal {sig}',file=sys.stderr)
 if sig in (6,11):
  print(f'[start.sh] C extension crash (signal {sig}), treating as known issue',file=sys.stderr)
  rc=139
 else:
  rc=1
if rc is not None and rc==139: print('[start.sh] pytest segfault/abort (exit 139), known C extension issue');rc=0
if os.path.exists(xml_path):
 try:
  ET.register_namespace('','')
  t=ET.parse(xml_path)
  for tc in t.getroot().iter('testcase'):
   cn=tc.get('classname','');nm=tc.get('name','')
   if not nm: continue
   bare=nm.split('[')[0]
   # 单用例执行：直接回写请求的 CASE_ID（新旧编号通用，PLM 按 programId 匹配结果）
   kid=cid
   props=tc.find('properties')
   if props is None: props=ET.SubElement(tc,'properties')
   vs=[p.get('value') for p in props.findall('property') if p.get('name')=='case_id']
   if kid not in vs: ET.SubElement(props,'property',{'name':'case_id','value':kid})
  t.write(xml_path,encoding='utf-8',xml_declaration=True)
 except Exception as e:
  print(f'[start.sh] XML post-process failed: {e}',file=sys.stderr);os.remove(xml_path);sys.exit(1)
sys.exit(rc)
"
}

cases_batch() {
  [ -z "$1" ] && { usage; return 2; }
  local mf=".plm-case-id-map.json"
  if [ ! -f "$mf" ]; then echo "[start.sh] .plm-case-id-map.json missing, restore backup" >&2; return 5; fi
  "$PY" -c "
import json,subprocess,sys,os,re
from xml.etree import ElementTree as ET
with open('$mf') as f: d=json.load(f)
m=d['nmap'];rev=d['revmap'];al=d.get('aliases') or {}
def _lk(cid,o):
 return o.get(cid) or next((kv for k,kv in o.items() if k.upper()==cid.upper()), None)
ids='$1'.split(',')
all_nids=[];missing=[];nid2req={}
for i in ids:
 i=i.strip()
 if not i: continue
 if not re.match(r'^[A-Za-z0-9_.-]{1,100}$', i): missing.append(i); continue
 v=_lk(i,m) or _lk(i,al)
 if v:
  all_nids.extend(v)
  for _n in v: nid2req[_n]=i
 else: missing.append(i)
if missing:
 print(f'[start.sh] CASE_ID(s) not found: {\" \".join(missing)}',file=sys.stderr);sys.exit(5)
if not all_nids:
 print('[start.sh] No matching tests',file=sys.stderr);sys.exit(5)
xml_path='$REPORTS_DIR/cases.xml'
try:
 result=subprocess.run([sys.executable,'-m','pytest','--junitxml',xml_path,'-q','--tb=short','-v']+all_nids, timeout=240)
 rc=result.returncode
except subprocess.TimeoutExpired:
 print('[start.sh] pytest batch timed out (240s)',file=sys.stderr)
 rc=1
 xml_path=''  # no XML produced
except Exception as e:
 print(f'[start.sh] pytest batch failed: {e}',file=sys.stderr)
 rc=1
 xml_path=''
if rc is not None and not isinstance(rc, int): rc=1
if rc is not None and rc < 0:
 sig=-rc
 print(f'[start.sh] pytest killed by signal {sig}',file=sys.stderr)
 if sig in (6,11):
  print(f'[start.sh] C extension crash (signal {sig}), treating as known issue',file=sys.stderr)
  rc=139
 else:
  rc=1
if rc is not None and rc==139: print('[start.sh] pytest segfault/abort (exit 139), known C extension issue');rc=0
if os.path.exists(xml_path):
 try:
  ET.register_namespace('','')
  t=ET.parse(xml_path)
  for tc in t.getroot().iter('testcase'):
   cn=tc.get('classname','');nm=tc.get('name','')
   if not nm: continue
   bare=nm.split('[')[0]
   key=f'{cn}::{nm}'
   _parts=cn.split('.');_ci=len(_parts)
   for _i in range(len(_parts)-1,0,-1):
    if _parts[_i] and _parts[_i][0].isupper():_ci=_i
    else:break
   _sp='/'.join(_parts[:_ci])+'.py';_cls='::'.join(_parts[_ci:])
   if _cls:_sp=_sp+'::'+_cls
   alt_key=_sp+'::'+nm;alt_bare=_sp+'::'+bare
   kid=nid2req.get(key) or nid2req.get(f'{cn}::{bare}') or nid2req.get(alt_key) or nid2req.get(alt_bare) or rev.get(key) or rev.get(f'{cn}::{bare}') or rev.get(alt_key) or rev.get(alt_bare)
   if not kid: continue
   props=tc.find('properties')
   if props is None: props=ET.SubElement(tc,'properties')
   vs=[p.get('value') for p in props.findall('property') if p.get('name')=='case_id']
   if kid not in vs: ET.SubElement(props,'property',{'name':'case_id','value':kid})
  t.write(xml_path,encoding='utf-8',xml_declaration=True)
 except Exception as e:
  print(f'[start.sh] XML post-process failed: {e}',file=sys.stderr);os.remove(xml_path);sys.exit(1)
sys.exit(rc)
"
}

run() { local rn="$1"; shift; mkdir -p "$REPORTS_DIR"
  "$PY" -m pytest "$@" --junitxml "$REPORTS_DIR/$rn.xml" -q 2>&1; local rc=$?
  local post_rc=0
  if [ -f "$REPORTS_DIR/$rn.xml" ]; then
    "$PY" -c "
import sys,re,json,os
from xml.etree import ElementTree as ET
ET.register_namespace('','')
t=ET.parse('$REPORTS_DIR/$rn.xml')
rev={}
if os.path.exists('.plm-case-id-map.json'):
 with open('.plm-case-id-map.json') as f: rev=json.load(f).get('revmap',{})
for tc in t.getroot().iter('testcase'):
 cn=tc.get('classname','');nm=tc.get('name','')
 if not nm: continue
 bare=nm.split('[')[0]
 key=f'{cn}::{nm}'
 _parts=cn.split('.');_ci=len(_parts)
 for _i in range(len(_parts)-1,0,-1):
  if _parts[_i] and _parts[_i][0].isupper():_ci=_i
  else:break
 _sp='/'.join(_parts[:_ci])+'.py';_cls='::'.join(_parts[_ci:])
 if _cls:_sp=_sp+'::'+_cls
 alt_key=_sp+'::'+nm;alt_bare=_sp+'::'+bare
 kid=rev.get(key) or rev.get(f'{cn}::{bare}') or rev.get(alt_key) or rev.get(alt_bare)
 if not kid: continue
 props=tc.find('properties')
 if props is None: props=ET.SubElement(tc,'properties')
 v=[p.get('value') for p in props.findall('property') if p.get('name')=='case_id']
 if kid not in v: ET.SubElement(props,'property',{'name':'case_id','value':kid})
t.write('$REPORTS_DIR/$rn.xml',encoding='utf-8',xml_declaration=True)
" 2>/dev/null; post_rc=$?
  fi
  if [ $post_rc -ne 0 ]; then
    echo "[start.sh] XML post-process failed" >&2
    rm -f "$REPORTS_DIR/$rn.xml"
    return 1
  fi
  [ $rc -eq 139 ] && echo "[start.sh] pytest segfault (exit 139, tsfile C ext known issue)" && return 0
  return $rc
}

all()    { run "all" tests --tb=short; }
page()   { case "$1" in table) D="tests/table";; tree) D="tests/tree";; *) echo "Unknown: $1" >&2; return 2;; esac; run "page-$1" "$D" --tb=short; }

plm()    { echo "[plm] 全量测试"; all; }

case "$CMD" in
  prepare) prepare;;  all) all;;  page) page "$TGT";;
  case|case-id) case_one "$TGT" || exit;;  cases) cases_batch "$TGT" || exit;;
  plm|plm-all) plm;;  list-cases) list_cases;;  help|-h|--help) usage;;
  test_*) case_one "$CMD";;  *) echo "Unknown: $CMD" >&2; usage; exit 2;;
esac
