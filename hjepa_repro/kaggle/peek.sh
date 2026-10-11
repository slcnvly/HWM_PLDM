#!/bin/bash
# usage: peek.sh <slug> ...   -> status + key log lines of each kernel (live stream snapshot while running)
K=~/.venvs/kaggle/bin/kaggle
for k in "$@"; do
  st=$($K kernels status seungwonryoo/$k 2>&1 | tail -1); echo "$st"
  if echo "$st" | grep -q RUNNING; then raw=$(timeout 40 $K kernels logs -f seungwonryoo/$k 2>&1); else raw=$($K kernels logs seungwonryoo/$k 2>&1); fi
  echo "$raw" | python3 -c "
import sys,json,re
raw=sys.stdin.read()
try: txt=''.join(x.get('data','') for x in json.loads(raw))
except Exception: txt=raw
L=txt.split('\n'); print('  loglen',len(txt))
print('\n'.join('  '+l[:300] for l in L if re.search(r'START|DONE|FAIL|PROGRESS|tasks=|WALL|STEP|FINAL|restore', l))[-3500:])"
done
