#!/bin/bash
# after the full tidemark run: baselines on all 400, ablations on the first 100 (T1/T2 interleaved)
cd "$(dirname "$0")"
export TIDEMARK_MODEL=qwen3-8b PYTHONIOENCODING=utf-8
run() { .venv/Scripts/python bench/stale.py --method $1 --out results/qwen3-8b/$1.jsonl --workers $2 ${3:+--limit $3} 2>&1 | grep -v "Loading weights"; }
for m in turns-plain turns; do echo "== $m $(date +%T)"; run $m 4; run $m 4; done
for m in noguard facts invalidate; do echo "== $m $(date +%T)"; run $m 24 100; run $m 24 100; done
echo "== done $(date +%T)"
