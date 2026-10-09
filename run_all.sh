#!/bin/bash
# generate answers for every method on all 400 STALE scenarios with the local model
cd "$(dirname "$0")"
export TIDEMARK_MODEL=qwen3-8b PYTHONIOENCODING=utf-8
for m in tidemark noguard turns-plain turns facts invalidate; do
  w=24; [[ $m == turns* ]] && w=4
  echo "== $m $(date +%T)"
  .venv/Scripts/python bench/stale.py --method $m --out results/qwen3-8b/$m.jsonl --workers $w 2>&1 | grep -v "Loading weights"
  .venv/Scripts/python bench/stale.py --method $m --out results/qwen3-8b/$m.jsonl --workers $w 2>&1 | grep -v "Loading weights"
done
echo "== done $(date +%T)"
