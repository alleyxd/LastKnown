#!/bin/bash
cd "$(dirname "$0")"
export TIDEMARK_MODEL=qwen3-8b PYTHONIOENCODING=utf-8
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done
for m in noguard tidemark; do
  echo "== $m $(date +%T)"
  for i in 1 2; do .venv/Scripts/python bench/stale.py --method $m --out results/qwen3-8b-v2/$m.jsonl --workers 24 --limit ${LIMIT:-100} 2>&1 | grep -v "Loading weights"; done
done
echo "== done $(date +%T)"
