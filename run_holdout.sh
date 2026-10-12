#!/bin/bash
# v2 on the remaining 300 scenarios, then judge everything in qwen3-8b-v2
cd "$(dirname "$0")"
docker stop tidemark-judge >/dev/null; docker start tidemark-llm >/dev/null
LIMIT=400 ./run_v2.sh
docker stop tidemark-llm >/dev/null; docker start tidemark-judge >/dev/null
until curl -sf localhost:8001/v1/models >/dev/null; do sleep 10; done
export PYTHONIOENCODING=utf-8
.venv/Scripts/python bench/judge.py results/qwen3-8b-v2/noguard.jsonl results/qwen3-8b-v2/tidemark.jsonl --judge-model qwen3-14b-awq --think --workers 12
(cd bench && ../.venv/Scripts/python reason.py ../results/qwen3-8b-v2/noguard.judged-qwen3-14b-awq.jsonl ../results/qwen3-8b-v2/tidemark.judged-qwen3-14b-awq.jsonl)
docker stop tidemark-judge >/dev/null
echo "== all done $(date +%T)"
