#!/bin/bash
# Reproduce the STALE tables: answers from the model under test, then the judge.
# Needs: the model server on :8000 (TIDEMARK_MODEL), later the judge on :8001, data/stale.json, ref/STALE.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 TIDEMARK_THINKING_SWITCH=${TIDEMARK_THINKING_SWITCH:-1}
out=results/${TIDEMARK_MODEL:?set TIDEMARK_MODEL}

for m in turns-plain turns tidemark noguard; do
  python bench/stale.py --method $m --out $out/$m.jsonl --workers ${WORKERS:-24}
done
for m in facts invalidate; do            # ablations, first 100 scenarios
  python bench/stale.py --method $m --out $out/$m.jsonl --workers ${WORKERS:-24} --limit 100
done

# switch the GPU to the judge (any OpenAI-compatible endpoint), then:
#   python bench/judge.py $out/*.jsonl --judge-model qwen3-14b-awq --base-url http://localhost:8001/v1 --think
#   (cd bench && python reason.py ../$out/*.judged-*.jsonl)
#   python bench/linkage.py $out/tidemark.jsonl
