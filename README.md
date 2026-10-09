# tidemark

Long-term memory for local LLM assistants that notices when something it remembers is no longer true.

Most memory layers are good at recall: ask what city the user mentioned and they find it. They are bad at the harder
case, where a later conversation quietly overturns an earlier one. The user said they live in Portland; months later
they mention shaking a scorpion out of their boot in the dry heat. Nobody said "I moved", but an assistant that still
recommends rainy-day activities in Portland is wrong.

The [STALE benchmark](https://arxiv.org/abs/2605.06527) (2026) measures exactly this, and published memory systems
(Mem0, Zep, A-Mem, LightMem) score 5 to 18% on it. tidemark is a small memory layer built around that failure, running
entirely on a local 8B model.

## How it works

```
session ──> extract facts ──> conflict check ──> ledger ──> answer
             + topic           stage 1: one-token     current facts
             + implied facts   "still true?"          + "no longer true" list
                               stage 2: short
                               reasoning on flagged
```

1. **Only the user's own words.** On STALE, 90% of a conversation is the assistant's replies, which say nothing about
   the user. Reading only user turns keeps everything about a user's life within an 8B model's context.
2. **Implied facts.** Each session is turned into short facts with a topic (home, work, religion, routine, ...). The
   model also writes down what the messages imply but never say: scorpions and dry heat give "the user lives in a hot
   desert region". This turns an implicit conflict into an explicit one that a cheap check can catch.
3. **Two-stage conflict check.** A new fact is compared with the most similar older facts. Stage 1 reads the answer to
   "is the earlier note still true?" from the next-token probabilities (one forward pass, no generation). Only flagged
   pairs go to stage 2, where the model reasons briefly before deciding. Stage 1 alone was badly calibrated: depending
   on the wording it flagged almost everything or almost nothing.
4. **The whole current state in the prompt.** A user's current state is small, so it always goes in full, with a list
   of facts that are no longer true. No retrieval step can miss the fact a question depends on.

```python
from tidemark import LLM, Memory, answer

mem = Memory(LLM(), embed)              # any OpenAI-compatible endpoint, any sentence embedder
mem.write(["I've been based in Seattle for the last few years."], "2023-01-10")
mem.write(["Just finished setting up utilities at my new place in Austin."], "2024-03-02")
print(mem.ledger.render())
```

## Results

(filled in from the final runs)

## Running the benchmark

```bash
uv venv && uv pip install -e ".[dev]"
git clone https://github.com/icedreamc/STALE ref/STALE                    # judge prompt and CUPMem reference
curl -L -o data/stale.json https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json

docker run -d --gpus all -p 8000:8000 --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B --served-model-name qwen3-8b --quantization fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.5 --max-model-len 32768 --enable-prefix-caching

TIDEMARK_MODEL=qwen3-8b python bench/stale.py --method tidemark --out results/qwen3-8b/tidemark.jsonl
python bench/judge.py results/qwen3-8b/tidemark.jsonl --judge-model qwen3-14b-awq --base-url http://localhost:8001/v1 --think
python bench/linkage.py results/qwen3-8b/tidemark.jsonl
```

`bench/conflict_dev.py` holds the hand-written fact pairs (not from any benchmark) used to choose the conflict-check
wording.

## License

MIT. STALE data is CC BY 4.0, its code MIT.
