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

STALE has 400 scenarios, each with three questions: is the old fact still true (state), a question built on the
outdated fact (false premise), and a task that silently depends on the new state (implicit). T1 conflicts are
stated, T2 have to be inferred. Everything below runs Qwen3-8B (FP8) locally on half of one RTX 5090.

Design changes were made while looking at the first 100 scenarios, so the main table is the other 300:

| Held-out 300 scenarios | State | False premise | Implicit | **STALE overall** | Right reason | Keeps new fact |
|---|---|---|---|---|---|---|
| All user messages, benchmark prompt | 6.7 | 0.7 | 19.3 | 8.9 | 2.5 | 93% |
| All user messages, our answer prompt | 15.7 | 1.0 | 21.0 | 12.6 | 6.5 | 97% |
| tidemark v1 | 66.0 | 41.7 | 34.0 | 47.2 | 38.4 | 53% |
| tidemark, no premise guard | 56.0 | 21.3 | 39.3 | 38.9 | 30.3 | 84% |
| **tidemark** | **63.7** | **57.3** | **44.0** | **55.0** | **48.7** | 65% |

Each column averages T1 and T2. Over all 400 scenarios tidemark scores 58.2 (`results/table_final.txt`).

For reference, the STALE paper reports Qwen3.5-9B with the full history at 14.6, Gemini-3.1-pro at 55.2, published
memory systems at 5.1 to 17.8, and its own CUPMem pipeline (GPT-4o-mini, about ten prompt types per session) at 68.0.
Those numbers use a Gemini judge; ours are graded by Qwen3-14B-AWQ with the benchmark's judge prompt unchanged, so the
comparison is indicative, not exact. Our baseline lands where the paper's similar-sized model does.

Two columns are not part of STALE, and both matter:

- **Keeps new fact.** STALE never checks that a still-true fact survives, so a memory that forgets everything scores
  well. We also ask whether the newer fact is still the case. The premise guard buys 36 points on false-premise
  questions and costs 19 points here; without it tidemark keeps 84%.
- **Right reason.** STALE's judge passes "the user may no longer live there" whatever the reason given. This column
  (state and false-premise questions) only counts answers that point to the actual change.

What the memory itself caught, with no judge involved (`bench/linkage.py`, all 400): the session with the old state
was correctly overturned by the session with the new one in 51% of scenarios (v1: 22.5%), overturned only by
something unrelated in 28%, and missed in 21%.

What each part adds, on the first 100 scenarios (v1 pipeline):

| | STALE overall | Keeps new fact |
|---|---|---|
| All user messages, our answer prompt | 11.3 | 97% |
| + extracted facts | 18.0 | 70% |
| + two-stage conflict check | 31.3 | 54% |
| + implied facts | 48.7 | 84% |
| + premise guard | 57.3 | 48% |

Cost per scenario (50 sessions): 50 extraction calls, about 450 one-token checks, and about 170 short reasoning calls
for flagged pairs. The memory it keeps is about 13K tokens against 22K for the raw user messages.

### Limitations

- One answer model (Qwen3-8B) and one judge (Qwen3-14B). The judge was spot-checked by hand on 12 scenarios and
  applied the rubric faithfully; the right-reason check is a second LLM judgment, not a human one.
- The retention question reuses the scenario's new-state message; when that message describes a moment ("laying out
  my yoga mat") a "no longer the case" answer can be fair, so 100% is not the ceiling.
- STALE's filler sessions come from many different people, so some "wrong" overturns in the linkage numbers are
  arguably correct readings of contradictory filler.

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
