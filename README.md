# LastKnown

[![tests](https://github.com/alleyxd/LastKnown/actions/workflows/tests.yml/badge.svg)](https://github.com/alleyxd/LastKnown/actions/workflows/tests.yml)

Long-term memory for LLM assistants that notices when something it remembers is no longer true.

Most memory layers are good at recall: ask what city the user mentioned and they find it. They are bad at the harder
case, where a later conversation quietly overturns an earlier one. The user said they live in Portland; months later
they mention shaking a scorpion out of their boot in the dry heat. Nobody said "I moved", but an assistant that still
recommends rainy-day activities in Portland is wrong.

On the [STALE benchmark](https://arxiv.org/abs/2605.06527), which tests exactly this, published memory systems (Mem0,
Zep, A-Mem, LightMem) score 5 to 18%. LastKnown scores 55% on held-out scenarios with an open 8B model (Qwen3-8B),
close to the 55.2% the paper reports for Gemini-3.1-pro reading the full history (with a different judge, so treat the
comparison as indicative). Details in [Results](#results) and the [paper](paper/lastknown.pdf).

It works with any OpenAI-compatible chat endpoint: OpenAI, vLLM, llama.cpp, Ollama, LM Studio, SGLang and others.

## Install

```bash
pip install "lastknown[local] @ git+https://github.com/alleyxd/LastKnown"
```

`[local]` adds sentence-transformers for on-device embeddings. Leave it out if you use an embeddings API.

## Quick start

```python
from lastknown import LLM, LocalEmbedder, Memory

llm = LLM(base_url="http://localhost:8000/v1", model="qwen3-8b", thinking_switch=True)   # vLLM; see below for others
mem = Memory(llm, LocalEmbedder())

mem.write(["I've been in Portland for years now, the rain doesn't even bother me anymore."], "2024-02-10")
mem.write(["Found another bark scorpion in my boot this morning. This dry heat drives them indoors."], "2024-09-03")

print(mem.context())
print(mem.ask("Since I live in Portland, what rainy-day activities should I plan?"))
mem.save("user.json")
```

```
What is true about the user now (most recent first):
- [2024-09-03] The user lives in a region with relentless dry heat. (inferred)
- [2024-09-03] The user has a toddler and a dog at home. (inferred)
...
No longer true (kept for history):
- [2024-02-10] The user lives in Portland. -> since 2024-09-03: The user lives in a region with relentless dry heat.

The assumption that you live in Portland is no longer accurate. Since 2024-09-03, ...
```

(real output from `examples/quickstart.py` with Qwen3-8B on vLLM)

## Using it in your assistant

Two calls per conversation:

```python
mem = Memory.load("user.json", llm, embed)        # or Memory(llm, embed) for a new user

# before each reply: put the memory in the system prompt
system = "You are a helpful assistant.\n\n" + mem.context()
notes = mem.premise_notes(user_message)           # optional: outdated facts the message takes for granted

# after the conversation (can run in the background): learn from what the user said
mem.write([m["content"] for m in conversation if m["role"] == "user"], date="2024-09-03")
mem.save("user.json")
```

`mem.ask(question)` does the first part for you and returns the model's answer.

## Choosing a server and model

Configure in code (`LLM(base_url=..., model=..., api_key=...)`) or with environment variables:

| Variable | Meaning |
|---|---|
| `LASTKNOWN_BASE_URL` | chat endpoint, default `http://localhost:8000/v1` |
| `LASTKNOWN_MODEL` | model name as the server knows it |
| `LASTKNOWN_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `LASTKNOWN_THINKING_SWITCH` | `1` to send `enable_thinking` per request (vLLM or SGLang serving Qwen3 or similar) |
| `LASTKNOWN_EMBED_MODEL`, `LASTKNOWN_EMBED_BASE_URL` | for `APIEmbedder` |

| Server | Example |
|---|---|
| OpenAI | `LLM(base_url="https://api.openai.com/v1", model="gpt-4o-mini")`, `APIEmbedder("text-embedding-3-small")` |
| Ollama | `LLM(base_url="http://localhost:11434/v1", model="llama3.1:8b")`, `APIEmbedder("nomic-embed-text", base_url="http://localhost:11434/v1")` |
| vLLM | `LLM(base_url="http://localhost:8000/v1", model="qwen3-8b", thinking_switch=True)` |
| llama.cpp | `LLM(base_url="http://localhost:8080/v1", model="local")` |
| LM Studio | `LLM(base_url="http://localhost:1234/v1", model="<loaded model>")` |

What the server needs, and what happens if it lacks it:

- **JSON-schema output** (`response_format`) for fact extraction. Without it, LastKnown asks for JSON in the prompt and
  parses the reply.
- **Logprobs** for the one-token conflict check. Without them it reads a short Yes/No reply instead. This works, but it
  is coarser: in our tests some changes were missed that the logprob version caught. Use a server with logprobs if
  you can (vLLM, llama.cpp, OpenAI).
- **Thinking models.** Models that think by default (Qwen3, DeepSeek-R1) should either run with
  `thinking_switch=True` on vLLM/SGLang or be served in non-thinking mode; otherwise the one-token check sees the start
  of the thinking block instead of an answer.

Results so far are for Qwen3-8B only. Larger models should extract and judge better; very small ones (under about 4B)
may miss implied facts.

## How it works

```
conversation ──> extract facts ──> conflict check ──────────> memory ──> answer
                  + topic           stage 1: "still true?"     current facts
                  + implied facts      one token, no generation  + "no longer true" list
                                    stage 2: short reasoning,
                                       flagged pairs only
```

1. **Only the user's own words.** On STALE, 90% of a conversation is the assistant's replies, which say nothing about
   the user. Reading only user turns keeps an 8B model within its context.
2. **Implied facts.** Each conversation becomes short facts with a topic (home, work, religion, routine, ...). The
   model also writes down what the messages imply but never say: scorpions and dry heat give "the user lives in a hot
   desert region". This turns an implicit conflict into an explicit one that a cheap check can catch.
3. **Two-stage conflict check.** Each new fact is compared with the most similar older facts. Stage 1 reads the answer
   to "is the earlier note still true?" from the next-token probabilities. Only flagged pairs go to stage 2, where the
   model reasons briefly before deciding. Stage 1 alone was badly calibrated: depending on the wording it flagged
   almost everything or almost nothing. Passing moments ("on a train right now") simply expire when a newer one
   arrives.
4. **The whole current state in the prompt.** A user's current state is small enough to always include in full,
   together with what is no longer true, so no retrieval step can miss the fact a question depends on.
5. **Premise guard.** Before answering, outdated facts close to the question are checked: does the question assume
   this? If so, the model is told what changed.

What it remembers is the user's situation: where they live, work, family, health, diet, beliefs, routines, plans. It
does not store what was said in a conversation (for that, use retrieval over the transcript) and it does not make the
model better at anything else.

## Results

STALE has 400 scenarios, each with three questions: is the old fact still true (state), a question built on the
outdated fact (false premise), and a task that silently depends on the new state (implicit). T1 conflicts are
stated, T2 have to be inferred. Everything below uses Qwen3-8B (FP8, about 16 GB of GPU memory).

Design changes were made while looking at the first 100 scenarios, so the main table is the other 300:

| Held-out 300 scenarios | State | False premise | Implicit | **STALE overall** | Right reason | Keeps new fact |
|---|---|---|---|---|---|---|
| All user messages, benchmark prompt | 6.7 | 0.7 | 19.3 | 8.9 | 2.5 | 93% |
| All user messages, our answer prompt | 15.7 | 1.0 | 21.0 | 12.6 | 6.5 | 97% |
| LastKnown v1 | 66.0 | 41.7 | 34.0 | 47.2 | 38.4 | 53% |
| LastKnown, no premise guard | 56.0 | 21.3 | 39.3 | 38.9 | 30.3 | 84% |
| **LastKnown** | **63.7** | **57.3** | **44.0** | **55.0** | **48.7** | 65% |

Each column averages T1 and T2. Over all 400 scenarios LastKnown scores 58.2 (`results/table_final.txt`).

For reference, the STALE paper reports Qwen3.5-9B with the full history at 14.6, Gemini-3.1-pro at 55.2, published
memory systems at 5.1 to 17.8, and its own CUPMem pipeline (GPT-4o-mini, about ten prompt types per session) at 68.0.
Those numbers use a Gemini judge; ours are graded by Qwen3-14B-AWQ with the benchmark's judge prompt unchanged, so the
comparison is indicative, not exact. Our baseline lands where the paper's similar-sized model does.

Two columns are not part of STALE, and both matter:

- **Keeps new fact.** STALE never checks that a still-true fact survives, so a memory that forgets everything scores
  well. We also ask whether the newer fact is still the case. The premise guard buys 36 points on false-premise
  questions and costs 19 points here; without it LastKnown keeps 84%.
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

Cost per scenario (50 conversations): 50 extraction calls, about 450 one-token checks and about 170 short reasoning
calls, all of which can run after the conversation. At answer time the memory adds about 13K tokens of context
(22K for the raw user messages) plus up to five one-token checks for the premise guard.

### Limitations

- One answer model (Qwen3-8B) and one judge (Qwen3-14B). The judge was spot-checked by hand on 12 scenarios and
  applied the rubric faithfully; the right-reason check is a second LLM judgment, not a human one.
- The retention question reuses the scenario's new-state message; when that message describes a moment ("laying out
  my yoga mat") a "no longer the case" answer can be fair, so 100% is not the ceiling.
- STALE's filler sessions come from many different people, so some "wrong" overturns in the linkage numbers are
  arguably correct readings of contradictory filler.
- The memory goes into the prompt in full. That is fine for dozens of conversations, but for years of history it
  should be trimmed to lasting facts.

## Reproducing the benchmark

```bash
git clone https://github.com/alleyxd/LastKnown && cd LastKnown
uv venv && uv pip install -e ".[dev]"
git clone https://github.com/icedreamc/STALE ref/STALE          # judge prompt
curl -L --create-dirs -o data/stale.json \
  https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json

docker run -d --gpus all -p 8000:8000 --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B --served-model-name qwen3-8b --quantization fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.5 --max-model-len 32768 --enable-prefix-caching

LASTKNOWN_MODEL=qwen3-8b bash bench/run_stale.sh      # about 10 GPU hours on an RTX 5090
```

`bench/run_stale.sh` lists the judging steps. `bench/conflict_dev.py` holds the hand-written fact pairs (not from any
benchmark) used to choose the conflict-check wording. `results/` has every judged answer behind the tables.

## Tests

```bash
pytest -q
```

## Citation

```bibtex
@misc{ashter2026lastknown,
  title  = {Implied Facts and Cheap Conflict Checks: Keeping Long-Term Memory Current with a Small Local Model},
  author = {Ashter, Ali},
  year   = {2026},
  url    = {https://github.com/alleyxd/LastKnown}
}
```

## License

MIT. STALE data is CC BY 4.0, its code MIT.
