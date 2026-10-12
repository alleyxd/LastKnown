"""Run a memory method on the STALE benchmark and save the three answers per scenario.

TIDEMARK_MODEL=qwen3-8b TIDEMARK_THINKING_SWITCH=1 python bench/stale.py --method tidemark --out results/qwen3-8b/tidemark.jsonl

methods:
  turns-plain  all user messages with dates, the benchmark's neutral prompt
  turns        all user messages with dates, our answer prompt
  facts        extracted facts with dates, no invalidation, no implied facts
  invalidate   facts + conflict check, no implied facts
  noguard      facts + implied facts + conflict check
  tidemark     noguard + premise guard (warns the model when a question assumes an outdated fact)
"""
import argparse
import os
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tidemark import LLM, LocalEmbedder, Memory, answer  # noqa: E402
from tidemark.memory import clip  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DIMS = ["dim1_query", "dim2_query", "dim3_query", "keep_query"]
# not part of STALE: checks that the newer fact is still treated as true, so forgetting everything is not rewarded
KEEP = 'Earlier I told you: "{new}" Is that still the case for me now, as far as you know?'


PLAIN = {
    "dim3_query": ("You are a helpful assistant. Review the following conversation history with the user, then respond "
                   "to the user's latest query directly.", "[Latest Query]"),
    "other": ("You are a helpful assistant. Review the following conversation history with the user, then accurately "
              "answer the question.", "[Question]"),
}


class Cache:
    """Extraction results keyed by prompt, so repeated distractor sessions are only processed once."""

    def __init__(self, path):
        self.path, self.lock, self.data = Path(path), threading.Lock(), {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                k, v = json.loads(line)
                self.data[k] = v

    def wrap(self, llm):
        chat = llm.chat

        def cached(messages, **kw):
            key = hashlib.sha1(json.dumps([messages, kw], sort_keys=True).encode()).hexdigest()
            if key in self.data:
                return self.data[key]
            out = chat(messages, **kw)
            with self.lock:
                self.data[key] = out
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps([key, out]) + "\n")
            return out
        return cached


def sessions(item):
    rows = sorted(zip(item["timestamps"], item["haystack_session"]), key=lambda r: r[0])
    for date, turns in rows:
        yield date, [m["content"] for m in turns if m["role"] == "user"]


def turns_text(item):
    lines = ["Messages the user sent in earlier conversations (oldest first):"]
    for date, msgs in sessions(item):
        lines += [f"[{date}] {clip(m)}" for m in msgs]
    return "\n".join(lines)


def query(item, d):
    return KEEP.format(new=item["M_new"]) if d == "keep_query" else item["probing_queries"][d]


def run_item(item, method, llm, embed, think):
    today = max(item["timestamps"])[:10]
    stats = {}
    if method == "turns-plain":
        history = turns_text(item)
        out = {}
        for d in DIMS:
            sys_prompt, tag = PLAIN["dim3_query" if d == "dim3_query" else "other"]
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"[Conversation History]\n{history}\n\n{tag}\n{query(item, d)}"}]
            out[d] = llm.chat(msgs, max_tokens=4096 if think else 1024, think=think)
        return out, stats
    if method == "turns":
        memory_text = turns_text(item)
    else:
        mem = Memory(llm, embed, implied=method in ("noguard", "tidemark"), invalidate=method != "facts")
        # extraction does not depend on memory state, so warm the cache for all sessions at once
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda s: mem.extract(s[1], s[0]), [s for s in sessions(item) if s[1]]))
        for date, msgs in sessions(item):
            if msgs:
                mem.write(msgs, date)
        memory_text = mem.ledger.render()
        stats = {"facts": len(mem.ledger.facts), "current": len(mem.ledger.current()), "checks": mem.checks, "verified": mem.verified,
                 "ledger": mem.ledger.to_json()}
    out = {}
    for d in DIMS:
        q = query(item, d)
        notes = mem.premise_notes(q, threshold=float(os.environ.get("TIDEMARK_GUARD", "0.5"))) if method == "tidemark" else []
        if notes:
            stats.setdefault("notes", {})[d] = notes
        out[d] = answer(llm, memory_text, q, today, notes=notes, think=think)
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["turns-plain", "turns", "facts", "invalidate", "noguard", "tidemark"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", default=HERE / "data" / "stale.json", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    items = json.loads(args.data.read_text(encoding="utf-8"))
    # interleave T1 and T2 so a --limit run covers both types
    t1, t2 = [x for x in items if x["type"] == "T1"], [x for x in items if x["type"] == "T2"]
    items = [x for pair in zip(t1, t2) for x in pair][: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["uid"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [x for x in items if x["uid"] not in done]
    print(f"{len(todo)} to run, {len(done)} done", flush=True)

    llm = LLM()
    llm.chat = Cache(HERE / "results" / "cache" / f"{llm.model}.jsonl").wrap(llm)
    embed = LocalEmbedder() if args.method not in ("turns", "turns-plain") else None
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_item, x, args.method, llm, embed, args.think): x for x in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            x = futs[fut]
            try:
                out, stats = fut.result()
            except Exception as e:  # keep going, the item is retried on the next run
                print(f"{x['uid']}: {type(e).__name__}: {e}", flush=True)
                continue
            row = {"uid": x["uid"], "type": x["type"], "method": args.method, "model": llm.model,
                   "target_model_responses": {d.replace("_query", "_response"): out[d] for d in DIMS}, **stats}
            with lock, args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
