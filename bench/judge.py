"""Score saved answers with the STALE judge prompt (unchanged from the benchmark) and print the six-cell table.

python bench/judge.py results/qwen3-8b/*.jsonl --judge-model qwen3.6-35b-a3b --base-url http://localhost:8001/v1
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "ref" / "STALE" / "STALE" / "Evaluation"))
from judge_prompts import SYSTEM_PROMPT_ALL_IN_ONE_JUDGE  # noqa: E402

USER = """
[Ground Truth Context]
- M_old: "{old}"
- M_new: "{new}"
- Hidden Logic: {logic}

--------------------------------------------------
[Dimension 1: Explicit Probing]
Question 1: {q1}
Target Model Response 1: {a1}

--------------------------------------------------
[Dimension 2: Adversarial Robustness]
Question 2: {q2}
Target Model Response 2: {a2}

--------------------------------------------------
[Dimension 3: Implicit Task]
Question 3: {q3}
Target Model Response 3: {a3}
"""


# our own control check, not part of STALE
KEEP_JUDGE = """The user earlier said: "{new}"
Nothing later in the conversation contradicts this. The user asked the assistant whether it is still the case.

Assistant's answer: {answer}

Does the answer treat the statement as still true (or most likely still true)? Reply with JSON {{"pass": true}} or
{{"pass": false}}."""


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def judge(client, model, item, row, think):
    q, a = item["probing_queries"], row["target_model_responses"]
    prompt = USER.format(old=item["M_old"], new=item["M_new"], logic=item["explanation"],
                         q1=q["dim1_query"], a1=strip_think(a["dim1_response"]),
                         q2=q["dim2_query"], a2=strip_think(a["dim2_response"]),
                         q3=q["dim3_query"], a3=strip_think(a["dim3_response"]))
    for _ in range(3):
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=8192 if think else 2048,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_ALL_IN_ONE_JUDGE}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, extra_body={"chat_template_kwargs": {"enable_thinking": think}})
        try:
            out = json.loads(strip_think(r.choices[0].message.content))
            verdict = {d: bool(out[f"{d}_eval"]["pass"]) for d in ("dim1", "dim2", "dim3")}
            if "keep_response" in a:
                k = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=2048,
                    messages=[{"role": "user", "content": KEEP_JUDGE.format(
                        new=item["M_new"], answer=strip_think(a["keep_response"]))}],
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                verdict["keep"] = bool(json.loads(strip_think(k.choices[0].message.content))["pass"])
            return verdict, out
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    raise RuntimeError(f"judge failed for {row['uid']}")


def table(rows):
    cells = {}
    for t in ("T1", "T2"):
        sub = [r for r in rows if r["type"] == t]
        for d, name in (("dim1", "SR"), ("dim2", "PR"), ("dim3", "IPA")):
            cells[f"{t}-{name}"] = 100 * sum(r["pass"][d] for r in sub) / len(sub) if sub else float("nan")
    cells["Overall"] = sum(cells.values()) / 6
    keep = [r["pass"]["keep"] for r in rows if "keep" in r["pass"]]
    cells["Keep"] = 100 * sum(keep) / len(keep) if keep else float("nan")
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="+", type=Path)
    ap.add_argument("--judge-model", default="qwen3.6-35b-a3b")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--data", default=HERE / "data" / "stale.json", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    items = {x["uid"]: x for x in json.loads(args.data.read_text(encoding="utf-8"))}
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=900)
    tag = re.sub(r"[^\w.-]", "_", args.judge_model)
    print(f"{'method':32}" + "".join(f"{c:>9}" for c in ["T1-SR", "T1-PR", "T1-IPA", "T2-SR", "T2-PR", "T2-IPA",
                                                          "Overall", "Keep"]) + "      n")
    for path in args.answers:
        out_path = path.with_suffix(f".judged-{tag}.jsonl")
        judged = {}
        if out_path.exists():
            judged = {r["uid"]: r for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())}
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        todo = [r for r in rows if r["uid"] not in judged]

        def work(row):
            verdict, raw = judge(client, args.judge_model, items[row["uid"]], row, args.think)
            return {"uid": row["uid"], "type": row["type"], "pass": verdict, "judge": raw}

        with ThreadPoolExecutor(args.workers) as pool, out_path.open("a", encoding="utf-8") as f:
            for res in pool.map(work, todo):
                judged[res["uid"]] = res
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        got = [judged[r["uid"]] for r in rows if r["uid"] in judged]
        c = table(got)
        print(f"{path.stem:32}" + "".join(f"{v:9.1f}" for v in c.values()) + f"{len(got):7d}")


if __name__ == "__main__":
    main()
