"""Add the stricter right-reason verdicts to already judged files (dim1/dim2 passes only).

python bench/reason.py results/qwen3-8b/*.judged-qwen3-14b-awq.jsonl
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from judge import HERE, REASON_JUDGE, strip_think


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged", nargs="+", type=Path)
    ap.add_argument("--judge-model", default="qwen3-14b-awq")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    items = {x["uid"]: x for x in json.loads((HERE / "data" / "stale.json").read_text(encoding="utf-8"))}
    client = OpenAI(base_url=args.base_url, api_key="none", timeout=600)

    for path in args.judged:
        answers = {r["uid"]: r for r in map(json.loads, Path(str(path).split(".judged-")[0] + ".jsonl")
                                            .read_text(encoding="utf-8").splitlines())}
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

        def work(row):
            item, a = items[row["uid"]], answers[row["uid"]]["target_model_responses"]
            for d in ("dim1", "dim2"):
                if f"{d}_reason" in row["pass"]:
                    continue
                if not row["pass"][d]:
                    row["pass"][f"{d}_reason"] = False
                    continue
                r = client.chat.completions.create(
                    model=args.judge_model, temperature=0, max_tokens=512,
                    messages=[{"role": "user", "content": REASON_JUDGE.format(
                        old=item["M_old"], new=item["M_new"], logic=item["explanation"],
                        question=item["probing_queries"][f"{d}_query"], answer=strip_think(a[f"{d}_response"]))}],
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                row["pass"][f"{d}_reason"] = bool(json.loads(strip_think(r.choices[0].message.content))["pass"])
            return row

        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(work, rows))
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(path.name, "done", flush=True)


if __name__ == "__main__":
    main()
