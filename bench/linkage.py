"""How often the memory itself caught the change, without any judge.

For each scenario: did a fact from the session with the old state get overturned by a fact from the session with the
new state (right), only by something unrelated (wrong), or not at all (missed)?

python bench/linkage.py results/qwen3-8b/lastknown.jsonl results/qwen3-8b-v2/lastknown.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main(paths):
    items = {x["uid"]: x for x in json.loads((HERE / "data" / "stale.json").read_text(encoding="utf-8"))}
    print(f"{'file':40}{'right':>8}{'wrong':>8}{'missed':>8}{'kept':>8}     n")
    for path in paths:
        c, n = Counter(), 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "ledger" not in row:
                break
            x, facts = items[row["uid"]], row["ledger"]
            old_date, new_date = (x["timestamps"][i] for i in x["relevant_session_index"])
            gone = [f for f in facts if f["date"] == old_date and f["replaced_by"] is not None
                    and not f.get("expired")]
            right = any(facts[f["replaced_by"]]["date"] == new_date for f in gone)
            c["right" if right else "wrong" if gone else "missed"] += 1
            new_facts = [f for f in facts if f["date"] == new_date and f["topic"] != "request"]
            c["kept"] += bool(new_facts) and all(f["replaced_by"] is None or f.get("expired") for f in new_facts)
            n += 1
        if n:
            print(f"{Path(path).parent.name + '/' + Path(path).stem:40}"
                  + "".join(f"{100 * c[k] / n:8.1f}" for k in ("right", "wrong", "missed", "kept")) + f"{n:6d}")


if __name__ == "__main__":
    main(sys.argv[1:])
