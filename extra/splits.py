"""
extra/splits.py — fixed dev / held-out split of every evaluation set (Task 4, Table 4b).

Each set's example indices are permuted with random.Random(SEED) and the first half (floor) becomes "dev", the
rest "heldout". The split is computed once and written to extra/splits.json together with the sha1 of every
example, so a changed data file is detected. Evaluations always run on the full sets; compare_results.py computes
dev / held-out metrics from the saved per-example scores.

  python -m extra.splits            # (re)writes extra/splits.json
"""

import json
import random
import sys

from extra.common import EXTRA_DIR, REPO_DIR, sha1_text, write_json

SEED = 0
OUT = EXTRA_DIR / "splits.json"


def _sets():
    u = REPO_DIR / "LLama_factory" / "data" / "utility"
    load = lambda p: json.loads(p.read_text())  # noqa: E731
    sets = {
        "mmlu": [d["question"] for d in load(u / "retain_mmlu.json")],
        "truthful": [d["question"] for d in load(u / "truthful.json")],
        "triviaqa": [d["question"] for d in load(u / "triviaqa.json")],
        "fluency": [d["instruction"] for d in load(u / "fluency.json")],
    }
    for c in ("news", "books"):
        d = REPO_DIR / "data" / c
        sets[f"{c}/verbmem"] = [x["prompt"] + x["gt"] for x in load(d / "verbmem" / "forget.json")]
        sets[f"{c}/knowmem_f"] = [x["question"] + x["answer"] for x in load(d / "knowmem" / "forget_qa.json")]
        sets[f"{c}/knowmem_r"] = [x["question"] + x["answer"] for x in load(d / "knowmem" / "retain_qa.json")]
        for s in ("forget", "retain", "holdout"):
            sets[f"{c}/privleak_{s}"] = load(d / "privleak" / f"{s}.json")
    return sets


def make():
    out = {"seed": SEED, "rule": "random.Random(seed).shuffle(indices); dev = first floor(n/2)", "sets": {}}
    for name, texts in _sets().items():
        idx = list(range(len(texts)))
        random.Random(f"{SEED}:{name}").shuffle(idx)
        h = len(idx) // 2
        out["sets"][name] = {"n": len(idx), "dev": sorted(idx[:h]), "heldout": sorted(idx[h:]),
                             "sha1": [sha1_text(t) for t in texts]}
    return out


def load():
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    write_json(make(), OUT)
    print(f"wrote {OUT}")
    sys.exit(0)
