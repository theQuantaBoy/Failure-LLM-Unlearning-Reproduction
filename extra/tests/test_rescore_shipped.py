"""
Re-score the authors' shipped per-example logs (FailureLLMUnlearning/temp/<name>/..., BOOKS) and check that
  * metrics/logger.py RougeEvalLogger on their (gt, response) pairs reproduces their agg.json mean_rougeL, and
  * compare_results._auc on their per-example Min-40% scores reproduces their auc.json forget_holdout_Min-40%.
This validates our scoring / dev-held-out recomputation path against the authors' own numbers. CPU, seconds.
    python -m extra.tests.test_rescore_shipped
"""

import json
import sys

from extra.common import REPO_DIR


def main() -> int:
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR.parent))
    from compare_results import _auc
    from metrics.logger import RougeEvalLogger

    ok = True
    for d in sorted((REPO_DIR / "temp").iterdir()):
        for m in ("verbmem_f", "knowmem_f", "knowmem_r"):
            if not (d / m / "log.json").exists():
                continue
            log = json.loads((d / m / "log.json").read_text())
            agg = json.loads((d / m / "agg.json").read_text())
            lg = RougeEvalLogger()
            for r in log:
                lg.log(r["prompt"], r["gt"], r["response"], question=r.get("question"))
            mine = sum(h["rougeL"] for h in lg.history) / len(lg.history)
            good = abs(mine - agg["mean_rougeL"]) < 1e-12
            ok &= good
            print(f"{d.name:28s} {m:10s} theirs {agg['mean_rougeL']:.6f} ours {mine:.6f} {'OK' if good else 'DIFF'}")
        p = d / "privleak"
        if (p / "log.json").exists():
            log = json.loads((p / "log.json").read_text())
            auc = json.loads((p / "auc.json").read_text())["forget_holdout_Min-40%"]
            mine = _auc([r["Min-40%"] for r in log["forget"]], [r["Min-40%"] for r in log["holdout"]])
            good = abs(mine - auc) < 1e-12
            ok &= good
            print(f"{d.name:28s} privleak   theirs {auc:.6f} ours {mine:.6f} {'OK' if good else 'DIFF'}")
    print("ALL OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
