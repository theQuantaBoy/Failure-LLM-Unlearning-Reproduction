"""
CPU test for W6 (loss-component recorder): for each algorithm, a few steps with logging_steps=1. The recorded
forget_term + retain_term must equal the loss the Trainer logs for that step (Trainer rounds it to 4 decimals),
and exactly two forward passes of the trainable model (forget, retain) must be seen per step.
(Bitwise "no change" of the recorder is checked by test_wrapper_cpu: wrapper vs the authors' unlearn.py.)

    python -m extra.tests.test_loss_components --model_dir M --tokenizer_dir T --forget F --retain R --work_dir W
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from extra.common import write_json

CASES = [("ga_gdr", []), ("npo_klr", []), ("ga_gdr_sure", []), ("npo_klr_sure", []), ("npo_klr_sure", ["--sure_fast"])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("model_dir", "tokenizer_dir", "forget", "retain", "work_dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    if work.exists():
        shutil.rmtree(work)
    res, ok = {}, True
    for algo, extra in CASES:
        tag = algo + ("_fast" if extra else "")
        lr, alpha = (1e-4, 20) if "sure" in algo else (1e-5, 2)
        out = work / tag
        rc = subprocess.run([sys.executable, "-m", "extra.unlearn_run", "--algo", algo, "--corpus", "books",
                             "--epochs", "1", "--lr", str(lr), "--alpha", str(alpha), "--threshold", "90",
                             "--per_device_batch_size", "2", "--max_len", "128", "--model_dir", a.model_dir,
                             "--tokenizer_dir", a.tokenizer_dir, "--data_file", a.forget, "--retain_data_file",
                             a.retain, "--out_dir", str(out), "--logs_dir", str(work / "logs"), "--cpu_test",
                             "--test_mode", "--max_steps", str(a.steps), "--save_strategy", "no",
                             "--logging_steps", "1", *extra]).returncode
        assert rc == 0, f"{tag} failed"
        comps = [json.loads(l) for l in (out / "loss_components.jsonl").read_text().splitlines()]
        logged = [h["loss"] for h in json.loads((out / "unlearn_run.json").read_text())["stats"]["log_history"]
                  if "loss" in h]
        rows = []
        for c, l in zip(comps, logged):
            good = abs(round(c["total"], 4) - l) <= max(2e-4, 1e-5 * abs(l)) and c["n_model_forwards"] == 2
            rows.append({"step": c["step"], "forget_term": c.get("forget_term"), "retain_term": c.get("retain_term"),
                         "recorded_total": c["total"], "trainer_logged_loss": l, "match": good})
        passed = len(rows) == a.steps and all(r["match"] for r in rows)
        ok &= passed
        res[tag] = {"PASS": passed, "steps": rows}
        print(tag, "PASS" if passed else "FAIL", json.dumps(rows))
    write_json(res, work / "loss_components_test.json")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
