"""
CPU test that the unlearning wrapper (extra/unlearn_run.py W1, W2) changes nothing, and that per-epoch
weight-only saving (save_only_model) does not touch the RNG or the result.

For each algorithm, from the same tiny target model and tiny data:
  authors   baselines/unlearn.py executed as __main__ (only iterative.device_count patched: no CUDA on CPU),
            i.e. the authors' save_strategy='epoch' with optimizer + RNG state in every checkpoint
  wrapper   python -m extra.unlearn_run --cpu_test (save_only_model=True, seed 42)
  nosave    wrapper with --test_mode --save_strategy no
Checks (bitwise):  final(authors) == final(wrapper) == final(nosave);  every checkpoint-<step> of wrapper equals
the authors' checkpoint-<step> weights;  wrapper checkpoints contain no optimizer/scheduler/RNG files;
RNG fingerprint at train end: wrapper == nosave.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from extra.common import REPO_DIR, write_json
from extra.tests.test_sure_equivalence import diff

AUTHORS_RUNNER = """
import runpy, sys
sys.path.insert(0, {base!r})
import baselines.iterative as it
it.device_count = lambda: 1          # CPU only: iterative.py:51-52 raises without CUDA
sys.argv = ['unlearn.py'] + {args!r}
runpy.run_path({script!r}, run_name='__main__')
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--algos", nargs="+", default=["ga_gdr", "npo_klr", "ga_gdr_sure", "npo_klr_sure"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    results, ok = {}, True
    for algo in a.algos:
        lr, alpha, thr = (1e-4, 20, 90) if "sure" in algo else (1e-5, 2, 90)
        hp = ["--algo", algo, "--model_dir", a.model_dir, "--tokenizer_dir", a.tokenizer_dir, "--data_file", a.forget,
              "--retain_data_file", a.retain, "--max_len", str(a.max_len), "--epochs", str(a.epochs), "--lr", str(lr),
              "--alpha", str(alpha), "--threshold", str(thr), "--per_device_batch_size", "2"]
        d = {k: work / algo / k for k in ("authors", "wrapper", "nosave")}
        for p in d.values():
            if p.exists():
                shutil.rmtree(p)
        runner = AUTHORS_RUNNER.format(base=str(REPO_DIR / "baselines"), args=hp + ["--out_dir", str(d["authors"])],
                                       script=str(REPO_DIR / "baselines" / "unlearn.py"))
        rc = subprocess.run([sys.executable, "-c", runner], cwd=REPO_DIR / "baselines").returncode
        assert rc == 0, f"authors' unlearn.py failed for {algo}"
        wrap = [sys.executable, "-m", "extra.unlearn_run", *hp, "--corpus", "books", "--cpu_test",
                "--logs_dir", str(work / "logs")]
        assert subprocess.run(wrap + ["--out_dir", str(d["wrapper"])]).returncode == 0
        assert subprocess.run(wrap + ["--out_dir", str(d["nosave"]), "--test_mode", "--save_strategy", "no"]).returncode == 0

        r = {"final_authors_vs_wrapper": diff(d["authors"], d["wrapper"]),
             "final_wrapper_vs_nosave": diff(d["wrapper"], d["nosave"])}
        ck_w = sorted(p.name for p in d["wrapper"].glob("checkpoint-*"))
        ck_a = sorted(p.name for p in d["authors"].glob("checkpoint-*"))
        r["checkpoints"] = ck_w
        r["checkpoints_match_authors_names"] = ck_w == ck_a
        r["checkpoint_diffs"] = {c: diff(d["authors"] / c, d["wrapper"] / c)["elements_differing"] for c in ck_w}
        r["wrapper_checkpoint_files"] = sorted({f.name for c in ck_w for f in (d["wrapper"] / c).iterdir()})
        r["authors_checkpoint_files"] = sorted({f.name for c in ck_a for f in (d["authors"] / c).iterdir()})
        forbidden = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
        r["wrapper_has_no_optimizer_or_rng"] = not (forbidden & set(r["wrapper_checkpoint_files"]))
        sw = json.loads((d["wrapper"] / "unlearn_run.json").read_text())["stats"]
        sn = json.loads((d["nosave"] / "unlearn_run.json").read_text())["stats"]
        r["rng_end_wrapper_eq_nosave"] = sw["rng_at_train_end"] == sn["rng_at_train_end"]
        r["sure_optimizer_step_calls"] = json.loads((d["wrapper"] / "unlearn_run.json").read_text())["counters"]
        passed = (r["final_authors_vs_wrapper"]["elements_differing"] == 0
                  and r["final_wrapper_vs_nosave"]["elements_differing"] == 0
                  and r["checkpoints_match_authors_names"] and not any(r["checkpoint_diffs"].values())
                  and r["wrapper_has_no_optimizer_or_rng"] and r["rng_end_wrapper_eq_nosave"] and len(ck_w) == a.epochs)
        r["PASS"] = passed
        ok &= passed
        results[algo] = r
        print(algo, "PASS" if passed else "FAIL", json.dumps(r, indent=1))
    write_json(results, work / "wrapper_test.json")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
