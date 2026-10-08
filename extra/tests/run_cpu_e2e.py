"""
End-to-end CPU test of the whole pipeline with a tiny random Llama and tiny data subsets (Phase 2, step 4a).
No GPU, no Modal, no large downloads (tokenizer files ~2 MB, wikitext-2 ~ 10 MB).

    python3 extra/tests/run_cpu_e2e.py            # uses .venvs/paper and .venvs/quant (see COMMANDS.md §0)
    python3 extra/tests/run_cpu_e2e.py --resume   # after an interruption: reuse passed steps and existing outputs

Steps (each logged with status and time in extra/tests/_out/e2e_summary.json):
  1  static check: fast SURE compute_loss is verbatim (test_fast_sure_verbatim)
  2  re-score the authors' shipped BOOKS logs (test_rescore_shipped)
  3  wrapper == authors' unlearn.py, bitwise, 4 algorithms, 2 epochs; per-epoch weight-only saving does not change
     results or RNG (test_wrapper_cpu)
  4  SURE: optimizer_step counter + mask audit + fast-path equivalence on CPU (test_sure_equivalence)
  5  unlearn: 4 BOOKS presets + 2 NEWS presets, 4 steps each (tiny data, max_len 128)
  6  quantize: RTN g128, GPTQ/AWQ g128 general, GPTQ/AWQ g32 books_retain (4 samples x 128 tokens)
  7  evaluate every model/variant with all 8 metrics on the first 2 examples of every set (--limit 2)
  8  determinism: one evaluation repeated, all output files must be byte-identical
  9  bnb4 (authors' bitsandbytes path) on CPU: expected to fail (bitsandbytes 0.43 has no CPU 4-bit); recorded
 10  compare_results.py on the produced results
"""

import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "extra" / "tests" / "_out"
PAPER_PY = str(PROJ / ".venvs" / "paper" / "bin" / "python")
QUANT_PY = str(PROJ / ".venvs" / "quant" / "bin" / "python")
REPO = PROJ / "FailureLLMUnlearning"
TOK_REPO = "NousResearch/Llama-2-7b-hf"  # ungated, byte-identical tokenizer.model (the gated meta-llama repo is not needed for CPU tests)
TOK_REV = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
summary = {"steps": []}
# The authors' repo tracks __pycache__/*.pyc files; never rewrite them (keeps `git status` clean).
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


RESUME = "--resume" in sys.argv  # reuse passing steps / existing outputs of an interrupted run
PREVIOUS = {}
if RESUME and (OUT / "e2e_summary.json").exists():
    PREVIOUS = {x["step"]: x for x in json.loads((OUT / "e2e_summary.json").read_text())["steps"] if x["ok"]}


def step(name, argv, py=PAPER_PY, expect_fail=False, env=None, done=None):
    """done: output file whose existence means the step already succeeded (used with --resume)."""
    if RESUME and name in PREVIOUS and (done is None or Path(done).exists()):
        summary["steps"].append({**PREVIOUS[name], "reused_from_previous_run": True})
        print(f"\n=== {name}  [reused: passed in the interrupted run]", flush=True)
        return True
    t0 = time.time()
    print(f"\n=== {name}\n$ {' '.join(map(str, argv))}", flush=True)
    rc = subprocess.run([py, *map(str, argv)], cwd=PROJ, env=env).returncode
    ok = (rc != 0) if expect_fail else (rc == 0)
    summary["steps"].append({"step": name, "rc": rc, "ok": ok, "expect_fail": expect_fail,
                             "seconds": round(time.time() - t0, 1)})
    (OUT / "e2e_summary.json").write_text(json.dumps(summary, indent=2))
    if not ok:
        print(f"!!! step failed: {name}")
    return ok


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = OUT / "tokenizer"
    tok.mkdir(exist_ok=True)
    for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        if not (tok / f).exists():
            urllib.request.urlretrieve(f"https://huggingface.co/{TOK_REPO}/resolve/{TOK_REV}/{f}", tok / f)
    data = OUT / "data"
    data.mkdir(exist_ok=True)
    for corpus in ("books", "news"):
        for src, dst in (("forget.txt", f"{corpus}_forget.txt"), ("retain1.txt", f"{corpus}_retain.txt")):
            (data / dst).write_text((REPO / "data" / corpus / "raw" / src).read_text()[:6000])
    if not (OUT / "tiny_target" / "config.json").exists():
        assert step("make tiny model", ["-m", "extra.tests.make_tiny", "--out", OUT / "tiny_target"])
    if not (OUT / "wikitext" / "dataset_info.json").exists():
        code = ("from datasets import load_dataset; d=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',"
                f"split='train'); d.save_to_disk(r'{OUT / 'wikitext'}')")
        assert step("download wikitext-2 train", ["-c", code], py=QUANT_PY)


def main() -> int:
    prepare()
    tok, tiny, data, logs = OUT / "tokenizer", OUT / "tiny_target", OUT / "data", OUT / "logs"
    res = OUT / "results"
    for p in (res, OUT / "ckpt", OUT / "quant"):
        if p.exists() and not RESUME:
            shutil.rmtree(p)

    step("1 fast SURE verbatim", ["-m", "extra.tests.test_fast_sure_verbatim"])
    step("2 rescore shipped logs", ["-m", "extra.tests.test_rescore_shipped"])
    step("3 wrapper == authors (bitwise), D12 saving", [
        "-m", "extra.tests.test_wrapper_cpu", "--model_dir", tiny, "--tokenizer_dir", tok,
        "--forget", data / "books_forget.txt", "--retain", data / "books_retain.txt", "--work_dir", OUT / "wrapper_test"])
    step("4 SURE dead-code + fast-path equivalence (CPU)", [
        "-m", "extra.tests.test_sure_equivalence", "--preset", "books_npo_klr_sure", "--steps", "4",
        "--model_dir", tiny, "--tokenizer_dir", tok, "--work_dir", OUT / "sure_eq", "--logs_dir", logs,
        "--passthrough", f"--cpu_test --max_len 128 --data_file {data/'books_forget.txt'} "
                         f"--retain_data_file {data/'books_retain.txt'}"])

    models = {}  # (corpus, name) -> dir
    for preset in ("books_ga_gdr", "books_npo_klr", "books_ga_gdr_sure", "books_npo_klr_sure",
                   "news_ga_gdr", "news_npo_klr"):
        corpus = preset.split("_")[0]
        out = OUT / "ckpt" / corpus / f"{preset}_s42"
        if step(f"5 unlearn {preset}", [
                "-m", "extra.unlearn_run", "--preset", preset, "--model_dir", tiny, "--tokenizer_dir", tok,
                "--out_dir", out, "--logs_dir", logs, "--cpu_test", "--max_len", "128",
                "--data_file", data / f"{corpus}_forget.txt", "--retain_data_file", data / f"{corpus}_retain.txt",
                "--test_mode", "--max_steps", "4", "--logging_steps", "1"], done=out / "unlearn_run.json"):
            models[(corpus, preset.split("_", 1)[1] + "_s42")] = out

    quant = {}  # (corpus, name, tag) -> dq dir
    qsrc = {("books", "target"): tiny, ("books", "npo_klr_sure_s42"): models.get(("books", "npo_klr_sure_s42")),
            ("news", "target"): tiny, ("news", "ga_gdr_s42"): models.get(("news", "ga_gdr_s42"))}
    qcfgs = {"books": [("rtn", 128, "general"), ("gptq", 128, "general"), ("awq", 128, "general"),
                       ("gptq", 32, "books_retain"), ("awq", 32, "books_retain")],
             "news": [("rtn", 128, "general")]}
    for (corpus, name), src in qsrc.items():
        if src is None:
            continue
        for method, gs, calib in qcfgs[corpus]:
            tag = f"{method}_g{gs}_{'nocalib' if method == 'rtn' else calib}"
            out = OUT / "quant" / f"{corpus}_{name}" / tag
            if step(f"6 quantize {corpus}/{name} {tag}", [
                    "-m", "extra.quantize_run", "--src", src, "--tokenizer_dir", tok, "--corpus", corpus,
                    "--method", method, "--group_size", str(gs), "--calib", calib, "--wikitext_dir", OUT / "wikitext",
                    "--n_samples", "4", "--seq_len", "128", "--out_dir", out, "--logs_dir", logs], py=QUANT_PY,
                    done=out / "quant_report.json"):
                quant[(corpus, name, tag)] = out / "dq"

    evals = [(c, "target", "bf16", tiny) for c in ("books", "news")]
    evals += [(c, n, "bf16", d) for (c, n), d in models.items()]
    evals += [(c, n, t, d) for (c, n, t), d in quant.items()]

    def ev(corpus, name, tag, model_dir, out_dir, quant_mode="none", expect_fail=False, reuse=True):
        argv = ["-m", "extra.eval_run", "--model_dir", model_dir, "--name", name, "--corpus", corpus,
                "--quant", quant_mode, "--tokenizer_dir", tok, "--limit", "2", "--out_dir", out_dir,
                "--logs_dir", logs]
        qrep = Path(model_dir).parent / "quant_report.json"
        if qrep.exists():
            argv += ["--quant_report", qrep]
        return step(f"7 eval {corpus}/{name}/{tag}", argv, expect_fail=expect_fail,
                    done=Path(out_dir) / "metrics.json" if reuse else "/nonexistent")

    for corpus, name, tag, d in evals:
        ev(corpus, name, tag, d, res / corpus / name / tag)

    # 8 determinism
    rep = OUT / "determinism_repeat"
    if rep.exists():
        shutil.rmtree(rep)
    ev("books", "target", "bf16", tiny, rep, reuse=False)
    a, b = res / "books" / "target" / "bf16", rep
    cmp = filecmp.dircmp(a, b)
    same = not cmp.diff_files and not cmp.left_only and not cmp.right_only
    files = sorted(p.name for p in a.iterdir())
    identical = all(filecmp.cmp(a / f, b / f, shallow=False) for f in files)
    summary["determinism"] = {"files": files, "byte_identical": bool(same and identical)}
    print("determinism:", summary["determinism"])

    # 9 bnb4 on CPU (expected failure)
    ev("books", "target", "bnb4", tiny, OUT / "bnb4_cpu", quant_mode="bnb4", expect_fail=True, reuse=False)

    # 10 tables
    step("10 compare_results", ["compare_results.py", "--results", res, "--out", OUT / "tables"])
    (OUT / "e2e_summary.json").write_text(json.dumps(summary, indent=2))
    bad = [s["step"] for s in summary["steps"] if not s["ok"]]
    if not summary["determinism"]["byte_identical"]:
        bad.append("8 determinism")
    print("\nE2E RESULT:", "ALL OK" if not bad else f"FAILED: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
