"""
A/B test: the authors' eval.py (A) vs extra/eval_run.py (B) on the same model, data and seed.

  A  FailureLLMUnlearning/eval.py executed unmodified as __main__ (runpy) from a COPY of the repo in which only the
     evaluation data files are truncated to their first N examples (the ICL files are not touched). The runner calls
     transformers.set_seed(seed) once before eval.py starts, and wraps the four utility functions so that the
     per-example predictions eval.py computes but does not save (json.dump commented out, e.g. eval_mmlu.py:75-77)
     are written to disk. The wrappers call the original functions with unchanged arguments.
  B  python -m extra.eval_run on the same copy, --seed_mode once --no_deterministic --verbmem_modes sample, i.e.
     the same single external seed and none of the additions E1-E3.

Tokenizer: eval.py hard-codes "meta-llama/Llama-2-7b-hf" (eval.py:78, LLAMA_DIR). Both runs resolve that id from
an offline HF cache. --tokenizer_fixture DIR builds such a cache from a local copy of the tokenizer files (used when
the gated meta-llama repo is not available). Without it, HF_HOME is left as is.

Compared: every per-example field the authors produce (VerbMem/KnowMem prompt, gt, response, ROUGE; PrivLeak
per-text scores and the full AUC table; MMLU predictions; TruthfulQA MC1/MC2; TriviaQA and fluency generations),
every agg.json, and every value of eval.py's CSV row against B's raw metrics. Verdict in <work_dir>/ab_verdict.json.

    python -m extra.tests.test_eval_ab --model_dir M --corpus books --n 3 --work_dir W --logs_dir L \
        [--tokenizer_fixture TOKDIR] [--quant bnb4]
"""

import argparse
import csv
import filecmp
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from extra.common import REPO_DIR, RunLog, write_json

META_ID, META_REV = "meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9"
UTIL = {"eval_mmlu": "retain_mmlu.json", "eval_truthfulqa": "truthful.json", "eval_triviaqa": "triviaqa.json",
        "eval_fluency": "fluency.json"}

RUNNER = r'''
import json, runpy, sys
sys.path.insert(0, {copy!r})
import transformers
transformers.set_seed({seed})                       # the single external seed
import LLama_factory.src.llmtuner.eval as E          # eval.py does `from LLama_factory.src.llmtuner.eval import ...`
def capture(name):
    orig = getattr(E, name)
    def wrapped(model, tokenizer, dataset, *args, **kw):
        out = orig(model, tokenizer, dataset, *args, **kw)      # unchanged call
        with open({work!r} + "/A_" + name + ".json", "w") as f:
            json.dump(dataset, f, default=float)               # predictions were added in place by orig
        return out
    setattr(E, name, wrapped)
for n in {names!r}:
    capture(n)
sys.argv = ["eval.py"] + {argv!r}
runpy.run_path({copy!r} + "/eval.py", run_name="__main__")
'''


def make_copy(work: Path, corpus: str, n: int) -> Path:
    copy = work / "repo_copy"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(REPO_DIR, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", "temp", "output.csv"))
    # code must be byte-identical
    bad = [str(p.relative_to(REPO_DIR)) for p in REPO_DIR.rglob("*.py")
           if ".git" not in p.parts and not filecmp.cmp(p, copy / p.relative_to(REPO_DIR), shallow=False)]
    assert not bad, f"copy differs: {bad}"
    d = copy / "data" / corpus
    for f in ("verbmem/forget.json", "privleak/forget.json", "privleak/retain.json", "privleak/holdout.json",
              "knowmem/forget_qa.json", "knowmem/retain_qa.json"):
        p = d / f
        p.write_text(json.dumps(json.loads(p.read_text())[:n]))
    for f in UTIL.values():
        p = copy / "LLama_factory" / "data" / "utility" / f
        p.write_text(json.dumps(json.loads(p.read_text())[:n]))
    return copy


def make_tokenizer_cache(work: Path, src: Path) -> Path:
    hf = work / "hf"
    repo = hf / "hub" / ("models--" + META_ID.replace("/", "--"))
    snap = repo / "snapshots" / META_REV
    snap.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, snap / f.name)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(META_REV)
    return hf


def eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return (math.isnan(a) and math.isnan(b)) or float(a) == float(b)
        except TypeError:
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and eq(v, b[k]) for k, v in a.items())
    return a == b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--corpus", default="books", choices=["news", "books"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant", default="none", choices=["none", "bnb4"])
    ap.add_argument("--tokenizer_fixture", help="dir with tokenizer files, served offline as " + META_ID)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--logs_dir", required=True)
    a = ap.parse_args(argv)
    work = Path(a.work_dir).resolve()
    a.model_dir = str(Path(a.model_dir).resolve()) if Path(a.model_dir).exists() else a.model_dir
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    runlog = RunLog(a.logs_dir, "test", f"eval_ab_{a.corpus}", vars(a))
    copy = make_copy(work, a.corpus, a.n)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if a.tokenizer_fixture:
        env.update(HF_HOME=str(make_tokenizer_cache(work, Path(a.tokenizer_fixture))), HF_HUB_OFFLINE="1")

    # ── A: authors' eval.py ──
    argv_a = ["--model_dirs", a.model_dir, "--names", "ab", "--corpus", a.corpus, "--out_file", str(work / "A.csv"),
              "--quantize_4bit", "1" if a.quant == "bnb4" else "0"]
    code = RUNNER.format(copy=str(copy), seed=a.seed, work=str(work), names=list(UTIL), argv=argv_a)
    rc_a = subprocess.run([sys.executable, "-c", code], cwd=copy, env=env).returncode
    # ── B: eval_run ──
    rc_b = subprocess.run([sys.executable, "-m", "extra.eval_run", "--model_dir", a.model_dir, "--name", "ab",
                           "--corpus", a.corpus, "--quant", a.quant, "--tokenizer_dir", META_ID, "--seed", str(a.seed),
                           "--seed_mode", "once", "--no_deterministic", "--verbmem_modes", "sample",
                           "--out_dir", str(work / "B"), "--logs_dir", a.logs_dir],
                          env={**env, "FAILUNL_REPO": str(copy)}).returncode
    if rc_a or rc_b:
        runlog.finish(1, rc_a=rc_a, rc_b=rc_b)
        print(f"FAIL: A rc={rc_a}, B rc={rc_b}")
        return 1

    B = work / "B"
    res, diffs = {}, []

    def check(name, x, y):
        ok = eq(x, y)
        res[name] = ok
        if not ok:
            diffs.append(name)

    # MUSE metrics: authors' temp/<name>/<metric>/{agg,log}.json. eval.py:189 re-binds `name` in
    # `for name, param in model.named_parameters()`, so <name> is the last parameter name ("lm_head.weight"),
    # not --names. We read whatever single directory eval.py created and record its name.
    tdirs = [p for p in (copy / "temp").iterdir() if p.is_dir()]
    assert len(tdirs) == 1, tdirs
    T = tdirs[0]
    res_extra = {"authors_temp_dir_name": T.name}
    for m, bf in (("verbmem_f", "verbmem_sample.json"), ("knowmem_f", "knowmem_f.json"), ("knowmem_r", "knowmem_r.json")):
        la, ga = json.loads((T / m / "log.json").read_text()), json.loads((T / m / "agg.json").read_text())
        b = json.loads((B / bf).read_text())
        check(f"{m}/agg", ga, b["agg"])
        check(f"{m}/n", len(la), len(b["per_example"]))
        for i, (ra, rb) in enumerate(zip(la, b["per_example"])):
            check(f"{m}/example{i}", ra, {k: rb[k] for k in ra})
    pa_log = json.loads((T / "privleak" / "log.json").read_text())
    pa_auc = json.loads((T / "privleak" / "auc.json").read_text())
    pb = json.loads((B / "privleak.json").read_text())
    check("privleak/auc_table", pa_auc, pb["auc"])
    for split in ("forget", "retain", "holdout"):
        for i, (ra, rb) in enumerate(zip(pa_log[split], pb["per_example"][split])):
            check(f"privleak/{split}{i}", ra, {k: rb[k] for k in ra})
    # utility per-example
    for fn, fname, fields in (("eval_mmlu", "mmlu.json", ("prediction",)),
                              ("eval_truthfulqa", "truthful.json", ("MC1", "MC2")),
                              ("eval_triviaqa", "triviaqa.json", ("prediction",)),
                              ("eval_fluency", "fluency.json", ("prediction",))):
        da = json.loads((work / f"A_{fn}.json").read_text())
        db = json.loads((B / fname).read_text())["per_example"]
        check(f"{fn}/n", len(da), len(db))
        for i, (ra, rb) in enumerate(zip(da, db)):
            check(f"{fn}/example{i}", {k: ra[k] for k in fields}, {k: rb[k] for k in fields})
    # eval.py CSV row vs B metrics (raw scales as eval.py reports them)
    row = next(csv.DictReader(open(work / "A.csv")))
    mb = json.loads((B / "metrics.json").read_text())["metrics"]
    for k_csv, k_b in (("verbmem_f", "verbmem_f"), ("privleak", "privleak"), ("knowmem_f", "knowmem_f"),
                       ("knowmem_r", "knowmem_r"), ("gen", "gen_raw"), ("flu", "flu_raw")):
        check(f"csv/{k_csv}", float(row[k_csv]), float(mb[k_b]))
    for k_csv, k_b in (("tru", "tru_raw"), ("fac", "fac_raw")):
        check(f"csv/{k_csv}", [float(x) for x in row[k_csv].strip("()").split(",")], mb[k_b])

    verdict = {**res_extra, "csv_name_column_A": row.get("name"), "identical": not diffs, "n_checks": len(res), "n_differences": len(diffs), "differences": diffs,
               "n_examples_per_set": a.n, "seed": a.seed, "quant": a.quant, "model_dir": a.model_dir,
               "csv_row_A": row}
    write_json(verdict, work / "ab_verdict.json")
    print(json.dumps({k: v for k, v in verdict.items() if k != "csv_row_A"}, indent=2))
    runlog.finish(0 if not diffs else 1, verdict=verdict)
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
