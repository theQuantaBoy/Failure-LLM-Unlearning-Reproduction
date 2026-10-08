"""
extra/eval_run.py — evaluate one checkpoint with the authors' own metric code, deterministically, saving every
per-example output.

Why not call eval.py directly: load_then_eval_models reads a global `args` (eval.py:188), eval_model does not
return per-example utility outputs (the json.dump calls are commented out, e.g. LLama_factory/.../eval_mmlu.py:75-77),
and VerbMem samples without a seed (metrics/verbmem.py:29-30). This wrapper therefore reproduces eval_model
(eval.py:16-162) step by step, calling the same functions with the same arguments and in the same order:

  model   utils.load_model(model_dir, name, quantize_4bit, quantize_8bit=0, alpha=5, corpus)  (eval.py:188)
          --quant none -> BF16 (utils.py:110-115); --quant bnb4 -> authors' bitsandbytes 4-bit path (utils.py:94-101).
          INT4 RTN/GPTQ/AWQ checkpoints from modal_quant.py are dequantized BF16 copies -> --quant none.
  tok     utils.load_tokenizer(tokenizer_dir) for the MUSE metrics (eval.py:193);
          AutoTokenizer(utility_tokenizer, padding_side='left'), pad=eos for the utility metrics (eval.py:78-79).
  order   model.eval(); no_grad{ MMLU(bs 1), TruthfulQA(bs 4), TriviaQA(bs 16), Fluency(bs 8) } (eval.py:75-87),
          then VerbMem, PrivLeak, KnowMem-forget, KnowMem-retain (eval.py:95-154).

Documented additions (none changes what is computed for a given model):
  E1  transformers.set_seed(seed) before every metric.
  E2  VerbMem 'sample' (primary): authors' call unchanged (do_sample=True, temperature=0.9); model.generate is
      wrapped so that example i is generated right after set_seed(seed + i) -> reproducible and independent of
      which other examples are evaluated. VerbMem 'greedy' (extra column): same call with do_sample=False and
      temperature removed (= upstream MUSE).
  E3  torch.use_deterministic_algorithms(True, warn_only=True), cudnn.benchmark=False (--no_deterministic disables).
  E4  Every per-example record is saved with its index in the source file and a sha1 of its text.
  E5  --limit N evaluates the first N examples of every set (CPU tests only; recorded).
  --diag_fp32_knowmem  DIAGNOSTIC, DEVIATION FROM THE AUTHORS' LOADER (not a reproduction result): load the model in
      FP32 (from_pretrained(torch_dtype=float32, device_map='auto'), i.e. utils.py:110-115 with float32 instead of
      bfloat16) and evaluate knowmem_f only. muse_bench's load_model passes no torch_dtype, so its FP32 checkpoints run
      in FP32; this tests whether BF16 explains the gap to the MUSE-copied KnowMem values. Needs ~27 GB (L40S, not
      A10). The output dir name must contain "fp32diag" so it can never overwrite a BF16 result.
  --seed_mode once (A/B test only, extra/tests/test_eval_ab.py): set_seed(seed) once before loading the model and
      nothing else (E1 and E2 off), i.e. exactly what "running eval.py after one external set_seed" does.

Scales in metrics.json follow eval.py (M1, M2, M4 x100; M3 by eval.py:120); utility scales:
(Gen=acc x100, Tru=MC2 x100 with MC1 alongside, Fac=F1 x100 with EM alongside, Flu=entropy x100).
"""

import argparse
import itertools
import json
import os
import sys
import traceback
from pathlib import Path

from extra.common import DEFAULT_EVAL_SEED, REPO_DIR, RunLog, sha1_text, write_json

UTILITY = ("gen", "tru", "fac", "flu")
MUSE = ("verbmem_f", "privleak", "knowmem_f", "knowmem_r")


def _ids(items, key=None):
    out = []
    for i, it in enumerate(items):
        text = it if key is None else (key(it) if callable(key) else it[key])
        out.append({"idx": i, "sha1": sha1_text(text if isinstance(text, str) else json.dumps(text, sort_keys=True))})
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--name", required=True, help="label of the model (row name in the tables)")
    p.add_argument("--corpus", required=True, choices=["news", "books"])
    p.add_argument("--quant", default="none", choices=["none", "bnb4"])
    p.add_argument("--quant_report", help="JSON written by extra/quantize_run.py (copied into metrics.json)")
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--utility_tokenizer_dir", help="default: --tokenizer_dir (authors hard-code Llama-2-7b-hf)")
    p.add_argument("--metrics", nargs="+", default=list(MUSE + UTILITY), choices=list(MUSE + UTILITY))
    p.add_argument("--verbmem_modes", nargs="+", default=["sample", "greedy"], choices=["sample", "greedy"])
    p.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    p.add_argument("--seed_mode", choices=["per_metric", "once"], default="per_metric",
                   help="per_metric = E1+E2 (default); once = A/B emulation of eval.py with one external seed")
    p.add_argument("--limit", type=int, help="E5, tests only")
    p.add_argument("--diag_fp32_knowmem", action="store_true",
                   help="DIAGNOSTIC (deviation from the authors' loader): FP32 load, knowmem_f only; see docstring")
    p.add_argument("--no_deterministic", action="store_true")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--logs_dir", required=True)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = vars(a).copy()
    runlog = RunLog(a.logs_dir, "eval", f"{a.corpus}_{a.name}", cfg)
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        res = run(a, out)
        runlog.finish(0, results=res)
        return 0
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        runlog.finish(1, error=tb)
        return 1


def run(a, out: Path) -> dict:
    if a.diag_fp32_knowmem:
        if a.quant != "none" or a.metrics != ["knowmem_f"]:
            raise SystemExit("--diag_fp32_knowmem requires --quant none --metrics knowmem_f")
        if "fp32diag" not in out.name:
            raise SystemExit(f"--diag_fp32_knowmem: out_dir name must contain 'fp32diag' (got {out.name})")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # resolve local paths before chdir (HF hub ids such as meta-llama/Llama-2-7b-hf are left as they are)
    for k in ("model_dir", "tokenizer_dir", "utility_tokenizer_dir", "quant_report"):
        v = getattr(a, k)
        if v and Path(v).exists():
            setattr(a, k, str(Path(v).resolve()))
    sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)  # the authors run eval.py from the repo root; data paths in constants.py are relative

    import torch
    import transformers
    from transformers import AutoTokenizer

    from constants import AUC_RETRAIN, DEFAULT_DATA, LLAMA_DIR
    from metrics.knowmem import eval as eval_knowmem
    from metrics.privleak import eval as eval_privleak
    from metrics.verbmem import eval as eval_verbmem
    from utils import load_model, load_tokenizer, read_json
    from LLama_factory.src.llmtuner.eval import eval_fluency, eval_mmlu, eval_triviaqa, eval_truthfulqa
    from LLama_factory.src.llmtuner.eval.eval_fluency import compute_n_gram_entropy
    from LLama_factory.src.llmtuner.eval.eval_triviaqa import EM, F1

    if not a.no_deterministic:  # E3
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    def lim(xs):
        return xs[: a.limit] if a.limit else xs

    once = a.seed_mode == "once"

    def seed():  # E1
        if not once:
            transformers.set_seed(a.seed)

    if once:
        transformers.set_seed(a.seed)

    meta = {"name": a.name, "corpus": a.corpus, "model_dir": a.model_dir, "quant": a.quant, "seed": a.seed,
            "seed_mode": a.seed_mode,
            "limit": a.limit, "deterministic": not a.no_deterministic, "authors_default_tokenizer": LLAMA_DIR}
    if a.quant_report:
        meta["quant_report"] = read_json(a.quant_report)

    # ── model and tokenizers (eval.py:188-193, eval.py:78-79) ──
    if a.diag_fp32_knowmem:  # DEVIATION (diagnostic only): utils.py:110-115 with float32 instead of bfloat16
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, device_map="auto", torch_dtype=torch.float32)
        meta["loader"] = "DEVIATION diag_fp32_knowmem: from_pretrained(torch_dtype=float32, device_map='auto')"
        meta["checkpoint_torch_dtype"] = read_json(str(Path(a.model_dir) / "config.json")).get("torch_dtype")
    else:
        model = load_model(a.model_dir, a.name, 1 if a.quant == "bnb4" else 0, 0, 5, corpus=a.corpus)
    dtypes = {}
    for _, prm in model.named_parameters():
        dtypes[str(prm.dtype)] = dtypes.get(str(prm.dtype), 0) + prm.numel()
    meta["param_dtypes"] = dtypes
    meta["quantization_config"] = str(getattr(model.config, "quantization_config", None))
    tokenizer = load_tokenizer(a.tokenizer_dir)
    model.eval()
    metrics = {}

    # ── utility metrics (eval.py:55-87) ──
    if any(m in a.metrics for m in UTILITY):
        udir = REPO_DIR / "LLama_factory" / "data" / "utility"
        with torch.no_grad():
            e_tokenizer = AutoTokenizer.from_pretrained(a.utility_tokenizer_dir or a.tokenizer_dir,
                                                        padding_side="left")
            e_tokenizer.pad_token = e_tokenizer.eos_token
            if "gen" in a.metrics:
                data = lim(json.loads((udir / "retain_mmlu.json").read_text()))
                ids = _ids(data, "question")
                seed()
                acc = eval_mmlu(model, e_tokenizer, data, batch_size=1, output_result_dir=None, use_prompt=False)
                per = [{**i, "prediction": d["prediction"], "answer": d["answer"],
                        "correct": bool(d["prediction"] == d["answer"]), "task": d["task"]} for i, d in zip(ids, data)]
                write_json({"acc": float(acc), "per_example": per}, out / "mmlu.json")
                metrics.update(gen=100 * float(acc), gen_raw=float(acc))
            if "tru" in a.metrics:
                data = lim(json.loads((udir / "truthful.json").read_text()))
                ids = _ids(data, "question")
                seed()
                mc1, mc2 = eval_truthfulqa(model, e_tokenizer, data, batch_size=4, output_result_dir=None,
                                           use_prompt=False)
                per = [{**i, "MC1": float(d["MC1"]), "MC2": float(d["MC2"])} for i, d in zip(ids, data)]
                write_json({"MC1": float(mc1), "MC2": float(mc2), "per_example": per}, out / "truthful.json")
                metrics.update(tru=100 * float(mc2), tru_mc1=100 * float(mc1), tru_raw=[float(mc1), float(mc2)])
            if "fac" in a.metrics:
                data = lim(json.loads((udir / "triviaqa.json").read_text()))
                ids = _ids(data, "question")
                seed()
                em, f1 = eval_triviaqa(model, e_tokenizer, data, batch_size=16, output_result_dir=None,
                                       use_prompt=False)
                emm, f1m = EM("em"), F1("F1")
                per = [{**i, "prediction": d["prediction"], "answers": d["answers"],
                        "f1": float(f1m._f1(d["prediction"], d["answers"])),
                        "em": float(emm._exact_match(d["prediction"], d["answers"]))} for i, d in zip(ids, data)]
                write_json({"EM": float(em), "F1": float(f1), "per_example": per}, out / "triviaqa.json")
                metrics.update(fac=100 * float(f1), fac_em=100 * float(em), fac_raw=[float(em), float(f1)])
            if "flu" in a.metrics:
                data = lim(json.loads((udir / "fluency.json").read_text()))
                ids = _ids(data, "instruction")
                seed()
                ent = eval_fluency(model, e_tokenizer, data, batch_size=8, output_result_dir=None, use_prompt=False)
                per = [{**i, "prediction": d["prediction"],
                        "entropy": float(compute_n_gram_entropy(d["prediction"]))} for i, d in zip(ids, data)]
                write_json({"entropy": float(ent), "per_example": per}, out / "fluency.json")
                metrics.update(flu=100 * float(ent), flu_raw=float(ent))

    files = DEFAULT_DATA[a.corpus]

    # ── M1 VerbMem (eval.py:96-107) ──
    if "verbmem_f" in a.metrics:
        data = lim(read_json(files["verbmem_forget_file"]))
        ids = _ids(data, lambda d: d["prompt"] + d["gt"])
        orig_generate = model.generate
        for mode in a.verbmem_modes:
            counter = itertools.count()

            def generate(*args, _mode=mode, _counter=counter, **kw):
                i = next(_counter)
                if not once:
                    transformers.set_seed(a.seed + i)  # E2
                if _mode == "greedy":
                    kw["do_sample"] = False
                    kw.pop("temperature", None)
                return orig_generate(*args, **kw)

            model.generate = generate
            seed()
            try:
                agg, log = eval_verbmem(prompts=[d["prompt"] for d in data], gts=[d["gt"] for d in data],
                                        model=model, tokenizer=tokenizer, max_new_tokens=128)
            finally:
                model.generate = orig_generate
            for i, rec in zip(ids, log):
                rec.update(i)
            write_json({"mode": mode, "agg": agg, "per_example": log}, out / f"verbmem_{mode}.json")
            key = "verbmem_f" if mode == "sample" else "verbmem_f_greedy"
            metrics[key] = agg["mean_rougeL"] * 100

    # ── M3 PrivLeak (eval.py:110-120) ──
    if "privleak" in a.metrics:
        f, r, h = (lim(read_json(files[k])) for k in
                   ("privleak_forget_file", "privleak_retain_file", "privleak_holdout_file"))
        seed()
        auc, log = eval_privleak(forget_data=f, retain_data=r, holdout_data=h, model=model, tokenizer=tokenizer)
        for split, texts in (("forget", f), ("retain", r), ("holdout", h)):
            for i, rec in zip(_ids(texts), log[split]):
                rec.update(i)
        key = "forget_holdout_Min-40%"
        write_json({"auc": auc, "per_example": log}, out / "privleak.json")
        metrics["privleak"] = (auc[key] - AUC_RETRAIN[a.corpus][key]) / AUC_RETRAIN[a.corpus][key] * 100
        metrics["privleak_auc"] = auc[key]
        metrics["privleak_auc_retrain_const"] = AUC_RETRAIN[a.corpus][key]

    # ── M2 / M4 KnowMem (eval.py:123-154) ──
    for m, qf, icf in (("knowmem_f", "knowmem_forget_qa_file", "knowmem_forget_qa_icl_file"),
                       ("knowmem_r", "knowmem_retain_qa_file", "knowmem_retain_qa_icl_file")):
        if m not in a.metrics:
            continue
        qa, icl = lim(read_json(files[qf])), read_json(files[icf])  # ICL examples are never truncated
        ids = _ids(qa, lambda d: d["question"] + d["answer"])
        seed()
        agg, log = eval_knowmem(questions=[d["question"] for d in qa], answers=[d["answer"] for d in qa],
                                icl_qs=[d["question"] for d in icl], icl_as=[d["answer"] for d in icl],
                                model=model, tokenizer=tokenizer, max_new_tokens=32)
        for i, rec in zip(ids, log):
            rec.update(i)
        write_json({"agg": agg, "per_example": log}, out / f"{m}.json")
        metrics[m] = agg["mean_rougeL"] * 100

    res = {"meta": meta, "metrics": metrics}
    write_json(res, out / "metrics.json")
    print(json.dumps(metrics, indent=2))
    return res


if __name__ == "__main__":
    sys.exit(main())
