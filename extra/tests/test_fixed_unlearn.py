"""
CPU unit tests for extra/fixed_unlearn.py (W7, opt-in fixes; FINDINGS.md §6). Tiny random Llama
(extra/tests/_out/tiny_target, FP32), no download, a few seconds per test.

  T1  all fixes off  ==  the authors' compute_loss (IterativeUnlearner and SURE): same loss, same gradients (bitwise)
  T2  kl=proper: KL >= 0, = 0 at theta = ref, invariant to adding a constant to either model's logits; the released
      raw-logit "KL" is negative for x = t + 1 and changes under a shift
  T3  npo=sequence: equals Eq. 8 built from HF's own CE loss; invariant to a logit shift; the released logit-level term
      is not; both = 2/beta*log 2 at theta = ref
  T4  sure_single_grad: update gradient == grad(L_f + alpha L_r); the released SURE gives 2 grad L_f + alpha grad L_r
  T5  sure_mask=step: the mask equals the authors' m_S dict (iterative.py:284-305) for the same gradients; after real
      Trainer steps every row with m = 0 is unchanged and rows with m = 1 do change
  T6  sure_mask=fixed: after 3 Trainer steps every row outside the theta_o mask is bitwise unchanged
  T7  observation (not a fix): pure-BF16 AdamW (the authors' setting: BF16 weights, bf16=True) drops most lr=1e-5
      updates by rounding; lr=1e-4 keeps most of them
  T8  CLI plumbing: unlearn_run with a FIX_PRESETS name records the fixes + deviation flag; the default preset path
      still reproduces the stored Phase 2 e2e checkpoint bitwise (books_npo_klr, 4 steps)
  T9  D2 audit through the smoke harness (extra/tests/test_sure_equivalence.py --runs A), same arguments as the
      Phase 2 e2e SURE test (4 steps, seed 42, tiny data, max_len 128): released books_npo_klr_sure -> rows changed
      outside m_S (b) > 0 and the per-step audit is identical to the stored Phase 2 one (released path unchanged);
      books_npo_klr_sure_masked -> (a) rows outside the applied mask with a non-zero gradient = 0 in every step;
      (b) rows changed outside it, (c) of those inside earlier, and the rows optimizer.step moved before the restore
      (AdamW momentum) are reported, not forced; both runs see the same first batch (step-1 CE identical)

    PYTHONDONTWRITEBYTECODE=1 .venvs/paper/bin/python -m extra.tests.test_fixed_unlearn [--only T1,T2] [--skip T8]
"""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from extra.common import PROJECT_DIR, REPO_DIR

OUT = PROJECT_DIR / "extra" / "tests" / "_out"
TINY, TOK, DATA = OUT / "tiny_target", OUT / "tokenizer", OUT / "data"
BETA = 0.1


def iterative_module():
    sys.path.insert(0, str(REPO_DIR / "baselines"))
    from baselines import iterative

    return iterative


def tiny_models(perturb=0.02, seed=0):
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
    model = copy.deepcopy(ref)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(perturb * torch.randn(p.shape, generator=g))
    ref.eval()
    return model, ref


def batch(seed=1, T=24):
    g = torch.Generator().manual_seed(seed)
    ids_f = torch.randint(3, 32000, (1, T), generator=g)
    ids_r = torch.randint(3, 32000, (1, T), generator=g)
    mk = lambda ids: {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids, dtype=torch.bool)}
    return mk(ids_f), mk(ids_r)


def make_trainer(cls, model, ref, algo, alpha, threshold=90, tmp=None, **kw):
    from transformers import TrainingArguments

    args = TrainingArguments(output_dir=tmp or tempfile.mkdtemp(), use_cpu=True, per_device_train_batch_size=1,
                             learning_rate=1e-3, report_to="none", save_strategy="no", lr_scheduler_type="constant",
                             optim="adamw_torch", **kw.pop("args", {}))
    extra = {"threshold": threshold} if "sure" in algo else {}
    t = cls(model=model, ref_model=ref if ("npo" in algo or "kl" in algo) else None, args=args,
            loss_type=algo, alpha=alpha, **extra, **kw)
    t.create_optimizer()
    return t


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def run_loss_and_backward(trainer, x):
    trainer.model.zero_grad(set_to_none=True)
    loss = trainer.compute_loss(trainer.model, x)
    loss.backward()  # = Trainer.training_step's accelerator.backward(loss) (trainer.py:3147), no scaler for fp32
    return loss.detach(), grads(trainer.model)


# ── tests ──────────────────────────────────────────────────────────────────────────────────────────────────────
def t1_equivalence():
    it = iterative_module()
    from extra.fixed_unlearn import make_classes

    FixedBase, FixedSURE = make_classes(it, {})
    out = {}
    for algo, alpha, (A, B) in [("npo_klr", 2, (it.IterativeUnlearner, FixedBase)),
                                ("ga_gdr", 100, (it.IterativeUnlearner, FixedBase)),
                                ("npo_klr_sure", 20, (it.SURE, FixedSURE)),
                                ("ga_gdr_sure", 400, (it.SURE, FixedSURE))]:
        model, ref = tiny_models()
        x = batch()
        la, ga = run_loss_and_backward(make_trainer(A, copy.deepcopy(model), ref, algo, alpha), x)
        lb, gb = run_loss_and_backward(make_trainer(B, copy.deepcopy(model), ref, algo, alpha), x)
        same_loss = torch.equal(la, lb)
        same_grads = ga.keys() == gb.keys() and all(torch.equal(ga[k], gb[k]) for k in ga)
        out[algo] = {"loss": float(la), "same_loss": same_loss, "same_grads": same_grads, "n_grads": len(ga)}
        assert same_loss and same_grads, (algo, out[algo])
    return out


def t2_kl():
    it = iterative_module()
    from extra.fixed_unlearn import make_classes

    g = torch.Generator().manual_seed(2)
    x = 3 * torch.randn(1, 16, 50, generator=g)
    t = 3 * torch.randn(1, 16, 50, generator=g)
    raw = lambda a, b: F.kl_div(a, b, reduction="batchmean", log_target=True)              # iterative.py:173-178
    proper = lambda a, b: F.kl_div(F.log_softmax(a, -1), F.softmax(b, -1), reduction="batchmean")  # :330-334
    r = {
        "raw(x,t)": float(raw(x, t)), "raw(x+5,t)": float(raw(x + 5, t)), "raw(t+1,t)": float(raw(t + 1, t)),
        "proper(x,t)": float(proper(x, t)), "proper(x+5,t-3)": float(proper(x + 5, t - 3)),
        "proper(t,t)": float(proper(t, t)),
    }
    assert r["raw(t+1,t)"] < 0 and abs(r["raw(x+5,t)"] - r["raw(x,t)"]) > 1
    assert r["proper(x,t)"] >= 0 and abs(r["proper(x,t)"] - r["proper(x+5,t-3)"]) < 1e-4 and abs(r["proper(t,t)"]) < 1e-5
    # inside the fixed base trainer: retain term = alpha * proper KL of the model's logits
    model, ref = tiny_models()
    xf, xr = batch()
    tr = make_trainer(make_classes(it, {"kl": "proper"})[0], model, ref, "npo_klr", 2)
    loss = tr.compute_loss(model, (xf, xr))
    with torch.no_grad():
        lf = -F.logsigmoid(BETA * (ref(**xf).logits - model(**xf).logits)).mean() * 2 / BETA
        kl = proper(model(**xr).logits, ref(**xr).logits)
    r["trainer_loss"], r["manual_loss"] = float(loss), float(lf + 2 * kl)
    assert torch.allclose(loss.detach(), lf + 2 * kl, rtol=1e-5, atol=1e-5), r
    return r


def t3_npo():
    it = iterative_module()
    from extra.fixed_unlearn import make_classes, sequence_logprob

    model, ref = tiny_models()
    xf, xr = batch()
    with torch.no_grad():
        out_m, out_r = model(**xf), ref(**xf)
        n_tok = xf["input_ids"].shape[1] - 1
        lp_m_hf, lp_r_hf = -out_m.loss * n_tok, -out_r.loss * n_tok          # from HF's own shifted CE
        lp_m, lp_r = sequence_logprob(out_m.logits, xf["labels"]), sequence_logprob(out_r.logits, xf["labels"])
        eq8 = -F.logsigmoid(-BETA * (lp_m_hf - lp_r_hf)).mean() * 2 / BETA
        seq = lambda lm, lr: -F.logsigmoid(BETA * (sequence_logprob(lr, xf["labels"]) - sequence_logprob(lm, xf["labels"]))).mean() * 2 / BETA
        logit = lambda lm, lr: -F.logsigmoid(BETA * (lr - lm)).mean() * 2 / BETA      # iterative.py:160-161
        r = {"seq_logprob_vs_hf_maxabs": float((lp_m - lp_m_hf).abs().max()),
             "eq8": float(eq8), "seq": float(seq(out_m.logits, out_r.logits)),
             "seq_shift": float(seq(out_m.logits + 7, out_r.logits)),
             "logit": float(logit(out_m.logits, out_r.logits)), "logit_shift": float(logit(out_m.logits + 7, out_r.logits)),
             "seq_at_ref": float(seq(out_r.logits, out_r.logits)), "logit_at_ref": float(logit(out_r.logits, out_r.logits)),
             "2/beta*log2": float(2 / BETA * torch.log(torch.tensor(2.0)))}
    assert r["seq_logprob_vs_hf_maxabs"] < 1e-2 and abs(r["eq8"] - r["seq"]) < 1e-3
    assert abs(r["seq_shift"] - r["seq"]) < 1e-3 and abs(r["logit_shift"] - r["logit"]) > 1
    assert abs(r["seq_at_ref"] - r["2/beta*log2"]) < 1e-4 and abs(r["logit_at_ref"] - r["2/beta*log2"]) < 1e-4
    tr = make_trainer(make_classes(it, {"npo": "sequence", "kl": "proper"})[0], model, ref, "npo_klr", 2)
    with torch.no_grad():
        loss = tr.compute_loss(model, (xf, xr))
        kl = F.kl_div(F.log_softmax(model(**xr).logits, -1), F.softmax(ref(**xr).logits, -1), reduction="batchmean")
    r["trainer_minus_manual"] = float(loss - (seq(out_m.logits, out_r.logits) + 2 * kl))
    assert abs(r["trainer_minus_manual"]) < 1e-3, r
    return r


def t4_single_grad():
    it = iterative_module()
    from extra.fixed_unlearn import make_classes

    r = {}
    for algo, alpha in [("npo_klr_sure", 20), ("ga_gdr_sure", 400)]:
        model, ref = tiny_models()
        xf, xr = batch()
        # independent reference gradients of L_f and L_r
        m0 = copy.deepcopy(model)
        out_f, out_r = m0(**xf), m0(**xr)
        if algo.startswith("npo"):
            with torch.no_grad():
                rf, rr = ref(**xf).logits, ref(**xr).logits
            Lf = -F.logsigmoid(BETA * (rf - out_f.logits)).mean() * 2 / BETA
            Lr = F.kl_div(F.log_softmax(out_r.logits, -1), F.softmax(rr, -1), reduction="batchmean")
        else:
            Lf, Lr = -out_f.loss, out_r.loss
        params = [p for _, p in m0.named_parameters()]
        names = [n for n, _ in m0.named_parameters()]
        gf = torch.autograd.grad(Lf, params, retain_graph=True, allow_unused=True)
        gr = torch.autograd.grad(Lr, params, allow_unused=True)
        z = lambda g, p: torch.zeros_like(p) if g is None else g
        want_single = {n: z(a, p) + alpha * z(b, p) for n, a, b, p in zip(names, gf, gr, params)}
        want_double = {n: 2 * z(a, p) + alpha * z(b, p) for n, a, b, p in zip(names, gf, gr, params)}

        def err(got, want):
            return max(float((got.get(n, torch.zeros_like(w)) - w).abs().max() / (w.abs().max() + 1e-12)) for n, w in want.items())

        _, g_auth = run_loss_and_backward(make_trainer(it.SURE, copy.deepcopy(model), ref, algo, alpha), (xf, xr))
        _, g_fix = run_loss_and_backward(make_trainer(make_classes(it, {"sure_single_grad": True})[1],
                                                      copy.deepcopy(model), ref, algo, alpha), (xf, xr))
        r[algo] = {"authors_vs_2gradLf+a*gradLr": err(g_auth, want_double), "authors_vs_single": err(g_auth, want_single),
                   "fixed_vs_single": err(g_fix, want_single), "fixed_vs_double": err(g_fix, want_double)}
        assert r[algo]["authors_vs_2gradLf+a*gradLr"] < 1e-4 and r[algo]["fixed_vs_single"] < 1e-4, r
        assert r[algo]["authors_vs_single"] > 1e-3 and r[algo]["fixed_vs_double"] > 1e-3, r
    return r


def _dataset(max_len=32):
    sys.path.insert(0, str(REPO_DIR / "baselines"))
    from baselines.dataset import ForgetRetainDataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK)
    tok.pad_token = tok.eos_token
    return ForgetRetainDataset(str(DATA / "books_forget.txt"), tokenizer=tok,
                               retain_file_path=str(DATA / "books_retain.txt"), max_len=max_len), tok


def _train(fixes, algo, alpha, threshold, steps):
    """Real Trainer.train() for `steps` steps; returns trainer, initial params, per-step (before, after, mask)."""
    it = iterative_module()
    from transformers import TrainerCallback
    from extra.fixed_unlearn import make_classes

    ds, tok = _dataset()
    model, ref = tiny_models()
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    log = []

    class Snap(TrainerCallback):
        def on_step_begin(self, args, state, control, model=None, **kw):
            self.before = {n: p.detach().clone() for n, p in model.named_parameters()}

        def on_step_end(self, args, state, control, model=None, **kw):
            log.append((self.before, {n: p.detach().clone() for n, p in model.named_parameters()},
                        {k: v.clone() for k, v in trainer.last_mask.items()} if trainer.last_mask else None))

    cls = make_classes(it, fixes)[1]
    trainer = make_trainer(cls, model, ref, algo, alpha, threshold=threshold, train_dataset=ds,
                           data_collator=ds.get_collate_fn(), tokenizer=tok,
                           args={"max_steps": steps, "max_grad_norm": 1.0, "logging_steps": 1})
    trainer.add_callback(Snap())  # added after _MaskRestore -> sees the restored parameters
    trainer.optimizer = None      # let train() build it as in a normal run
    trainer.train()
    return trainer, init, log


def _check_rows(before, after, mask):
    changed_outside = changed_inside = rows_outside = rows_inside = 0
    for n, a in after.items():
        keep = mask.get(n, torch.zeros(a.shape[0], dtype=torch.bool))
        diff = (a != before[n])
        ch = diff.reshape(diff.shape[0], -1).any(1) if diff.dim() > 1 else diff
        changed_outside += int((ch & ~keep).sum()); rows_outside += int((~keep).sum())
        changed_inside += int((ch & keep).sum()); rows_inside += int(keep.sum())
    return {"rows_outside_mask": rows_outside, "changed_outside_mask": changed_outside,
            "rows_inside_mask": rows_inside, "changed_inside_mask": changed_inside}


def t5_mask_step():
    it = iterative_module()
    from extra.fixed_unlearn import saliency_mask

    # (a) same mask as the authors' dict for the same gradients
    model, ref = tiny_models()
    tr = make_trainer(it.SURE, model, ref, "npo_klr_sure", 20, threshold=90)
    tr.compute_loss(model, batch())  # authors' code: first backward + m_S dict; grads now = grad L_f
    mine = saliency_mask(model, 90)
    mism = sum(int(tr.m_S[f"{n}.{i}"] != float(v)) for n, m in mine.items() for i, v in enumerate(m.tolist()))
    r = {"authors_m_S_entries": len(tr.m_S), "mine_entries": sum(m.numel() for m in mine.values()),
         "mismatches": mism, "salient_frac": sum(int(m.sum()) for m in mine.values()) / sum(m.numel() for m in mine.values())}
    assert mism == 0 and r["authors_m_S_entries"] == r["mine_entries"], r
    # (b) real Trainer steps: rows with m = 0 never change within a step
    trainer, _, log = _train({"sure_single_grad": True, "sure_mask": "step"}, "npo_klr_sure", 20, 90, steps=3)
    r["steps"] = [_check_rows(b, a, m) for b, a, m in log]
    for s in r["steps"]:
        assert s["changed_outside_mask"] == 0 and s["changed_inside_mask"] > 0, r
    r["mask_stats"] = trainer.mask_stats[:3]
    return r


def t6_mask_fixed():
    trainer, init, log = _train({"sure_single_grad": True, "sure_mask": "fixed", "sure_mask_batches": 2},
                                "ga_gdr_sure", 400, 99, steps=3)
    final = log[-1][1]
    r = {"vs_initial": _check_rows(init, final, trainer._fixed_mask), "mask_stats": trainer.mask_stats}
    assert r["vs_initial"]["changed_outside_mask"] == 0 and r["vs_initial"]["changed_inside_mask"] > 0, r
    return r


def t7_bf16_rounding():
    g = torch.Generator().manual_seed(0)
    w0 = 0.02 * torch.randn(1_000_000, generator=g)  # Llama-like weight scale (assumption, see report)
    grad = torch.randn(1_000_000, generator=g)
    r = {}
    for dtype in (torch.bfloat16, torch.float32):
        for lr in (1e-5, 1e-4):
            p = torch.nn.Parameter(w0.to(dtype).clone())
            opt = torch.optim.AdamW([p], lr=lr, weight_decay=0.0)
            p.grad = grad.to(dtype)
            opt.step()
            r[f"{str(dtype).split('.')[-1]}_lr{lr:g}_frac_changed"] = float((p.detach() != w0.to(dtype)).float().mean())
    assert r["float32_lr1e-05_frac_changed"] > 0.99 and r["bfloat16_lr1e-05_frac_changed"] < 0.3
    assert r["bfloat16_lr0.0001_frac_changed"] > 0.8
    return r


def t8_cli():
    import os

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    common = ["--model_dir", str(TINY), "--tokenizer_dir", str(TOK), "--cpu_test", "--max_len", "128",
              "--data_file", str(DATA / "books_forget.txt"), "--retain_data_file", str(DATA / "books_retain.txt"),
              "--test_mode", "--max_steps", "4", "--logging_steps", "1"]
    r = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for preset in ("books_npo_klr_sure_masked", "books_npo_klr_properkl"):
            out = tmp / preset
            cmd = [sys.executable, "-m", "extra.unlearn_run", "--preset", preset, "--out_dir", str(out),
                   "--logs_dir", str(tmp / "logs")] + common
            p = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
            assert p.returncode == 0, p.stderr[-3000:]
            rec = json.loads((out / "unlearn_run.json").read_text())
            r[preset] = {"fixes": rec["config"].get("fixes"), "deviation": rec["config"].get("deviation"),
                         "global_step": rec["stats"].get("global_step")}
            assert rec["config"].get("deviation") and rec["stats"]["global_step"] == 4
        # default path unchanged: books_npo_klr, same arguments as the Phase 2 e2e run
        ref_dir = OUT / "ckpt" / "books" / "books_npo_klr_s42"
        out = tmp / "default"
        cmd = [sys.executable, "-m", "extra.unlearn_run", "--preset", "books_npo_klr", "--out_dir", str(out),
               "--logs_dir", str(tmp / "logs")] + common
        p = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        rec = json.loads((out / "unlearn_run.json").read_text())
        assert "fixes" not in rec["config"] and "deviation" not in rec["config"]
        if (ref_dir / "model.safetensors").exists():
            from safetensors.torch import load_file

            a, b = load_file(str(ref_dir / "model.safetensors")), load_file(str(out / "model.safetensors"))
            r["default_vs_phase2_e2e_bitwise"] = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        else:
            r["default_vs_phase2_e2e_bitwise"] = "reference checkpoint missing"
    return r


def t9_d2_audit():
    import os

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    passthrough = (f"--cpu_test --max_len 128 --data_file {DATA / 'books_forget.txt'} "
                   f"--retain_data_file {DATA / 'books_retain.txt'}")  # = extra/tests/run_cpu_e2e.py step 4
    stored = json.loads((OUT / "sure_eq" / "verdict.json").read_text())
    r = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        v = {}
        for preset in ("books_npo_klr_sure", "books_npo_klr_sure_masked"):
            cmd = [sys.executable, "-m", "extra.tests.test_sure_equivalence", "--preset", preset, "--steps", "4",
                   "--model_dir", str(TINY), "--tokenizer_dir", str(TOK), "--work_dir", str(tmp / preset),
                   "--logs_dir", str(tmp / "logs"), "--runs", "A", "--passthrough", passthrough]
            p = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
            v[preset] = json.loads((tmp / preset / "verdict.json").read_text())
            v[preset]["_rc"] = p.returncode
            v[preset]["_lc"] = [json.loads(x) for x in (tmp / preset / "A" / "loss_components.jsonl").read_text().splitlines()]
        rel, fix = v["books_npo_klr_sure"], v["books_npo_klr_sure_masked"]
        r["released"] = {"rc": rel["_rc"], "rows_changed_outside_mask": rel["D2_rows_changed_outside_mask_total"],
                         "dead_code": rel["D2_mask_is_dead_code"],
                         "audit_identical_to_phase2": rel["D2_mask_audit"] == stored["D2_mask_audit"],
                         "phase2_total": stored["D2_rows_changed_outside_mask_total"]}
        r["fixed"] = {"rc": fix["_rc"], "applied": fix["D2_fixed_mask_applied"],
                      "a_rows_outside_mask_nonzero_grad": fix["D2a_rows_outside_mask_nonzero_grad_total"],
                      "b_rows_changed_outside_mask": fix["D2_rows_changed_outside_mask_total"],
                      "c_of_b_inside_earlier": fix["D2c_rows_changed_outside_mask_prev_inside_total"],
                      "moved_by_optimizer_before_restore": fix["D2_rows_moved_by_optimizer_before_restore_total"],
                      "moved_before_restore_inside_earlier":
                          fix["D2_rows_moved_by_optimizer_before_restore_prev_inside_total"],
                      "rows_changed_inside_mask": fix["D2_rows_changed_inside_mask_total"],
                      "per_step": fix["D2_per_step"], "steps_audited": fix["D2_steps_audited"],
                      "mask_source": fix["D2_mask_audit"][0].get("mask_source")}
        a1, b1 = rel["_lc"][0], fix["_lc"][0]
        r["same_first_batch"] = a1["ce_forget"] == b1["ce_forget"] and a1["ce_retain"] == b1["ce_retain"]
        assert rel["_rc"] == 0 and rel["D2_mask_is_dead_code"] and rel["D2_rows_changed_outside_mask_total"] > 0, r
        assert r["released"]["audit_identical_to_phase2"], r
        assert fix["_rc"] == 0 and fix["D2_fixed_mask_applied"], r
        assert fix["D2a_rows_outside_mask_nonzero_grad_total"] == 0 and fix["D2_rows_changed_inside_mask_total"] > 0, r
        assert fix["D2_steps_audited"] == 4 and r["same_first_batch"], r
    return r


TESTS = {"T1": t1_equivalence, "T2": t2_kl, "T3": t3_npo, "T4": t4_single_grad, "T5": t5_mask_step,
         "T6": t6_mask_fixed, "T7": t7_bf16_rounding, "T8": t8_cli, "T9": t9_d2_audit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--out", default=str(OUT / "fixed_unlearn_tests.json"))
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(TESTS)
    names = [n for n in names if not (a.skip and n in a.skip.split(","))]
    res, ok = {}, True
    for n in names:
        try:
            res[n] = {"ok": True, "result": TESTS[n]()}
        except Exception:
            ok = False
            res[n] = {"ok": False, "error": traceback.format_exc()}
        print(f"{n}: {'OK' if res[n]['ok'] else 'FAIL'}  {json.dumps(res[n].get('result', res[n].get('error')), default=str)[:600]}")
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
