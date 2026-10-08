"""
extra/unlearn_run.py — run the authors' unlearning code (baselines/baselines/iterative.py::unlearn) unchanged.

It is the equivalent of `cd baselines && python unlearn.py --algo ... ` (baselines/unlearn.py:38-50), with these
documented wrappers. None of them changes the optimisation (tests in extra/tests/ check this):

  W1  TrainingArguments factory, visible only inside iterative.py (iterative.py:54 calls
      `transformers.TrainingArguments(...)`). It keeps every argument the authors pass and adds
        save_only_model=True  (per-epoch checkpoints hold BF16 weights only, no optimizer/scheduler/RNG
                               state; transformers 4.40 trainer.py:2752-2758)
        seed=<--seed>         (default 42 = the TrainingArguments default the authors use implicitly)
      Test-only overrides, rejected unless --test_mode: max_steps, save_strategy="no", logging_steps.
  W2  Subclasses of IterativeUnlearner / SURE that add a TrainerCallback (wall time, s/step, peak GPU memory,
      progress written to the run log every 25 steps) and, for SURE, count calls to SURE.optimizer_step
      (iterative.py:339; the D2 check of extra/tests/test_sure_equivalence.py expects 0).
  W3  --audit_mask (SURE only, diagnostics): before/after each step, copies selected parameters to CPU and
      counts rows that changed although the step's saliency mask m_S is 0 for them. With a W7 mask fix
      (sure_mask step/fixed) the mask read is the one actually applied in that step (FixedSURE.last_mask,
      extra/fixed_unlearn.py; rows of a parameter without forget gradient count as 0, as in its training_step);
      those records carry "mask_source". The released-code records are unchanged.
  W4  --sure_fast (SURE only; D3 check in extra/tests/test_sure_equivalence.py): compute_loss is the authors' code verbatim (iterative.py:240-337) except that
      the per-neuron dict / percentile / mask construction (iterative.py:284-305), whose result m_S is never
      read by anything that runs, is skipped. loss_f.backward(retain_graph=True) and optimizer.zero_grad() are
      kept. Allowed only after extra/tests/test_sure_equivalence.py passes on GPU.
  W6  Loss-component recorder (read-only): a forward hook on the trainable model records the CE loss of its two
      calls per step (1st = forget batch, 2nd = retain batch; iterative.py:122-135 / 254-316), and iterative.py's
      `F` is replaced by a proxy whose logsigmoid / kl_div call the real functions and return the SAME tensor while
      recording its value (NPO term iterative.py:161 / 272, KL term :173-178 / :330-334). Per step:
      forget_term, retain_term, their sum and the raw pieces go to <out_dir>/loss_components.jsonl.
  W5  --cpu_test: iterative.unlearn raises without CUDA (iterative.py:51-52); for local CPU tests only,
      iterative.device_count is replaced by `lambda: 1`.
  W7  OPT-IN fixes (DEVIATION from the released code; FINDINGS.md §6): a FIX_PRESETS name
      (extra/fixed_unlearn.py) or --fix_kl/--fix_npo/--fix_sure_single_grad/--fix_sure_mask. The wrapped trainers
      then derive from fixed_unlearn.make_classes(...) instead of the authors' classes. Without a fix nothing changes.

Usage (repo layout as in modal_app.py):
  python -m extra.unlearn_run --preset books_npo_klr_sure --out_dir /vol/runs/ckpt/books/books_npo_klr_sure_s42 \
      --model_dir <local snapshot of MUSE-books_target> --tokenizer_dir <local snapshot of the Llama-2 tokenizer> \
      --logs_dir /vol/runs/logs
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from extra.common import MAX_LEN, PRESETS, REPO_DIR, DEFAULT_TRAIN_SEED, RunLog, write_json
from extra.fixed_unlearn import FIX_PRESETS, NO_FIXES, any_fix, make_classes, validate as validate_fixes


class _ModuleProxy:
    """Forwards attribute access to a real module, except for the names given as overrides."""

    def __init__(self, module, **overrides):
        self._module = module
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._module, name)


def _rng_fingerprint() -> str:
    import random

    import numpy as np
    import torch

    h = hashlib.sha1()
    h.update(repr(random.getstate()).encode())
    h.update(np.random.get_state()[1].tobytes())
    h.update(torch.random.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        h.update(torch.cuda.random.get_rng_state().numpy().tobytes())
    return h.hexdigest()


def _make_callback(runlog, stats):
    import torch
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            stats["train_begin"] = time.time()
            stats["max_steps"] = state.max_steps
            stats["rng_at_train_begin"] = _rng_fingerprint()

        def on_step_end(self, args, state, control, **kw):
            stats["global_step"] = state.global_step
            if state.global_step in (1, 2, 3, 5, 10) or state.global_step % 25 == 0 \
                    or state.global_step == state.max_steps:
                el = time.time() - stats["train_begin"]
                stats["s_per_step"] = round(el / max(state.global_step, 1), 3)
                if torch.cuda.is_available():
                    devs = range(torch.cuda.device_count())
                    alloc = [torch.cuda.max_memory_allocated(d) / 1e9 for d in devs]
                    resv = [torch.cuda.max_memory_reserved(d) / 1e9 for d in devs]
                    total = [torch.cuda.get_device_properties(d).total_memory / 1e9 for d in devs]
                    stats["peak_gpu_mem_gb"] = round(max(alloc), 2)  # max over devices, allocated by tensors
                    stats["gpu_mem"] = {"peak_allocated_gb": [round(x, 2) for x in alloc],
                                        "peak_reserved_gb": [round(x, 2) for x in resv],
                                        "device_total_gb": [round(x, 2) for x in total],
                                        "min_headroom_gb": round(min(t - r for t, r in zip(total, resv)), 2)}
                if "loss_components_last" in stats:
                    stats["loss_components_recent"] = stats["loss_components_last"]
                stats["last_logs"] = state.log_history[-3:]
                runlog.update(progress=dict(stats))

        def on_train_end(self, args, state, control, **kw):
            stats["rng_at_train_end"] = _rng_fingerprint()
            stats["log_history"] = state.log_history

    return ProgressCallback()


class _LossRecorder:
    """W6: read-only capture of the pieces the authors' compute_loss sums (see docstring)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ce, self.logsig_mean, self.kl = [], [], []

    def hook(self, module, args, output):  # forward hook: never modifies the output
        loss = getattr(output, "loss", None)
        if loss is not None:
            self.ce.append(float(loss.detach().float()))

    def functional_proxy(self, real_F):
        rec = self

        def logsigmoid(*a, **k):
            out = real_F.logsigmoid(*a, **k)
            rec.logsig_mean.append(float(out.detach().float().mean()))
            return out

        def kl_div(*a, **k):
            out = real_F.kl_div(*a, **k)
            rec.kl.append(float(out.detach().float()))
            return out

        return _ModuleProxy(real_F, logsigmoid=logsigmoid, kl_div=kl_div)

    def components(self, loss_type: str, alpha, beta) -> dict:
        parts = loss_type.split("_")
        d = {"ce_forget": self.ce[0] if self.ce else None,
             "ce_retain": self.ce[1] if len(self.ce) > 1 else None,
             "npo_logsigmoid_mean": self.logsig_mean[0] if self.logsig_mean else None,
             "kl_raw": self.kl[0] if self.kl else None, "n_model_forwards": len(self.ce)}
        if "ga" in parts:
            d["forget_term"] = -d["ce_forget"]                      # iterative.py:157 / 262
        elif "npo" in parts:
            d["forget_term"] = -d["npo_logsigmoid_mean"] * 2 / beta  # iterative.py:161 / 272
        if "gdr" in parts:
            d["retain_term"] = alpha * d["ce_retain"]               # iterative.py:167 / 320
        elif "klr" in parts:
            d["retain_term"] = alpha * d["kl_raw"]                  # iterative.py:179 / 335
        d["total"] = d.get("forget_term", 0.0) + d.get("retain_term", 0.0)
        return d


def _make_loss_callback(recorder, trainer_ref, out_path: Path, stats):
    from transformers import TrainerCallback

    class LossComponentsCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            recorder.reset()

        def on_step_end(self, args, state, control, **kw):
            t = trainer_ref[0]
            d = {"step": state.global_step, **recorder.components(t.loss_type, t.alpha, t.beta)}
            with open(out_path, "a") as fh:
                fh.write(json.dumps(d) + "\n")
            stats["loss_components_last"] = d

    return LossComponentsCallback()


def _make_mask_audit(trainer_ref, names, stats):
    """W3: counts rows that changed in a step although that step's m_S marks them 0."""
    import torch
    from transformers import TrainerCallback

    class MaskAudit(TrainerCallback):
        def __init__(self):
            self.before = {}

        def on_step_begin(self, args, state, control, model=None, **kw):
            params = dict(model.named_parameters())
            self.before = {n: params[n].detach().to("cpu", copy=True) for n in names if n in params}

        def on_step_end(self, args, state, control, model=None, **kw):
            t = trainer_ref[0]
            fix = getattr(t, "fix", None)
            applied = bool(fix) and fix.get("sure_mask", "none") != "none"  # W7 path: audit the applied mask
            m_S = t.m_S
            params = dict(model.named_parameters())
            if applied:
                last = t.last_mask
                rec = {"step": state.global_step, "mask_available": last is not None,
                       "mask_source": "applied mask (fixed_unlearn FixedSURE.last_mask)", "params": {}}
            else:
                rec = {"step": state.global_step, "mask_available": m_S is not None, "params": {}}
            for n, b in self.before.items():
                a = params[n].detach().cpu()
                diff = (a != b)
                changed = diff.reshape(diff.shape[0], -1).any(dim=1) if diff.dim() > 1 else diff
                if applied:
                    if last is None:
                        mask = torch.full((a.shape[0],), float("nan"))
                    elif n in last:
                        mask = last[n].detach().cpu().float()
                    else:  # no forget gradient -> masked to 0 in FixedSURE.training_step
                        mask = torch.zeros(a.shape[0])
                else:
                    mask = torch.tensor([m_S.get(f"{n}.{i}", 0.0) if m_S else float("nan") for i in range(a.shape[0])])
                rec["params"][n] = {
                    "rows": int(a.shape[0]),
                    "rows_changed": int(changed.sum()),
                    "rows_in_mask": int((mask == 1).sum()),
                    "rows_changed_outside_mask": int((changed & (mask == 0)).sum()),
                    "rows_unchanged_inside_mask": int((~changed & (mask == 1)).sum()),
                }
                if applied:  # W7 diagnostics (fixed_unlearn FixedSURE.mask_diag); released records unchanged
                    d = t.mask_diag.get(n, {})
                    prev = t._ever_in_before.get(n)
                    prev = prev.detach().cpu() if prev is not None else torch.zeros(a.shape[0], dtype=torch.bool)
                    rec["params"][n].update({
                        "rows_outside_mask_nonzero_grad": d.get("rows_outside_mask_nonzero_grad"),       # (a)
                        "rows_changed_outside_mask_prev_inside": int((changed & (mask == 0) & prev).sum()),  # (c)
                        "rows_moved_by_optimizer_outside_mask": d.get("rows_moved_by_optimizer_outside_mask", 0),
                        "rows_moved_by_optimizer_outside_mask_prev_inside":
                            d.get("rows_moved_by_optimizer_outside_mask_prev_inside", 0),
                        "rows_outside_mask_prev_inside": int(((mask == 0) & prev).sum()),
                    })
            stats.setdefault("mask_audit", []).append(rec)
            self.before = {}

    return MaskAudit()


_FAST_F = None


def _fast_sure_compute_loss(self, model, x, return_outputs=False):
    """W4: iterative.py:240-337 verbatim, minus the m_S construction (iterative.py:284-305)."""
    import torch

    F = _FAST_F  # = iterative.F (the W6 proxy), so the recorder also sees the fast path's terms

    x_f, x_r = x

    loss_components = self.loss_type.split('_')

    loss = 0
    outputs_f = outputs_r = None

    # Reset saliency mask
    self.m_S = None

    ### Compute loss on forget data ###
    if 'ga' in loss_components or 'npo' in loss_components:
        # Compute loss on forget data
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f.get('labels', x_f['input_ids'].clone()),
            attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
        )

        if 'ga' in loss_components:
            # Gradient Ascent on forget data
            loss_f = -outputs_f.loss
        elif 'npo' in loss_components:
            # NPO loss on forget data
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f.get('labels', x_f['input_ids'].clone()),
                    attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
                )
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
        else:
            raise ValueError("Unknown loss component for forget data.")

        loss += loss_f

        # Zero existing gradients
        self.optimizer.zero_grad()

        # Backward pass for loss_f to get gradients
        loss_f.backward(retain_graph=True)

        # [W4] iterative.py:284-305 (neuron_grad_norms dict, np.percentile, m_S dict) skipped: m_S is only read
        # by SURE.optimizer_step, which transformers 4.40 never calls (trainer.py:2266). self.m_S stays None.

    else:
        raise ValueError("No valid forget data loss component found in loss_type.")

    ### Compute loss on retain data ###
    if 'gdr' in loss_components or 'klr' in loss_components:
        outputs_r = model(
            x_r['input_ids'],
            labels=x_r.get('labels', x_r['input_ids'].clone()),
            attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
        )
    if 'gdr' in loss_components:
        # Gradient Descent on retain data
        loss_r = outputs_r.loss
        loss += self.alpha * loss_r  # Use self.alpha to weight the retain data loss

    if 'klr' in loss_components:
        # KL Divergence Regularization on retain data
        with torch.no_grad():
            outputs_r_ref = self.ref_model(
                x_r['input_ids'],
                labels=x_r.get('labels', x_r['input_ids'].clone()),
                attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
            )
        kl_r = F.kl_div(
            F.log_softmax(outputs_r.logits, dim=-1),
            F.softmax(outputs_r_ref.logits, dim=-1),
            reduction='batchmean'
        )
        loss += self.alpha * kl_r

    return (loss, outputs_f) if return_outputs else loss


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS) + sorted(FIX_PRESETS),
                   help="hyperparameters from paper Tables 4/5; FIX_PRESETS names add W7 fixes (deviation)")
    p.add_argument("--algo")
    p.add_argument("--corpus", choices=["news", "books"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--alpha", type=int, help="int, as in baselines/unlearn.py:97")
    p.add_argument("--threshold", type=int)
    p.add_argument("--per_device_batch_size", type=int)
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--data_file", help="default: <repo>/data/<corpus>/raw/forget.txt")
    p.add_argument("--retain_data_file", help="default: <repo>/data/<corpus>/raw/retain1.txt")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--logs_dir", required=True)
    p.add_argument("--run_name")
    p.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    p.add_argument("--sure_fast", action="store_true", help="W4 / D3")
    p.add_argument("--audit_mask", action="store_true", help="W3 (SURE only)")
    p.add_argument("--audit_params", nargs="*", default=None,
                   help="parameter names for W3 (default: layer 0 and last layer q_proj/down_proj + lm_head)")
    p.add_argument("--cpu_test", action="store_true", help="W5")
    p.add_argument("--test_mode", action="store_true", help="allow the test-only overrides below")
    p.add_argument("--max_steps", type=int)
    # W7 (deviation from the released code; see extra/fixed_unlearn.py)
    p.add_argument("--fix_kl", choices=["raw", "proper"])
    p.add_argument("--fix_npo", choices=["logits", "sequence"])
    p.add_argument("--fix_sure_single_grad", action="store_true", default=None)
    p.add_argument("--fix_sure_mask", choices=["none", "step", "fixed"])
    p.add_argument("--fix_sure_mask_batches", type=int)
    p.add_argument("--save_strategy")
    p.add_argument("--logging_steps", type=int)
    return p


def resolve_config(a) -> dict:
    cfg = dict(PRESETS[a.preset]) if a.preset in PRESETS else dict(FIX_PRESETS[a.preset]) if a.preset else {}
    for k_arg, k_cfg in [("algo", "algo"), ("corpus", "corpus"), ("epochs", "epochs"), ("lr", "lr"),
                         ("alpha", "alpha"), ("threshold", "threshold"), ("per_device_batch_size", "bs")]:
        v = getattr(a, k_arg)
        if v is not None:
            cfg[k_cfg] = v
    missing = [k for k in ("algo", "corpus", "epochs", "lr", "alpha", "threshold", "bs") if k not in cfg]
    if missing:
        raise SystemExit(f"missing hyperparameters {missing}: pass --preset or set them explicitly")
    data = REPO_DIR / "data" / cfg["corpus"] / "raw"
    cfg.update(
        max_len=a.max_len, model_dir=a.model_dir, tokenizer_dir=a.tokenizer_dir,
        data_file=a.data_file or str(data / "forget.txt"),
        retain_data_file=a.retain_data_file or str(data / "retain1.txt"),
        out_dir=a.out_dir, seed=a.seed, sure_fast=a.sure_fast, audit_mask=a.audit_mask,
        cpu_test=a.cpu_test, test_overrides={k: getattr(a, k) for k in ("max_steps", "save_strategy", "logging_steps")
                                             if getattr(a, k) is not None},
    )
    if cfg["test_overrides"] and not a.test_mode:
        raise SystemExit(f"{sorted(cfg['test_overrides'])} are test-only overrides; add --test_mode")
    if (a.sure_fast or a.audit_mask) and "sure" not in cfg["algo"]:
        raise SystemExit("--sure_fast/--audit_mask apply to *_sure algorithms only")
    fixes = dict(cfg.pop("fixes", NO_FIXES))
    for k in ("kl", "npo", "sure_single_grad", "sure_mask", "sure_mask_batches"):
        v = getattr(a, f"fix_{k}")
        if v is not None:
            fixes[k] = v
    if any_fix(fixes):
        if a.preset in PRESETS:  # modal_app.unlearn_job names the ckpt dir after the preset: never reuse a released one
            raise SystemExit("--fix_* flags need a FIX_PRESETS name (own checkpoint dir), not a released preset")
        try:
            validate_fixes(fixes, cfg["algo"])
        except ValueError as e:
            raise SystemExit(str(e))
        if a.sure_fast:
            raise SystemExit("--sure_fast is for the released SURE only; not combinable with fixes")
        if a.audit_mask and fixes["sure_mask"] == "none":
            raise SystemExit("--audit_mask with fixes needs a mask fix (sure_mask step/fixed): nothing is applied")
        cfg["fixes"] = fixes
        cfg["deviation"] = "W7 fixes active: NOT the authors' code as released"
    elif a.preset in FIX_PRESETS:
        cfg["deviation"] = FIX_PRESETS[a.preset]["deviation"]
    return cfg


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = resolve_config(a)
    name = a.run_name or Path(a.out_dir).name
    runlog = RunLog(a.logs_dir, "unlearn", name, cfg)
    stats = {}
    try:
        # Same import context as `cd baselines && python unlearn.py` (unlearn.py:1-6).
        sys.path.insert(0, str(REPO_DIR / "baselines"))
        import transformers
        from baselines import iterative

        # W1 ------------------------------------------------------------------------------------------
        real_ta = transformers.TrainingArguments
        captured = {}

        def training_arguments(**authors_kwargs):
            captured["authors_kwargs"] = {k: str(v) for k, v in authors_kwargs.items()}
            kw = dict(authors_kwargs)
            overrides = {"save_only_model": True, "seed": cfg["seed"]}
            overrides.update(cfg["test_overrides"])
            kw.update(overrides)
            captured["overrides"] = overrides
            ta = real_ta(**kw)
            captured["effective"] = {k: str(v) for k, v in ta.to_dict().items()}
            return ta

        iterative.transformers = _ModuleProxy(transformers, TrainingArguments=training_arguments)

        # W6 ------------------------------------------------------------------------------------------
        global _FAST_F
        recorder = _LossRecorder()
        iterative.F = recorder.functional_proxy(iterative.F)
        _FAST_F = iterative.F
        Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
        comp_path = Path(cfg["out_dir"]) / "loss_components.jsonl"
        if comp_path.exists():
            comp_path.unlink()

        # W2/W3/W4 ------------------------------------------------------------------------------------
        trainer_ref = []
        counters = {"sure_optimizer_step_calls": 0}
        callback = _make_callback(runlog, stats)
        audit_names = a.audit_params

        def _attach_recorder(trainer):
            trainer.model.register_forward_hook(recorder.hook)
            trainer.add_callback(_make_loss_callback(recorder, trainer_ref, comp_path, stats))

        base_cls, sure_cls = iterative.IterativeUnlearner, iterative.SURE
        if "fixes" in cfg:  # W7
            base_cls, sure_cls = make_classes(iterative, cfg["fixes"])

        class BaseWrapped(base_cls):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                trainer_ref.append(self)
                self.add_callback(callback)
                _attach_recorder(self)

        class SUREWrapped(sure_cls):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                trainer_ref.append(self)
                self.add_callback(callback)
                _attach_recorder(self)
                if cfg["audit_mask"]:
                    names = audit_names
                    if not names:
                        n_layers = self.model.config.num_hidden_layers
                        names = [f"model.layers.{i}.{m}" for i in (0, n_layers - 1)
                                 for m in ("self_attn.q_proj.weight", "mlp.down_proj.weight")] + ["lm_head.weight"]
                    if getattr(self, "fix", None) and self.fix.get("sure_mask", "none") != "none":
                        self.audit_names = set(names)  # W7 diagnostics; the released SURE is not touched
                    self.add_callback(_make_mask_audit(trainer_ref, names, stats))

            def optimizer_step(self, *args, **kw):
                counters["sure_optimizer_step_calls"] += 1
                return super().optimizer_step(*args, **kw)

        if cfg["sure_fast"]:
            SUREWrapped.compute_loss = _fast_sure_compute_loss

        iterative.IterativeUnlearner = BaseWrapped
        iterative.SURE = SUREWrapped

        if cfg["cpu_test"]:  # W5
            iterative.device_count = lambda: 1

        # Same call as baselines/unlearn.py:38-50.
        iterative.unlearn(
            cfg["model_dir"], cfg["data_file"], cfg["out_dir"],
            retain_data_file=cfg["retain_data_file"],
            loss_type=cfg["algo"],
            per_device_batch_size=cfg["bs"],
            epochs=cfg["epochs"],
            learning_rate=cfg["lr"],
            max_len=cfg["max_len"],
            tokenizer_dir=cfg["tokenizer_dir"],
            resume_from_checkpoint=False,
            alpha=cfg["alpha"],
            threshold=cfg["threshold"],
        )
        stats["train_end"] = time.time()
        if trainer_ref and getattr(trainer_ref[0], "mask_stats", None):
            stats["fix_mask_stats"] = trainer_ref[0].mask_stats
        write_json({"config": cfg, "training_arguments": captured, "counters": counters,
                    "stats": stats}, Path(cfg["out_dir"]) / "unlearn_run.json")
        runlog.finish(0, training_arguments=captured, counters=counters, progress=stats)
        print(f"[unlearn_run] done; SURE.optimizer_step calls = {counters['sure_optimizer_step_calls']}")
        return 0
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        runlog.finish(1, error=tb, progress=stats)
        return 1


if __name__ == "__main__":
    sys.exit(main())
