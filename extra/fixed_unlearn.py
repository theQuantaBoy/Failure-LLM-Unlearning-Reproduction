"""
extra/fixed_unlearn.py — OPT-IN corrected versions of the authors' unlearning losses (FINDINGS.md §6).

Everything here is a DEVIATION from "the authors' code as released". It is used only when a FIX_PRESETS name or a
--fix_* flag is passed to extra/unlearn_run.py; the existing presets (extra/common.py::PRESETS) and every existing
result are untouched. Results trained with a fix must be reported separately from the reproduction.

Fixes (each one switchable on its own; all off = the authors' code, proved by extra/tests/test_fixed_unlearn.py T1):

  kl="proper"          Base trainer (IterativeUnlearner) KLR term. Released: F.kl_div(logits, ref_logits,
                       reduction='batchmean', log_target=True) on RAW logits (iterative.py:173-178) = sum exp(t)(t-x)/B,
                       which is not a KL, is unbounded below and not shift-invariant. Fix: the KL the authors' own SURE
                       trainer uses (iterative.py:330-334): kl_div(log_softmax(logits), softmax(ref_logits), 'batchmean')
                       = KL(p_ref || p_theta) summed over tokens / batch. Paper Eq. 10 (App. B.2).
  npo="sequence"       NPO forget term. Released: logsigmoid(beta * (ref_logits - logits)).mean() over every vocab entry of
                       every position (iterative.py:160-161 / 271-272): it ignores the labels and is not shift-invariant.
                       Fix: paper Eq. 8 (App. B.1), log f(x) = sum over tokens of log p(x_t | x_<t) (labels shifted by
                       one, -100 ignored): -2/beta * mean_x logsigmoid(-beta * (log f_theta(x) - log f_target(x))).
  sure_single_grad     SURE computes loss_f.backward(retain_graph=True) inside compute_loss (iterative.py:282) and returns
                       loss = L_f + alpha L_r, which the Trainer back-propagates again (trainer.py:3147): the update uses
                       2 grad L_f + alpha grad L_r. Fix: the first backward is used for the saliency only and its
                       gradients are discarded, so the update uses grad(L_f + alpha L_r) (paper Eq. 1). When no mask is
                       requested the first backward is skipped altogether.
  sure_mask="step"     The mask m_S is built every step (iterative.py:284-305) but only read in SURE.optimizer_step
                       (iterative.py:339-357), which transformers' Trainer never calls (4.40 trainer.py:2266 calls
                       self.optimizer.step() directly). Fix: the same per-step mask (same per-row norms, same
                       np.percentile threshold, same ">= gamma" rule) is applied to the gradients right after the
                       backward (so clipping sees masked gradients) AND to the update: rows with m = 0 are restored after
                       optimizer.step, i.e. theta_{t+1} = theta_t + m_t * Delta_theta_t (paper Eq. 6 per step).
  sure_mask="fixed"    Paper Eq. 4 literally: s_i = ||grad_{theta_i} L_forget(theta; D_forget)|| at theta = theta_o, i.e. one
                       mask from the forget gradient summed over the forget set (first `sure_mask_batches` forget chunks,
                       0 = all) before the first step, fixed for the whole run. Masked rows never receive a gradient, so
                       AdamW (weight decay 0) never moves them.

What sure_mask="step" does with AdamW state (read before interpreting row-change audits): it does NOT reset or mask
the optimizer state. Gradients of rows with m = 0 are zeroed before clipping, so their exp_avg / exp_avg_sq just decay
(beta1 = 0.9, beta2 = 0.999), and optimizer.step can still move such a row by its stored momentum; the row is then
put back bitwise by _MaskRestore. When the row is salient again, its update includes that decayed momentum.
Diagnostics (only when `audit_names` is set, e.g. by unlearn_run --audit_mask): per step and audited parameter,
  rows_outside_mask_nonzero_grad          rows with m = 0 whose gradient is non-zero after masking (must be 0)
  rows_moved_by_optimizer_outside_mask    rows with m = 0 that optimizer.step changed before the restore (momentum)
  ..._prev_inside                         of those, rows that were inside the mask in an earlier step
They are read by the W3 audit (extra/unlearn_run.py), which also counts rows changed AFTER the restore.

Granularity follows the code, not the paper text: the code masks rows ("neurons") of every parameter and elements
of 1-D parameters (iterative.py:289-298); the paper describes module-level masks (Sec. 6.1). Kept as coded so that the
paper's thresholds (Table 5) keep their meaning.
"""

import numpy as np

from extra.common import PRESETS

FIX_KEYS = ("kl", "npo", "sure_single_grad", "sure_mask", "sure_mask_batches")
NO_FIXES = {"kl": "raw", "npo": "logits", "sure_single_grad": False, "sure_mask": "none", "sure_mask_batches": 0}


def _fx(base: str, **fixes) -> dict:
    assert set(fixes) <= set(FIX_KEYS), fixes
    return dict(PRESETS[base], fixes={**NO_FIXES, **fixes}, base_preset=base,
                deviation="W7 fixes active: NOT the authors' code as released")


# New presets only; the hyperparameters are those of the base preset (paper Tables 4/5).
FIX_PRESETS = {
    "books_npo_klr_properkl": _fx("books_npo_klr", kl="proper"),
    "books_npo_klr_properkl_seqnpo": _fx("books_npo_klr", kl="proper", npo="sequence"),
    "books_npo_klr_sure_single": _fx("books_npo_klr_sure", sure_single_grad=True),
    "books_npo_klr_sure_masked": _fx("books_npo_klr_sure", sure_single_grad=True, sure_mask="step"),
    "books_npo_klr_sure_maskfixed": _fx("books_npo_klr_sure", sure_single_grad=True, sure_mask="fixed"),
    "books_ga_gdr_sure_single": _fx("books_ga_gdr_sure", sure_single_grad=True),
    "books_ga_gdr_sure_masked": _fx("books_ga_gdr_sure", sure_single_grad=True, sure_mask="step"),
    "news_npo_klr_properkl": _fx("news_npo_klr", kl="proper"),
    # Provenance test, NOT a fix: at commit 1882021 (2024-10-20) iterative.py defined IterativeUnlearner twice and the
    # second definition (git show 1882021:baselines/baselines/iterative.py, lines 510-652) -- the code that became
    # SURE in 5731982 -- overrode the first, so a base 'npo_klr' run used the proper KL, the doubled forget gradient
    # and the (dead) mask. Running the released SURE class with the base hyperparameters reproduces that path.
    "books_npo_klr_as1882021": dict(PRESETS["books_npo_klr"], algo="npo_klr_sure", base_preset="books_npo_klr",
                                    deviation="provenance test: released SURE class with books_npo_klr hyperparameters "
                                              "(= effective base trainer at commit 1882021); not Table 4's setting"),
}


def any_fix(fixes: dict | None) -> bool:
    return bool(fixes) and any(fixes.get(k, v) != v for k, v in NO_FIXES.items() if k != "sure_mask_batches")


def validate(fixes: dict, algo: str) -> None:
    if fixes["kl"] not in ("raw", "proper") or fixes["npo"] not in ("logits", "sequence") \
            or fixes["sure_mask"] not in ("none", "step", "fixed"):
        raise ValueError(f"unknown fix value in {fixes}")
    if "rmu" in algo:
        raise ValueError("fixes are not defined for rmu")
    if (fixes["sure_single_grad"] or fixes["sure_mask"] != "none") and "sure" not in algo:
        raise ValueError("sure_* fixes apply to *_sure algorithms only")
    if fixes["kl"] == "proper" and ("sure" in algo or "klr" not in algo):
        raise ValueError("kl=proper applies to the base trainer's klr term only (SURE's KL is already proper)")
    if fixes["npo"] == "sequence" and "npo" not in algo:
        raise ValueError("npo=sequence applies to npo* algorithms only")


def sequence_logprob(logits, labels):
    """log f(x) = sum_t log p(x_t | x_<t) per sequence, with the causal shift HF uses for `labels` (-100 ignored)."""
    import torch
    import torch.nn.functional as F

    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    keep = labels != -100
    lp = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * keep).sum(-1)


def row_norms(model):
    """Per-row gradient norms exactly as iterative.py:286-298 (FP32 copy, L2 over all dims but the first; |g| for 1-D)."""
    out = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().float()
            n = grad.norm(2, dim=list(range(1, grad.dim()))) if grad.dim() > 1 else grad.abs()
            out.append((name, n))
    return out


def saliency_mask(model, threshold) -> dict:
    """{param name: bool row tensor, True = salient (updated)}; gamma = np.percentile over all rows (iterative.py:301-305)."""
    norms = row_norms(model)
    allv = np.concatenate([n.cpu().numpy() for _, n in norms])
    gamma = np.percentile(allv, threshold)
    return {name: (n >= gamma).to(n.device) for name, n in norms}


def _row_view(mask_rows, t):
    return mask_rows.view([t.shape[0]] + [1] * (t.dim() - 1)) if t.dim() > 1 else mask_rows


def make_classes(iterative, fixes: dict):
    """Subclasses of the authors' IterativeUnlearner and SURE implementing `fixes` (all off = authors' behaviour)."""
    import torch
    from transformers import TrainerCallback

    fixes = {**NO_FIXES, **(fixes or {})}

    def x_get(x, k, default):
        return x[k] if k in x else default

    class FixedIterativeUnlearner(iterative.IterativeUnlearner):
        """iterative.py:116-196 (rmu branch removed); changes marked [FIX-KL] / [FIX-NPO]."""

        fix = fixes

        def compute_loss(self, model, x, return_outputs=False):
            F = iterative.F  # resolved at call time, so the W6 recorder proxy (extra/unlearn_run.py) still records
            x_f, x_r = x
            outputs_f = model(
                x_f['input_ids'],
                labels=x_get(x_f, 'labels', x_f['input_ids'].clone()),
                attention_mask=x_get(x_f, 'attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
            )
            loss_f = outputs_f.loss

            if 'gdr' in self.loss_type or 'klr' in self.loss_type:
                outputs_r = model(
                    x_r['input_ids'],
                    labels=x_get(x_r, 'labels', x_r['input_ids'].clone()),
                    attention_mask=x_get(x_r, 'attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
                )
                loss_r = outputs_r.loss

            if 'klf' in self.loss_type or 'npo' in self.loss_type:
                with torch.no_grad():
                    outputs_f_ref = self.ref_model(
                        x_f['input_ids'],
                        labels=x_get(x_f, 'labels', x_f['input_ids'].clone()),
                        attention_mask=x_get(x_f, 'attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
                    )

            if 'klr' in self.loss_type:
                with torch.no_grad():
                    outputs_r_ref = self.ref_model(
                        x_r['input_ids'],
                        labels=x_get(x_r, 'labels', x_r['input_ids'].clone()),
                        attention_mask=x_get(x_r, 'attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
                    )

            loss = 0
            if 'ga' in self.loss_type:
                loss += -loss_f
            elif 'npo' in self.loss_type:
                if self.fix["npo"] == "sequence":  # [FIX-NPO] paper Eq. 8
                    labels_f = x_get(x_f, 'labels', x_f['input_ids'])
                    neg_log_ratio = (sequence_logprob(outputs_f_ref.logits, labels_f)
                                     - sequence_logprob(outputs_f.logits, labels_f))
                else:
                    neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
                loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

            if 'gdr' in self.loss_type:
                loss += loss_r * self.alpha
            if 'klf' in self.loss_type:
                raise NotImplementedError("KL forget not implemented yet!")
            if 'klr' in self.loss_type:
                if self.fix["kl"] == "proper":  # [FIX-KL] = iterative.py:330-334, paper Eq. 10
                    kl_r = F.kl_div(
                        F.log_softmax(outputs_r.logits, dim=-1),
                        F.softmax(outputs_r_ref.logits, dim=-1),
                        reduction='batchmean'
                    )
                else:
                    kl_r = F.kl_div(
                        outputs_r.logits,
                        outputs_r_ref.logits,
                        reduction='batchmean',
                        log_target=True
                    )
                loss += kl_r * self.alpha
            if 'rmu' in self.loss_type:
                raise NotImplementedError("rmu is not covered by extra/fixed_unlearn.py")
            return (loss, outputs_f) if return_outputs else loss

    class _MaskRestore(TrainerCallback):
        """sure_mask='step': put rows with m = 0 back to their pre-step values (runs after optimizer.step)."""

        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_begin(self, args, state, control, model=None, **kw):
            if self.trainer.fix["sure_mask"] == "fixed":
                self.trainer._build_fixed_mask(model)

        def on_step_end(self, args, state, control, model=None, **kw):
            t = self.trainer
            if t._saved_rows:
                params = dict(model.named_parameters())
                with torch.no_grad():
                    for name, (keep, saved) in t._saved_rows.items():
                        p = params[name]
                        if t.audit_names and name in t.audit_names:  # diagnostics only; read before the restore
                            cur = p.data[~keep]
                            moved = (cur != saved).reshape(cur.shape[0], -1).any(dim=1) if cur.dim() > 1 \
                                else (cur != saved)
                            prev = t._ever_in_before.get(name)
                            prev_out = prev[~keep] if prev is not None else torch.zeros_like(moved)
                            d = t.mask_diag.setdefault(name, {})
                            d["rows_moved_by_optimizer_outside_mask"] = int(moved.sum())
                            d["rows_moved_by_optimizer_outside_mask_prev_inside"] = int((moved & prev_out).sum())
                        p.data[~keep] = saved
                t._saved_rows = {}

    class FixedSURE(iterative.SURE):
        """iterative.py:240-337; changes marked [FIX-NPO] / [FIX-MASK] / [FIX-SINGLE]."""

        fix = fixes

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self._step_mask = None
            self._fixed_mask = None
            self._saved_rows = {}
            self.last_mask = None
            self.mask_stats = []
            self.audit_names = None      # set by extra/unlearn_run.py --audit_mask (diagnostics only)
            self.mask_diag = {}          # per audited parameter, current step
            self._ever_in = {}           # rows inside the applied mask in any step so far (audited parameters)
            self._ever_in_before = {}    # the same, as of the start of the current step
            if self.fix["sure_mask"] != "none":
                self.add_callback(_MaskRestore(self))

        def _forget_outputs_and_loss(self, model, x_f, loss_components, F):
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f.get('labels', x_f['input_ids'].clone()),
                attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
            )
            if 'ga' in loss_components:
                loss_f = -outputs_f.loss
            elif 'npo' in loss_components:
                with torch.no_grad():
                    outputs_f_ref = self.ref_model(
                        x_f['input_ids'],
                        labels=x_f.get('labels', x_f['input_ids'].clone()),
                        attention_mask=x_f.get('attention_mask', torch.ones_like(x_f['input_ids'], dtype=torch.bool))
                    )
                if self.fix["npo"] == "sequence":  # [FIX-NPO] paper Eq. 8
                    labels_f = x_f.get('labels', x_f['input_ids'])
                    neg_log_ratio = (sequence_logprob(outputs_f_ref.logits, labels_f)
                                     - sequence_logprob(outputs_f.logits, labels_f))
                else:
                    neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
                loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            else:
                raise ValueError("Unknown loss component for forget data.")
            return outputs_f, loss_f

        def compute_loss(self, model, x, return_outputs=False):
            F = iterative.F
            x_f, x_r = x
            loss_components = self.loss_type.split('_')
            loss = 0
            outputs_f = outputs_r = None
            self.m_S = None  # the authors' dict is never built here; SURE.optimizer_step stays dead code

            if 'ga' in loss_components or 'npo' in loss_components:
                outputs_f, loss_f = self._forget_outputs_and_loss(model, x_f, loss_components, F)
                loss += loss_f
                need_first_backward = self.fix["sure_mask"] == "step" or not self.fix["sure_single_grad"]
                if need_first_backward:
                    self.optimizer.zero_grad()
                    loss_f.backward(retain_graph=True)
                    if self.fix["sure_mask"] == "step":  # [FIX-MASK] same mask as iterative.py:284-305
                        self._step_mask = saliency_mask(model, self.threshold)
                    if self.fix["sure_single_grad"]:  # [FIX-SINGLE] discard grad L_f of the saliency pass
                        model.zero_grad(set_to_none=True)
            else:
                raise ValueError("No valid forget data loss component found in loss_type.")

            if 'gdr' in loss_components or 'klr' in loss_components:
                outputs_r = model(
                    x_r['input_ids'],
                    labels=x_r.get('labels', x_r['input_ids'].clone()),
                    attention_mask=x_r.get('attention_mask', torch.ones_like(x_r['input_ids'], dtype=torch.bool))
                )
            if 'gdr' in loss_components:
                loss_r = outputs_r.loss
                loss += self.alpha * loss_r
            if 'klr' in loss_components:
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

        # [FIX-MASK] -----------------------------------------------------------------------------------------
        def _build_fixed_mask(self, model):
            """Eq. 4 at theta_o: grad of L_forget summed over the forget chunks, then the per-row threshold."""
            F = iterative.F
            loss_components = self.loss_type.split('_')
            ds, collate = self.train_dataset, self.data_collator
            n = len(ds) if not self.fix["sure_mask_batches"] else min(len(ds), self.fix["sure_mask_batches"])
            model.zero_grad(set_to_none=True)
            for i in range(n):
                x_f, _ = self._prepare_inputs(collate([ds[i]]))
                with self.compute_loss_context_manager():
                    _, loss_f = self._forget_outputs_and_loss(model, x_f, loss_components, F)
                loss_f.backward()
            self._fixed_mask = saliency_mask(model, self.threshold)
            model.zero_grad(set_to_none=True)
            self.mask_stats.append({"fixed_mask_batches": n, **self._mask_summary(self._fixed_mask)})

        @staticmethod
        def _mask_summary(mask):
            rows = sum(int(m.numel()) for m in mask.values())
            kept = sum(int(m.sum()) for m in mask.values())
            return {"rows": rows, "salient_rows": kept, "salient_frac": kept / max(rows, 1)}

        def training_step(self, model, inputs):
            loss = super().training_step(model, inputs)  # compute_loss + the Trainer's backward
            mode = self.fix["sure_mask"]
            if mode == "none":
                return loss
            mask = self._step_mask if mode == "step" else self._fixed_mask
            params = dict(model.named_parameters())
            self.mask_diag = {}
            self._ever_in_before = {n: v.clone() for n, v in self._ever_in.items()}
            with torch.no_grad():
                for name, p in params.items():
                    keep = mask.get(name)
                    if keep is None:  # no forget gradient for this parameter -> m = 0 (iterative.py:347 default 0.0)
                        keep = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
                    if p.grad is not None:
                        p.grad.mul_(_row_view(keep, p.grad).to(p.grad.dtype))
                    if self.audit_names and name in self.audit_names:  # diagnostics only
                        g = p.grad
                        nz = (torch.zeros(p.shape[0], dtype=torch.bool, device=p.device) if g is None
                              else (g != 0).reshape(g.shape[0], -1).any(dim=1) if g.dim() > 1 else (g != 0))
                        self.mask_diag[name] = {"rows_outside_mask_nonzero_grad": int((nz & ~keep).sum())}
                        self._ever_in[name] = self._ever_in.get(name, torch.zeros_like(keep)) | keep
                    if mode == "step" and not bool(keep.all()):
                        self._saved_rows[name] = (keep, p.data[~keep].clone())
            self.last_mask = mask  # read by extra/tests/test_fixed_unlearn.py
            if mode == "step":
                self._step_mask = None
                if len(self.mask_stats) < 50:
                    self.mask_stats.append(self._mask_summary(mask))
            return loss

    return FixedIterativeUnlearner, FixedSURE
