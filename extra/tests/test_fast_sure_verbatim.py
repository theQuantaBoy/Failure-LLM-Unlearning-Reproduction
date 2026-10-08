"""
Static check for W4 / D3: extra.unlearn_run._fast_sure_compute_loss must equal SURE.compute_loss
(baselines/baselines/iterative.py:240-337) line for line, except the skipped block iterative.py:283-305
(the neuron_grad_norms dict, np.percentile and the m_S dict) and the function-local imports.
Comment-only and blank lines are ignored on both sides.  Run:  python -m extra.tests.test_fast_sure_verbatim
"""

import inspect
import sys

from extra.common import REPO_DIR


def code_lines(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(ln.rstrip())
    return out


def main() -> int:
    src = (REPO_DIR / "baselines" / "baselines" / "iterative.py").read_text().split("\n")
    # 1-based line numbers: body 241-337, skipped 283-305
    original = src[240:282] + src[305:337]
    assert src[281].strip() == "loss_f.backward(retain_graph=True)", src[281]
    assert src[304].strip().startswith("self.m_S = {neuron_name"), src[304]
    assert src[336].strip() == "return (loss, outputs_f) if return_outputs else loss", src[336]

    from extra.unlearn_run import _fast_sure_compute_loss

    mine = inspect.getsource(_fast_sure_compute_loss).split("\n")
    body = mine[mine.index('    """W4: iterative.py:240-337 verbatim, minus the m_S construction (iterative.py:284-305)."""') + 1:]
    # function-local name bindings: `import torch`, and `F` bound to iterative.F (the read-only W6 proxy around
    # torch.nn.functional; it returns the same tensors) instead of `import torch.nn.functional as F`
    body = [ln for ln in body if ln.strip() not in ("import torch", "import torch.nn.functional as F")
            and not ln.strip().startswith("F = _FAST_F")]
    # the authors' method is indented one level deeper (class body)
    a = [ln[4:] if ln.startswith("    ") else ln for ln in code_lines(original)]
    b = code_lines(body)
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"first difference at code line {i}:\n  authors: {x!r}\n  fast:    {y!r}")
                break
        print(f"authors {len(a)} code lines, fast {len(b)} code lines")
        print("FAIL")
        return 1
    print(f"OK: {len(a)} code lines identical; only iterative.py:283-305 (m_S construction) is skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
