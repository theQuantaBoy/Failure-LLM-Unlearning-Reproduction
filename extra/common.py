"""
extra/common.py — shared constants and helpers for the wrappers in extra/.

Pure standard library so it can be imported from the paper image (Python 3.10, transformers 4.40),
the quantization image (Python 3.12, transformers 5.x) and locally by compare_results.py.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXTRA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXTRA_DIR.parent
# The authors' repo. Inside Modal containers it is mounted at /root/FailureLLMUnlearning (see modal_app.py);
# FAILUNL_REPO overrides the location for local CPU tests.
REPO_DIR = Path(os.environ.get("FAILUNL_REPO", PROJECT_DIR / "FailureLLMUnlearning")).resolve()

# ── Pinned Hugging Face artifacts (queried 2026-10-03; Reproducibility.md §3) ────────────────────────────
MODELS = {
    "news_target": ("muse-bench/MUSE-news_target", "a2f39769e9a0b98ec1cdd12f65e9962502208935"),
    "news_retrain": ("muse-bench/MUSE-news_retrain", "324ef49ee0a038078aba7d8de831edf57235c9b3"),
    "books_target": ("muse-bench/MUSE-books_target", "c8dd3fb23a726762ec66d277933c7cff6767f3c2"),
    "books_retrain": ("muse-bench/MUSE-books_retrain", "1d67430e4e8bdf2a65823740e909792519175ac2"),
}
# Two tokenizer sources: meta-llama (gated) is the authors' tokenizer; the NousResearch mirror has a byte-identical
# tokenizer.model. FAILUNL_TOKENIZER selects which entry is used.
TOKENIZERS = {
    "meta": ("meta-llama/Llama-2-7b-hf", "01c7f73d771dfac7d292323805ebc428287df4f9"),
    "nous": ("NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"),
}
TOKENIZER_KEY = os.environ.get("FAILUNL_TOKENIZER", "meta")
DATASETS = {
    "muse_news": ("muse-bench/MUSE-News", "506bd5b150b92814d45e4404a82f120ab2d748bf"),
    "muse_books": ("muse-bench/MUSE-Books", "051ba90319e920d410d87cfdbd61f25843c1b892"),
    # General-text calibration (GPTQ / AWQ). Revision is resolved and recorded by modal_app.py::download.
    "wikitext": ("Salesforce/wikitext", None),
}

# ── Unlearning presets: paper App. D.1 Table 4 (base) and App. D.2 Table 5 (SURE, BOOKS) ────────────────
# bs = per_device_batch_size (= global batch: 1 GPU, no gradient accumulation; Reproducibility.md §4); max_len and retain1 from the scripts.
PRESETS = {
    "news_ga_gdr": dict(corpus="news", algo="ga_gdr", epochs=10, lr=1e-5, alpha=1, threshold=90, bs=2),
    "news_npo_klr": dict(corpus="news", algo="npo_klr", epochs=10, lr=1e-5, alpha=1, threshold=90, bs=2),
    "books_ga_gdr": dict(corpus="books", algo="ga_gdr", epochs=5, lr=1e-5, alpha=100, threshold=90, bs=1),
    "books_npo_klr": dict(corpus="books", algo="npo_klr", epochs=5, lr=1e-5, alpha=2, threshold=90, bs=1),
    "books_ga_gdr_sure": dict(corpus="books", algo="ga_gdr_sure", epochs=5, lr=1e-4, alpha=400, threshold=99, bs=1),
    "books_npo_klr_sure": dict(corpus="books", algo="npo_klr_sure", epochs=5, lr=1e-4, alpha=20, threshold=90, bs=1),
}
MAX_LEN = 2048  # baselines/scripts/unlearn_*.sh
DEFAULT_TRAIN_SEED = 42  # transformers TrainingArguments default; the authors never set a seed
DEFAULT_EVAL_SEED = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def git_rev(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        # Inside Modal the repo is copied without .git; the revision is baked in at image build time.
        return os.environ.get("FAILUNL_REPO_REV", "unknown")


def versions(pkgs=("torch", "transformers", "accelerate", "bitsandbytes", "datasets", "tokenizers",
                   "numpy", "scipy", "scikit-learn", "rouge-score", "nltk", "huggingface-hub",
                   "llmcompressor", "compressed-tensors", "safetensors")) -> dict:
    from importlib import metadata

    out = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def gpu_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
    except Exception:
        pass
    return {"devices": ["cpu"]}


class RunLog:
    """One JSON log file per run (never append to a shared log).

    Written at start (status=running) and rewritten at the end (status, exit code, end time, results).
    """

    def __init__(self, logs_dir, kind: str, name: str, config: dict):
        self.path = Path(logs_dir).resolve() / f"{stamp()}_{kind}_{name}.json"
        self.t0 = time.time()
        self.data = {
            "kind": kind,
            "name": name,
            "status": "running",
            "start": utc_now(),
            "argv": sys.argv,
            "config": config,
            "repo_rev": git_rev(REPO_DIR),
            "versions": versions(),
            "gpu": gpu_info(),
            "host": platform.node(),
        }
        write_json(self.data, self.path)

    def update(self, **kw) -> None:
        self.data.update(kw)
        write_json(self.data, self.path)

    def finish(self, exit_code: int, **kw) -> None:
        self.data.update(kw)
        self.data.update(
            status="ok" if exit_code == 0 else "failed",
            exit_code=exit_code,
            end=utc_now(),
            minutes=round((time.time() - self.t0) / 60, 2),
        )
        write_json(self.data, self.path)
