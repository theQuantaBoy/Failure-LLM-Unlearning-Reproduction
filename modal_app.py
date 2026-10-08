"""
modal_app.py — run the authors' code (FailureLLMUnlearning @ 10131ae) on Modal: downloads, unlearning, evaluation.
GPTQ/AWQ/RTN-INT4 live in modal_quant.py (different library versions). See COMMANDS.md for the full sequence.

One-time setup (laptop):
    pip install modal==1.5.5 && modal setup
    modal secret create huggingface HF_TOKEN=hf_...      # an HF account with access to meta-llama/Llama-2-7b-hf

Every entrypoint *spawns* the job and returns; with --detach the job survives a dropped connection
Follow progress with `modal app logs failunl` or the dashboard, and in the
per-run JSON logs on the Volume:  modal volume ls failunl-runs logs

    modal run --detach modal_app.py::download --what models,tokenizer,wikitext,verify_data
    modal run --detach modal_app.py::train --preset books_npo_klr --seed 42
    modal run --detach modal_app.py::evaluate --model hf:books_target --corpus books --name target --quant none
    modal run --detach modal_app.py::evaluate --model ckpt/books/books_npo_klr_s42 --corpus books \
        --name npo_klr --quant bnb4
    modal run --detach modal_app.py::smoke_sure                  # D2 + D3 GPU tests (extra/tests/test_sure_equivalence.py)
    modal run modal_app.py::env_info
    # move a final checkpoint to another account through a private HF repo (COMMANDS.md §8)
    MODAL_PROFILE=<source> modal run --detach modal_app.py::push_ckpt --run books_npo_klr_sure_s42
    MODAL_PROFILE=<target> modal run --detach modal_app.py::pull_ckpt --run books_npo_klr_sure_s42 \
        --revision <hf_commit from the source's ckpt/books/<run>/failunl_transfer_manifest.json>

Layout on Volume failunl-runs (/vol/runs):
    ckpt/<corpus>/<preset>_s<seed>/{checkpoint-<step>/ (BF16 weights per epoch), final model files (= last epoch,
        written by trainer.save_model, iterative.py:90), unlearn_run.json}
    quant/<source-tag>/<method>_g<gs>_<calib>[_s<seed>]/{compressed/, dq/, quant_report.json}   (modal_quant.py)
    results/<corpus>/<name>/<tag>/{metrics.json, *.json per-example}
    logs/<stamp>_<kind>_<name>.json   (one per run, wrapper RunLog)   logs/<stamp>_<kind>_<name>.out (stdout)
Volume failunl-hfcache (/vol/hf): HF hub cache (pinned revisions) + calib/wikitext2_train (save_to_disk).
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

# ─── Settings ───────────────────────────────────────────────────────────────────────────────────────────────
TRAIN_GPU = "H200"  # 141 GB: fits every run without changing the code (Reproducibility.md §2)
EVAL_GPU = "A10"  # 24 GB is enough for BF16 inference of a 7B model
TRAIN_TIMEOUT_H = 5  # longest estimate (BOOKS NPO_KLR+SURE as-is) ~3 h; caps a hung H200 job at ~$25
EVAL_TIMEOUT_H = 3
TORCH_VERSION = "2.2.0"  # environment.yml
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
RESOLVE_AS_OF = "2024-10-25"  # day after the authors' environment.yml commit (631fcec, 2024-10-24)
REPO_REV = "10131ae25f55f1d8feb744eabb235ffc3f094b1b"
TOKENIZER_KEY = os.environ.get("FAILUNL_TOKENIZER", "meta")  # "meta" or "nous" (extra/common.py TOKENIZERS)

HERE = Path(__file__).parent
LOCAL_REPO = HERE / "FailureLLMUnlearning"
REMOTE_REPO = "/root/FailureLLMUnlearning"
REMOTE_PROJ = "/root/proj"  # contains extra/ (importable as package `extra`)
HF_DIR = "/vol/hf"
RUNS_DIR = "/vol/runs"

app = modal.App("failunl")
hf_vol = modal.Volume.from_name("failunl-hfcache", create_if_missing=True)
runs_vol = modal.Volume.from_name("failunl-runs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")
VOLUMES = {HF_DIR: hf_vol, RUNS_DIR: runs_vol}

COMMON_ENV = {
    "HF_HOME": HF_DIR,
    "HF_TOKEN_PATH": "/tmp/hf-auth/token",  # never persist a token into the cache Volume
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "FAILUNL_REPO": REMOTE_REPO,
    "FAILUNL_REPO_REV": REPO_REV,
    "FAILUNL_TOKENIZER": TOKENIZER_KEY,
    "PYTHONPATH": REMOTE_PROJ,
}
REPO_IGNORE = [".git", "**/__pycache__", "temp", "output.csv"]

paper_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("uv")
    .add_local_file(HERE / "extra" / "requirements-paper.txt", "/tmp/requirements-paper.txt", copy=True)
    .run_commands(
        f"uv pip install --system --index-url {TORCH_INDEX} torch=={TORCH_VERSION}",
        f"uv pip install --system --exclude-newer {RESOLVE_AS_OF} -r /tmp/requirements-paper.txt",
        # R6: nltk 3.9.1 word_tokenize needs punkt_tab; eval_fluency.py:45 only downloads punkt.
        "python -c \"import nltk; [nltk.download(p, download_dir='/usr/share/nltk_data') for p in ('punkt','punkt_tab')]\"",
    )
    .env({**COMMON_ENV, "HF_HUB_OFFLINE": "1", "NLTK_DATA": "/usr/share/nltk_data",
          "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(LOCAL_REPO, REMOTE_REPO, ignore=REPO_IGNORE)
    .add_local_dir(HERE / "extra", f"{REMOTE_PROJ}/extra", ignore=["**/__pycache__", "tests/_out*"])
)

dl_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands("uv pip install --system --exclude-newer 2026-10-01 datasets==5.0.1 huggingface_hub hf_xet")
    .env(COMMON_ENV)
    .add_local_dir(LOCAL_REPO, REMOTE_REPO, ignore=REPO_IGNORE)
    .add_local_dir(HERE / "extra", f"{REMOTE_PROJ}/extra", ignore=["**/__pycache__", "tests/_out*"])
)

# Checkpoint transfer between accounts (push_ckpt / pull_ckpt). Pinned to the versions extra/tests/test_ckpt_transfer.py
# runs against (.venvs/quant). HF_HOME and the xet chunk cache live on container disk: nothing goes to failunl-hfcache.
xfer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands("uv pip install --system --exclude-newer 2026-10-01 huggingface_hub==1.33.0 hf_xet==1.6.0")
    .env({**COMMON_ENV, "HF_HOME": "/tmp/hf-home", "HF_XET_CACHE": "/tmp/hf-xet"})
    .add_local_dir(HERE / "extra", f"{REMOTE_PROJ}/extra", ignore=["**/__pycache__", "tests/_out*"])
)
XFER_TIMEOUT_H = 2  # 12.6 GiB per BOOKS checkpoint; expected 5-20 min (COMMANDS.md §8)


# ─── helpers (run inside containers)────────────────────────────────────────────────────────────────────────
def snapshot(key: str) -> str:
    """Local path of a pinned HF snapshot in the cache Volume (downloaded by `download`)."""
    sys.path.insert(0, REMOTE_PROJ)
    from extra.common import MODELS, TOKENIZERS

    repo, rev = MODELS[key] if key in MODELS else TOKENIZERS[key]
    p = Path(HF_DIR) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if not p.exists():
        raise FileNotFoundError(f"{repo}@{rev} not in cache: run `modal run modal_app.py::download` first")
    return str(p)


def resolve_model(ref: str) -> str:
    """'hf:<key>' -> pinned snapshot; otherwise a path relative to the runs Volume."""
    if ref.startswith("hf:"):
        return snapshot(ref[3:])
    p = Path(RUNS_DIR) / ref
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"no HF checkpoint at {p}")
    return str(p)


def default_name(model: str, corpus: str) -> str:
    """Row key used by compare_results.py: 'target' or '<method>_s<seed>'.
    hf:books_target -> target; ckpt/books/books_npo_klr_s42 -> npo_klr_s42;
    quant/books_npo_klr_s42/gptq_g128_general/dq -> npo_klr_s42;
    ckpt/news/news_ga_gdr_s42/checkpoint-408 -> ga_gdr_s42 (per-epoch checkpoint: name of the training run)."""
    if model.startswith("hf:"):
        key = model[3:]
        return "target" if key.endswith("_target") else key.split("_", 1)[1]
    parts = Path(model).parts
    if parts[-1].startswith("checkpoint-"):
        parts = parts[:-1]
    src = parts[1] if parts[0] == "quant" else parts[-1]
    return "target" if src == f"{corpus}_target" else src.removeprefix(f"{corpus}_")


def default_tag(model: str, quant: str, seed: int, model_dir: str = "", extra_args: str = "") -> str:
    """bf16 | bnb4 | <quant dir name> (e.g. gptq_g128_general), plus _ep<epoch> for a per-epoch checkpoint
    (epoch from its trainer_state.json, written even with save_only_model, transformers trainer.py:2776-2778;
    _ckpt<step> where that file is not readable, e.g. the local preview line) and _e<seed> for a non-default
    eval seed. The FP32 KnowMem diagnostic (--diag_fp32_knowmem, a deviation) is filed as fp32diag_knowmem."""
    if "--diag_fp32_knowmem" in extra_args:
        return "fp32diag_knowmem" if seed == 0 else f"fp32diag_knowmem_e{seed}"
    parts = Path(model).parts
    tag = parts[2] if parts and parts[0] == "quant" else ("bnb4" if quant == "bnb4" else "bf16")
    if parts and parts[-1].startswith("checkpoint-"):
        state = Path(model_dir) / "trainer_state.json" if model_dir else None
        tag += (f"_ep{round(json.loads(state.read_text())['epoch'])}" if state and state.exists()
                else f"_ckpt{parts[-1].removeprefix('checkpoint-')}")
    return tag if seed == 0 else f"{tag}_e{seed}"


def run_logged(argv: list, out_name: str) -> int:
    """Run a wrapper as a subprocess; stdout+stderr are streamed to the Modal log AND to logs/<out_name>.out."""
    from datetime import datetime, timezone

    logs = Path(RUNS_DIR) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = logs / f"{stamp}_{out_name}.out"
    print(f"▶ {' '.join(shlex.quote(x) for x in argv)}\n  stdout → {out_path}", flush=True)
    with open(out_path, "w") as fh:
        fh.write(f"# cmd: {shlex.join(argv)}\n# start: {stamp}\n")
        fh.flush()
        proc = subprocess.Popen(argv, cwd=REMOTE_PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = proc.wait()
        fh.write(f"# exit code: {rc}\n")
    runs_vol.commit()
    return rc


# ─── downloads (CPU only) ───────────────────────────────────────────────────────────────────────────────────
@app.function(image=dl_image, cpu=4.0, memory=16384, timeout=4 * 3600, secrets=[HF_SECRET], volumes=VOLUMES)
def download_job(what: str) -> None:
    import json

    from huggingface_hub import HfApi, snapshot_download

    sys.path.insert(0, REMOTE_PROJ)
    from extra.common import DATASETS, MODELS, TOKENIZERS, write_json

    items = [w.strip() for w in what.split(",") if w.strip()]
    report = {}
    if "models" in items or any(i in MODELS for i in items):
        keys = list(MODELS) if "models" in items else [i for i in items if i in MODELS]
        for k in keys:
            repo, rev = MODELS[k]
            p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors"])
            report[k] = {"repo": repo, "revision": rev, "path": p}
            hf_vol.commit()
    if "tokenizer" in items:
        for k in [TOKENIZER_KEY]:
            repo, rev = TOKENIZERS[k]
            p = snapshot_download(repo, revision=rev,
                                  allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json",
                                                  "generation_config.json"])
            report[f"tokenizer_{k}"] = {"repo": repo, "revision": rev, "path": p}
    if "wikitext" in items:
        from datasets import load_dataset

        repo = DATASETS["wikitext"][0]
        sha = HfApi().dataset_info(repo).sha
        ds = load_dataset(repo, "wikitext-2-raw-v1", split="train", revision=sha)
        out = Path(HF_DIR) / "calib" / "wikitext2_train"
        ds.save_to_disk(str(out))
        write_json({"repo": repo, "config": "wikitext-2-raw-v1", "split": "train", "revision": sha,
                    "rows": len(ds)}, out / "SOURCE.json")
        report["wikitext"] = {"repo": repo, "revision": sha, "rows": len(ds), "path": str(out)}
    if "verify_data" in items:
        report["verify_data"] = _verify_data()
    hf_vol.commit()
    write_json(report, Path(RUNS_DIR) / "logs" / "download_report.json")
    runs_vol.commit()
    print(json.dumps(report, indent=2))


def _verify_data() -> dict:
    """The committed data/ must equal load_data.py's output at the pinned dataset revisions."""
    import json

    from datasets import load_dataset

    from extra.common import DATASETS

    repo_data = Path(REMOTE_REPO) / "data"
    res = {}
    for corpus, key in (("news", "muse_news"), ("books", "muse_books")):
        repo, rev = DATASETS[key]
        ld = lambda cfg, split: load_dataset(repo, cfg, split=split, revision=rev)  # noqa: E731
        checks = {}
        for split in ("forget_qa", "retain_qa", "forget_qa_icl", "retain_qa_icl"):
            d = ld("knowmem", split)
            mine = [{"question": q, "answer": a} for q, a in zip(d["question"], d["answer"])]
            checks[f"knowmem/{split}"] = mine == json.loads((repo_data / corpus / "knowmem" / f"{split}.json").read_text())
        d = ld("verbmem", "forget")
        mine = [{"prompt": p, "gt": g} for p, g in zip(d["prompt"], d["gt"])]
        checks["verbmem/forget"] = mine == json.loads((repo_data / corpus / "verbmem" / "forget.json").read_text())
        for split in ("forget", "retain", "holdout"):
            checks[f"privleak/{split}"] = ld("privleak", split)["text"] == json.loads(
                (repo_data / corpus / "privleak" / f"{split}.json").read_text())
        for split in ("forget", "holdout", "retain1", "retain2"):
            raw = ld("raw", split)["text"]
            checks[f"raw/{split}.json"] = raw == json.loads((repo_data / corpus / "raw" / f"{split}.json").read_text())
            checks[f"raw/{split}.txt"] = "\n\n".join(raw) == (repo_data / corpus / "raw" / f"{split}.txt").read_text()
        res[corpus] = {"revision": rev, "all_equal": all(checks.values()), "checks": checks}
    return res


# ─── unlearning ─────────────────────────────────────────────────────────────────────────────────────────────
@app.function(image=paper_image, gpu=TRAIN_GPU, cpu=4.0, memory=32768, timeout=TRAIN_TIMEOUT_H * 3600,
              volumes=VOLUMES)
def unlearn_job(preset: str, seed: int = 42, sure_fast: bool = False, extra_args: str = "",
                out_rel: str = "") -> int:
    sys.path.insert(0, REMOTE_PROJ)
    from extra.common import PRESETS
    from extra.fixed_unlearn import FIX_PRESETS  # W7 opt-in fixes (deviation); own names -> own ckpt dirs

    corpus = {**PRESETS, **FIX_PRESETS}[preset]["corpus"]
    out_rel = out_rel or f"ckpt/{corpus}/{preset}_s{seed}"
    argv = [sys.executable, "-m", "extra.unlearn_run", "--preset", preset, "--seed", str(seed),
            "--model_dir", snapshot(f"{corpus}_target"), "--tokenizer_dir", snapshot(TOKENIZER_KEY),
            "--out_dir", f"{RUNS_DIR}/{out_rel}", "--logs_dir", f"{RUNS_DIR}/logs"]
    if sure_fast:
        argv.append("--sure_fast")
    argv += shlex.split(extra_args)
    rc = run_logged(argv, f"unlearn_{Path(out_rel).name}")
    if rc != 0:
        raise RuntimeError(f"unlearning failed with exit code {rc}")
    return rc


# ─── evaluation ─────────────────────────────────────────────────────────────────────────────────────────────
@app.function(image=paper_image, gpu=EVAL_GPU, cpu=4.0, memory=32768, timeout=EVAL_TIMEOUT_H * 3600,
              volumes=VOLUMES)
def eval_job(model: str, corpus: str, name: str = "", quant: str = "none", tag: str = "", seed: int = 0,
             extra_args: str = "") -> int:
    model_dir = resolve_model(model)
    name = name or default_name(model, corpus)
    tag = tag or default_tag(model, quant, seed, model_dir, extra_args)
    argv = [sys.executable, "-m", "extra.eval_run", "--model_dir", model_dir, "--name", name, "--corpus", corpus,
            "--quant", quant, "--seed", str(seed), "--tokenizer_dir", snapshot(TOKENIZER_KEY),
            "--out_dir", f"{RUNS_DIR}/results/{corpus}/{name}/{tag}", "--logs_dir", f"{RUNS_DIR}/logs"]
    qrep = Path(model_dir).parent / "quant_report.json"
    if qrep.exists():
        argv += ["--quant_report", str(qrep)]
    argv += shlex.split(extra_args)
    rc = run_logged(argv, f"eval_{corpus}_{name}_{tag}")
    if rc != 0:
        raise RuntimeError(f"evaluation failed with exit code {rc}")
    return rc


# ─── checkpoint transfer between Modal accounts via a private HF repo (CPU only; extra/ckpt_transfer.py) ─────
def _xfer(kind: str, run: str, corpus: str, config: dict, fn) -> dict:
    sys.path.insert(0, REMOTE_PROJ)
    from extra.common import RunLog

    if corpus not in ("books", "news"):
        raise ValueError(f"corpus must be books or news, not {corpus!r}")
    log = RunLog(f"{RUNS_DIR}/logs", kind, run, {"run": run, "corpus": corpus, **config})
    try:
        rec = fn()
    except BaseException as e:
        log.finish(1, error=repr(e))
        runs_vol.commit()
        raise
    log.finish(0, result=rec)
    runs_vol.commit()
    return {k: rec[k] for k in ("repo_id", "hf_commit", "total_bytes", "status")}


@app.function(image=xfer_image, cpu=4.0, memory=16384, timeout=XFER_TIMEOUT_H * 3600, secrets=[HF_SECRET],
              volumes={RUNS_DIR: runs_vol})
def push_ckpt_job(run: str, corpus: str = "books", repo: str = "", overwrite: bool = False) -> dict:
    """ckpt/<corpus>/<run>/ final files -> private HF repo (default <hf user>/failunl-<run>), one commit, verified."""
    from huggingface_hub import HfApi

    from extra.ckpt_transfer import push

    return _xfer("push_ckpt", run, corpus, {"repo": repo, "overwrite": overwrite},
                 lambda: push(HfApi(), f"{RUNS_DIR}/ckpt/{corpus}/{run}", run, corpus, repo, overwrite))


@app.function(image=xfer_image, cpu=4.0, memory=16384, timeout=XFER_TIMEOUT_H * 3600, secrets=[HF_SECRET],
              volumes={RUNS_DIR: runs_vol})
def pull_ckpt_job(run: str, corpus: str = "books", repo: str = "", revision: str = "main",
                  overwrite: bool = False) -> dict:
    """private HF repo -> ckpt/<corpus>/<run>/ (same path as in the source account); sha256 checked, raises on any diff."""
    from huggingface_hub import HfApi

    from extra.ckpt_transfer import pull

    return _xfer("pull_ckpt", run, corpus, {"repo": repo, "revision": revision, "overwrite": overwrite},
                 lambda: pull(HfApi(), f"{RUNS_DIR}/ckpt/{corpus}", run, corpus, repo, revision, overwrite))


# ─── GPU smoke tests (D2, D3, per-epoch saving)──────────────────────────────────────────────────────────────────────────
@app.function(image=paper_image, gpu=TRAIN_GPU, cpu=4.0, memory=32768, timeout=int(1.5 * 3600), volumes=VOLUMES)
def smoke_sure_job(preset: str = "books_npo_klr_sure", steps: int = 20, runs: str = "ABC") -> int:
    """D2: optimizer_step counter + mask audit; D3: original x2 vs fast x1 (extra/tests/test_sure_equivalence.py).
    W7 mask presets (FIX_PRESETS, deviation): runs="A" only; D2 then checks the applied mask holds."""
    sys.path.insert(0, REMOTE_PROJ)
    from extra.common import PRESETS
    from extra.fixed_unlearn import FIX_PRESETS

    corpus = {**PRESETS, **FIX_PRESETS}[preset]["corpus"]
    argv = [sys.executable, "-m", "extra.tests.test_sure_equivalence", "--preset", preset, "--steps", str(steps),
            "--model_dir", snapshot(f"{corpus}_target"), "--tokenizer_dir", snapshot(TOKENIZER_KEY),
            "--work_dir", f"{RUNS_DIR}/smoke/{preset}", "--logs_dir", f"{RUNS_DIR}/logs", "--delete_weights"]
    if runs != "ABC":
        argv += ["--runs", runs]
    return run_logged(argv, f"smoke_sure_{preset}")


@app.function(image=paper_image, gpu=TRAIN_GPU, cpu=4.0, memory=32768, timeout=2 * 3600, volumes=VOLUMES)
def smoke_train_job(preset: str = "news_npo_klr", steps: int = 30) -> int:
    """Measures s/step and peak memory of one real configuration (max_steps, no checkpoints kept)."""
    sys.path.insert(0, REMOTE_PROJ)
    from extra.common import PRESETS
    from extra.fixed_unlearn import FIX_PRESETS  # W7 presets (deviation) are smoke-testable too

    corpus = {**PRESETS, **FIX_PRESETS}[preset]["corpus"]
    out = f"{RUNS_DIR}/smoke/train_{preset}"
    argv = [sys.executable, "-m", "extra.unlearn_run", "--preset", preset, "--model_dir",
            snapshot(f"{corpus}_target"), "--tokenizer_dir", snapshot(TOKENIZER_KEY), "--out_dir", out,
            "--logs_dir", f"{RUNS_DIR}/logs", "--test_mode", "--max_steps", str(steps), "--save_strategy", "no",
            "--logging_steps", "1"]
    rc = run_logged(argv, f"smoke_train_{preset}")
    subprocess.run(["bash", "-c", f"rm -f {out}/*.safetensors"], check=False)  # keep logs, drop 13.5 GB weights
    runs_vol.commit()
    return rc


@app.function(image=paper_image, gpu=EVAL_GPU, cpu=4.0, memory=32768, timeout=3600, volumes=VOLUMES)
def eval_ab_job(model: str = "hf:books_target", corpus: str = "books", n: int = 5, quant: str = "none") -> int:
    """A/B: authors' eval.py vs extra/eval_run.py on the same model and first-n examples (extra/tests/test_eval_ab.py)."""
    argv = [sys.executable, "-m", "extra.tests.test_eval_ab", "--model_dir", resolve_model(model), "--corpus", corpus,
            "--n", str(n), "--quant", quant, "--work_dir", f"{RUNS_DIR}/smoke/eval_ab_{corpus}_{quant}",
            "--logs_dir", f"{RUNS_DIR}/logs"]
    # eval.py hard-codes the hub id meta-llama/Llama-2-7b-hf (eval.py:78, LLAMA_DIR). Offline, an id resolves through
    # refs/main, which snapshot_download(revision=<sha>) does not write. So always serve the pinned snapshot's files
    # under that id from a private offline cache (also covers FAILUNL_TOKENIZER=nous).
    argv += ["--tokenizer_fixture", snapshot(TOKENIZER_KEY)]
    return run_logged(argv, f"smoke_eval_ab_{corpus}_{quant}")


@app.function(image=paper_image, gpu=EVAL_GPU, timeout=600, volumes={RUNS_DIR: runs_vol})
def env_info_job() -> str:
    cmd = ("nvidia-smi; python --version; python -m pip freeze; "
           "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
           "torch.cuda.get_device_name(0))\"")
    text = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = Path(RUNS_DIR) / "logs" / "environment_paper.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.stdout + text.stderr)
    runs_vol.commit()
    return text.stdout


# ─── local entrypoints (spawn and return) ───────────────────────────────────────────────────────────────────
def _gpu(gpu: str, count: int) -> str:
    return f"{gpu}:{count}" if count > 1 else gpu


def _spawned(call, what: str) -> None:
    """Print the call id, then wait for the job.

    Always launch with `modal run --detach`: without it, the client's disconnect at the end of the entrypoint (or a
    dropped connection) terminates all running tasks of the ephemeral app (modal 1.5.5 runner.py:332-333). Waiting
    here means a forgotten --detach only kills the job if the connection drops, not immediately; with --detach the
    job keeps running after Ctrl-C / disconnect (runner.py:495-497, 515-519) and this wait is merely cosmetic.
    """
    print(f"spawned {what}: function call {call.object_id}\n"
          f"follow: `modal app logs failunl` or the dashboard; per-run logs: `modal volume ls failunl-runs logs`\n"
          f"waiting for it to finish (safe to close the terminal only if you used `modal run --detach`) ...",
          flush=True)
    result = call.get()
    print(f"finished {what}: return value {result!r}")


@app.local_entrypoint()
def download(what: str = "models,tokenizer,wikitext,verify_data"):
    _spawned(download_job.spawn(what), f"download {what}")


@app.local_entrypoint()
def train(preset: str, seed: int = 42, sure_fast: bool = False, gpu: str = TRAIN_GPU, gpu_count: int = 1,
          extra_args: str = ""):
    fn = unlearn_job.with_options(gpu=_gpu(gpu, gpu_count))
    _spawned(fn.spawn(preset, seed, sure_fast, extra_args), f"unlearn {preset} seed={seed} on {_gpu(gpu, gpu_count)}")


@app.local_entrypoint()
def evaluate(model: str, corpus: str, name: str = "", quant: str = "none", tag: str = "", seed: int = 0,
             gpu: str = EVAL_GPU, extra_args: str = ""):
    if "--diag_fp32_knowmem" in extra_args and gpu == EVAL_GPU:
        gpu = "L40S"  # FP32 7B = ~27 GB of weights: does not fit the A10's 24 GB
    fn = eval_job.with_options(gpu=gpu)
    _spawned(fn.spawn(model, corpus, name, quant, tag, seed, extra_args),
             f"eval {corpus}/{name or default_name(model, corpus)}/{tag or default_tag(model, quant, seed, '', extra_args)}"
             f" on {gpu}")


@app.local_entrypoint()
def smoke_sure(preset: str = "books_npo_klr_sure", steps: int = 20, gpu: str = TRAIN_GPU, runs: str = "ABC"):
    _spawned(smoke_sure_job.with_options(gpu=gpu).spawn(preset, steps, runs), f"SURE smoke test {preset} ({runs})")


@app.local_entrypoint()
def smoke_train(preset: str = "news_npo_klr", steps: int = 30, gpu: str = TRAIN_GPU, gpu_count: int = 1):
    _spawned(smoke_train_job.with_options(gpu=_gpu(gpu, gpu_count)).spawn(preset, steps), f"train smoke {preset}")


@app.local_entrypoint()
def eval_ab(model: str = "hf:books_target", corpus: str = "books", n: int = 5, quant: str = "none",
            gpu: str = EVAL_GPU):
    _spawned(eval_ab_job.with_options(gpu=gpu).spawn(model, corpus, n, quant), f"eval A/B {model} ({quant}, n={n})")


@app.local_entrypoint()
def push_ckpt(run: str, corpus: str = "books", repo: str = "", overwrite: bool = False):
    _spawned(push_ckpt_job.spawn(run, corpus, repo, overwrite),
             f"push ckpt/{corpus}/{run} -> HF {repo or '(default repo)'}")


@app.local_entrypoint()
def pull_ckpt(run: str, corpus: str = "books", repo: str = "", revision: str = "main", overwrite: bool = False):
    if revision == "main":
        print("WARNING: --revision not given: pulling the newest commit of the repo, not necessarily the one push_ckpt "
              "verified. Pass --revision <hf_commit> from the source manifest (COMMANDS.md §11).", flush=True)
    _spawned(pull_ckpt_job.spawn(run, corpus, repo, revision, overwrite),
             f"pull HF {repo or '(default repo)'}@{revision} -> ckpt/{corpus}/{run}")


@app.local_entrypoint()
def env_info():
    print(env_info_job.remote())
