"""
extra/ckpt_transfer.py — move a final checkpoint between Modal accounts through a PRIVATE Hugging Face repo.

    push (source account):  ckpt/<corpus>/<run>/ final files  ->  hf://<user>/failunl-<run>  (one commit)
    pull (target account):  hf://<user>/failunl-<run>@<commit>  ->  ckpt/<corpus>/<run>/      (same Volume path)

What moves (top-level regular files only; checkpoint-*/ epoch dirs are never read):
    config.json, generation_config.json (sampling defaults used by VerbMem), *.safetensors (+ index),
    tokenizer files, unlearn_run.json (provenance: preset, seed, training arguments).
What stays: checkpoint-*/, training_args.bin (pickle), loss_components.jsonl, anything else (listed as "excluded").

Integrity:
    push  sha256 + size of every file -> MANIFEST (committed with the files, and written into the run dir on the
          Volume together with the HF commit id). After the commit, each remote file is checked against the manifest
          (LFS/xet files: lfs.sha256; small files: git blob sha1). Any difference raises.
    pull  the repo's file list must equal the manifest; files are downloaded to a staging dir, sha256 recomputed,
          then moved to ckpt/<corpus>/<run>/ and hashed AGAIN there. Any missing / extra / size / sha256 difference
          raises; the target path only appears after the staging copy verified.
The transfer does not change any byte of the files that downstream jobs read, so results are unaffected.

Pure standard library at import time; huggingface_hub is used only through the `api` argument (an HfApi, or the
fake in extra/tests/test_ckpt_transfer.py) and the CommitOperation* classes imported inside push().
"""

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

from extra.common import read_json, utc_now, write_json

MANIFEST = "failunl_transfer_manifest.json"  # in the HF repo AND in ckpt/<corpus>/<run>/ on both Volumes
PULL_REPORT = "failunl_pull_report.json"  # target Volume only
TRANSFER_PATTERNS = ("config.json", "generation_config.json", "*.safetensors", "model.safetensors.index.json",
                     "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                     "added_tokens.json", "unlearn_run.json")
REPO_HOUSEKEEPING = {".gitattributes"}  # created by create_repo; never part of the checkpoint
CHUNK = 16 << 20


# ── files and hashes ─────────────────────────────────────────────────────────────────────────────────────────
def select_files(run_dir) -> tuple:
    """(transferred, excluded): sorted names of top-level files. Directories (checkpoint-*) are not descended."""
    run_dir = Path(run_dir)
    keep, skip = [], []
    for p in sorted(run_dir.iterdir()):
        if not p.is_file() or p.name in (MANIFEST, PULL_REPORT):
            continue
        (keep if any(fnmatch(p.name, pat) for pat in TRANSFER_PATTERNS) else skip).append(p.name)
    return keep, skip


def check_complete(run_dir, files) -> None:
    """Refuse a run that has not finished or whose weights are incomplete."""
    run_dir, files = Path(run_dir), set(files)
    problems = [f"missing {n}" for n in ("config.json", "generation_config.json", "unlearn_run.json")
                if n not in files]
    if "model.safetensors.index.json" in files:
        shards = set(read_json(run_dir / "model.safetensors.index.json")["weight_map"].values())
        problems += [f"missing shard {s} (listed in model.safetensors.index.json)" for s in sorted(shards - files)]
    elif "model.safetensors" not in files:
        problems.append("no model.safetensors and no model.safetensors.index.json")
    if problems:
        raise ValueError(f"{run_dir} is not a complete final checkpoint: " + "; ".join(problems))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path) -> str:
    """git object id of a non-LFS file (= `git hash-object`), as reported by the Hub as blob_id."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hash_files(run_dir, names, workers: int = 4) -> dict:
    """{name: {"size", "sha256"}}; hashlib releases the GIL, so shards hash in parallel."""
    run_dir = Path(run_dir)
    with ThreadPoolExecutor(workers) as ex:
        digests = dict(zip(names, ex.map(lambda n: sha256_file(run_dir / n), names)))
    return {n: {"size": (run_dir / n).stat().st_size, "sha256": digests[n]} for n in names}


def verify_dir(run_dir, manifest: dict) -> list:
    """Problems (empty = identical): missing / extra / size / sha256, against manifest["files"]."""
    run_dir = Path(run_dir)
    want = manifest["files"]
    have = {p.name for p in run_dir.iterdir() if p.is_file() and p.name not in (MANIFEST, PULL_REPORT)}
    problems = [f"missing {n}" for n in sorted(set(want) - have)]
    problems += [f"unexpected file {n}" for n in sorted(have - set(want))]
    common = sorted(set(want) & have)
    got = hash_files(run_dir, common)
    for n in common:
        if got[n]["size"] != want[n]["size"]:
            problems.append(f"size mismatch {n}: {got[n]['size']} != manifest {want[n]['size']}")
        elif got[n]["sha256"] != want[n]["sha256"]:
            problems.append(f"sha256 mismatch {n}: {got[n]['sha256']} != manifest {want[n]['sha256']}")
    return problems


def default_repo(api, run: str) -> str:
    return f"{api.whoami()['name']}/failunl-{run}"


def _token_role(api):
    try:
        return api.whoami()["auth"]["accessToken"]["role"]
    except (KeyError, TypeError):
        return None


def _repo_files(api, repo_id: str, revision: str) -> set:
    return set(api.list_repo_files(repo_id, revision=revision)) - REPO_HOUSEKEEPING


# ── push ─────────────────────────────────────────────────────────────────────────────────────────────────────
def push(api, run_dir, run: str, corpus: str, repo_id: str = "", overwrite: bool = False, log=print) -> dict:
    """Upload the final checkpoint in ONE commit to a private repo, verify it remotely, write the Volume manifest."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.errors import RepositoryNotFoundError

    run_dir = Path(run_dir)
    files, excluded = select_files(run_dir)
    check_complete(run_dir, files)
    if _token_role(api) == "read":
        raise PermissionError("the HF token in Secret 'huggingface' is a READ token: pushing needs a write token "
                              "(or a fine-grained token with write access to your own repos)")
    repo_id = repo_id or default_repo(api, run)

    try:
        info = api.repo_info(repo_id)
        existed = True
    except RepositoryNotFoundError:
        existed = False
    if not existed:
        api.create_repo(repo_id, private=True, repo_type="model")
        info = api.repo_info(repo_id)
    if info.private is not True:
        raise PermissionError(f"{repo_id} is not private (private={info.private!r}): refusing to upload")
    stale = _repo_files(api, repo_id, info.sha) if existed else set()
    if stale and not overwrite:
        raise FileExistsError(f"{repo_id} already holds {len(stale)} files: pass --overwrite to replace them")

    log(f"hashing {len(files)} files of {run_dir} ...")
    manifest = {
        "format": 1, "run": run, "corpus": corpus, "source_path": f"ckpt/{corpus}/{run}", "repo_id": repo_id,
        "created": utc_now(), "hash": "sha256", "files": hash_files(run_dir, files), "excluded": excluded,
    }
    manifest["total_bytes"] = sum(f["size"] for f in manifest["files"].values())
    write_json({**manifest, "status": "hashed, not uploaded"}, run_dir / MANIFEST)

    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(run_dir / n)) for n in files]
    ops.append(CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=json.dumps(manifest, indent=2).encode()))
    ops += [CommitOperationDelete(path_in_repo=n) for n in sorted(stale - set(files) - {MANIFEST})]
    log(f"uploading {manifest['total_bytes'] / 2**30:.2f} GiB to {repo_id} (private) in one commit ...")
    commit = api.create_commit(repo_id, operations=ops, commit_message=f"failunl checkpoint {corpus}/{run}")

    problems = remote_check(api, repo_id, commit.oid, run_dir, manifest)
    record = {**manifest, "status": "ok" if not problems else "REMOTE MISMATCH", "hf_commit": commit.oid,
              "remote_check": problems or "all files match (size + lfs sha256 / git blob sha1)"}
    write_json(record, run_dir / MANIFEST)
    if problems:
        raise RuntimeError(f"remote copy of {repo_id}@{commit.oid} differs from the Volume: {problems}")
    log(f"pushed {repo_id}@{commit.oid}: {len(files)} files verified remotely")
    return record


def remote_check(api, repo_id: str, revision: str, run_dir, manifest: dict) -> list:
    """Compare the committed files with the manifest without downloading them."""
    run_dir, want = Path(run_dir), manifest["files"]
    problems = []
    remote = _repo_files(api, repo_id, revision)
    if remote != set(want) | {MANIFEST}:
        problems.append(f"repo files {sorted(remote)} != manifest files + {MANIFEST}")
    for info in api.get_paths_info(repo_id, sorted(want), revision=revision):
        w = want[info.path]
        if info.size != w["size"]:
            problems.append(f"remote size {info.path}: {info.size} != {w['size']}")
        elif info.lfs is not None:
            if info.lfs.sha256 != w["sha256"]:
                problems.append(f"remote sha256 {info.path}: {info.lfs.sha256} != {w['sha256']}")
        elif info.blob_id != git_blob_sha1(run_dir / info.path):
            problems.append(f"remote git blob {info.path}: {info.blob_id} != local")
    return problems


# ── pull ─────────────────────────────────────────────────────────────────────────────────────────────────────
def pull(api, ckpt_root, run: str, corpus: str, repo_id: str = "", revision: str = "main", overwrite: bool = False,
         log=print) -> dict:
    """Download to <ckpt_root>/.pull_<run>, verify, move to <ckpt_root>/<run>, verify again; raise on any difference."""
    ckpt_root = Path(ckpt_root)
    dst, staging = ckpt_root / run, ckpt_root / f".pull_{run}"
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists: pass --overwrite to replace it")
    repo_id = repo_id or default_repo(api, run)
    info = api.repo_info(repo_id, revision=revision)
    sha = info.sha  # pin every read to one commit, even if the repo changes meanwhile
    if staging.exists():
        shutil.rmtree(staging)

    api.hf_hub_download(repo_id, MANIFEST, revision=sha, local_dir=str(staging))
    manifest = read_json(staging / MANIFEST)
    if (manifest.get("run"), manifest.get("corpus")) != (run, corpus):
        raise ValueError(f"{repo_id} holds {manifest.get('corpus')}/{manifest.get('run')}, not {corpus}/{run}")
    remote = _repo_files(api, repo_id, sha)
    if remote != set(manifest["files"]) | {MANIFEST}:
        raise ValueError(f"{repo_id}@{sha}: repo files {sorted(remote)} != manifest files + {MANIFEST}")

    log(f"downloading {manifest['total_bytes'] / 2**30:.2f} GiB from {repo_id}@{sha} ...")
    api.snapshot_download(repo_id, revision=sha, local_dir=str(staging),
                          allow_patterns=sorted(manifest["files"]) + [MANIFEST])
    shutil.rmtree(staging / ".cache", ignore_errors=True)  # huggingface_hub local_dir bookkeeping
    problems = verify_dir(staging, manifest)
    if problems:
        raise RuntimeError(f"downloaded copy in {staging} differs from {repo_id}@{sha}'s manifest: {problems} "
                           f"(staging dir kept for inspection; delete it before retrying)")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(staging), str(dst))
    problems = verify_dir(dst, manifest)
    report = {"repo_id": repo_id, "requested_revision": revision, "hf_commit": sha, "private": info.private,
              "pulled": utc_now(), "files": len(manifest["files"]), "total_bytes": manifest["total_bytes"],
              "verify_staging": "ok", "verify_final": problems or "ok", "status": "ok" if not problems else "FAILED"}
    write_json(report, dst / PULL_REPORT)
    if problems:
        raise RuntimeError(f"{dst} differs from the manifest after the move: {problems}")
    log(f"pulled {repo_id}@{sha} -> {dst}: {len(manifest['files'])} files, sha256 verified twice")
    return report
