"""
Offline test of extra/ckpt_transfer.py (push / pull through a private HF repo) against a fake Hub.

The fake stores every commit as a directory, reports LFS sha256 / git blob sha1 like the Hub's paths-info, and can
corrupt one file on upload or on download. Checks: file selection (no checkpoint-*/), completeness guard, git blob
sha1 == `git hash-object`, refusals (read token, public repo, existing repo, existing target, wrong run, extra repo
file), a clean round trip (pulled files byte-identical to the source), and that every corruption raises with the
target path left absent.

    .venvs/quant/bin/python -m extra.tests.test_ckpt_transfer          # needs huggingface_hub (quant venv)
"""

import filecmp
import fnmatch
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx
from huggingface_hub.errors import RepositoryNotFoundError

from extra import ckpt_transfer as ct

LFS_MIN = 1024  # the fake stores files >= 1 KiB as LFS (the Hub's threshold is larger; irrelevant here)


class FakeHub:
    def __init__(self, root: Path, role: str = "write"):
        self.root, self.role = root, role
        self.repos = {}  # repo_id -> {"private": bool, "commits": [oid, ...]}
        self.corrupt_upload = self.corrupt_download = self.truncate_download = None

    # -- helpers
    def _dir(self, repo_id, oid):
        return self.root / repo_id.replace("/", "__") / oid

    def _resolve(self, repo_id, revision):
        if repo_id not in self.repos:
            raise RepositoryNotFoundError(f"{repo_id} not found", response=httpx.Response(
                404, request=httpx.Request("GET", f"https://hf.test/api/models/{repo_id}")))
        commits = self.repos[repo_id]["commits"]
        return commits[-1] if revision in (None, "main") else next(c for c in commits if c == revision)

    # -- the HfApi subset used by ckpt_transfer
    def whoami(self):
        return {"name": "tester", "auth": {"accessToken": {"role": self.role}}}

    def create_repo(self, repo_id, private=None, repo_type=None):
        self.repos[repo_id] = {"private": bool(private), "commits": []}
        self._commit(repo_id, {".gitattributes": b"*.safetensors filter=lfs\n"}, [])

    def repo_info(self, repo_id, revision=None):
        sha = self._resolve(repo_id, revision)
        return SimpleNamespace(private=self.repos[repo_id]["private"], sha=sha)

    def list_repo_files(self, repo_id, revision=None):
        d = self._dir(repo_id, self._resolve(repo_id, revision))
        return sorted(p.name for p in d.iterdir())

    def _commit(self, repo_id, adds: dict, deletes: list):
        commits = self.repos[repo_id]["commits"]
        oid = hashlib.sha1(f"{repo_id}{len(commits)}".encode()).hexdigest()
        new = self._dir(repo_id, oid)
        if commits:
            shutil.copytree(self._dir(repo_id, commits[-1]), new)
        else:
            new.mkdir(parents=True)
        for name, data in adds.items():
            (new / name).write_bytes(data)
        for name in deletes:
            (new / name).unlink()
        commits.append(oid)
        return oid

    def create_commit(self, repo_id, operations, commit_message=""):
        adds, deletes = {}, []
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                deletes.append(op.path_in_repo)
                continue
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            if op.path_in_repo == self.corrupt_upload:
                data = _flip(data)
            adds[op.path_in_repo] = data
        return SimpleNamespace(oid=self._commit(repo_id, adds, deletes))

    def get_paths_info(self, repo_id, paths, revision=None):
        d = self._dir(repo_id, self._resolve(repo_id, revision))
        out = []
        for p in paths:
            data = (d / p).read_bytes()
            lfs = (SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
                   if len(data) >= LFS_MIN else None)
            out.append(SimpleNamespace(path=p, size=len(data), lfs=lfs,
                                       blob_id=hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()))
        return out

    def _fetch(self, repo_id, revision, name, local_dir):
        dst = Path(local_dir) / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = (self._dir(repo_id, self._resolve(repo_id, revision)) / name).read_bytes()
        if name == self.corrupt_download:
            data = _flip(data)
        if name == self.truncate_download:
            data = data[:-1]
        dst.write_bytes(data)
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)

    def hf_hub_download(self, repo_id, filename, revision=None, local_dir=None):
        self._fetch(repo_id, revision, filename, local_dir)

    def snapshot_download(self, repo_id, revision=None, local_dir=None, allow_patterns=None):
        for name in self.list_repo_files(repo_id, revision):
            if allow_patterns is None or any(fnmatch.fnmatch(name, pat) for pat in allow_patterns):
                self._fetch(repo_id, revision, name, local_dir)


def _flip(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 1])


def make_run(run_dir: Path) -> None:
    """A finished BOOKS run as unlearn_run.py + trainer.save_model write it (2 shards, tokenizer, epoch dir)."""
    run_dir.mkdir(parents=True)
    shards = {"model-00001-of-00002.safetensors": b"\x01" * 5000, "model-00002-of-00002.safetensors": b"\x02" * 3000}
    for name, data in shards.items():
        (run_dir / name).write_bytes(data)
    (run_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 8000}, "weight_map": {"a.weight": "model-00001-of-00002.safetensors",
                                                           "b.weight": "model-00002-of-00002.safetensors"}}))
    for name, text in {"config.json": '{"torch_dtype": "bfloat16"}', "generation_config.json": '{"do_sample": true}',
                       "tokenizer.json": "{}" * 600, "tokenizer_config.json": "{}", "special_tokens_map.json": "{}",
                       "unlearn_run.json": '{"config": {"seed": 42}}', "loss_components.jsonl": "{}\n",
                       "training_args.bin": "pickle"}.items():
        (run_dir / name).write_text(text)
    (run_dir / "tokenizer.model").write_bytes(b"\x07" * 2000)
    ep = run_dir / "checkpoint-553"
    ep.mkdir()
    (ep / "model.safetensors").write_bytes(b"\x03" * 4000)
    (ep / "trainer_state.json").write_text('{"epoch": 1.0}')


def expect(exc, fn, *args, contains="", **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        assert contains in str(e), f"{exc.__name__} raised but without {contains!r}: {e}"
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def main() -> int:
    quiet = lambda *a: None  # noqa: E731
    run, corpus = "books_npo_klr_sure_s42", "books"
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_root, dst_root = tmp / "acctA" / "ckpt" / corpus, tmp / "acctC" / "ckpt" / corpus
        src = src_root / run
        make_run(src)

        # 1 selection: top level only, no epoch dirs, no pickle / loss log
        keep, skip = ct.select_files(src)
        assert "checkpoint-553" not in keep + skip
        assert set(skip) == {"loss_components.jsonl", "training_args.bin"}, skip
        assert {"config.json", "generation_config.json", "tokenizer.model", "unlearn_run.json",
                "model.safetensors.index.json"} <= set(keep), keep
        results["select"] = {"transferred": keep, "excluded": skip}

        # 2 completeness guard
        expect(ValueError, ct.check_complete, src, [f for f in keep if f != "model-00002-of-00002.safetensors"],
               contains="missing shard model-00002-of-00002.safetensors")
        expect(ValueError, ct.check_complete, src, [f for f in keep if f != "unlearn_run.json"],
               contains="missing unlearn_run.json")
        results["complete_guard"] = "ok"

        # 3 git blob id matches git itself
        for name in ("config.json", "tokenizer.model"):
            git = subprocess.run(["git", "hash-object", str(src / name)], capture_output=True, text=True,
                                 check=True).stdout.strip()
            assert ct.git_blob_sha1(src / name) == git, name
        results["git_blob_sha1"] = "ok"

        # 4 refusals before any upload
        expect(PermissionError, ct.push, FakeHub(tmp / "hub_r", role="read"), src, run, corpus, log=quiet,
               contains="READ token")
        pub = FakeHub(tmp / "hub_p")
        pub.create_repo(f"tester/failunl-{run}", private=False)
        expect(PermissionError, ct.push, pub, src, run, corpus, log=quiet, contains="not private")
        results["refusals_push"] = "ok"

        # 5 clean push; volume manifest records the commit
        hub = FakeHub(tmp / "hub")
        rec = ct.push(hub, src, run, corpus, log=quiet)
        repo = f"tester/failunl-{run}"
        assert rec["status"] == "ok" and rec["repo_id"] == repo, rec
        assert set(hub.list_repo_files(repo)) == set(keep) | {ct.MANIFEST, ".gitattributes"}
        vol_manifest = json.loads((src / ct.MANIFEST).read_text())
        assert vol_manifest["hf_commit"] == hub.repo_info(repo).sha and vol_manifest["status"] == "ok"
        assert ct.select_files(src)[0] == keep  # the manifest on the Volume is never transferred itself
        expect(FileExistsError, ct.push, hub, src, run, corpus, log=quiet, contains="--overwrite")
        rec2 = ct.push(hub, src, run, corpus, overwrite=True, log=quiet)
        assert rec2["status"] == "ok"
        results["push"] = "ok"

        # 6 corruption during upload is caught by the remote check (LFS sha256 and git blob paths)
        for victim in ("model-00001-of-00002.safetensors", "config.json"):
            bad = FakeHub(tmp / f"hub_bad_{victim}")
            bad.corrupt_upload = victim
            expect(RuntimeError, ct.push, bad, src, run, corpus, log=quiet, contains=victim)
            assert json.loads((src / ct.MANIFEST).read_text())["status"] == "REMOTE MISMATCH"
        ct.push(hub, src, run, corpus, overwrite=True, log=quiet)  # leave a good Volume manifest behind
        results["upload_corruption"] = "caught (lfs + blob)"

        # 7 clean pull: byte-identical, no epoch dirs, pull report ok
        rep = ct.pull(hub, dst_root, run, corpus, log=quiet)
        dst = dst_root / run
        assert rep["status"] == "ok" and rep["hf_commit"] == hub.repo_info(repo).sha
        for name in keep:
            assert filecmp.cmp(src / name, dst / name, shallow=False), name
        assert not (dst / "checkpoint-553").exists() and not (dst / ".cache").exists()
        assert not (dst_root / f".pull_{run}").exists()
        expect(FileExistsError, ct.pull, hub, dst_root, run, corpus, log=quiet, contains="--overwrite")
        results["pull"] = "ok (byte-identical)"

        # 7b pinned revision (the documented default): an older verified commit is pulled even after a newer push
        old_sha = hub.repo_info(repo).sha
        (src / "config.json").write_text('{"torch_dtype": "bfloat16", "changed": true}')
        ct.push(hub, src, run, corpus, overwrite=True, log=quiet)
        assert hub.repo_info(repo).sha != old_sha
        pinned = ct.pull(hub, tmp / "acct_pinned", run, corpus, revision=old_sha, log=quiet)
        assert pinned["hf_commit"] == old_sha
        assert filecmp.cmp(dst / "config.json", tmp / "acct_pinned" / run / "config.json", shallow=False)
        assert not filecmp.cmp(src / "config.json", tmp / "acct_pinned" / run / "config.json", shallow=False)
        results["pull_pinned_revision"] = "ok (older commit, not main)"

        # 8 corruption / truncation during download: raises, target absent, staging kept
        for attr, victim, msg in (("corrupt_download", "model-00002-of-00002.safetensors", "sha256 mismatch"),
                                  ("truncate_download", "config.json", "size mismatch")):
            root = tmp / f"acct_{attr}" / "ckpt" / corpus
            setattr(hub, attr, victim)
            expect(RuntimeError, ct.pull, hub, root, run, corpus, log=quiet, contains=f"{msg} {victim}")
            setattr(hub, attr, None)
            assert not (root / run).exists() and (root / f".pull_{run}").exists()
        results["download_corruption"] = "caught (sha256 + size)"

        # 9 repo content does not match its manifest / wrong run
        hub._commit(repo, {"README.md": b"hi"}, [])
        expect(ValueError, ct.pull, hub, tmp / "acct_extra", run, corpus, log=quiet, contains="README.md")
        expect(ValueError, ct.pull, hub, tmp / "acct_wrong", "books_ga_gdr_s42", corpus, log=quiet,
               repo_id=repo, contains="not books/books_ga_gdr_s42")
        results["pull_refusals"] = "ok"

    print(json.dumps(results, indent=2))
    print("test_ckpt_transfer: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
