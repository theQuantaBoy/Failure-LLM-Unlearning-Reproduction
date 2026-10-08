"""
extra/calib.py — calibration sets for GPTQ / AWQ (Task 3 and Task 4), identical for both quantizers.

Sources
  general       wikitext-2-raw-v1 train (Salesforce/wikitext): "texts from Wikipedia" (paper Sec. 4.4, p.7).
  books_retain  MUSE BOOKS raw/retain1.txt + raw/retain2.txt (Harry Potter FanWiki), i.e. retain data only.

Exclusion (Task 4: forget examples and evaluation examples are excluded from calibration)
  A word 13-gram ban set is built from: raw/forget.txt, verbmem/forget.json (prompt+gt), privleak/{forget,retain,
  holdout}.json, knowmem/{forget,retain}_qa.json (question+answer) of the corpus. Text is split into paragraphs
  (lines); every paragraph sharing at least one 13-gram with the ban set is dropped *before* tokenisation, for
  both sources. (Measured on BOOKS: privleak/retain.json is entirely contained in retain2, so this filter is
  needed.)

Sampling (same for every quantizer → matched examples, counts, lengths and seed)
  Kept paragraphs are joined with "\n", tokenised without special tokens, cut into consecutive non-overlapping
  windows of seq_len-1 tokens, each prefixed with BOS (as the unlearning data, dataset.py:232-239). n_samples
  windows are drawn with random.Random(seed).sample. Every sample's sha1 is recorded.
"""

import hashlib
import json
import random
import re
from pathlib import Path

N_GRAM = 13


def _words(text):
    return re.findall(r"\w+", text.lower())


def _grams(words, n=N_GRAM):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def ban_set(repo_dir: Path, corpus: str) -> set:
    d = Path(repo_dir) / "data" / corpus
    texts = [(d / "raw" / "forget.txt").read_text()]
    texts += [x["prompt"] + " " + x["gt"] for x in json.loads((d / "verbmem" / "forget.json").read_text())]
    for s in ("forget", "retain", "holdout"):
        texts += json.loads((d / "privleak" / f"{s}.json").read_text())
    for s in ("forget_qa", "retain_qa"):
        texts += [x["question"] + " " + x["answer"] for x in json.loads((d / "knowmem" / f"{s}.json").read_text())]
    ban = set()
    for t in texts:
        ban |= _grams(_words(t))
    return ban


def source_paragraphs(source: str, repo_dir: Path, corpus: str, wikitext_rows=None):
    if source == "books_retain":
        if corpus != "books":
            raise ValueError("books_retain calibration is defined for the BOOKS corpus only")
        d = Path(repo_dir) / "data" / "books" / "raw"
        text = (d / "retain1.txt").read_text() + "\n" + (d / "retain2.txt").read_text()
        return text.split("\n")
    if source == "general":
        if wikitext_rows is None:
            raise ValueError("general calibration needs the wikitext rows")
        return list(wikitext_rows)
    raise ValueError(f"unknown calibration source {source!r}")


def build(source: str, tokenizer, repo_dir: Path, corpus: str, n_samples: int, seq_len: int, seed: int,
          wikitext_rows=None):
    """Returns (list of token-id lists, report dict)."""
    ban = ban_set(repo_dir, corpus)
    paras = [p for p in source_paragraphs(source, repo_dir, corpus, wikitext_rows) if p.strip()]
    kept = [p for p in paras if not (_grams(_words(p)) & ban)]
    ids = tokenizer("\n".join(kept), add_special_tokens=False)["input_ids"]
    w = seq_len - 1
    windows = [ids[i:i + w] for i in range(0, len(ids) - w + 1, w)]
    if len(windows) < n_samples:
        raise ValueError(f"{source}: only {len(windows)} clean windows of {seq_len} tokens, {n_samples} requested")
    pick = sorted(random.Random(seed).sample(range(len(windows)), n_samples))
    bos = tokenizer.bos_token_id
    samples = [[bos] + windows[i] for i in pick]
    report = {
        "source": source,
        "corpus_for_exclusion": corpus,
        "n_gram": N_GRAM,
        "paragraphs_total": len(paras),
        "paragraphs_kept": len(kept),
        "words_total": sum(len(_words(p)) for p in paras),
        "words_kept": sum(len(_words(p)) for p in kept),
        "tokens_kept": len(ids),
        "windows_available": len(windows),
        "n_samples": n_samples,
        "seq_len": seq_len,
        "seed": seed,
        "bos_prefixed": True,
        "window_indices": pick,
        "sample_sha1": [hashlib.sha1(json.dumps(s).encode()).hexdigest() for s in samples],
    }
    report["set_sha1"] = hashlib.sha1("".join(report["sample_sha1"]).encode()).hexdigest()
    return samples, report
