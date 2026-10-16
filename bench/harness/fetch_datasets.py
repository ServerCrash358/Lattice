#!/usr/bin/env python3
"""Phase 0 dataset fetcher (spec/phase-0.md deliverable #3).

Downloads:
  - ann-benchmarks SIFT1M / GloVe-100 / GIST1M (hdf5, §11.3) -- handled by
    fetch_ann_benchmarks.sh alongside this script; this file covers the
    parts that need Python: the BEIR subset and the frozen embedding model.
  - A single small BEIR dataset (default: scifact) as the Phase 0 labelled
    retrieval subset (finding 34: NOT the full MS MARCO passage set).
  - nomic-embed-text-v1.5 weights (D-VEC-07), via sentence-transformers.

Run from the venv created in bench/.venv:
    bench/.venv/bin/python bench/harness/fetch_datasets.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch_beir_subset(dataset_name: str = "scifact") -> Path:
    """Downloads one small BEIR dataset (D-ARCH-13/finding 34: scoped, not
    the full MS MARCO corpus). scifact is ~5K docs, ~300 queries -- small
    enough to embed and index on this machine in minutes, not days."""
    from beir import util

    out_dir = DATA_DIR / "beir"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    print(f"[beir] fetching {dataset_name} from {url}")
    data_path = util.download_and_unzip(url, str(out_dir))
    print(f"[beir] extracted to {data_path}")
    return Path(data_path)


def fetch_embedding_model(model_id: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
    """Downloads and caches the frozen D-VEC-07 embedding model weights via
    sentence-transformers, and does a one-vector smoke test to confirm the
    768-d Matryoshka-capable model loads and encodes correctly."""
    from sentence_transformers import SentenceTransformer

    print(f"[model] fetching {model_id}")
    model = SentenceTransformer(model_id, trust_remote_code=True)
    vec = model.encode(["Lattice is a disk-first vector database."])
    dim = vec.shape[-1]
    print(f"[model] loaded OK, embedding dim = {dim}")
    if dim != 768:
        print(f"[model] WARNING: expected 768-d per D-VEC-07/D-ARCH-02, got {dim}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beir-dataset", default="scifact",
                     help="BEIR dataset name (default: scifact, a small subset per finding 34)")
    ap.add_argument("--skip-beir", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    if not args.skip_beir:
        fetch_beir_subset(args.beir_dataset)
    if not args.skip_model:
        fetch_embedding_model()

    print("Done.")


if __name__ == "__main__":
    main()
