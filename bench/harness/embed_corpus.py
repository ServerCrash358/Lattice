#!/usr/bin/env python3
"""Embeds the synthetic structured corpus (gen_synthetic_corpus.py's output)
into 768-d vectors via D-VEC-07's frozen model, and exports the result in
the same flat float32 binary format export_sift_binary.py already produces
(train.f32/test.f32/meta.txt) -- so bench/cpp's existing readers (e.g.
bench_rabitq_real_data) work against real text embeddings with zero changes.

This is the "text set" gen_synthetic_corpus.py's docstring flagged as a
separate pass (Phase 3 exit gate item 8: RaBitQ's error bound validated on
SIFT1M AND a real text-embedding set, not just synthetic uniform-random
vectors).

Usage:
    bench/.venv/bin/python bench/harness/embed_corpus.py \\
        --input bench/data/synthetic/corpus_100k.jsonl \\
        --out-dir bench/data/text_bin \\
        --limit 5000 --num-queries 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("bench/data/synthetic/corpus_100k.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench/data/text_bin"))
    ap.add_argument("--limit", type=int, default=5000,
                     help="rows to use as the 'database' (train) set")
    ap.add_argument("--num-queries", type=int, default=200,
                     help="separate held-out rows embedded as the 'query' (test) set")
    ap.add_argument("--model-id", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.limit + args.num_queries:
                break
    if len(rows) < args.limit + args.num_queries:
        raise SystemExit(
            f"corpus has only {len(rows)} rows, need {args.limit + args.num_queries} "
            f"(--limit + --num-queries) -- regenerate a bigger corpus with gen_synthetic_corpus.py")

    train_rows = rows[: args.limit]
    query_rows = rows[args.limit: args.limit + args.num_queries]

    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {args.model_id} ...")
    model = SentenceTransformer(args.model_id, trust_remote_code=True)

    def embed_texts(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    print(f"[embed] encoding {len(train_rows)} database rows ...")
    train_vecs = embed_texts([r["text"] for r in train_rows])
    print(f"[embed] encoding {len(query_rows)} query rows ...")
    test_vecs = embed_texts([r["text"] for r in query_rows])

    dim = train_vecs.shape[-1]
    if dim != 768:
        print(f"[embed] WARNING: expected 768-d per D-VEC-07, got {dim}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_vecs.tofile(args.out_dir / "train.f32")
    test_vecs.tofile(args.out_dir / "test.f32")
    with open(args.out_dir / "meta.txt", "w") as f:
        # neighbors/gt_k are unused by bench_rabitq_real_data (it recomputes
        # exact dot products itself), written as 0 0 to keep the file
        # format identical to export_sift_binary.py's so the same C++
        # reader works unmodified against either dataset.
        f.write(f"{train_vecs.shape[0]} {dim}\n")
        f.write(f"{test_vecs.shape[0]} {dim}\n")
        f.write("0 0\n")

    print(f"[embed] wrote {train_vecs.shape[0]} train + {test_vecs.shape[0]} test vectors "
          f"(dim={dim}) to {args.out_dir}")


if __name__ == "__main__":
    main()
