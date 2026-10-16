#!/usr/bin/env python3
"""Phase 0 deliverable: synthetic structured corpus with known ground truth
(spec/phase-0.md deliverable #3, spec §11.4 item 4).

Generates a structured corpus where relevance to a query is known by
construction (not by human judgment), used for the filtered-search and
grouped-top-k benchmarks in later phases. Each record has:
  - id, group_id (document), version
  - typed scalar fields: tenant (int), category (enum), created_at (timestamp)
  - a 768-d dense vector embedding a short synthetic "fact" sentence, using
    the same frozen model as D-VEC-07 (nomic-embed-text-v1.5)
  - ground-truth: the query template and predicate that should retrieve it

This does NOT use an LLM to generate free text (that's a heavier Phase 0
follow-up per spec §11.4 item 4's "generate via LLM" suggestion) -- this
first pass uses templated sentences over a fixed vocabulary, which already
gives exact, checkable ground truth and is enough to exercise the filtered
selectivity benchmark (D-QRY-01) end to end. Swap in LLM-generated text
later without changing the schema.

Usage:
    bench/.venv/bin/python bench/harness/gen_synthetic_corpus.py \\
        --rows 100000 --out bench/data/synthetic/corpus_100k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES = ["engineering", "finance", "legal", "product", "research", "sales", "support"]
TOPICS = [
    "the quarterly revenue forecast", "a new authentication vulnerability",
    "customer churn in the enterprise segment", "the Vamana graph index build time",
    "GDPR data deletion requirements", "the Q3 hiring plan", "a filed patent application",
    "the disk-resident beam search latency", "a contract renewal with a vendor",
    "the RaBitQ compression error bound", "an outage postmortem", "the onboarding funnel",
]
SUBJECTS = ["the engineering team", "legal counsel", "the finance department",
            "a customer", "the research group", "an auditor", "the on-call engineer"]


def make_row(row_id: int, rng: random.Random, n_tenants: int) -> dict:
    tenant = rng.randint(0, n_tenants - 1)
    category = rng.choice(CATEGORIES)
    topic = rng.choice(TOPICS)
    subject = rng.choice(SUBJECTS)
    # Deterministic "fact" sentence -- ground truth is exactly this template,
    # so a query for (tenant, category, topic) has a known, checkable answer set.
    text = f"{subject} discussed {topic} in a {category} review for tenant {tenant}."
    return {
        "id": f"row-{row_id}",
        "group_id": f"doc-{row_id // 5}",  # 5 rows/doc on average, exercises D-RET-07 grouping
        "version": 1,
        "tenant": tenant,
        "category": category,
        "topic": topic,
        "created_at": 1_700_000_000 + row_id,  # monotonically increasing synthetic timestamp
        "text": text,
        # Ground truth: this row is the unique known-correct hit for this exact
        # (tenant, category, topic) triple among rows sharing it by construction.
        "ground_truth_key": f"{tenant}:{category}:{topic}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--tenants", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=Path("bench/data/synthetic/corpus_100k.jsonl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.rows):
            row = make_row(i, rng, args.tenants)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.rows} rows to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print("Note: text embedding into 768-d vectors is a separate pass "
          "(embed_corpus.py) -- this script only generates the structured "
          "rows + ground truth, so it stays fast and reproducible without a GPU.")


if __name__ == "__main__":
    main()
