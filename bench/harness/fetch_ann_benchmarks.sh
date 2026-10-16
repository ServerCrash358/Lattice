#!/usr/bin/env bash
# Phase 0 dataset fetcher (spec/phase-0.md deliverable #3, spec §11.3).
# Downloads the three standard ann-benchmarks hdf5 datasets used for the
# Phase 0 exit gate (SIFT1M) and later recall/latency benchmarks (GloVe-100,
# GIST1M -- 960-d, closest published dim to Lattice's 768-d target).
set -euo pipefail

DATA_DIR="$(dirname "${BASH_SOURCE[0]}")/../data/ann-benchmarks"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

declare -A DATASETS=(
  [sift-128-euclidean]=525128288
  [glove-100-angular]=485413888
  [gist-960-euclidean]=3844648288
)

for name in "${!DATASETS[@]}"; do
  f="${name}.hdf5"
  expected="${DATASETS[$name]}"
  if [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null || echo 0)" = "$expected" ]; then
    echo "[skip] $f already present ($expected bytes)"
    continue
  fi
  echo "[fetch] $f (~$((expected / 1024 / 1024)) MB) from https://ann-benchmarks.com/$f"
  # -f: fail on HTTP errors instead of saving an error page; download to a temp name so an
  # interrupted fetch never leaves a partial file under the final name.
  curl -fsSL --retry 3 -o "$f.part" "https://ann-benchmarks.com/$f"
  mv "$f.part" "$f"
  actual="$(stat -c%s "$f")"
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: $f size mismatch (expected $expected, got $actual)" >&2
    exit 1
  fi
  echo "[ok] $f verified"
done
