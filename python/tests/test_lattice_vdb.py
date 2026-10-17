import json

import numpy as np
import pytest
from click.testing import CliRunner

import lattice_vdb
from lattice_vdb.cli import main


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((3000, 24)).astype(np.float32)


@pytest.fixture(scope="module")
def built(tmp_path_factory, data):
    path = str(tmp_path_factory.mktemp("idx") / "test.lattice")
    lattice_vdb.build_index(data, path, metric="l2", r=24, threads=1)
    return path


def brute(data, q, k):
    return np.argsort(((data - q) ** 2).sum(1))[:k]


def test_search_recall(data, built):
    idx = lattice_vdb.Index(built, preload=True)
    assert len(idx) == 3000 and idx.dim == 24
    hits = 0
    for i in range(0, 3000, 100):
        q = data[i] + 0.01
        hits += len(set(idx.search(q, k=10, ef_search=128).tolist()) & set(brute(data, q, 10).tolist()))
    assert hits / 300 >= 0.95


def test_search_batch_matches_single(data, built):
    idx = lattice_vdb.Index(built)
    batch = idx.search_batch(data[:20], k=5, ef_search=64, threads=4)
    for i in range(20):
        assert batch[i].tolist() == idx.search(data[i], k=5, ef_search=64).tolist()


def test_cosine_normalizes(tmp_path, data):
    path = str(tmp_path / "cos.lattice")
    lattice_vdb.build_index(data * 5.0, path, metric="cosine", r=24, threads=1)
    idx = lattice_vdb.Index(path)
    assert idx.metric == "cosine"
    assert idx.search(data[7] * 3.0, k=1, ef_search=128)[0] == 7  # scale-invariant


def test_bad_inputs(tmp_path, data):
    with pytest.raises(ValueError):
        lattice_vdb.build_index(data, str(tmp_path / "x"), metric="hamming")
    with pytest.raises(Exception):
        lattice_vdb.Index(str(tmp_path / "missing.lattice"))


def test_cli_end_to_end(tmp_path, data):
    runner = CliRunner()
    vecs = tmp_path / "v.npy"
    np.save(vecs, data)
    queries = tmp_path / "q.npy"
    np.save(queries, data[:5] + 0.01)
    gt = tmp_path / "gt.npy"
    np.save(gt, np.stack([brute(data, q, 10) for q in data[:50] + 0.01]))
    qbench = tmp_path / "qb.npy"
    np.save(qbench, data[:50] + 0.01)
    index = str(tmp_path / "cli.lattice")

    r = runner.invoke(main, ["build", str(vecs), index, "--r", "24", "--threads", "1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["info", index])
    assert r.exit_code == 0 and "count" in r.output and "3000" in r.output
    r = runner.invoke(main, ["verify", index])
    assert r.exit_code == 0 and r.output.startswith("OK")
    r = runner.invoke(main, ["search", index, str(queries), "--k", "3", "--format", "json"])
    assert r.exit_code == 0
    assert len(json.loads(r.output)) == 5
    r = runner.invoke(main, ["bench", index, str(qbench), str(gt), "--ef", "64", "--threads", "1,4"])
    assert r.exit_code == 0, r.output
    assert "recall@k" in r.output
    r = runner.invoke(main, ["--help"])
    for cmd in ("build", "search", "info", "bench", "fetch", "verify"):
        assert cmd in r.output


def test_verify_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.lattice"
    bad.write_bytes(b"\xff" * 100)
    r = CliRunner().invoke(main, ["verify", str(bad)])
    assert r.exit_code != 0
