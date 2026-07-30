import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from run_initial_node_kmeans_baseline import (
    evaluate_initial_node_kmeans_features,
    hard_node_cluster_statistics,
    load_initial_node_features_for_kmeans,
    run_one,
)


def _toy_node_features():
    return np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )


def test_initial_node_kmeans_uses_node_features_directly():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    result = evaluate_initial_node_kmeans_features(
        _toy_node_features(),
        labels,
        K=2,
        seed=3,
        sample_size=-1,
        lloyd_iters=5,
        device="cpu",
    )
    assert result["method"] == "initial_node_kmeans"
    assert result["uses_edges"] is False
    assert result["uses_time"] is False
    assert result["uses_incidence_projection"] is False
    assert result["uses_training"] is False
    assert result["Macro_F1"] == pytest.approx(1.0)
    assert result["F1"] == pytest.approx(result["Macro_F1"])
    assert result["node_hard_active_clusters"] == 2


def test_initial_node_kmeans_requires_existing_node2vec_when_requested(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing required Node2Vec"):
        load_initial_node_features_for_kmeans(
            str(tmp_path / "missing.emb"),
            num_nodes=4,
            fallback_dim=2,
            seed=1,
            require_pretrained_node2vec=True,
        )


def test_hard_node_cluster_statistics_counts_and_ratios():
    stats = hard_node_cluster_statistics(np.asarray([1, 1, 0, 2, 2]), K=4)
    assert stats["node_hard_counts"] == [1, 2, 2, 0]
    assert sum(stats["node_hard_counts"]) == 5
    assert sum(stats["node_hard_ratios"]) == pytest.approx(1.0)
    assert stats["node_hard_active_clusters"] == 3
    assert stats["node_hard_empty_clusters"] == 1
    assert stats["node_hard_largest_ratio"] == pytest.approx(0.4)
    assert stats["node_hard_smallest_nonempty_ratio"] == pytest.approx(0.2)


def test_initial_node_kmeans_run_writes_only_compact_outputs(tmp_path):
    root = tmp_path
    ds = root / "dataset" / "toy"
    ds.mkdir(parents=True)
    (ds / "toy.txt").write_text("0 1 0\n2 3 1\n0 1 2\n2 3 3\n", encoding="utf-8")
    (ds / "node2label.txt").write_text("0 0\n1 0\n2 1\n3 1\n", encoding="utf-8")
    emb = root / "pretrain" / "toy_feature.emb"
    emb.parent.mkdir()
    features = _toy_node_features()
    emb.write_text(
        "\n".join(f"{idx} {row[0]} {row[1]}" for idx, row in enumerate(features)) + "\n",
        encoding="utf-8",
    )
    cfg = {
        "method": "initial_node_kmeans",
        "dataset": "toy",
        "seed": 3,
        "node_count": 4,
        "event_count": 4,
        "class_count": 2,
        "node2vec_path": str(emb),
        "fallback_dim": 2,
        "node_kmeans_sample_size": -1,
        "node_kmeans_lloyd_iters": 5,
        "assign_chunk_size": 16,
        "device": "cpu",
        "require_pretrained_node2vec": 1,
    }
    out_dir = root / "logs" / "initial_node_kmeans" / "test"
    result = run_one(root, out_dir, cfg, resume=False)
    run_dir = out_dir / "toy" / "seed_3"
    assert result["status"] == "success"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "diagnostic.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "train.log").exists()
    payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "predictions" not in serialized
    assert "edge_hard_labels" not in serialized
    assert payload["metrics"]["uses_incidence_projection"] is False
