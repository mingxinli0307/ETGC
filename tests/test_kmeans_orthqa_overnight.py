import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import edge_model
from edge_model import (
    assign_all_to_centers,
    fit_kmeans_centers,
    initialize_cluster_output_from_prototypes,
    initialize_cluster_output_random_event,
    initialize_cluster_output_random_orthogonal,
)
from edge_train import edge_hard_labels_to_node_predictions
from edge_uniform_diagnostic import cluster_volume_statistics
from edge_proximity import compute_edge_ppr_cached
from summarize_kmeans_orthqa_runs import is_success_result, make_plan, stable_run_key, write_failure


ETGCModel = getattr(edge_model, "Edge" + "Hi" + "No" + "SModel")


def _initial(num_nodes=8, dim=6):
    rng = np.random.RandomState(7)
    return rng.normal(0.0, 0.1, size=(num_nodes, dim)).astype(np.float32)


def test_random_orthogonal_initialization_rows_are_unit_and_low_correlation():
    model = ETGCModel(_initial(), 3, 8, 8, 6, 4, False, cluster_output_bias_mode="zero")
    stats = initialize_cluster_output_random_orthogonal(model, 4, seed=13)
    weight = model.cluster_output.weight.detach()
    gram = weight @ weight.t()
    assert stats["strict_orthogonal_rows"] is True
    assert torch.allclose(torch.linalg.norm(weight, dim=1), torch.ones(4), atol=1e-6)
    assert torch.max(torch.abs(gram - torch.eye(4))) < 1e-5
    assert torch.count_nonzero(model.cluster_output.bias.detach()) == 0


def test_random_orthogonal_handles_k_greater_than_hidden_dim():
    model = ETGCModel(_initial(), 3, 8, 8, 3, 5, False, cluster_output_bias_mode="none")
    stats = initialize_cluster_output_random_orthogonal(model, 5, seed=14)
    assert stats["strict_orthogonal_rows"] is False
    assert tuple(model.cluster_output.weight.shape) == (5, 3)
    assert torch.isfinite(model.cluster_output.weight).all()


def test_random_event_initialization_uses_unique_events_and_seed():
    model1 = ETGCModel(_initial(), 3, 8, 8, 5, 4, False, cluster_output_bias_mode="zero")
    model2 = ETGCModel(_initial(), 3, 8, 8, 5, 4, False, cluster_output_bias_mode="zero")
    hidden = torch.randn(20, 5)
    s1 = initialize_cluster_output_random_event(model1, hidden, 4, seed=9)
    s2 = initialize_cluster_output_random_event(model2, hidden, 4, seed=9)
    assert s1["random_event_unique_count"] == 4
    assert s1["random_event_index_checksum"] == s2["random_event_index_checksum"]
    assert torch.allclose(model1.cluster_output.weight, model2.cluster_output.weight)


def test_kmeans_lloyd_iterations_zero_one_and_full_are_explicit():
    torch.manual_seed(0)
    features = torch.cat([torch.randn(30, 3) - 3, torch.randn(30, 3) + 3], dim=0)
    c0, s0 = fit_kmeans_centers(features, 2, seed=4, max_iters=0)
    c1, s1 = fit_kmeans_centers(features, 2, seed=4, max_iters=1)
    c10, s10 = fit_kmeans_centers(features, 2, seed=4, max_iters=10)
    assert s0["kmeans_lloyd_iters_run"] == 0
    assert s1["kmeans_lloyd_iters_run"] == 1
    assert s10["kmeans_lloyd_iters_run"] >= 1
    assert tuple(c0.shape) == tuple(c1.shape) == tuple(c10.shape) == (2, 3)
    assert torch.isfinite(c10).all()


def test_prototype_lloyd_zero_does_not_fallback_to_ten():
    model = ETGCModel(_initial(), 3, 8, 8, 5, 4, False, cluster_output_bias_mode="zero")
    hidden = torch.randn(25, 5)
    stats = initialize_cluster_output_from_prototypes(model, hidden, 4, seed=5, lloyd_iters=0)
    assert stats["prototype_lloyd_iters"] == 0
    assert stats["kmeans_lloyd_iters_run"] == 0


def test_direct_kmeans_assignment_and_projection_match_explicit_accumulation():
    hidden = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]], dtype=torch.float32)
    centers = torch.tensor([[0.0, 0.0], [5.0, 5.0]], dtype=torch.float32)
    labels = assign_all_to_centers(hidden, centers).cpu().numpy()
    src = np.array([0, 1, 2, 3])
    dst = np.array([1, 2, 3, 0])
    pred = edge_hard_labels_to_node_predictions(labels, src, dst, 4, 2)
    q = np.eye(2, dtype=np.float32)[labels]
    S = np.zeros((4, 2), dtype=np.float32)
    np.add.at(S, src, q)
    np.add.at(S, dst, q)
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-8)
    assert np.array_equal(pred, S.argmax(axis=1))


def test_init_final_ari_q_drift_and_weight_drift_formulas():
    edge_init = np.array([0, 0, 1, 1])
    edge_final = np.array([0, 1, 1, 1])
    q_init = np.eye(2, dtype=np.float32)[edge_init]
    q_final = np.eye(2, dtype=np.float32)[edge_final]
    drift = np.linalg.norm(q_final - q_init) / np.sqrt(q_init.shape[0])
    assert adjusted_rand_score(edge_init, edge_final) < 1.0
    assert drift > 0.0
    w0 = torch.eye(2)
    w1 = torch.eye(2) * 2
    weight_drift = torch.linalg.norm(w1 - w0) / torch.linalg.norm(w0)
    assert float(weight_drift) > 0.0


def test_cluster_volume_statistics_soft_and_hard_are_finite():
    Q = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
    degree = torch.tensor([1.0, 2.0, 3.0])
    stats = cluster_volume_statistics(Q, degree)
    assert 0.0 <= stats["cluster_volume_min_ratio"] <= stats["cluster_volume_max_ratio"] <= 1.0
    assert stats["cluster_volume_coefficient_of_variation"] >= 0.0
    assert stats["mean_top1_probability"] > stats["mean_top2_probability"]


def test_w_e_cache_key_depends_on_forest_seed_not_model_seed(tmp_path):
    src = np.array([0, 1, 2, 0], dtype=np.int64)
    dst = np.array([1, 2, 0, 2], dtype=np.int64)
    times = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    args = dict(
        dataset="toy_cache",
        src=src,
        dst=dst,
        times=times,
        num_nodes=3,
        cache_dir=str(tmp_path),
        method="temporal_state_forest",
        alpha=0.2,
        T=2,
        forest_samples=1,
        edge_neighbor_k=-1,
        edge_ppr_topk=-1,
        beta=5.0,
        seed=20260725,
        quiet=True,
    )
    _, _, _, s1 = compute_edge_ppr_cached(**args)
    _, _, _, s2 = compute_edge_ppr_cached(**args)
    args_other_forest = dict(args)
    args_other_forest["seed"] = 20260726
    _, _, _, s3 = compute_edge_ppr_cached(**args_other_forest)
    assert s1["config_hash"] == s2["config_hash"]
    assert s1["config_hash"] != s3["config_hash"]


def test_run_key_resume_and_failure_helpers(tmp_path):
    cfg = {
        "dataset": "school",
        "init_mode": "random_orthogonal",
        "lloyd_iters": 0,
        "penalty_type": "orthqa",
        "penalty_weight": 1.0,
        "warmup_epochs": 0,
        "lambda_prox": 0.0,
        "model_seed": 42,
        "prototype_seed": 42,
        "forest_seed": 20260725,
        "direct_kmeans_eval": 0,
        "init_only": 0,
        "epoch": 30,
    }
    cfg_model_changed = dict(cfg)
    cfg_model_changed["model_seed"] = 43
    assert stable_run_key(cfg) != stable_run_key(cfg_model_changed)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    assert is_success_result(run_dir)
    failed_dir = tmp_path / "failed"
    write_failure(failed_dir, cfg, "runtime_error", "traceback tail", 2, 1.5)
    payload = json.loads((failed_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "runtime_error"


def test_plan_contains_required_groups_and_deduplicable_configs(tmp_path):
    plan = make_plan(ROOT, tmp_path, ["A", "B", "C", "D"])
    names = {cfg["name"] for cfg in plan}
    assert {"A1_direct_kmeans", "A2_prototype_init_only", "B1_random_orthogonal"}.issubset(names)
    assert any(cfg["penalty_type"] == "orthqa" and cfg["group"] == "C" for cfg in plan)
    assert any(cfg["lambda_prox"] == 1.0 and cfg["group"] == "D" for cfg in plan)
