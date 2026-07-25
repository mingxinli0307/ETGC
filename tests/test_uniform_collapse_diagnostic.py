import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import edge_trace_mincut_loss_global, project_edge_assignments_to_nodes_global
from edge_model import EdgeHiNoSModel
from edge_uniform_diagnostic import (
    cluster_head_gradient_diagnostics,
    compute_uniform_collapse_stage,
    hard_cluster_statistics,
    logits_statistics,
    q_margin_statistics,
    q_uniform_distance_statistics,
    uniform_delta,
)


def _toy_affinity():
    rows = np.array([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int64)
    cols = np.array([1, 0, 2, 1, 3, 2, 3, 0], dtype=np.int64)
    data = np.array([0.7, 0.7, 0.5, 0.5, 0.9, 0.9, 0.2, 0.2], dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float32)


def test_hard_cluster_statistics_counts_ratios_and_active_clusters():
    Q = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )
    stats = hard_cluster_statistics(Q, 3, "edge")
    assert stats["edge_hard_counts"] == [1, 2, 1]
    assert np.allclose(stats["edge_hard_ratios"], [0.25, 0.5, 0.25])
    assert stats["num_active_edge_clusters"] == 3
    assert stats["num_empty_hard_edge_clusters"] == 0
    assert abs(stats["largest_edge_cluster_ratio"] - 0.5) < 1e-12
    assert abs(stats["smallest_nonempty_edge_cluster_ratio"] - 0.25) < 1e-12


def test_q_margin_statistics_match_known_probabilities():
    Q = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.34, 0.33, 0.33],
        ],
        dtype=torch.float32,
    )
    margins = torch.tensor([0.7, 0.3, 0.01], dtype=torch.float32)
    stats = q_margin_statistics(Q)
    assert abs(stats["q_margin_mean"] - float(margins.mean())) < 1e-6
    assert abs(stats["q_margin_p50"] - float(torch.quantile(margins, 0.5))) < 1e-6
    assert abs(stats["q_margin_p90"] - float(torch.quantile(margins, 0.9))) < 1e-6
    assert abs(stats["q_margin_lt_1e2_ratio"] - (1.0 / 3.0)) < 1e-6
    assert abs(stats["q_margin_lt_1e_2_ratio"] - (1.0 / 3.0)) < 1e-6


def test_uniform_distance_is_zero_for_uniform_q():
    Q = torch.full((5, 4), 0.25, dtype=torch.float32)
    stats = q_uniform_distance_statistics(Q)
    assert abs(stats["q_uniform_l2_mean"]) < 1e-7
    assert abs(stats["q_uniform_fro_normalized"]) < 1e-7
    assert abs(stats["q_uniform_l1_mean"]) < 1e-7
    assert abs(stats["q_uniform_kl_mean"]) < 1e-6
    assert abs(stats["q_entropy_gap"]) < 1e-6


def test_logits_statistics_match_torch_reference():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    row_std = logits.std(dim=1, unbiased=False)
    cluster_mean = logits.mean(dim=0)
    cluster_std = logits.std(dim=0, unbiased=False)
    stats = logits_statistics(logits)
    assert abs(stats["logits_global_std"] - float(logits.std(unbiased=False))) < 1e-7
    assert abs(stats["logits_row_std_mean"] - float(row_std.mean())) < 1e-7
    assert abs(stats["logits_cluster_mean_std"] - float(cluster_mean.std(unbiased=False))) < 1e-7
    assert abs(stats["logits_cluster_std_mean"] - float(cluster_std.mean())) < 1e-7
    assert stats["logits_std_unbiased"] is False


def test_gradient_diagnostics_do_not_populate_parameter_grad():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    initial = rng.normal(0.0, 0.1, size=(4, 5)).astype(np.float32)
    model = EdgeHiNoSModel(initial, 3, 6, 8, 5, 3, False)
    src = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 3], dtype=torch.long)
    _, Q, _ = model(src, dst, torch.randn(4, 3), return_logits=True)
    W = _toy_affinity()
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    _, cut_loss, orth_loss = edge_trace_mincut_loss_global(Q, W, degree, 3, row_block_size=2)
    params = [p for p in model.cluster_head.parameters() if p.requires_grad]
    stats = cluster_head_gradient_diagnostics(cut_loss, orth_loss, params)
    assert stats["cut_grad_none_param_count"] == 0
    assert stats["orth_grad_none_param_count"] == 0
    assert np.isfinite(stats["cut_cluster_head_grad_l2"])
    assert np.isfinite(stats["orth_cluster_head_grad_l2"])
    assert stats["cut_cluster_head_grad_l2"] >= 0.0
    assert stats["orth_cluster_head_grad_l2"] >= 0.0
    assert all(param.grad is None for param in params)


def test_first_update_delta_uses_reforwarded_q():
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    initial = rng.normal(0.0, 0.1, size=(4, 5)).astype(np.float32)
    model = EdgeHiNoSModel(initial, 3, 6, 8, 5, 2, False)
    optimizer = torch.optim.SGD(model.cluster_head.parameters(), lr=0.5)
    src = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 3], dtype=torch.long)
    time_feat = torch.randn(4, 3)
    _, Q0, logits0 = model(src, dst, time_feat, return_logits=True)
    initial_stats = compute_uniform_collapse_stage("initial", Q0, logits0, src, dst, 4, 2)
    loss = -torch.log(Q0[:, 0].clamp_min(1e-8)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, Q1, logits1 = model(src, dst, time_feat, return_logits=True)
    after_stats = compute_uniform_collapse_stage("after", Q1, logits1, src, dst, 4, 2)
    delta = uniform_delta(initial_stats, after_stats)
    assert not torch.allclose(Q0.detach(), Q1.detach())
    assert abs(delta["q_uniform_l2_mean"] - (after_stats["q_uniform_l2_mean"] - initial_stats["q_uniform_l2_mean"])) < 1e-12


def test_diagnostic_projection_matches_explicit_incidence():
    torch.manual_seed(2)
    Q = torch.softmax(torch.randn(4, 3), dim=1)
    src = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 3], dtype=torch.long)
    B = torch.zeros((4, 4), dtype=Q.dtype)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        B[u, i] += 1.0
        B[v, i] += 1.0
    S_ref = B @ Q
    S_ref = S_ref / S_ref.sum(dim=1, keepdim=True).clamp_min(1e-8)
    S_got = project_edge_assignments_to_nodes_global(Q, src, dst, num_nodes=4)
    assert float((S_ref - S_got).abs().max()) < 1e-6
