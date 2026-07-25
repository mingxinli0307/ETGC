import inspect
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import edge_orthqa_penalty_global, edge_trace_mincut_loss_global, trace_mincut_orthogonality_loss
from edge_model import (
    EdgeHiNoSModel,
    initialize_cluster_output_from_prototypes,
    torch_kmeans_plus_plus,
)
from edge_uniform_diagnostic import q_rank_statistics


def _initial(num_nodes=6, dim=5):
    rng = np.random.RandomState(123)
    return rng.normal(0.0, 0.1, size=(num_nodes, dim)).astype(np.float32)


def _inputs():
    src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    time_feat = torch.randn(4, 3)
    return src, dst, time_feat


def test_cluster_output_bias_modes_have_expected_parameters_and_shapes():
    src, dst, time_feat = _inputs()
    models = {
        mode: EdgeHiNoSModel(_initial(), 3, 7, 8, 5, 4, False, cluster_output_bias_mode=mode)
        for mode in ["default", "zero", "none"]
    }
    assert models["default"].cluster_output.bias is not None
    assert models["zero"].cluster_output.bias is not None
    assert torch.count_nonzero(models["zero"].cluster_output.bias.detach()) == 0
    assert models["zero"].cluster_output.bias.requires_grad
    assert models["none"].cluster_output.bias is None
    for model in models.values():
        _, Q, logits = model(src, dst, time_feat, return_logits=True)
        assert tuple(Q.shape) == (4, 4)
        assert tuple(logits.shape) == (4, 4)


def test_none_bias_has_no_output_bias_parameter_in_optimizer():
    model = EdgeHiNoSModel(_initial(), 3, 7, 8, 5, 4, False, cluster_output_bias_mode="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert model.cluster_output.bias is None
    assert id(model.cluster_output.weight) in ids


def test_layernorm_normalizes_only_cluster_input_rows():
    torch.manual_seed(0)
    model = EdgeHiNoSModel(_initial(), 3, 7, 8, 5, 4, False, cluster_input_norm="layernorm")
    src, dst, time_feat = _inputs()
    R, _, _ = model(src, dst, time_feat, return_logits=True)
    cluster_input, _ = model.cluster_hidden_from_edge_repr(R)
    assert isinstance(model.cluster_input_norm, torch.nn.LayerNorm)
    assert model.cluster_input_norm.elementwise_affine is False
    assert torch.allclose(cluster_input.mean(dim=1), torch.zeros(cluster_input.size(0)), atol=1e-5)
    assert torch.allclose(cluster_input.var(dim=1, unbiased=False), torch.ones(cluster_input.size(0)), atol=5e-3)


def test_none_input_norm_matches_manual_old_cluster_path():
    torch.manual_seed(1)
    model = EdgeHiNoSModel(_initial(), 3, 7, 8, 5, 4, False, cluster_input_norm="none")
    src, dst, time_feat = _inputs()
    R, Q, logits = model(src, dst, time_feat, return_logits=True)
    cluster_input, hidden = model.cluster_hidden_from_edge_repr(R)
    manual_logits = model.cluster_output(hidden)
    assert torch.allclose(cluster_input, R)
    assert torch.allclose(logits, manual_logits)
    assert torch.allclose(Q, torch.softmax(manual_logits, dim=-1))


def test_prototype_kmeans_is_reproducible_and_finite():
    torch.manual_seed(2)
    features = torch.randn(40, 6)
    c1 = torch_kmeans_plus_plus(features, 4, seed=7, sample_size=30, max_iters=5)
    c2 = torch_kmeans_plus_plus(features, 4, seed=7, sample_size=30, max_iters=5)
    assert tuple(c1.shape) == (4, 6)
    assert torch.allclose(c1, c2)
    assert torch.isfinite(c1).all()
    assert float(torch.pdist(c1).max()) > 0.0


def test_prototype_initializes_output_weight_without_replacing_parameter_or_using_labels():
    torch.manual_seed(3)
    model = EdgeHiNoSModel(_initial(), 3, 7, 8, 5, 4, False, cluster_output_bias_mode="zero")
    hidden = torch.randn(50, 5)
    before_weight_id = id(model.cluster_output.weight)
    expected = torch_kmeans_plus_plus(hidden, 4, seed=11, sample_size=40, max_iters=4)
    stats = initialize_cluster_output_from_prototypes(model, hidden, 4, seed=11, sample_size=40, lloyd_iters=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert "labels" not in inspect.signature(initialize_cluster_output_from_prototypes).parameters
    assert before_weight_id == id(model.cluster_output.weight)
    assert before_weight_id in opt_ids
    assert torch.allclose(model.cluster_output.weight.detach(), expected, atol=1e-6)
    assert torch.count_nonzero(model.cluster_output.bias.detach()) == 0
    assert stats["prototype_shape"] == [4, 5]
    assert all("label" not in key.lower() and "pred" not in key.lower() for key in stats)


def test_q_rank_statistics_detect_rank_one_and_multicluster_q():
    p = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    rank_one = p.repeat(8, 1)
    stats_one = q_rank_statistics(rank_one)
    assert abs(stats_one["q_rank1_energy_ratio"] - 1.0) < 1e-8
    assert stats_one["q_centered_energy"] < 1e-10
    assert stats_one["q_numerical_rank"] == 1

    multi = torch.eye(3).repeat(3, 1)
    stats_multi = q_rank_statistics(multi)
    assert stats_multi["q_effective_rank"] > 2.5
    assert stats_multi["q_numerical_rank"] > 1
    assert stats_multi["q_centered_energy"] > 0.0
    assert stats_multi["q_centered_effective_rank"] > 1.0


def test_orthqa_matches_dense_reference():
    Q = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]], dtype=torch.float32)
    degree = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    sqrt_k = 2.0 ** 0.5
    total_volume = degree.sum()
    cluster_volume_sqrt = torch.sqrt((degree[:, None] * Q.square()).sum(dim=0) + 1e-12)
    reference = (sqrt_k - cluster_volume_sqrt.sum() / torch.sqrt(total_volume + 1e-12)) / (sqrt_k - 1.0)
    got = edge_orthqa_penalty_global(Q, degree)
    assert abs(float(got - reference)) < 1e-6


def test_orthqa_balanced_one_hot_is_zero_and_unbalanced_is_positive():
    degree = torch.ones(6)
    balanced = torch.tensor(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        dtype=torch.float32,
    )
    unbalanced = torch.tensor(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=torch.float32,
    )
    assert abs(float(edge_orthqa_penalty_global(balanced, degree))) < 1e-5
    assert float(edge_orthqa_penalty_global(unbalanced, degree)) > 0.0


def test_orthqa_rank_one_rows_are_one_for_uniform_and_nonuniform_p():
    degree = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for p in [torch.full((3,), 1.0 / 3.0), torch.tensor([0.7, 0.2, 0.1])]:
        Q = p.view(1, -1).repeat(4, 1)
        assert abs(float(edge_orthqa_penalty_global(Q, degree)) - 1.0) < 1e-5


def test_orthqa_soft_assignment_backward_and_exceptions():
    Q = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.55, 0.45]], dtype=torch.float32, requires_grad=True)
    degree = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float32)
    loss = edge_orthqa_penalty_global(Q, degree)
    loss.backward()
    assert torch.isfinite(loss)
    assert Q.grad is not None
    assert torch.isfinite(Q.grad).all()
    with pytest.raises(ValueError, match="K > 1"):
        edge_orthqa_penalty_global(torch.ones(3, 1), torch.ones(3))
    with pytest.raises(FloatingPointError, match="total_volume"):
        edge_orthqa_penalty_global(torch.ones(3, 2), torch.zeros(3))


def test_orth_branch_compatibility_and_default_orth_type():
    torch.manual_seed(4)
    Q = torch.softmax(torch.randn(4, 3), dim=1)
    W = torch.eye(4).to_sparse()
    degree = torch.ones(4)
    expected_orth = trace_mincut_orthogonality_loss(Q, 3)
    default_total, default_cut, default_orth = edge_trace_mincut_loss_global(Q, W, degree, 3)
    explicit_total, explicit_cut, explicit_orth = edge_trace_mincut_loss_global(Q, W, degree, 3, orth_type="orth")
    assert abs(float(default_orth - expected_orth)) < 1e-6
    assert abs(float(explicit_orth - expected_orth)) < 1e-6
    assert abs(float(default_total - explicit_total)) < 1e-6
    assert abs(float(default_cut - explicit_cut)) < 1e-6


def test_trace_mincut_selects_orthqa_branch_when_requested():
    torch.manual_seed(5)
    Q = torch.softmax(torch.randn(5, 3), dim=1)
    W = torch.eye(5).to_sparse()
    degree = torch.ones(5)
    expected_orthqa = edge_orthqa_penalty_global(Q, degree)
    total, cut, selected_orth = edge_trace_mincut_loss_global(Q, W, degree, 3, orth_type="orthqa")
    assert abs(float(selected_orth - expected_orthqa)) < 1e-6
    assert abs(float(total - (cut + expected_orthqa))) < 1e-6
