import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_main import build_parser
from edge_model import EdgeHiNoSModel
from edge_proximity import build_edge_ncut_affinity, compute_edge_ppr_cached, sparse_symmetric_union_knn
from edge_time import build_edge_time_features
from edge_train import EdgeHiNoSTrainer


def test_current_timestamp_features_ignore_history_inputs():
    times = np.array([0.1, 0.4, 0.9], dtype=np.float32)
    src_a = np.array([0, 0, 1], dtype=np.int64)
    dst_a = np.array([1, 2, 2], dtype=np.int64)
    src_b = np.array([2, 1, 0], dtype=np.int64)
    dst_b = np.array([0, 2, 1], dtype=np.int64)
    current_a = build_edge_time_features(src_a, dst_a, times, 7, mode="current")
    current_b = build_edge_time_features(src_b, dst_b, times, 7, mode="current")
    history_a = build_edge_time_features(src_a, dst_a, times, 7, mode="history")
    history_b = build_edge_time_features(src_b, dst_b, times, 7, mode="history")
    assert np.allclose(current_a, current_b)
    assert current_a[:, 0].tolist() == pytest.approx(times.tolist())
    assert not np.allclose(history_a, history_b)


def test_direct_representation_is_node_node_current_time_concat():
    H = np.arange(12, dtype=np.float32).reshape(3, 4)
    model = EdgeHiNoSModel(
        H,
        time_dim=3,
        edge_dim=5,
        edge_hidden_dim=6,
        cluster_hidden_dim=4,
        K=2,
        directed=False,
        edge_encoder_mode="direct_node_time",
        cluster_head_type="cosine_prototype",
    )
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 0])
    time_feat = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    got = model.build_direct_node_time_event_repr(src, dst, time_feat)
    expected = torch.cat([model.node_emb[src], model.node_emb[dst], time_feat], dim=1)
    assert tuple(got.shape) == (2, 11)
    assert torch.allclose(got, expected)


def test_cosine_prototype_logits_temperature_and_softmax():
    H = np.eye(3, dtype=np.float32)
    model = EdgeHiNoSModel(
        H,
        time_dim=1,
        edge_dim=4,
        edge_hidden_dim=4,
        cluster_hidden_dim=4,
        K=2,
        directed=False,
        edge_encoder_mode="direct_node_time",
        cluster_head_type="cosine_prototype",
        prototype_temperature=0.5,
        hier_ncut_h=2,
    )
    assert model.cluster_output.bias is None
    with torch.no_grad():
        model.cluster_output.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    src = torch.tensor([0])
    dst = torch.tensor([1])
    time_feat = torch.tensor([[0.0]])
    r, q, logits = model(src, dst, time_feat, return_logits=True)
    r_hat = r / torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1e-12)
    c_hat = model.cluster_output.weight / torch.linalg.norm(model.cluster_output.weight, dim=1, keepdim=True).clamp_min(1e-12)
    expected_logits = (r_hat @ c_hat.t()) / 0.5
    assert torch.allclose(logits, expected_logits, atol=1e-6)
    assert torch.allclose(q.sum(dim=1), torch.ones(1), atol=1e-6)


def test_cosine_prototypes_use_module_random_initialization():
    features = np.random.RandomState(0).normal(size=(5, 3)).astype(np.float32)
    torch.manual_seed(11)
    model1 = EdgeHiNoSModel(
        features, 2, 4, 4, 4, 3, False,
        edge_encoder_mode="direct_node_time", cluster_head_type="cosine_prototype",
    )
    torch.manual_seed(11)
    model2 = EdgeHiNoSModel(
        features, 2, 4, 4, 4, 3, False,
        edge_encoder_mode="direct_node_time", cluster_head_type="cosine_prototype",
    )
    torch.testing.assert_close(model1.cluster_output.weight, model2.cluster_output.weight)
    assert model1.cluster_output.weight.shape == (model1.H, model1.cluster_input_dim)
    assert torch.isfinite(model1.cluster_output.weight).all()
    assert float(model1.cluster_output.weight.std(unbiased=False)) > 0.0






def test_symmetric_union_knn_semantics_and_degree_after_sparsification():
    W = sp.csr_matrix(
        np.array(
            [
                [0.0, 5.0, 4.0, 0.0],
                [5.0, 0.0, 1.0, 3.0],
                [4.0, 1.0, 0.0, 2.0],
                [0.0, 3.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    out = sparse_symmetric_union_knn(W, 1)
    dense = out.toarray()
    assert isinstance(out, sp.csr_matrix)
    assert np.allclose(dense, dense.T)
    assert dense[0, 1] == pytest.approx(5.0)
    assert dense[0, 2] == pytest.approx(4.0)
    assert dense[1, 3] == pytest.approx(3.0)
    assert dense[0, 1] != pytest.approx(2.5)
    assert np.allclose(sparse_symmetric_union_knn(W, -1).toarray(), W.toarray())
    Pi = sp.csr_matrix(np.array([[1.0, 5.0, 4.0], [5.0, 1.0, 1.0], [4.0, 1.0, 1.0]], dtype=np.float32))
    final_W = build_edge_ncut_affinity(Pi, 1, affinity_sparsify="symmetric_union_knn")
    degree = np.asarray(final_W.sum(axis=1)).ravel()
    assert np.allclose(degree, np.asarray(final_W.toarray().sum(axis=1)).ravel())
    assert np.allclose(final_W.toarray(), final_W.toarray().T)
    assert np.allclose(final_W.diagonal(), 0.0)


def test_cache_key_separates_row_topk_and_symmetric_union_knn(tmp_path):
    src = np.array([0, 1, 2], dtype=np.int64)
    dst = np.array([1, 2, 0], dtype=np.int64)
    times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    common = dict(
        dataset="toy",
        src=src,
        dst=dst,
        times=times,
        num_nodes=3,
        cache_dir=str(tmp_path),
        method="truncated",
        alpha=0.2,
        T=1,
        forest_samples=1,
        edge_neighbor_k=-1,
        edge_ppr_topk=1,
        beta=5.0,
        seed=7,
        quiet=True,
    )
    _, _, _, row_stats = compute_edge_ppr_cached(**common, affinity_sparsify="row_topk")
    _, _, _, sym_stats = compute_edge_ppr_cached(**common, affinity_sparsify="symmetric_union_knn")
    assert row_stats["pi_config_hash"] == sym_stats["pi_config_hash"]
    assert row_stats["w_config_hash"] != sym_stats["w_config_hash"]
