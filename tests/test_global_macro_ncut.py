import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import (
    edge_ncut_loss,
    edge_ncut_loss_global,
    edge_trace_mincut_loss_global,
    project_edge_assignments_to_nodes_global,
    projection_loss_global,
    scipy_csr_to_torch_sparse_coo,
)
from edge_metrics import align_predicted_labels, evaluate_node_clustering
from edge_model import EdgeHiNoSModel
from edge_proximity import build_temporal_edge_event_transition, sparse_row_topk
from edge_train import build_optimizer_for_node_emb_mode


def _toy_affinity():
    rows = np.array([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int64)
    cols = np.array([1, 0, 2, 1, 3, 2, 3, 0], dtype=np.int64)
    data = np.array([0.7, 0.7, 0.5, 0.5, 0.9, 0.9, 0.2, 0.2], dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float32)


def test_macro_f1_matches_sklearn_after_alignment():
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    pred = np.asarray([2, 2, 0, 1, 1, 1], dtype=np.int64)
    aligned = align_predicted_labels(labels, pred)
    expected = f1_score(labels, aligned, average="macro", zero_division=0)
    metrics = evaluate_node_clustering(labels, pred)
    assert abs(metrics["Macro_F1"] - expected) < 1e-12
    assert metrics["F1"] == metrics["Macro_F1"]


def test_global_ncut_matches_full_union_local_loss_and_backward():
    torch.manual_seed(0)
    W = _toy_affinity()
    logits = torch.randn(4, 3, requires_grad=True)
    Q = torch.softmax(logits, dim=1)
    local = edge_ncut_loss(Q, np.arange(4, dtype=np.int64), W, 3)
    global_loss = edge_ncut_loss_global(Q, W, 3, row_block_size=2)
    assert abs(float(local.detach()) - float(global_loss.detach())) < 1e-5
    global_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_global_ncut_backpropagates_to_model_parameters():
    torch.manual_seed(1)
    rng = np.random.RandomState(1)
    initial = rng.normal(0.0, 0.1, size=(4, 5)).astype(np.float32)
    model = EdgeHiNoSModel(
        initial_node_features=initial,
        time_dim=3,
        edge_dim=6,
        edge_hidden_dim=8,
        cluster_hidden_dim=5,
        K=2,
        directed=False,
    )
    src = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 3], dtype=torch.long)
    time_feat = torch.randn(4, 3)
    _, Q = model(src, dst, time_feat)
    loss = edge_ncut_loss_global(Q, _toy_affinity(), 2, row_block_size=2)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


def test_trace_mincut_matches_dense_reference_sparse_and_block():
    torch.manual_seed(2)
    W = _toy_affinity()
    Q = torch.softmax(torch.randn(4, 3), dim=1)
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    W_dense = torch.from_numpy(W.toarray()).float()
    degree_t = torch.from_numpy(degree).float()
    lambda_orth = 0.7

    reference_cut = -torch.trace(Q.t() @ W_dense @ Q) / torch.trace(Q.t() @ torch.diag(degree_t) @ Q)
    QtQ = Q.t() @ Q
    reference_orth = torch.linalg.norm(
        QtQ / torch.linalg.norm(QtQ, ord="fro") - torch.eye(3) / (3.0 ** 0.5),
        ord="fro",
    )
    reference_total = reference_cut + lambda_orth * reference_orth

    total_block, cut_block, orth_block = edge_trace_mincut_loss_global(
        Q, W, degree, 3, lambda_orth=lambda_orth, row_block_size=2
    )
    W_sparse = scipy_csr_to_torch_sparse_coo(W, Q.device, Q.dtype)
    total_sparse, cut_sparse, orth_sparse = edge_trace_mincut_loss_global(
        Q, W_sparse, degree, 3, lambda_orth=lambda_orth, row_block_size=2
    )

    assert abs(float(cut_block - reference_cut)) < 1e-5
    assert abs(float(orth_block - reference_orth)) < 1e-5
    assert abs(float(total_block - reference_total)) < 1e-5
    assert abs(float(cut_sparse - reference_cut)) < 1e-5
    assert abs(float(orth_sparse - reference_orth)) < 1e-5
    assert abs(float(total_sparse - reference_total)) < 1e-5


def test_trace_numerator_equals_q_times_wq():
    torch.manual_seed(3)
    W = torch.from_numpy(_toy_affinity().toarray()).float()
    Q = torch.softmax(torch.randn(4, 2), dim=1)
    trace_value = torch.trace(Q.t() @ W @ Q)
    q_wq_value = (Q * (W @ Q)).sum()
    assert abs(float(trace_value - q_wq_value)) < 1e-6


def test_trace_denominator_equals_degree_weighted_q_square():
    torch.manual_seed(4)
    W = torch.from_numpy(_toy_affinity().toarray()).float()
    degree = W.sum(dim=1)
    Q = torch.softmax(torch.randn(4, 2), dim=1)
    trace_value = torch.trace(Q.t() @ torch.diag(degree) @ Q)
    weighted_value = (degree.unsqueeze(1) * Q.square()).sum()
    assert abs(float(trace_value - weighted_value)) < 1e-6


def test_trace_orthogonality_term_matches_formula():
    torch.manual_seed(5)
    Q = torch.softmax(torch.randn(5, 3), dim=1)
    QtQ = Q.t() @ Q
    expected = torch.linalg.norm(
        QtQ / torch.linalg.norm(QtQ, ord="fro") - torch.eye(3) / (3.0 ** 0.5),
        ord="fro",
    )
    W = sp.eye(5, format="csr", dtype=np.float32)
    total, cut, got = edge_trace_mincut_loss_global(Q, W, np.ones(5, dtype=np.float32), 3)
    del total, cut
    assert abs(float(got - expected)) < 1e-6


def test_trace_mincut_projection_joint_backward_is_finite_and_nonzero():
    torch.manual_seed(6)
    W = _toy_affinity()
    logits = torch.randn(4, 3, requires_grad=True)
    Q = torch.softmax(logits, dim=1)
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    cluster_loss, cut_loss, orth_loss = edge_trace_mincut_loss_global(Q, W, degree, 3, row_block_size=2)
    proj_loss = projection_loss_global(
        Q,
        torch.tensor([0, 1, 2, 0], dtype=torch.long),
        torch.tensor([1, 2, 3, 3], dtype=torch.long),
        num_nodes=4,
    )
    loss = cluster_loss + proj_loss
    loss.backward()
    assert torch.isfinite(cluster_loss)
    assert torch.isfinite(cut_loss)
    assert torch.isfinite(orth_loss)
    assert torch.isfinite(proj_loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_global_projection_matches_explicit_incidence_matrix():
    torch.manual_seed(7)
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


def test_node_embedding_optimizer_modes():
    rng = np.random.RandomState(8)
    initial = rng.normal(0.0, 0.1, size=(4, 5)).astype(np.float32)

    frozen_model = EdgeHiNoSModel(initial, 3, 6, 8, 5, 2, False)
    frozen_opt, frozen_info = build_optimizer_for_node_emb_mode(frozen_model, 1e-4, "frozen", 1e-5)
    _, Q = frozen_model(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.randn(2, 3))
    Q.sum().backward()
    assert frozen_info["node_emb_trainable"] is False
    assert frozen_model.node_emb.grad is None
    assert all(id(frozen_model.node_emb) not in {id(p) for p in group["params"]} for group in frozen_opt.param_groups)

    small_model = EdgeHiNoSModel(initial, 3, 6, 8, 5, 2, False)
    small_opt, small_info = build_optimizer_for_node_emb_mode(small_model, 1e-4, "small_lr", 1e-5)
    group_lrs = sorted(group["lr"] for group in small_opt.param_groups)
    group_param_ids = [id(p) for group in small_opt.param_groups for p in group["params"]]
    assert small_info["node_emb_trainable"] is True
    assert group_lrs == [1e-5, 1e-4]
    assert len(group_param_ids) == len(set(group_param_ids))

    full_model = EdgeHiNoSModel(initial, 3, 6, 8, 5, 2, False)
    full_opt, full_info = build_optimizer_for_node_emb_mode(full_model, 1e-4, "full", 1e-5)
    assert full_info["node_emb_trainable"] is True
    assert [group["lr"] for group in full_opt.param_groups] == [1e-4]


def test_sparse_row_topk_negative_keeps_matrix():
    mat = sp.csr_matrix(
        (
            np.array([0.1, 0.3, 0.2, 0.4], dtype=np.float32),
            (np.array([0, 0, 1, 2]), np.array([1, 2, 2, 0])),
        ),
        shape=(3, 3),
        dtype=np.float32,
    )
    out = sparse_row_topk(mat, -1)
    diff = (out - mat).tocoo()
    assert out.shape == mat.shape
    assert out.nnz == mat.nnz
    assert diff.nnz == 0 or np.allclose(diff.data, 0.0)


def test_temporal_transition_chunked_matches_reference():
    src = np.asarray([0, 0, 1, 2, 0, 1], dtype=np.int64)
    dst = np.asarray([1, 2, 2, 0, 1, 0], dtype=np.int64)
    times = np.asarray([0.0, 0.1, 0.2, 0.3, 0.3, 0.4], dtype=np.float32)
    num_nodes = 3
    beta = 2.0
    edge_neighbor_k = -1

    incident = [[] for _ in range(num_nodes)]
    for eid, (u, v) in enumerate(zip(src, dst)):
        incident[int(u)].append(int(eid))
        if int(v) != int(u):
            incident[int(v)].append(int(eid))
    positions = []
    for events in incident:
        events.sort(key=lambda eid: (float(times[eid]), int(eid)))
        positions.append({int(eid): pos for pos, eid in enumerate(events)})

    weights = {}
    for i in range(len(src)):
        for node in (int(src[i]), int(dst[i])):
            events = incident[node]
            pos = positions[node].get(int(i))
            candidates, raw = [], []
            if pos is not None:
                for j in events[pos + 1 :]:
                    dt_raw = float(times[j]) - float(times[i])
                    if dt_raw > 0.0 or (abs(dt_raw) <= 1e-9 and int(j) > int(i)):
                        candidates.append(int(j))
                        raw.append(float(np.exp(-beta * max(dt_raw, 0.0))))
            if not candidates:
                weights[(int(i), int(i))] = weights.get((int(i), int(i)), 0.0) + 0.5
                continue
            raw = np.asarray(raw, dtype=np.float64)
            probs = raw / raw.sum()
            for j, prob in zip(candidates, probs):
                weights[(int(i), int(j))] = weights.get((int(i), int(j)), 0.0) + 0.5 * float(prob)

    rows, cols, data = zip(*((i, j, v) for (i, j), v in weights.items()))
    ref = sp.csr_matrix((data, (rows, cols)), shape=(len(src), len(src)), dtype=np.float32)
    ref = ref.multiply(1.0 / np.asarray(ref.sum(axis=1)).ravel()[:, None]).tocsr()
    got = build_temporal_edge_event_transition(src, dst, times, num_nodes, edge_neighbor_k, beta)
    assert got.shape == ref.shape
    assert np.allclose(got.toarray(), ref.toarray(), atol=1e-7)


if __name__ == "__main__":
    test_macro_f1_matches_sklearn_after_alignment()
    test_global_ncut_matches_full_union_local_loss_and_backward()
    test_global_ncut_backpropagates_to_model_parameters()
    test_sparse_row_topk_negative_keeps_matrix()
    test_temporal_transition_chunked_matches_reference()
    print("test_global_macro_ncut: ok")
