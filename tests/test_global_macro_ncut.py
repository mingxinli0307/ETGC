import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import edge_ncut_loss, edge_ncut_loss_global
from edge_metrics import align_predicted_labels, evaluate_node_clustering
from edge_model import EdgeHiNoSModel
from edge_proximity import build_temporal_edge_event_transition, sparse_row_topk


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
