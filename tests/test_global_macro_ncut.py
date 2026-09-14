import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import (project_edge_assignments_to_nodes_global, projection_loss_global, scipy_csr_to_torch_sparse_coo)
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
