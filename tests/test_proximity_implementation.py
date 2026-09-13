import csv
import os
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import edge_ppr_proximity_loss, edge_ppr_proximity_loss_preindexed
from edge_main import build_parser
from edge_model import EdgeHiNoSModel
from edge_train import EdgeHiNoSTrainer


def _direct_model() -> EdgeHiNoSModel:
    features = np.arange(30, dtype=np.float32).reshape(6, 5) / 10.0
    return EdgeHiNoSModel(
        features,
        time_dim=3,
        edge_dim=8,
        edge_hidden_dim=9,
        cluster_hidden_dim=7,
        K=3,
        directed=False,
        edge_encoder_mode="direct_node_time",
    )


def test_encode_ids_returns_event_repr_without_calling_cluster_head():
    model = _direct_model()
    trainer = object.__new__(EdgeHiNoSTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.src_t = torch.tensor([0, 2, 4, 1])
    trainer.dst_t = torch.tensor([1, 3, 5, 4])
    trainer.time_feat_t = torch.tensor(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.4, 0.5, 0.6], [0.7, 0.1, 0.2]]
    )
    ids = np.array([0, 2, 3], dtype=np.int64)

    with mock.patch.object(model.cluster_output, "forward", side_effect=AssertionError("cluster head called")):
        event_repr = trainer._encode_ids(ids)

    ids_t = torch.from_numpy(ids)
    expected = torch.cat(
        [
            model.node_emb[trainer.src_t.index_select(0, ids_t)],
            model.node_emb[trainer.dst_t.index_select(0, ids_t)],
            trainer.time_feat_t.index_select(0, ids_t),
        ],
        dim=-1,
    )
    assert tuple(event_repr.shape) == (3, model.event_repr_dim)
    assert model.event_repr_dim == 2 * model.node_dim + model.time_dim
    assert torch.allclose(event_repr, expected)

    _, q = model(
        trainer.src_t.index_select(0, ids_t),
        trainer.dst_t.index_select(0, ids_t),
        trainer.time_feat_t.index_select(0, ids_t),
    )
    assert tuple(q.shape) == (3, 3)


def test_cosine_proximity_matches_repeated_norm_reference_and_backward():
    r_actual = torch.tensor(
        [[0.2, 0.5, 0.7], [0.6, 0.1, 0.4], [0.3, 0.8, 0.2], [0.9, 0.2, 0.5]],
        requires_grad=True,
    )
    r_reference = r_actual.detach().clone().requires_grad_(True)
    rows = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cols = np.array([1, 2, 0, 3, 0, 3, 1, 2])
    values = np.array([0.7, 0.3, 0.7, 0.3, 0.3, 0.7, 0.3, 0.7], dtype=np.float32)
    pi_e = sp.csr_matrix((values, (rows, cols)), shape=(4, 4))
    local_index = {event_id: event_id for event_id in range(4)}
    batch_ids = np.arange(4, dtype=np.int64)
    eps = 1e-8

    loss_actual = edge_ppr_proximity_loss(
        r_actual,
        local_index,
        batch_ids,
        pi_e,
        4,
        np.random.RandomState(11),
        torch.device("cpu"),
        similarity_mode="cosine",
        eps=eps,
    )

    anchors, positives, weights = [], [], []
    for event_id in batch_ids.tolist():
        start, end = pi_e.indptr[event_id], pi_e.indptr[event_id + 1]
        for neighbor, weight in zip(pi_e.indices[start:end], pi_e.data[start:end]):
            if int(neighbor) == int(event_id) or int(neighbor) not in local_index:
                continue
            anchors.append(local_index[event_id])
            positives.append(local_index[int(neighbor)])
            weights.append(float(weight))
    rng = np.random.RandomState(11)
    negative_global = rng.randint(0, 4, size=len(anchors))
    negatives = [local_index.get(int(event_id), int(rng.randint(0, len(local_index)))) for event_id in negative_global]
    anchor_t = torch.tensor(anchors)
    positive_t = torch.tensor(positives)
    negative_t = torch.tensor(negatives)
    weight_t = torch.tensor(weights)

    def repeated_norm_cosine(left, right):
        return (left * right).sum(dim=-1) / (
            torch.linalg.norm(left, dim=-1) * torch.linalg.norm(right, dim=-1) + eps
        )

    positive_score = repeated_norm_cosine(r_reference[anchor_t], r_reference[positive_t])
    negative_score = repeated_norm_cosine(r_reference[anchor_t], r_reference[negative_t])
    loss_reference = (weight_t * F.softplus(-positive_score)).mean() + F.softplus(negative_score).mean()

    assert torch.allclose(loss_actual, loss_reference, atol=1e-7, rtol=1e-6)
    loss_actual.backward()
    loss_reference.backward()
    assert torch.isfinite(r_actual.grad).all()
    assert torch.allclose(r_actual.grad, r_reference.grad, atol=1e-7, rtol=1e-6)


def _trainer_with_proximity_csr(pi_e: sp.csr_matrix) -> EdgeHiNoSTrainer:
    trainer = object.__new__(EdgeHiNoSTrainer)
    trainer.device = torch.device("cpu")
    trainer.Pi_E = pi_e.tocsr(copy=True)
    trainer.Pi_E.sort_indices()
    trainer.data = SimpleNamespace(num_events=trainer.Pi_E.shape[0])
    trainer._prox_indptr_np = np.asarray(trainer.Pi_E.indptr, dtype=np.int64)
    trainer._prox_indices_np = np.asarray(trainer.Pi_E.indices, dtype=np.int64)
    trainer._prox_data_np = np.asarray(trainer.Pi_E.data, dtype=np.float32)
    trainer._prox_global_to_local_np = np.full(trainer.data.num_events, -1, dtype=np.int64)
    return trainer


def _random_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_vectorized_proximity_batch_matches_historical_pairs_rng_loss_and_gradient():
    rows = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4])
    cols = np.array([0, 1, 4, 0, 3, 2, 4, 0, 2, 4, 1, 3])
    values = np.linspace(0.1, 1.2, rows.size, dtype=np.float32)
    pi_e = sp.csr_matrix((values, (rows, cols)), shape=(5, 5))
    trainer = _trainer_with_proximity_csr(pi_e)
    batch_ids = np.array([3, 0, 2], dtype=np.int64)
    union_ids = trainer._batch_union_ids(batch_ids)
    local_index = {int(event_id): index for index, event_id in enumerate(union_ids.tolist())}

    anchors, positives, weights = [], [], []
    for event_id in batch_ids.tolist():
        start, end = pi_e.indptr[event_id], pi_e.indptr[event_id + 1]
        for neighbor, weight in zip(pi_e.indices[start:end], pi_e.data[start:end]):
            if int(neighbor) == int(event_id) or int(neighbor) not in local_index:
                continue
            anchors.append(local_index[event_id])
            positives.append(local_index[int(neighbor)])
            weights.append(float(weight))
    reference_rng = np.random.RandomState(29)
    negative_global = reference_rng.randint(0, pi_e.shape[0], size=len(anchors))
    negatives = np.asarray(
        [
            local_index.get(int(event_id), int(reference_rng.randint(0, len(local_index))))
            for event_id in negative_global
        ],
        dtype=np.int64,
    )

    vectorized_rng = np.random.RandomState(29)
    prepared = trainer._prepare_proximity_batch(batch_ids, vectorized_rng)

    assert np.array_equal(prepared["union_ids"].numpy(), union_ids)
    assert np.array_equal(prepared["anchors"].numpy(), np.asarray(anchors))
    assert np.array_equal(prepared["positives"].numpy(), np.asarray(positives))
    assert np.array_equal(prepared["negatives"].numpy(), negatives)
    assert np.array_equal(prepared["weights"].numpy(), np.asarray(weights, dtype=np.float32))
    assert prepared["pair_count"] == len(anchors)
    assert _random_states_equal(reference_rng.get_state(), vectorized_rng.get_state())
    assert np.all(trainer._prox_global_to_local_np == -1)

    generator = torch.Generator().manual_seed(7)
    r_fast = torch.randn(len(union_ids), 6, generator=generator, requires_grad=True)
    r_reference = r_fast.detach().clone().requires_grad_(True)
    loss_fast = edge_ppr_proximity_loss_preindexed(
        r_fast,
        prepared["anchors"],
        prepared["positives"],
        prepared["negatives"],
        prepared["weights"],
        similarity_mode="cosine",
    )
    loss_reference = edge_ppr_proximity_loss(
        r_reference,
        local_index,
        batch_ids,
        pi_e,
        pi_e.shape[0],
        np.random.RandomState(29),
        torch.device("cpu"),
        similarity_mode="cosine",
    )
    assert torch.equal(loss_fast, loss_reference)
    loss_fast.backward()
    loss_reference.backward()
    assert torch.equal(r_fast.grad, r_reference.grad)


def test_proximity_runtime_fields_are_part_of_epoch_metrics():
    fieldnames = EdgeHiNoSTrainer._metrics_fieldnames()
    expected = {
        "prox_forward_backward_seconds",
        "prox_pair_build_seconds",
        "prox_pair_build_wait_seconds",
        "prox_h2d_seconds",
        "prox_encode_seconds",
        "prox_loss_forward_seconds",
        "prox_backward_seconds",
        "prox_optimizer_step_seconds",
        "prox_pair_count",
        "prox_optimizer_steps",
    }
    assert expected.issubset(fieldnames)


def test_one_epoch_global_cosine_proximity_smoke(tmp_path):
    data_dir = tmp_path / "dataset" / "toy"
    data_dir.mkdir(parents=True)
    (data_dir / "toy.txt").write_text(
        "0 1 0.0\n1 2 0.2\n2 3 0.4\n3 0 0.6\n0 2 0.8\n1 3 1.0\n",
        encoding="utf-8",
    )
    (data_dir / "node2label.txt").write_text("0 0\n1 0\n2 1\n3 1\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--dataset", "toy",
            "--data_root", str(tmp_path / "dataset"),
            "--cache_dir", str(tmp_path / "cache"),
            "--output_dir", str(output_dir),
            "--epoch", "1",
            "--batch_size", "2",
            "--edge_ppr_method", "truncated",
            "--edge_ppr_topk", "-1",
            "--time_dim", "3",
            "--edge_encoder_mode", "direct_node_time",
            "--cluster_head_type", "cosine_prototype",
            "--prototype_init_mode", "random",
            "--cluster_loss_type", "matrix_ncut",
            "--orth_type", "orthqa",
            "--ncut_scope", "global",
            "--node_emb_mode", "full",
            "--lambda_prox", "1",
            "--lambda_esg", "0",
            "--prox_similarity_mode", "cosine",
            "--prox_warmup_epochs", "0",
            "--quiet", "1",
        ]
    )
    args.model_seed = args.seed
    args.prototype_seed = args.seed
    args.forest_seed = args.seed
    trainer = EdgeHiNoSTrainer(args)
    trainer.train()

    with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as reader:
        rows = list(csv.DictReader(reader))
    assert len(rows) == 1
    assert int(rows[0]["prox_optimizer_steps"]) > 0
    assert int(rows[0]["prox_pair_count"]) > 0
    assert float(rows[0]["prox_pair_build_seconds"]) >= 0.0
    assert float(rows[0]["prox_pair_build_wait_seconds"]) >= 0.0
    assert float(rows[0]["prox_encode_seconds"]) >= 0.0
    assert float(rows[0]["prox_loss_forward_seconds"]) >= 0.0
    assert float(rows[0]["prox_backward_seconds"]) >= 0.0
    assert float(rows[0]["prox_optimizer_step_seconds"]) >= 0.0
    assert float(rows[0]["prox_forward_backward_seconds"]) >= 0.0
    assert np.isfinite(float(rows[0]["prox_loss"]))


def test_zero_orth_during_proximity_warmup_logs_uncomputed_global_losses(tmp_path):
    data_dir = tmp_path / "dataset" / "toy"
    data_dir.mkdir(parents=True)
    (data_dir / "toy.txt").write_text(
        "0 1 0.0\n1 2 0.2\n2 3 0.4\n3 0 0.6\n0 2 0.8\n1 3 1.0\n",
        encoding="utf-8",
    )
    (data_dir / "node2label.txt").write_text("0 0\n1 0\n2 1\n3 1\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--dataset", "toy",
            "--data_root", str(tmp_path / "dataset"),
            "--cache_dir", str(tmp_path / "cache"),
            "--output_dir", str(output_dir),
            "--epoch", "1",
            "--batch_size", "2",
            "--edge_ppr_method", "truncated",
            "--edge_ppr_topk", "-1",
            "--time_dim", "3",
            "--edge_encoder_mode", "direct_node_time",
            "--cluster_head_type", "cosine_prototype",
            "--prototype_init_mode", "random",
            "--cluster_loss_type", "matrix_ncut",
            "--orth_type", "orthqa",
            "--ncut_scope", "global",
            "--node_emb_mode", "full",
            "--lambda_prox", "1",
            "--lambda_orth", "0",
            "--lambda_esg", "0.1",
            "--prox_similarity_mode", "cosine",
            "--prox_warmup_epochs", "5",
            "--global_warmup_epochs", "0",
            "--loss_formulation_diagnostic", "1",
            "--quiet", "1",
        ]
    )
    args.model_seed = args.seed
    args.prototype_seed = args.seed
    args.forest_seed = args.seed

    EdgeHiNoSTrainer(args).train()

    with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as reader:
        rows = list(csv.DictReader(reader))
    assert len(rows) == 1
    assert np.isnan(float(rows[0]["weighted_cut_loss"]))
    assert np.isnan(float(rows[0]["weighted_orth_loss"]))
    assert int(rows[0]["prox_optimizer_steps"]) > 0
