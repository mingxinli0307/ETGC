import json
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from edge_losses import (
    edge_orthqa_penalty_global,
    edge_ppr_proximity_loss,
    edge_trace_mincut_loss_global,
    project_edge_assignments_to_nodes_global,
    projection_loss_global,
)
from edge_model import EdgeHiNoSModel, load_pretrained_node_features
from edge_train import build_optimizer_for_node_emb_mode
from summarize_direct_node_time_all_datasets import (
    CONFIGS,
    COMMON_CONFIG,
    discover_datasets,
    make_plan,
    make_run_config,
    stable_config_checksum,
)


def _node_features(num_nodes=5, dim=4):
    return (np.arange(num_nodes * dim, dtype=np.float32).reshape(num_nodes, dim) / 10.0) + 0.1


def _direct_model(node_mode="full", K=3):
    model = EdgeHiNoSModel(
        _node_features(),
        time_dim=2,
        edge_dim=6,
        edge_hidden_dim=7,
        cluster_hidden_dim=5,
        K=K,
        directed=False,
        cluster_output_bias_mode="none",
        cluster_input_norm="layernorm",
        edge_encoder_mode="direct_node_time",
    )
    opt, info = build_optimizer_for_node_emb_mode(model, 1e-2, node_mode, 1e-4)
    return model, opt, info


def _toy_inputs():
    src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    time_feat = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=torch.float32)
    return src, dst, time_feat


def _toy_affinity():
    rows = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int64)
    cols = np.array([1, 0, 2, 1, 3, 2, 0, 3], dtype=np.int64)
    data = np.array([0.5, 0.5, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4], dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float32)


def test_node2vec_initialization_matches_given_matrix_and_missing_required_fails(tmp_path):
    H0 = _node_features(3, 2)
    emb = tmp_path / "toy_feature.emb"
    emb.write_text("\n".join(f"{i} {row[0]} {row[1]}" for i, row in enumerate(H0)) + "\n", encoding="utf-8")
    loaded = load_pretrained_node_features(str(emb), 3, 8, seed=1, require_existing=True)
    model = EdgeHiNoSModel(loaded, 2, 6, 7, 5, 2, False, edge_encoder_mode="direct_node_time")
    assert np.allclose(loaded, H0)
    assert torch.allclose(model.node_emb.detach(), torch.from_numpy(H0))
    with pytest.raises(FileNotFoundError, match="Missing required Node2Vec"):
        load_pretrained_node_features(str(tmp_path / "missing.emb"), 3, 8, seed=1, require_existing=True)


def test_direct_node_time_is_ordered_raw_concat_and_time_sensitive():
    model, _, _ = _direct_model()
    src, dst, time_feat = _toy_inputs()
    expected = torch.cat(
        [
            model.node_emb.detach().index_select(0, src),
            model.node_emb.detach().index_select(0, dst),
            time_feat,
        ],
        dim=-1,
    )
    got = model.build_direct_node_time_event_repr(src, dst, time_feat)
    forward_repr, _ = model(src, dst, time_feat)
    reversed_repr = model.build_direct_node_time_event_repr(dst, src, time_feat)
    changed_time = model.build_direct_node_time_event_repr(src[:1], dst[:1], torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(got, expected)
    assert torch.allclose(forward_repr, expected)
    assert not torch.allclose(got, reversed_repr)
    assert not torch.allclose(got[:1], changed_time)


def test_direct_mode_has_no_edge_mlp_parameters_or_optimizer_membership():
    model, opt, info = _direct_model("full")
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert model.edge_mlp is None
    assert info["optimizer_groups"]
    assert id(model.node_emb) in opt_ids
    assert len(opt_ids) == sum(len(group["params"]) for group in opt.param_groups)


def test_proximity_backward_updates_node_path_not_cluster_output():
    model, _, _ = _direct_model("full")
    src, dst, time_feat = _toy_inputs()
    r, _ = model(src, dst, time_feat)
    pi = sp.csr_matrix(
        (
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0])),
        ),
        shape=(4, 4),
    )
    loss = edge_ppr_proximity_loss(
        r,
        {0: 0, 1: 1, 2: 2, 3: 3},
        np.array([0, 1, 2, 3], dtype=np.int64),
        pi,
        4,
        np.random.RandomState(3),
        torch.device("cpu"),
    )
    loss.backward()
    assert model.node_emb.grad is not None
    assert torch.isfinite(model.node_emb.grad).all()
    assert float(model.node_emb.grad.abs().sum()) > 0.0
    assert model.cluster_output.weight.grad is None or float(model.cluster_output.weight.grad.abs().sum()) == 0.0


def test_cut_orth_orthqa_and_projection_backpropagate_to_node_embedding():
    src, dst, time_feat = _toy_inputs()
    W = _toy_affinity()
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    for idx, loss_name in enumerate(["cut", "orth", "orthqa", "projection"]):
        rng = np.random.RandomState(100 + idx)
        model = EdgeHiNoSModel(
            rng.normal(0.0, 0.4, size=(5, 4)).astype(np.float32),
            time_dim=2,
            edge_dim=6,
            edge_hidden_dim=7,
            cluster_hidden_dim=5,
            K=3,
            directed=False,
            cluster_output_bias_mode="none",
            cluster_input_norm="none",
            edge_encoder_mode="direct_node_time",
        )
        model.node_emb.requires_grad_(True)
        _, Q = model(src, dst, time_feat)
        _, cut_loss, orth_loss = edge_trace_mincut_loss_global(Q, W, degree, 3, row_block_size=2)
        if loss_name == "cut":
            loss = cut_loss
        elif loss_name == "orth":
            loss = orth_loss
        elif loss_name == "orthqa":
            loss = edge_orthqa_penalty_global(Q, torch.from_numpy(degree))
        else:
            loss = projection_loss_global(Q, src, dst, num_nodes=5)
        loss.backward()
        assert model.node_emb.grad is not None, loss_name
        assert torch.isfinite(model.node_emb.grad).all(), loss_name
        assert float(model.node_emb.grad.abs().sum()) > 0.0, loss_name
        assert model.cluster_hidden.weight.grad is not None, loss_name
        assert model.cluster_output.weight.grad is not None, loss_name


def test_frozen_full_and_small_lr_node_embedding_modes():
    src, dst, time_feat = _toy_inputs()
    frozen, frozen_opt, frozen_info = _direct_model("frozen")
    before = frozen.node_emb.detach().clone()
    _, Q = frozen(src, dst, time_feat)
    loss = projection_loss_global(Q, src, dst, num_nodes=5)
    frozen_opt.zero_grad()
    loss.backward()
    frozen_opt.step()
    assert frozen_info["node_emb_trainable"] is False
    assert all(id(frozen.node_emb) not in {id(p) for p in group["params"]} for group in frozen_opt.param_groups)
    assert torch.allclose(before, frozen.node_emb.detach())

    full, full_opt, _ = _direct_model("full")
    before_full = full.node_emb.detach().clone()
    r, _ = full(src, dst, time_feat)
    loss = r.square().mean()
    full_opt.zero_grad()
    loss.backward()
    full_opt.step()
    assert not torch.allclose(before_full, full.node_emb.detach())

    small, small_opt, small_info = _direct_model("small_lr")
    group_lrs = sorted(float(group["lr"]) for group in small_opt.param_groups)
    group_ids = [id(p) for group in small_opt.param_groups for p in group["params"]]
    assert group_lrs == [1e-4, 1e-2]
    assert small_info["node_emb_lr"] == 1e-4
    assert len(group_ids) == len(set(group_ids))


def test_mlp_default_path_still_uses_edge_mlp_and_shape():
    model = EdgeHiNoSModel(_node_features(), 2, 6, 7, 5, 3, False)
    src, dst, time_feat = _toy_inputs()
    R, Q = model(src, dst, time_feat)
    assert model.edge_encoder_mode == "mlp"
    assert model.edge_mlp is not None
    assert tuple(R.shape) == (4, 6)
    assert tuple(Q.shape) == (4, 3)


def test_direct_projection_matches_explicit_accumulation_and_reforwards_after_prox():
    model, opt, _ = _direct_model("full")
    src, dst, time_feat = _toy_inputs()
    _, Q0 = model(src, dst, time_feat)
    S = project_edge_assignments_to_nodes_global(Q0, src, dst, 5)
    explicit = torch.zeros_like(S)
    explicit.index_add_(0, src, Q0)
    explicit.index_add_(0, dst, Q0)
    explicit = explicit / explicit.sum(dim=1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(S, explicit)

    pi = sp.csr_matrix((np.ones(4, dtype=np.float32), (np.arange(4), np.roll(np.arange(4), -1))), shape=(4, 4))
    r, _ = model(src, dst, time_feat)
    loss = edge_ppr_proximity_loss(
        r,
        {0: 0, 1: 1, 2: 2, 3: 3},
        np.array([0, 1, 2, 3], dtype=np.int64),
        pi,
        4,
        np.random.RandomState(8),
        torch.device("cpu"),
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    _, Q1 = model(src, dst, time_feat)
    assert not torch.allclose(Q0.detach(), Q1.detach())
    assert torch.isfinite(Q1).all()


def _write_toy_dataset(root):
    ds = root / "dataset" / "toy"
    emb = root / "emb"
    ds.mkdir(parents=True)
    emb.mkdir(parents=True)
    ds.joinpath("toy.txt").write_text("0 1 0.0\n1 2 0.1\n2 0 0.2\n", encoding="utf-8")
    ds.joinpath("node2label.txt").write_text("0 0\n1 1\n2 1\n", encoding="utf-8")
    emb.joinpath("toy_feature.emb").write_text(
        "0 0.1 0.2\n1 0.3 0.4\n2 0.5 0.6\n",
        encoding="utf-8",
    )


def test_inventory_plan_and_checksum_include_direct_mode(tmp_path):
    _write_toy_dataset(tmp_path)
    inventory = discover_datasets(tmp_path, COMMON_CONFIG)
    assert inventory[0]["dataset_name"] == "toy"
    assert inventory[0]["usable"] is True
    assert inventory[0]["node2vec_exists"] is True
    assert inventory[0]["node2vec_shape"] == [3, 2]
    plan = make_plan(inventory, "", "", None, tmp_path, tmp_path / "logs")
    assert len(plan) == 5 * 3
    assert {cfg["config"] for cfg in plan} == set(CONFIGS)
    e0 = make_run_config("toy", "E0", 42, tmp_path, tmp_path / "logs")
    e1 = make_run_config("toy", "E1", 42, tmp_path, tmp_path / "logs")
    assert stable_config_checksum(e0) != stable_config_checksum(e1)
    assert e1["edge_encoder_mode"] == "direct_node_time"
    assert e1["cluster_init_mode"] == "random_orthogonal"


def test_inventory_marks_missing_node2vec_unusable(tmp_path):
    ds = tmp_path / "dataset" / "toy"
    ds.mkdir(parents=True)
    ds.joinpath("toy.txt").write_text("0 1 0.0\n", encoding="utf-8")
    ds.joinpath("node2label.txt").write_text("0 0\n1 1\n", encoding="utf-8")
    inventory = discover_datasets(tmp_path, COMMON_CONFIG)
    assert inventory[0]["usable"] is False
    assert "Node2Vec" in inventory[0]["failure_reason"]
