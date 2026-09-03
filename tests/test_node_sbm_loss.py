import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import node_sbm_reconstruction_loss_global
from edge_main import build_parser
from edge_node_prior import (
    build_adaptive_node_prior,
    build_component_structural_node_prior,
    temporal_block_validation_auc,
)
from edge_train import EdgeHiNoSTrainer


class _ZeroLogitModel(torch.nn.Module):
    def forward(
        self,
        src,
        dst,
        time_feat,
        return_logits=False,
        return_cluster_hidden=False,
    ):
        edge_repr = torch.zeros((src.numel(), 2), dtype=time_feat.dtype)
        logits = torch.zeros((src.numel(), 3), dtype=time_feat.dtype)
        q = torch.softmax(logits, dim=1)
        result = [edge_repr, q]
        if return_logits:
            result.append(logits)
        if return_cluster_hidden:
            result.append(torch.ones((src.numel(), 2), dtype=time_feat.dtype))
        return tuple(result)


def test_node_sbm_loss_is_finite_symmetric_and_backpropagates():
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.5],
            [0.0, 3.0],
            [0.5, 2.0],
        ],
        requires_grad=True,
    )
    assignments = torch.softmax(logits, dim=1)
    pos_src = torch.tensor([0, 1, 2, 3])
    pos_dst = torch.tensor([1, 0, 3, 2])
    neg_src = torch.tensor([0, 1, 2, 3])
    neg_dst = torch.tensor([2, 3, 0, 1])

    loss, block_logits, stats = node_sbm_reconstruction_loss_global(
        assignments,
        pos_src,
        pos_dst,
        neg_src,
        neg_dst,
        directed=False,
    )

    assert torch.isfinite(loss)
    assert torch.allclose(block_logits, block_logits.t(), atol=1e-7)
    assert stats["node_sbm_block_symmetry_error"] < 1e-7
    assert stats["node_sbm_positive_score_mean"] > stats["node_sbm_negative_score_mean"]
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(torch.linalg.norm(logits.grad)) > 0.0


def test_node_sbm_cli_is_opt_in():
    args = build_parser().parse_args([])
    assert args.lambda_node_sbm == 0.0
    assert args.node_sbm_negative_ratio == 1.0
    assert args.lambda_node_prior == 0.0
    assert args.node_prior_mode == "none"
    assert args.node_prior_logit_strength == 0.0
    assert args.node_prior_event_role == "source"


def test_adaptive_node_prior_is_deterministic_and_label_free():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    src = torch.tensor([0, 1, 0, 2, 3, 2, 0, 2]).numpy()
    dst = torch.tensor([1, 0, 1, 3, 2, 3, 2, 0]).numpy()
    times = torch.arange(len(src), dtype=torch.float32).numpy()

    first, first_info = build_adaptive_node_prior(
        features,
        src,
        dst,
        times,
        K=2,
        restarts=4,
        seed=11,
        lloyd_iters=5,
    )
    second, second_info = build_adaptive_node_prior(
        features,
        src,
        dst,
        times,
        K=2,
        restarts=4,
        seed=11,
        lloyd_iters=5,
    )

    assert torch.equal(first, second)
    assert first_info == second_info
    assert first_info["node_prior_active_clusters"] == 2
    auc = temporal_block_validation_auc(first.numpy(), src, dst, times, K=2)
    assert 0.0 <= auc <= 1.0


def test_component_structural_prior_preserves_disconnected_residue():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
            [0.0, -1.0],
            [0.1, -0.9],
        ]
    )
    # Nodes 0..5 form the giant component; nodes 6..7 form a minor one.
    src = torch.tensor([0, 1, 2, 3, 4, 6]).numpy()
    dst = torch.tensor([1, 2, 3, 4, 5, 7]).numpy()
    labels, info = build_component_structural_node_prior(
        features,
        src,
        dst,
        K=3,
        seed=17,
        kmeans_restarts=3,
        bisecting_restarts=3,
    )

    assert info["node_prior_mode_effective"] == "component_preserving_bisecting_kmeans"
    assert info["node_prior_connected_components"] == 2
    assert info["node_prior_active_clusters"] == 3
    assert labels[6].item() == 2
    assert labels[7].item() == 2


def test_component_structural_prior_uses_global_kmeans_for_many_components():
    features = torch.eye(6)
    # No edge joins distinct nodes, hence component_count > K.
    nodes = torch.arange(6).numpy()
    labels, info = build_component_structural_node_prior(
        features,
        nodes,
        nodes,
        K=2,
        seed=19,
        kmeans_restarts=3,
        bisecting_restarts=3,
    )

    assert info["node_prior_mode_effective"] == "global_l2_kmeans"
    assert info["node_prior_connected_components"] == 6
    assert info["node_prior_active_clusters"] == 2


def test_component_structural_prior_can_force_global_kmeans():
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, 0.1]]
    )
    src = torch.tensor([0, 1, 2, 4]).numpy()
    dst = torch.tensor([1, 2, 3, 5]).numpy()
    _, info = build_component_structural_node_prior(
        features,
        src,
        dst,
        K=3,
        seed=23,
        kmeans_restarts=3,
        bisecting_restarts=3,
        preserve_components=False,
    )

    assert info["node_prior_connected_components"] == 2
    assert info["node_prior_mode_effective"] == "global_l2_kmeans"
    assert info["node_prior_preserve_components"] is False


def test_structural_prior_logit_bias_guides_event_q_by_source_node():
    trainer = EdgeHiNoSTrainer.__new__(EdgeHiNoSTrainer)
    trainer.model = _ZeroLogitModel()
    trainer.node_prior_t = torch.tensor([2, 0, 1])
    trainer.node_prior_logit_strength = 8.0
    trainer.node_prior_event_role = "source"
    trainer.K = 3
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 0])
    time_feat = torch.zeros((3, 1))

    edge_repr, q, logits, hidden = trainer._forward_event_tensors(
        src,
        dst,
        time_feat,
        return_logits=True,
        return_cluster_hidden=True,
    )

    assert edge_repr.shape == (3, 2)
    assert hidden.shape == (3, 2)
    assert torch.equal(q.argmax(dim=1), trainer.node_prior_t[src])
    assert torch.equal(logits.argmax(dim=1), trainer.node_prior_t[src])
    assert torch.allclose(q.sum(dim=1), torch.ones(3))


def test_structural_prior_can_average_source_and_destination_roles():
    trainer = EdgeHiNoSTrainer.__new__(EdgeHiNoSTrainer)
    trainer.model = _ZeroLogitModel()
    trainer.node_prior_t = torch.tensor([2, 0, 1])
    trainer.node_prior_logit_strength = 8.0
    trainer.node_prior_event_role = "mean_endpoints"
    trainer.K = 3
    src = torch.tensor([0])
    dst = torch.tensor([1])

    _, q, logits = trainer._forward_event_tensors(
        src,
        dst,
        torch.zeros((1, 1)),
        return_logits=True,
    )

    assert torch.allclose(logits, torch.tensor([[4.0, 0.0, 4.0]]))
    assert torch.allclose(q, torch.softmax(logits, dim=1))
