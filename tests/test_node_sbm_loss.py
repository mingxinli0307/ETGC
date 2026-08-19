import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import node_sbm_reconstruction_loss_global
from edge_main import build_parser
from edge_node_prior import build_adaptive_node_prior, temporal_block_validation_auc


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
