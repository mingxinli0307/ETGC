import inspect
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edge_losses import edge_matrix_ncut_loss_global, edge_trace_mincut_loss_global
from edge_proximity import build_edge_ncut_affinity, sparse_symmetry_error


def _toy_problem():
    raw_pi = sp.csr_matrix(
        np.array(
            [
                [0.8, 0.7, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1],
                [0.1, 0.6, 0.8, 0.0, 0.3, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.7, 0.9, 0.0, 0.1, 0.0, 0.0],
                [0.0, 0.4, 0.2, 0.5, 0.7, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.1, 0.9, 0.8, 0.0, 0.2],
                [0.1, 0.0, 0.0, 0.0, 0.4, 0.6, 0.9, 0.0],
                [0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.7, 0.8],
                [0.6, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.5],
            ],
            dtype=np.float32,
        )
    )
    pi_cut = build_edge_ncut_affinity(raw_pi, edge_ppr_topk=-1)
    degree = np.asarray(pi_cut.sum(axis=1)).ravel().astype(np.float32)
    logits = torch.tensor(
        [
            [2.0, 0.2, -0.4],
            [1.4, 0.5, -0.1],
            [0.4, 1.7, -0.3],
            [0.1, 1.3, 0.2],
            [-0.2, 0.5, 1.8],
            [0.0, 0.2, 1.5],
            [0.8, -0.1, 1.0],
            [1.1, 0.0, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    return pi_cut, degree, logits


def test_matrix_ncut_matches_dense_c_form_and_has_finite_gradient():
    pi_cut, degree, logits = _toy_problem()
    q = torch.softmax(logits, dim=1)
    eps = 1e-6

    pi_dense = torch.from_numpy(pi_cut.toarray())
    d_dense = torch.diag(torch.from_numpy(degree))
    a = q.t() @ d_dense @ q + eps * torch.eye(3)
    b = q.t() @ (d_dense - pi_dense) @ q
    expected = torch.trace(torch.linalg.solve(a, b))

    diagnostics = {}
    total, actual, _ = edge_matrix_ncut_loss_global(
        q,
        pi_cut,
        degree,
        K=3,
        lambda_orth=0.25,
        eps=eps,
        orth_type="orthqa",
        diagnostics=diagnostics,
    )

    assert abs(float(actual.detach() - expected.detach())) < 1e-5
    assert torch.isfinite(actual)
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
    assert sparse_symmetry_error(pi_cut) < 1e-7
    assert np.all(degree >= 0.0)
    assert diagnostics["matrix_ncut_cut_loss_finite"] is True

    # The production implementation applies D_Pi by row scaling and solves A X = B.
    source = inspect.getsource(edge_matrix_ncut_loss_global)
    assert "torch.inverse" not in source
    assert "torch.diag(" not in source
    assert ".to_dense(" not in source
    assert "torch.linalg.solve" in source


def test_matrix_ncut_is_not_legacy_scalar_trace_ratio():
    pi_cut, degree, logits = _toy_problem()
    q = torch.softmax(logits, dim=1)
    _, matrix_cut, _ = edge_matrix_ncut_loss_global(
        q, pi_cut, degree, K=3, lambda_orth=0.0, eps=1e-6, orth_type="orthqa"
    )
    _, legacy_cut, _ = edge_trace_mincut_loss_global(
        q, pi_cut, degree, K=3, lambda_orth=0.0, eps=1e-6, orth_type="orthqa"
    )
    assert abs(float(matrix_cut.detach() - legacy_cut.detach())) > 1e-3
