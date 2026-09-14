import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


warnings.filterwarnings(
    "ignore",
    message="Sparse invariant checks are implicitly disabled.*",
    category=UserWarning,
)
if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
    torch.sparse.check_sparse_tensor_invariants.disable()


def scipy_csr_to_torch_sparse_coo(
    mat: sp.spmatrix,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mat = mat.tocoo()
    indices = torch.stack(
        [
            torch.from_numpy(mat.row.astype(np.int64, copy=False)),
            torch.from_numpy(mat.col.astype(np.int64, copy=False)),
        ],
        dim=0,
    ).to(device=device)
    values = torch.from_numpy(mat.data.astype(np.float32, copy=False)).to(device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=mat.shape, device=device, dtype=dtype).coalesce()


def _as_degree_tensor(degree, Q_all: torch.Tensor) -> torch.Tensor:
    if isinstance(degree, torch.Tensor):
        return degree.to(device=Q_all.device, dtype=Q_all.dtype)
    degree_np = np.asarray(degree).reshape(-1)
    return torch.from_numpy(degree_np).to(device=Q_all.device, dtype=Q_all.dtype)


def matrix_ncut_qtdq_diagnostics(
    Q_all: torch.Tensor,
    degree,
    eps: float = 1e-8,
) -> dict:
    """Return read-only conditioning diagnostics for the matrix-Ncut solve matrix."""
    if Q_all.dim() != 2:
        raise ValueError(f"Q_all must be 2D, got shape={tuple(Q_all.shape)}")
    degree_t = _as_degree_tensor(degree, Q_all)
    if int(degree_t.numel()) != int(Q_all.size(0)):
        raise ValueError(
            f"degree length={degree_t.numel()} does not match Q_all rows={Q_all.size(0)}"
        )
    k = int(Q_all.size(1))
    q_diag = Q_all.detach().to(dtype=torch.float64)
    degree_diag = degree_t.detach().to(dtype=torch.float64)
    qtdq = q_diag.t().mm(degree_diag.unsqueeze(1) * q_diag)
    qtdq = 0.5 * (qtdq + qtdq.t())
    solve_matrix = qtdq + float(eps) * torch.eye(
        k,
        dtype=qtdq.dtype,
        device=Q_all.device,
    )
    eigvals = torch.linalg.eigvalsh(qtdq)
    regularized_eigvals = torch.linalg.eigvalsh(solve_matrix)
    finite = bool(torch.isfinite(eigvals).all())
    min_eig = float(eigvals.min().detach().cpu()) if eigvals.numel() else 0.0
    max_eig = float(eigvals.max().detach().cpu()) if eigvals.numel() else 0.0
    condition = max_eig / min_eig if finite and min_eig > 0.0 else float("inf")
    regularized_min = float(regularized_eigvals.min().detach().cpu()) if regularized_eigvals.numel() else 0.0
    regularized_max = float(regularized_eigvals.max().detach().cpu()) if regularized_eigvals.numel() else 0.0
    regularized_condition = (
        regularized_max / regularized_min
        if bool(torch.isfinite(regularized_eigvals).all()) and regularized_min > 0.0
        else float("inf")
    )
    return {
        "qtdq_min_eigenvalue": min_eig,
        "qtdq_max_eigenvalue": max_eig,
        "qtdq_condition_number": condition,
        "qtdq_regularized_condition_number": regularized_condition,
        "qtdq_eigenvalues_finite": finite,
        "qtdq_regularization_eps": float(eps),
    }


def _check_scalar_finite(name: str, value: torch.Tensor, stats: dict) -> None:
    if not torch.isfinite(value).all():
        details = ", ".join(f"{k}={v}" for k, v in stats.items())
        raise FloatingPointError(f"{name} is not finite: {details}")


def _sparse_wq_product(W_E, Q_all: torch.Tensor, row_block_size: int = 65536) -> tuple:
    if isinstance(W_E, torch.Tensor):
        if W_E.shape[0] != W_E.shape[1] or int(W_E.shape[0]) != int(Q_all.size(0)):
            raise ValueError(f"W_E shape={tuple(W_E.shape)} does not match Q_all rows={Q_all.size(0)}")
        if W_E.layout == torch.strided:
            W_dense = W_E.to(device=Q_all.device, dtype=Q_all.dtype)
            return W_dense @ Q_all, int(torch.count_nonzero(W_dense))
        if W_E.layout not in (torch.sparse_coo, torch.sparse_csr):
            raise ValueError(f"W_E tensor must be sparse COO/CSR, got layout={W_E.layout}")
        W_sparse = W_E.to(device=Q_all.device, dtype=Q_all.dtype)
        return torch.sparse.mm(W_sparse, Q_all), int(W_sparse._nnz())

    W_csr = W_E.tocsr()
    m = int(W_csr.shape[0])
    if W_csr.shape[0] != W_csr.shape[1] or m != int(Q_all.size(0)):
        raise ValueError(f"W_E shape={W_csr.shape} does not match Q_all rows={Q_all.size(0)}")
    block_size = max(1, int(row_block_size))
    chunks = []
    for start in range(0, m, block_size):
        end = min(m, start + block_size)
        sub = W_csr[start:end].tocoo()
        if sub.nnz == 0:
            chunks.append(Q_all.new_zeros((end - start, int(Q_all.size(1)))))
            continue
        indices = torch.stack(
            [
                torch.from_numpy(sub.row.astype(np.int64, copy=False)),
                torch.from_numpy(sub.col.astype(np.int64, copy=False)),
            ],
            dim=0,
        ).to(device=Q_all.device)
        values = torch.from_numpy(sub.data).to(
            device=Q_all.device, dtype=Q_all.dtype
        )
        W_block = torch.sparse_coo_tensor(
            indices,
            values,
            size=(end - start, m),
            device=Q_all.device,
            dtype=Q_all.dtype,
        ).coalesce()
        chunks.append(torch.sparse.mm(W_block, Q_all))
    return torch.cat(chunks, dim=0), int(W_csr.nnz)


def edge_expected_structural_score_gain_loss_global(
    Q_all: torch.Tensor,
    W_E,
    degree,
    tau: float = 0.5,
    eps: float = 1e-8,
    row_block_size: int = 65536,
) -> tuple:
    """Expected Structural Score Gain on the global sparse edge affinity."""
    if Q_all.dim() != 2:
        raise ValueError(f"Q_all must be 2D, got shape={tuple(Q_all.shape)}")
    m = int(Q_all.size(0))
    if m <= 0:
        raise ValueError("Q_all must contain at least one edge event")
    if float(tau) < 0.0:
        raise ValueError(f"tau must be nonnegative, got {tau}")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    WQ, _nnz = _sparse_wq_product(W_E, Q_all, int(row_block_size))
    degree_t = _as_degree_tensor(degree, Q_all)
    if int(degree_t.numel()) != m:
        raise ValueError(f"degree length={degree_t.numel()} does not match Q_all rows={m}")

    # Apply P = D^{-1} W_E without materializing P or a dense W_E.
    R = WQ / degree_t.clamp_min(float(eps)).unsqueeze(1)
    cluster_mass = Q_all.sum(dim=0)
    actual = (Q_all * R).sum(dim=0)
    expected = (cluster_mass / float(m)) * R.sum(dim=0)
    gain = (actual - expected) / (cluster_mass + float(eps)).pow(float(tau))
    esg_loss = -gain.mean()
    stats = {
        "mean_gain": float(gain.detach().mean().cpu()),
        "min_gain": float(gain.detach().min().cpu()),
        "max_gain": float(gain.detach().max().cpu()),
    }
    _check_scalar_finite("expected structural score gain loss", esg_loss, stats)
    return esg_loss, stats


def cform_ncut_loss(
    Q: torch.Tensor,
    Pi,
    degree,
    eps: float = 1e-8,
    row_block_size: int = 65536,
    pi_q: torch.Tensor = None,
) -> tuple:
    """Pure C-form Tr[solve(Q.T D Q + eps I, Q.T (D-Pi) Q)].

    Accumulate in float64 so eps survives near-rank-one soft assignments.
    Casts preserve autograd. Optional pi_q reuses the affinity product.
    No regularizer or objective coefficient is included here.
    """
    if Q.ndim != 2 or min(Q.shape) <= 0 or eps <= 0:
        raise ValueError("C-form requires nonempty 2D Q and positive eps")
    Q = Q.to(torch.float64)
    degree = _as_degree_tensor(degree, Q)
    if degree.shape != (Q.shape[0],):
        raise ValueError("Degree must contain one row sum per assignment")
    if pi_q is None:
        pi_q, _ = _sparse_wq_product(Pi, Q, row_block_size)
    if pi_q.shape != Q.shape:
        raise ValueError("PiQ must have the same shape as Q")
    DQ = degree[:, None] * Q
    A = Q.T @ DQ + eps * torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype)
    B = Q.T @ (DQ - pi_q)
    ncut_loss = torch.trace(torch.linalg.solve(A, B))
    if not torch.isfinite(ncut_loss):
        raise FloatingPointError("C-form Ncut is not finite")
    return ncut_loss, {"solve_dtype": str(A.dtype), "eps": float(eps)}


def hierarchical_ncut_terms(q1, p1, q2, Pi_E, degree_E, eps=1e-8, row_block_size=65536):
    """Two C-forms and the exact differentiable coarse graph; no loss weights."""
    p1 = p1.to(torch.float64)
    q2 = q2.to(torch.float64)
    degree_E = _as_degree_tensor(degree_E, p1)
    PiP1, _ = _sparse_wq_product(Pi_E, p1, row_block_size)
    fine_ncut_loss, _ = cform_ncut_loss(p1, Pi_E, degree_E, eps, row_block_size, pi_q=PiP1)
    Pi_H = p1.T @ PiP1
    degree_H = Pi_H.sum(dim=1)
    coarse_ncut_loss, _ = cform_ncut_loss(q2, Pi_H, degree_H, eps)
    # Read-only diagnostics; neither C_H nor Delta_H enters either loss.
    with torch.no_grad():
        C_H = p1.T @ (degree_E[:, None] * p1)
        Delta_H = torch.diag(degree_H) - C_H
        volume = degree_E.sum().clamp_min(eps)
        q1_softness = (degree_E * (1 - q1.to(torch.float64).square().sum(1))).sum() / volume
        p1_softness = (degree_E * (1 - p1.square().sum(1))).sum() / volume
        diagnostics = dict(q1_softness=float(q1_softness.cpu()),
                           p1_softness=float(p1_softness.cpu()),
                           coarse_discrepancy_trace=float(torch.trace(Delta_H).cpu()))
    return fine_ncut_loss, coarse_ncut_loss, Pi_H, degree_H, diagnostics


def trace_mincut_orthogonality_loss(Q_all: torch.Tensor, K: int, eps: float = 1e-12) -> torch.Tensor:
    QtQ = Q_all.t().mm(Q_all)
    QtQ_norm = torch.linalg.norm(QtQ, ord="fro").clamp_min(float(eps))
    QtQ_normalized = QtQ / QtQ_norm
    target = torch.eye(int(K), dtype=Q_all.dtype, device=Q_all.device) / (float(K) ** 0.5)
    return torch.linalg.norm(QtQ_normalized - target, ord="fro")


def edge_ppr_proximity_loss_preindexed(
    r_union: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    weights: torch.Tensor,
    similarity_mode: str = "cosine",
    node_emb: torch.Tensor = None,
    src_union: torch.Tensor = None,
    dst_union: torch.Tensor = None,
    time_feat_union: torch.Tensor = None,
    prox_role_ss_weight: float = 0.25,
    prox_role_dd_weight: float = 0.25,
    prox_role_ds_weight: float = 1.0,
    prox_role_sd_weight: float = 0.0,
    prox_role_time_weight: float = 0.25,
    prox_temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Evaluate the existing proximity objective from prepared local pairs."""
    if anchors.numel() == 0:
        return r_union.sum() * 0.0

    a = anchors.long()
    p = positives.long()
    neg = negatives.long()
    w = weights.to(dtype=torch.float32)
    if not (a.device == p.device == neg.device == w.device == r_union.device):
        raise ValueError("Pre-indexed proximity tensors must be on the same device as r_union")
    if not (a.numel() == p.numel() == neg.numel() == w.numel()):
        raise ValueError("Pre-indexed proximity pair tensors must have equal lengths")

    mode = str(similarity_mode).lower()
    if mode == "event_dot":
        pos_score = (r_union[a] * r_union[p]).sum(dim=-1)
        neg_score = (r_union[a] * r_union[neg]).sum(dim=-1)
    elif mode == "cosine":
        r_norm = torch.linalg.norm(r_union, dim=-1)
        pos_score = (r_union[a] * r_union[p]).sum(dim=-1) / (
            r_norm[a] * r_norm[p] + float(eps)
        )
        neg_score = (r_union[a] * r_union[neg]).sum(dim=-1) / (
            r_norm[a] * r_norm[neg] + float(eps)
        )
    elif mode == "role_aware":
        pos_score = role_aware_event_scores(
            a,
            p,
            node_emb=node_emb,
            src_union=src_union,
            dst_union=dst_union,
            time_feat_union=time_feat_union,
            prox_role_ss_weight=prox_role_ss_weight,
            prox_role_dd_weight=prox_role_dd_weight,
            prox_role_ds_weight=prox_role_ds_weight,
            prox_role_sd_weight=prox_role_sd_weight,
            prox_role_time_weight=prox_role_time_weight,
            prox_temperature=prox_temperature,
            eps=eps,
        )
        neg_score = role_aware_event_scores(
            a,
            neg,
            node_emb=node_emb,
            src_union=src_union,
            dst_union=dst_union,
            time_feat_union=time_feat_union,
            prox_role_ss_weight=prox_role_ss_weight,
            prox_role_dd_weight=prox_role_dd_weight,
            prox_role_ds_weight=prox_role_ds_weight,
            prox_role_sd_weight=prox_role_sd_weight,
            prox_role_time_weight=prox_role_time_weight,
            prox_temperature=prox_temperature,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported prox_similarity_mode: {similarity_mode}")
    return (w * F.softplus(-pos_score)).mean() + F.softplus(neg_score).mean()


def edge_ppr_proximity_loss(
    r_union: torch.Tensor,
    local_index: dict,
    batch_ids: np.ndarray,
    Pi_E: sp.csr_matrix,
    num_events: int,
    rng: np.random.RandomState,
    device: torch.device,
    similarity_mode: str = "cosine",
    node_emb: torch.Tensor = None,
    src_union: torch.Tensor = None,
    dst_union: torch.Tensor = None,
    time_feat_union: torch.Tensor = None,
    prox_role_ss_weight: float = 0.25,
    prox_role_dd_weight: float = 0.25,
    prox_role_ds_weight: float = 1.0,
    prox_role_sd_weight: float = 0.0,
    prox_role_time_weight: float = 0.25,
    prox_temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compatibility wrapper retaining the historical Python pair builder."""
    anchors, positives, weights = [], [], []
    for eid in batch_ids.tolist():
        start, end = Pi_E.indptr[eid], Pi_E.indptr[eid + 1]
        for nbr, weight in zip(Pi_E.indices[start:end], Pi_E.data[start:end]):
            if int(nbr) == int(eid) or int(nbr) not in local_index:
                continue
            anchors.append(local_index[int(eid)])
            positives.append(local_index[int(nbr)])
            weights.append(float(weight))
    if not anchors:
        return r_union.sum() * 0.0

    a = torch.as_tensor(anchors, dtype=torch.long, device=device)
    p = torch.as_tensor(positives, dtype=torch.long, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    neg_global = rng.randint(0, num_events, size=len(anchors))
    neg = torch.as_tensor(
        [local_index.get(int(e), int(rng.randint(0, len(local_index)))) for e in neg_global],
        dtype=torch.long,
        device=device,
    )
    return edge_ppr_proximity_loss_preindexed(
        r_union,
        a,
        p,
        neg,
        w,
        similarity_mode=similarity_mode,
        node_emb=node_emb,
        src_union=src_union,
        dst_union=dst_union,
        time_feat_union=time_feat_union,
        prox_role_ss_weight=prox_role_ss_weight,
        prox_role_dd_weight=prox_role_dd_weight,
        prox_role_ds_weight=prox_role_ds_weight,
        prox_role_sd_weight=prox_role_sd_weight,
        prox_role_time_weight=prox_role_time_weight,
        prox_temperature=prox_temperature,
        eps=eps,
    )


def _cosine_pair(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    return (x * y).sum(dim=-1) / (
        torch.linalg.norm(x, dim=-1) * torch.linalg.norm(y, dim=-1) + float(eps)
    )


def role_aware_event_scores(
    anchor_local: torch.Tensor,
    other_local: torch.Tensor,
    node_emb: torch.Tensor,
    src_union: torch.Tensor,
    dst_union: torch.Tensor,
    time_feat_union: torch.Tensor,
    prox_role_ss_weight: float = 0.25,
    prox_role_dd_weight: float = 0.25,
    prox_role_ds_weight: float = 1.0,
    prox_role_sd_weight: float = 0.0,
    prox_role_time_weight: float = 0.25,
    prox_temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    if node_emb is None or src_union is None or dst_union is None or time_feat_union is None:
        raise ValueError("role_aware proximity requires node_emb, src_union, dst_union, and time_feat_union")
    if float(prox_temperature) <= 0.0:
        raise ValueError(f"prox_temperature must be positive, got {prox_temperature}")

    anchor_local = anchor_local.long()
    other_local = other_local.long()
    src_union = src_union.long()
    dst_union = dst_union.long()

    s_i = node_emb.index_select(0, src_union.index_select(0, anchor_local))
    d_i = node_emb.index_select(0, dst_union.index_select(0, anchor_local))
    s_j = node_emb.index_select(0, src_union.index_select(0, other_local))
    d_j = node_emb.index_select(0, dst_union.index_select(0, other_local))
    t_i = time_feat_union.index_select(0, anchor_local)
    t_j = time_feat_union.index_select(0, other_local)

    w_ss = float(prox_role_ss_weight)
    w_dd = float(prox_role_dd_weight)
    w_ds = float(prox_role_ds_weight)
    w_sd = float(prox_role_sd_weight)
    w_t = float(prox_role_time_weight)
    weight_sum = max(w_ss + w_dd + w_ds + w_sd + w_t, float(eps))
    score_raw = (
        w_ss * _cosine_pair(s_i, s_j, eps)
        + w_dd * _cosine_pair(d_i, d_j, eps)
        + w_ds * _cosine_pair(d_i, s_j, eps)
        + w_sd * _cosine_pair(s_i, d_j, eps)
        + w_t * _cosine_pair(t_i, t_j, eps)
    )
    return (score_raw / weight_sum) / float(prox_temperature)


def projection_loss(Q_batch: torch.Tensor, src_batch: torch.Tensor, dst_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
    K = Q_batch.size(1)
    S = torch.zeros((num_nodes, K), dtype=Q_batch.dtype, device=Q_batch.device)
    S.index_add_(0, src_batch.long(), Q_batch)
    S.index_add_(0, dst_batch.long(), Q_batch)
    S = S / S.sum(dim=1, keepdim=True).clamp_min(1e-8)
    su = S.index_select(0, src_batch.long())
    sv = S.index_select(0, dst_batch.long())
    return -(Q_batch * torch.log(su * sv + 1e-8)).sum(dim=1).mean()


def project_edge_assignments_to_nodes_global(
    Q_all: torch.Tensor,
    src_all: torch.Tensor,
    dst_all: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    K = Q_all.size(1)
    S_raw = torch.zeros((int(num_nodes), K), dtype=Q_all.dtype, device=Q_all.device)
    S_raw.index_add_(0, src_all.long(), Q_all)
    S_raw.index_add_(0, dst_all.long(), Q_all)
    return S_raw / S_raw.sum(dim=1, keepdim=True).clamp_min(float(eps))


def projection_loss_global(
    Q_all: torch.Tensor,
    src_all: torch.Tensor,
    dst_all: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    S_all = project_edge_assignments_to_nodes_global(Q_all, src_all, dst_all, num_nodes, eps)
    su = S_all.index_select(0, src_all.long())
    sv = S_all.index_select(0, dst_all.long())
    return -(Q_all * torch.log(su * sv + float(eps))).sum(dim=1).mean()
