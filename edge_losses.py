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
    degree_np = np.asarray(degree, dtype=np.float32).reshape(-1)
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
        raise FloatingPointError(f"{name} is not finite in trace mincut loss: {details}")


def _sparse_block_wq_numerator(W_E: sp.csr_matrix, Q_all: torch.Tensor, row_block_size: int) -> torch.Tensor:
    W_E = W_E.tocsr()
    m = int(W_E.shape[0])
    block_size = max(1, int(row_block_size))
    numerator = Q_all.new_zeros(())
    for start in range(0, m, block_size):
        end = min(m, start + block_size)
        sub = W_E[start:end].tocoo()
        if sub.nnz == 0:
            continue
        indices = torch.stack(
            [
                torch.from_numpy(sub.row.astype(np.int64, copy=False)),
                torch.from_numpy(sub.col.astype(np.int64, copy=False)),
            ],
            dim=0,
        ).to(device=Q_all.device)
        values = torch.from_numpy(sub.data.astype(np.float32, copy=False)).to(
            device=Q_all.device, dtype=Q_all.dtype
        )
        W_block = torch.sparse_coo_tensor(
            indices,
            values,
            size=(end - start, m),
            device=Q_all.device,
            dtype=Q_all.dtype,
        ).coalesce()
        WQ_block = torch.sparse.mm(W_block, Q_all)
        numerator = numerator + (Q_all[start:end] * WQ_block).sum()
    return numerator


def _sparse_wq_product(W_E, Q_all: torch.Tensor, row_block_size: int = 65536) -> tuple:
    if isinstance(W_E, torch.Tensor):
        if W_E.shape[0] != W_E.shape[1] or int(W_E.shape[0]) != int(Q_all.size(0)):
            raise ValueError(f"W_E shape={tuple(W_E.shape)} does not match Q_all rows={Q_all.size(0)}")
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
        values = torch.from_numpy(sub.data.astype(np.float32, copy=False)).to(
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


def edge_matrix_ncut_loss_global(
    Q_all: torch.Tensor,
    W_E,
    degree,
    K: int,
    lambda_orth: float = 1.0,
    eps: float = 1e-8,
    row_block_size: int = 65536,
    orth_type: str = "orthqa",
) -> tuple:
    if Q_all.dim() != 2:
        raise ValueError(f"Q_all must be 2D, got shape={tuple(Q_all.shape)}")
    m = int(Q_all.size(0))
    k = int(Q_all.size(1))
    if k != int(K):
        raise ValueError(f"Q_all has K={k}, expected K={K}")
    WQ, nnz = _sparse_wq_product(W_E, Q_all, int(row_block_size))
    degree_t = _as_degree_tensor(degree, Q_all)
    if int(degree_t.numel()) != m:
        raise ValueError(f"degree length={degree_t.numel()} does not match Q_all rows={m}")
    DQ = degree_t.unsqueeze(1) * Q_all
    eye = torch.eye(k, dtype=Q_all.dtype, device=Q_all.device)
    A = Q_all.t().mm(DQ) + float(eps) * eye
    B = Q_all.t().mm(DQ - WQ)
    X = torch.linalg.solve(A, B)
    ncut_loss = torch.trace(X)
    selected_orth_type = str(orth_type).lower()
    if selected_orth_type == "orth":
        penalty_loss = trace_mincut_orthogonality_loss(Q_all, int(K), eps=float(eps))
    elif selected_orth_type == "orthqa":
        penalty_loss = edge_orthqa_penalty_global(Q_all, degree_t, eps=float(eps))
    else:
        raise ValueError(f"Unsupported orth_type: {orth_type}")
    total_cluster_loss = ncut_loss + float(lambda_orth) * penalty_loss
    stats = {
        "M": m,
        "K": k,
        "W_nnz": nnz,
        "degree_min": float(degree_t.detach().min().cpu()) if degree_t.numel() else 0.0,
        "degree_max": float(degree_t.detach().max().cpu()) if degree_t.numel() else 0.0,
        "A_min": float(A.detach().min().cpu()) if A.numel() else 0.0,
        "A_max": float(A.detach().max().cpu()) if A.numel() else 0.0,
        "B_min": float(B.detach().min().cpu()) if B.numel() else 0.0,
        "B_max": float(B.detach().max().cpu()) if B.numel() else 0.0,
        "ncut_loss": float(ncut_loss.detach().cpu()),
        "penalty_loss": float(penalty_loss.detach().cpu()),
        "total_cluster_loss": float(total_cluster_loss.detach().cpu()),
    }
    _check_scalar_finite("matrix_ncut loss", ncut_loss, stats)
    _check_scalar_finite("matrix_ncut penalty", penalty_loss, stats)
    _check_scalar_finite("matrix_ncut total_cluster_loss", total_cluster_loss, stats)
    return total_cluster_loss, ncut_loss, penalty_loss


def edge_trace_mincut_loss_global(
    Q_all: torch.Tensor,
    W_E,
    degree,
    K: int,
    lambda_orth: float = 1.0,
    eps: float = 1e-12,
    row_block_size: int = 65536,
    orth_type: str = "orth",
) -> tuple:
    """Global trace mincut over all temporal edge events.

    Computes O(nnz(W_E) K + M K^2) work without materializing dense W_E or D_E.
    The legacy sum-of-ratios normalized association loss is intentionally kept
    separate in edge_ncut_loss_global for compatibility and diagnostics.
    """
    if Q_all.dim() != 2:
        raise ValueError(f"Q_all must be 2D, got shape={tuple(Q_all.shape)}")
    m = int(Q_all.size(0))
    k = int(Q_all.size(1))
    if k != int(K):
        raise ValueError(f"Q_all has K={k}, expected K={K}")

    if isinstance(W_E, torch.Tensor):
        if W_E.shape[0] != W_E.shape[1] or int(W_E.shape[0]) != m:
            raise ValueError(f"W_E shape={tuple(W_E.shape)} does not match Q_all rows={m}")
        if W_E.layout not in (torch.sparse_coo, torch.sparse_csr):
            raise ValueError(f"W_E tensor must be sparse COO/CSR, got layout={W_E.layout}")
        W_sparse = W_E.to(device=Q_all.device, dtype=Q_all.dtype)
        WQ = torch.sparse.mm(W_sparse, Q_all)
        numerator = (Q_all * WQ).sum()
        nnz = int(W_sparse._nnz())
    else:
        W_csr = W_E.tocsr()
        if W_csr.shape[0] != W_csr.shape[1] or int(W_csr.shape[0]) != m:
            raise ValueError(f"W_E shape={W_csr.shape} does not match Q_all rows={m}")
        numerator = _sparse_block_wq_numerator(W_csr, Q_all, int(row_block_size))
        nnz = int(W_csr.nnz)

    degree_t = _as_degree_tensor(degree, Q_all)
    if int(degree_t.numel()) != m:
        raise ValueError(f"degree length={degree_t.numel()} does not match Q_all rows={m}")
    denominator = (degree_t.unsqueeze(1) * Q_all.square()).sum()
    denom_value = float(denominator.detach().cpu())
    stats = {
        "M": m,
        "K": int(K),
        "W_nnz": nnz,
        "numerator": float(numerator.detach().cpu()),
        "denominator": denom_value,
        "degree_min": float(degree_t.detach().min().cpu()) if degree_t.numel() else 0.0,
        "degree_max": float(degree_t.detach().max().cpu()) if degree_t.numel() else 0.0,
        "Q_min": float(Q_all.detach().min().cpu()) if Q_all.numel() else 0.0,
        "Q_max": float(Q_all.detach().max().cpu()) if Q_all.numel() else 0.0,
    }
    _check_scalar_finite("trace_mincut numerator", numerator, stats)
    _check_scalar_finite("trace_mincut denominator", denominator, stats)
    if denom_value <= 0.0:
        details = ", ".join(f"{key}={value}" for key, value in stats.items())
        raise FloatingPointError(f"trace_mincut denominator is not positive: {details}")

    cut_loss = -numerator / (denominator + float(eps))
    selected_orth_type = str(orth_type).lower()
    if selected_orth_type == "orth":
        orth_loss = trace_mincut_orthogonality_loss(Q_all, int(K), eps=float(eps))
    elif selected_orth_type == "orthqa":
        orth_loss = edge_orthqa_penalty_global(Q_all, degree_t, eps=float(eps))
    else:
        raise ValueError(f"Unsupported orth_type: {orth_type}")
    total_cluster_loss = cut_loss + float(lambda_orth) * orth_loss

    stats.update(
        {
            "cut_loss": float(cut_loss.detach().cpu()),
            "orth_loss": float(orth_loss.detach().cpu()),
            "orth_type": selected_orth_type,
            "total_cluster_loss": float(total_cluster_loss.detach().cpu()),
        }
    )
    _check_scalar_finite("trace_mincut cut_loss", cut_loss, stats)
    _check_scalar_finite("trace_mincut orth_loss", orth_loss, stats)
    _check_scalar_finite("trace_mincut total_cluster_loss", total_cluster_loss, stats)
    return total_cluster_loss, cut_loss, orth_loss


def trace_mincut_orthogonality_loss(Q_all: torch.Tensor, K: int, eps: float = 1e-12) -> torch.Tensor:
    QtQ = Q_all.t().mm(Q_all)
    QtQ_norm = torch.linalg.norm(QtQ, ord="fro").clamp_min(float(eps))
    QtQ_normalized = QtQ / QtQ_norm
    target = torch.eye(int(K), dtype=Q_all.dtype, device=Q_all.device) / (float(K) ** 0.5)
    return torch.linalg.norm(QtQ_normalized - target, ord="fro")


def edge_orthqa_penalty_global(Q_all: torch.Tensor, degree, eps: float = 1e-12) -> torch.Tensor:
    if Q_all.dim() != 2:
        raise ValueError(f"Q_all must be 2D, got shape={tuple(Q_all.shape)}")
    K = int(Q_all.size(1))
    if K <= 1:
        raise ValueError(f"edge_orthqa_penalty_global requires K > 1, got K={K}")
    degree_t = _as_degree_tensor(degree, Q_all)
    if int(degree_t.numel()) != int(Q_all.size(0)):
        raise ValueError(f"degree length={degree_t.numel()} does not match Q_all rows={Q_all.size(0)}")
    total_volume = degree_t.sum()
    stats = {
        "M": int(Q_all.size(0)),
        "K": K,
        "total_volume": float(total_volume.detach().cpu()),
        "degree_min": float(degree_t.detach().min().cpu()) if degree_t.numel() else 0.0,
        "degree_max": float(degree_t.detach().max().cpu()) if degree_t.numel() else 0.0,
        "Q_min": float(Q_all.detach().min().cpu()) if Q_all.numel() else 0.0,
        "Q_max": float(Q_all.detach().max().cpu()) if Q_all.numel() else 0.0,
    }
    _check_scalar_finite("orthqa total_volume", total_volume, stats)
    if float(total_volume.detach().cpu()) <= 0.0:
        details = ", ".join(f"{key}={value}" for key, value in stats.items())
        raise FloatingPointError(f"orthqa total_volume must be positive: {details}")
    weighted_square = degree_t.unsqueeze(1) * Q_all.square()
    cluster_volume_sqrt = torch.sqrt(weighted_square.sum(dim=0) + float(eps))
    normalized_sum = cluster_volume_sqrt.sum() / torch.sqrt(total_volume + float(eps))
    sqrt_k = float(K) ** 0.5
    orthqa_loss = (sqrt_k - normalized_sum) / (sqrt_k - 1.0)
    stats.update(
        {
            "normalized_sum": float(normalized_sum.detach().cpu()),
            "orthqa_loss": float(orthqa_loss.detach().cpu()),
        }
    )
    _check_scalar_finite("orthqa normalized_sum", normalized_sum, stats)
    _check_scalar_finite("orthqa loss", orthqa_loss, stats)
    return orthqa_loss


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
    neg = torch.as_tensor([local_index.get(int(e), int(rng.randint(0, len(local_index)))) for e in neg_global],
                          dtype=torch.long, device=device)

    mode = str(similarity_mode).lower()
    if mode == "event_dot":
        pos_score = (r_union[a] * r_union[p]).sum(dim=-1)
        neg_score = (r_union[a] * r_union[neg]).sum(dim=-1)
    elif mode == "cosine":
        pos_score = _cosine_pair(r_union[a], r_union[p], eps)
        neg_score = _cosine_pair(r_union[a], r_union[neg], eps)
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


def node_embedding_anchor_loss(node_emb: torch.Tensor, node_emb_initial: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(node_emb, node_emb_initial.to(device=node_emb.device, dtype=node_emb.dtype))


def edge_ncut_loss(Q_union: torch.Tensor, union_ids: np.ndarray, W_E: sp.csr_matrix, K: int) -> torch.Tensor:
    sub = W_E[union_ids][:, union_ids].tocoo()
    if sub.nnz == 0:
        return Q_union.sum() * 0.0
    device = Q_union.device
    row = torch.from_numpy(sub.row.astype(np.int64, copy=False)).to(device)
    col = torch.from_numpy(sub.col.astype(np.int64, copy=False)).to(device)
    val = torch.from_numpy(sub.data.astype(np.float32, copy=False)).to(device)
    q_i = Q_union.index_select(0, row)
    q_j = Q_union.index_select(0, col)
    assoc = (val.unsqueeze(1) * q_i * q_j).sum(dim=0)
    degree_np = np.asarray(sub.tocsr().sum(axis=1)).ravel().astype(np.float32)
    degree = torch.from_numpy(degree_np).to(device)
    vol = (degree.unsqueeze(1) * Q_union).sum(dim=0)
    return float(K) - (assoc / (vol + 1e-8)).sum()


def edge_ncut_loss_global(
    Q_all: torch.Tensor,
    W_E: sp.csr_matrix,
    K: int,
    row_block_size: int = 65536,
) -> torch.Tensor:
    W_E = W_E.tocsr()
    if W_E.shape[0] != W_E.shape[1]:
        raise ValueError(f"W_E must be square, got shape={W_E.shape}")
    if W_E.shape[0] != int(Q_all.size(0)):
        raise ValueError(f"Q_all rows={Q_all.size(0)} do not match W_E shape={W_E.shape}")
    if W_E.nnz == 0:
        return Q_all.sum() * 0.0

    device = Q_all.device
    dtype = Q_all.dtype
    m = int(W_E.shape[0])
    block_size = max(1, int(row_block_size))
    assoc = Q_all.new_zeros(int(K))

    for start in range(0, m, block_size):
        end = min(m, start + block_size)
        sub = W_E[start:end].tocoo()
        if sub.nnz == 0:
            continue
        row_np = sub.row.astype(np.int64, copy=False) + start
        col_np = sub.col.astype(np.int64, copy=False)
        val_np = sub.data.astype(np.float32, copy=False)
        row = torch.from_numpy(row_np).to(device=device)
        col = torch.from_numpy(col_np).to(device=device)
        val = torch.from_numpy(val_np).to(device=device, dtype=dtype)
        q_i = Q_all.index_select(0, row)
        q_j = Q_all.index_select(0, col)
        assoc = assoc + (val.unsqueeze(1) * q_i * q_j).sum(dim=0)

    degree_np = np.asarray(W_E.sum(axis=1)).ravel().astype(np.float32)
    degree = torch.from_numpy(degree_np).to(device=device, dtype=dtype)
    vol = (degree.unsqueeze(1) * Q_all).sum(dim=0)
    return float(K) - (assoc / (vol + 1e-8)).sum()


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


def balance_loss(Q: torch.Tensor, K: int) -> torch.Tensor:
    qtq = Q.t().mm(Q)
    qtq = qtq / torch.linalg.norm(qtq, ord="fro").clamp_min(1e-8)
    target = torch.eye(K, dtype=Q.dtype, device=Q.device) / (float(K) ** 0.5)
    return torch.linalg.norm(qtq - target, ord="fro")
