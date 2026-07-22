import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def edge_ppr_proximity_loss(
    r_union: torch.Tensor,
    local_index: dict,
    batch_ids: np.ndarray,
    Pi_E: sp.csr_matrix,
    num_events: int,
    rng: np.random.RandomState,
    device: torch.device,
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

    pos_score = (r_union[a] * r_union[p]).sum(dim=-1)
    neg_score = (r_union[a] * r_union[neg]).sum(dim=-1)
    return (w * F.softplus(-pos_score)).mean() + F.softplus(neg_score).mean()


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


def projection_loss(Q_batch: torch.Tensor, src_batch: torch.Tensor, dst_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
    K = Q_batch.size(1)
    S = torch.zeros((num_nodes, K), dtype=Q_batch.dtype, device=Q_batch.device)
    S.index_add_(0, src_batch.long(), Q_batch)
    S.index_add_(0, dst_batch.long(), Q_batch)
    S = S / S.sum(dim=1, keepdim=True).clamp_min(1e-8)
    su = S.index_select(0, src_batch.long())
    sv = S.index_select(0, dst_batch.long())
    return -(Q_batch * torch.log(su * sv + 1e-8)).sum(dim=1).mean()


def balance_loss(Q: torch.Tensor, K: int) -> torch.Tensor:
    qtq = Q.t().mm(Q)
    qtq = qtq / torch.linalg.norm(qtq, ord="fro").clamp_min(1e-8)
    target = torch.eye(K, dtype=Q.dtype, device=Q.device) / (float(K) ** 0.5)
    return torch.linalg.norm(qtq - target, ord="fro")
