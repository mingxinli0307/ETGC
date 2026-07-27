import csv
import json
import math
import os
from typing import Iterable, Optional

import torch

from edge_losses import project_edge_assignments_to_nodes_global
from edge_metrics import evaluate_node_clustering
from edge_model import feature_common_variation_statistics


EPS = 1e-12
LOGITS_STD_UNBIASED = False


def _float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _nullable_float(value):
    if value is None:
        return None
    return _float(value)


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return _float(torch.quantile(values.detach().float(), float(q)))


def _ratios_from_counts(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum().clamp_min(1)
    return counts.to(dtype=torch.float64) / total.to(dtype=torch.float64)


def hard_cluster_statistics_from_labels(labels: torch.Tensor, K: int, prefix: str) -> dict:
    labels = labels.detach().long().reshape(-1)
    K = int(K)
    counts = torch.bincount(labels, minlength=K)[:K].to(dtype=torch.long)
    ratios = _ratios_from_counts(counts)
    active_mask = counts > 0
    nonempty_ratios = ratios[active_mask]
    total = int(labels.numel())
    ratio_sum = _float(ratios.sum()) if ratios.numel() else 0.0
    if int(counts.sum().item()) != total:
        raise ValueError(f"{prefix} hard count sum mismatch: count_sum={int(counts.sum())} total={total}")
    if total > 0 and abs(ratio_sum - 1.0) >= 1e-6:
        raise ValueError(f"{prefix} hard ratio sum mismatch: ratio_sum={ratio_sum}")
    return {
        f"{prefix}_hard_counts": [int(x) for x in counts.detach().cpu().tolist()],
        f"{prefix}_hard_ratios": [float(x) for x in ratios.detach().cpu().tolist()],
        f"num_active_{prefix}_clusters": int(active_mask.sum().item()),
        f"largest_{prefix}_cluster_ratio": _float(ratios.max()) if ratios.numel() else 0.0,
        f"smallest_nonempty_{prefix}_cluster_ratio": _float(nonempty_ratios.min()) if nonempty_ratios.numel() else 0.0,
        f"num_empty_hard_{prefix}_clusters": int((counts == 0).sum().item()),
    }


def hard_cluster_statistics(Q: torch.Tensor, K: int, prefix: str) -> dict:
    return hard_cluster_statistics_from_labels(torch.argmax(Q.detach(), dim=1), K, prefix)


def q_margin_statistics(Q: torch.Tensor) -> dict:
    Q = Q.detach()
    if Q.size(1) < 2:
        margins = torch.zeros((Q.size(0),), dtype=Q.dtype, device=Q.device)
    else:
        top2 = torch.topk(Q, k=2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
    stats = {
        "q_margin_mean": _float(margins.mean()) if margins.numel() else 0.0,
        "q_margin_std": _float(margins.std(unbiased=False)) if margins.numel() else 0.0,
        "q_margin_min": _float(margins.min()) if margins.numel() else 0.0,
        "q_margin_max": _float(margins.max()) if margins.numel() else 0.0,
        "q_margin_p25": _quantile(margins, 0.25),
        "q_margin_p50": _quantile(margins, 0.50),
        "q_margin_p75": _quantile(margins, 0.75),
        "q_margin_p90": _quantile(margins, 0.90),
        "q_margin_p95": _quantile(margins, 0.95),
        "q_margin_p99": _quantile(margins, 0.99),
        "q_margin_lt_1e_6_ratio": _float((margins < 1e-6).float().mean()) if margins.numel() else 0.0,
        "q_margin_lt_1e_5_ratio": _float((margins < 1e-5).float().mean()) if margins.numel() else 0.0,
        "q_margin_lt_1e_4_ratio": _float((margins < 1e-4).float().mean()) if margins.numel() else 0.0,
        "q_margin_lt_1e_3_ratio": _float((margins < 1e-3).float().mean()) if margins.numel() else 0.0,
        "q_margin_lt_1e_2_ratio": _float((margins < 1e-2).float().mean()) if margins.numel() else 0.0,
    }
    stats["q_margin_lt_1e6_ratio"] = stats["q_margin_lt_1e_6_ratio"]
    stats["q_margin_lt_1e5_ratio"] = stats["q_margin_lt_1e_5_ratio"]
    stats["q_margin_lt_1e4_ratio"] = stats["q_margin_lt_1e_4_ratio"]
    stats["q_margin_lt_1e3_ratio"] = stats["q_margin_lt_1e_3_ratio"]
    stats["q_margin_lt_1e2_ratio"] = stats["q_margin_lt_1e_2_ratio"]
    return stats


def logits_statistics(logits: torch.Tensor) -> dict:
    logits = logits.detach()
    if logits.numel() == 0:
        return {
            "logits_std_unbiased": LOGITS_STD_UNBIASED,
            "logits_global_std": 0.0,
            "logits_row_std_mean": 0.0,
            "logits_row_std_median": 0.0,
            "logits_row_std_min": 0.0,
            "logits_row_std_max": 0.0,
            "logits_cluster_mean_std": 0.0,
            "logits_cluster_std_mean": 0.0,
            "logits_within_cluster_event_std": 0.0,
            "logits_bias_to_event_variation_ratio": 0.0,
            "logits_abs_max": 0.0,
            "logits_mean": 0.0,
        }
    row_std = logits.std(dim=1, unbiased=LOGITS_STD_UNBIASED)
    cluster_mean = logits.mean(dim=0)
    cluster_std = logits.std(dim=0, unbiased=LOGITS_STD_UNBIASED)
    mean_std = _float(cluster_mean.std(unbiased=LOGITS_STD_UNBIASED))
    within_std = _float(cluster_std.mean()) if cluster_std.numel() else 0.0
    return {
        "logits_std_unbiased": LOGITS_STD_UNBIASED,
        "logits_global_std": _float(logits.std(unbiased=LOGITS_STD_UNBIASED)),
        "logits_row_std_mean": _float(row_std.mean()) if row_std.numel() else 0.0,
        "logits_row_std_median": _float(torch.median(row_std)) if row_std.numel() else 0.0,
        "logits_row_std_min": _float(row_std.min()) if row_std.numel() else 0.0,
        "logits_row_std_max": _float(row_std.max()) if row_std.numel() else 0.0,
        "logits_cluster_mean_std": mean_std,
        "logits_cluster_std_mean": within_std,
        "logits_within_cluster_event_std": within_std,
        "logits_bias_to_event_variation_ratio": mean_std / (within_std + EPS),
        "logits_abs_max": _float(logits.abs().max()),
        "logits_mean": _float(logits.mean()),
    }


def q_uniform_distance_statistics(Q: torch.Tensor, eps: float = EPS) -> dict:
    Q = Q.detach()
    M = max(1, int(Q.size(0)))
    K = max(1, int(Q.size(1)))
    uniform_value = 1.0 / float(K)
    diff = Q - uniform_value
    row_l2 = torch.linalg.norm(diff, dim=1)
    row_l1 = diff.abs().sum(dim=1)
    entropy = -(Q * torch.log(Q.clamp_min(float(eps)))).sum(dim=1)
    entropy_mean = _float(entropy.mean()) if entropy.numel() else 0.0
    entropy_max = float(math.log(K))
    kl = (Q * (torch.log(Q.clamp_min(float(eps))) + entropy_max)).sum(dim=1)
    return {
        "q_uniform_l2_mean": _float(row_l2.mean()) if row_l2.numel() else 0.0,
        "q_uniform_fro_normalized": _float(torch.linalg.norm(diff, ord="fro") / (float(M) ** 0.5)),
        "q_uniform_l1_mean": _float(row_l1.mean()) if row_l1.numel() else 0.0,
        "q_uniform_max_abs": _float(diff.abs().max()) if diff.numel() else 0.0,
        "q_uniform_kl_mean": _float(kl.mean()) if kl.numel() else 0.0,
        "q_entropy_mean": entropy_mean,
        "q_entropy_max": entropy_max,
        "q_entropy_gap": entropy_max - entropy_mean,
    }


def q_rank_statistics(Q: torch.Tensor, eps: float = EPS) -> dict:
    Q_detached = Q.detach().to(dtype=torch.float64)
    K = int(Q_detached.size(1)) if Q_detached.dim() == 2 else 0
    if Q_detached.numel() == 0 or K == 0:
        return {
            "q_rank1_energy_ratio": 0.0,
            "q_second_energy_ratio": 0.0,
            "q_effective_rank": 0.0,
            "q_numerical_rank": 0,
            "q_centered_energy": 0.0,
            "q_centered_effective_rank": 0.0,
            "q_centered_numerical_rank": 0,
        }
    eigvals = torch.linalg.eigvalsh(Q_detached.t().mm(Q_detached)).clamp_min(0.0)
    eigvals = torch.flip(eigvals, dims=[0])
    total = eigvals.sum().clamp_min(float(eps))
    probs = eigvals / total
    effective_rank = torch.exp(-(probs * torch.log(probs.clamp_min(float(eps)))).sum())
    second = eigvals[1] / total if K > 1 else eigvals.new_zeros(())
    numerical_rank = int((eigvals > (1e-6 * eigvals[0].clamp_min(float(eps)))).sum().item())

    centered = Q_detached - Q_detached.mean(dim=0, keepdim=True)
    centered_eig = torch.linalg.eigvalsh(centered.t().mm(centered)).clamp_min(0.0)
    centered_eig = torch.flip(centered_eig, dims=[0])
    centered_energy = centered_eig.sum()
    if float(centered_energy.cpu()) <= eps:
        centered_effective_rank = 0.0
        centered_numerical_rank = 0
    else:
        centered_probs = centered_eig / centered_energy.clamp_min(float(eps))
        centered_effective_rank = _float(
            torch.exp(-(centered_probs * torch.log(centered_probs.clamp_min(float(eps)))).sum())
        )
        centered_numerical_rank = int(
            (centered_eig > (1e-6 * centered_eig[0].clamp_min(float(eps)))).sum().item()
        )
    return {
        "q_rank1_energy_ratio": _float(eigvals[0] / total),
        "q_second_energy_ratio": _float(second),
        "q_effective_rank": _float(effective_rank),
        "q_numerical_rank": numerical_rank,
        "q_centered_energy": _float(centered_energy),
        "q_centered_effective_rank": centered_effective_rank,
        "q_centered_numerical_rank": centered_numerical_rank,
    }


def prefixed_feature_statistics(features: torch.Tensor, suffix: str) -> dict:
    stats = feature_common_variation_statistics(features)
    return {f"{key}_{suffix}": value for key, value in stats.items()}


def cluster_volume_statistics(Q: torch.Tensor, degree: torch.Tensor, eps: float = EPS) -> dict:
    Q = Q.detach()
    degree = degree.detach().to(device=Q.device, dtype=Q.dtype).reshape(-1)
    if degree.numel() != Q.size(0):
        raise ValueError(f"degree length={degree.numel()} does not match Q rows={Q.size(0)}")
    volume = (degree.unsqueeze(1) * Q).sum(dim=0)
    volume_ratio = volume / volume.sum().clamp_min(float(eps))
    hard = torch.argmax(Q, dim=1)
    hard_volume = torch.zeros((Q.size(1),), dtype=Q.dtype, device=Q.device)
    hard_volume.index_add_(0, hard.long(), degree)
    hard_ratio = hard_volume / hard_volume.sum().clamp_min(float(eps))
    top2 = torch.topk(Q, k=min(2, Q.size(1)), dim=1).values
    top1 = top2[:, 0] if top2.numel() else Q.new_zeros((Q.size(0),))
    top2_value = top2[:, 1] if Q.size(1) > 1 else Q.new_zeros((Q.size(0),))
    margin = top1 - top2_value
    entropy = -(Q * torch.log(Q.clamp_min(float(eps)))).sum(dim=1)
    return {
        "cluster_volume_min_ratio": _float(volume_ratio.min()) if volume_ratio.numel() else 0.0,
        "cluster_volume_max_ratio": _float(volume_ratio.max()) if volume_ratio.numel() else 0.0,
        "cluster_volume_std": _float(volume_ratio.std(unbiased=False)) if volume_ratio.numel() else 0.0,
        "cluster_volume_coefficient_of_variation": _float(
            volume_ratio.std(unbiased=False) / volume_ratio.mean().clamp_min(float(eps))
        ) if volume_ratio.numel() else 0.0,
        "hard_cluster_volume_min_ratio": _float(hard_ratio.min()) if hard_ratio.numel() else 0.0,
        "hard_cluster_volume_max_ratio": _float(hard_ratio.max()) if hard_ratio.numel() else 0.0,
        "hard_cluster_volume_cv": _float(
            hard_ratio.std(unbiased=False) / hard_ratio.mean().clamp_min(float(eps))
        ) if hard_ratio.numel() else 0.0,
        "mean_assignment_entropy": _float(entropy.mean()) if entropy.numel() else 0.0,
        "mean_top1_probability": _float(top1.mean()) if top1.numel() else 0.0,
        "mean_top2_probability": _float(top2_value.mean()) if top2_value.numel() else 0.0,
        "mean_top1_top2_margin": _float(margin.mean()) if margin.numel() else 0.0,
    }


def node_embedding_drift_statistics(current: torch.Tensor, initial: torch.Tensor, eps: float = EPS) -> dict:
    cur = current.detach()
    init = initial.detach().to(device=cur.device, dtype=cur.dtype)
    if cur.shape != init.shape:
        raise ValueError(f"node embedding shape mismatch: current={tuple(cur.shape)} initial={tuple(init.shape)}")
    norms = torch.linalg.norm(cur, dim=1)
    drift = cur - init
    drift_fro = torch.linalg.norm(drift, ord="fro")
    init_fro = torch.linalg.norm(init, ord="fro").clamp_min(float(eps))
    cosine = torch.nn.functional.cosine_similarity(cur, init, dim=1, eps=float(eps))
    return {
        "node_embedding_norm_mean": _float(norms.mean()) if norms.numel() else 0.0,
        "node_embedding_norm_std": _float(norms.std(unbiased=False)) if norms.numel() else 0.0,
        "node_embedding_drift_fro_normalized": _float(drift_fro / (max(1, int(cur.size(0))) ** 0.5)),
        "node_embedding_relative_drift": _float(drift_fro / init_fro),
        "node_embedding_cosine_to_initial_mean": _float(cosine.mean()) if cosine.numel() else 0.0,
    }


def edge_repr_block_norm_statistics(
    edge_repr: torch.Tensor,
    node_dim: int,
    time_dim: int,
    edge_encoder_mode: str,
    raw_time_feat: Optional[torch.Tensor] = None,
    direct_time_scale: float = 1.0,
) -> dict:
    if str(edge_encoder_mode).lower() != "direct_node_time":
        return {
            "source_block_norm_mean": "",
            "destination_block_norm_mean": "",
            "time_block_norm_mean": "",
            "raw_time_block_norm_mean": "",
            "scaled_time_block_norm_mean": "",
            "scaled_time_to_node_ratio": "",
        }
    node_dim = int(node_dim)
    time_dim = int(time_dim)
    expected = 2 * node_dim + time_dim
    if edge_repr.dim() != 2 or int(edge_repr.size(1)) != expected:
        raise ValueError(
            f"direct_node_time edge representation must have dim={expected}, got shape={tuple(edge_repr.shape)}"
        )
    src_block = edge_repr[:, :node_dim]
    dst_block = edge_repr[:, node_dim : 2 * node_dim]
    time_block = edge_repr[:, 2 * node_dim :]
    source_norm = torch.linalg.norm(src_block, dim=1).mean() if src_block.numel() else edge_repr.new_zeros(())
    destination_norm = torch.linalg.norm(dst_block, dim=1).mean() if dst_block.numel() else edge_repr.new_zeros(())
    scaled_time_norm = torch.linalg.norm(time_block, dim=1).mean() if time_block.numel() else edge_repr.new_zeros(())
    if raw_time_feat is not None:
        raw_time = raw_time_feat.detach().to(device=edge_repr.device, dtype=edge_repr.dtype)
        if raw_time.shape[0] != edge_repr.shape[0]:
            raise ValueError(
                f"raw_time_feat rows={raw_time.shape[0]} do not match edge_repr rows={edge_repr.shape[0]}"
            )
        raw_time_norm = torch.linalg.norm(raw_time, dim=1).mean() if raw_time.numel() else edge_repr.new_zeros(())
    else:
        scale = float(direct_time_scale)
        raw_time_norm = scaled_time_norm / scale if abs(scale) > EPS else edge_repr.new_zeros(())
    node_norm = 0.5 * (source_norm + destination_norm)
    return {
        "source_block_norm_mean": _float(source_norm),
        "destination_block_norm_mean": _float(destination_norm),
        "time_block_norm_mean": _float(scaled_time_norm),
        "raw_time_block_norm_mean": _float(raw_time_norm),
        "scaled_time_block_norm_mean": _float(scaled_time_norm),
        "scaled_time_to_node_ratio": _float(scaled_time_norm / node_norm.clamp_min(float(EPS))),
    }


def node_event_degree_drift_statistics(
    src,
    dst,
    num_nodes: int,
    current: torch.Tensor,
    initial: torch.Tensor,
    eps: float = EPS,
) -> dict:
    cur = current.detach().cpu()
    init = initial.detach().cpu().to(dtype=cur.dtype)
    if cur.shape != init.shape:
        raise ValueError(f"node embedding shape mismatch: current={tuple(cur.shape)} initial={tuple(init.shape)}")
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    degree = torch.zeros((int(num_nodes),), dtype=torch.float64)
    one_src = torch.ones_like(src_t, dtype=torch.float64)
    one_dst = torch.ones_like(dst_t, dtype=torch.float64)
    degree.index_add_(0, src_t, one_src)
    degree.index_add_(0, dst_t, one_dst)
    drift = torch.linalg.norm(cur - init, dim=1).to(dtype=torch.float64)

    def _median(x: torch.Tensor) -> float:
        return _float(torch.median(x)) if x.numel() else 0.0

    def _rankdata(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x)
        ranks = torch.empty_like(x, dtype=torch.float64)
        n = int(x.numel())
        i = 0
        while i < n:
            j = i + 1
            while j < n and x[order[j]] == x[order[i]]:
                j += 1
            rank = 0.5 * float(i + j - 1)
            ranks[order[i:j]] = rank
            i = j
        return ranks

    if int(degree.numel()) > 1 and float(degree.std(unbiased=False)) > eps and float(drift.std(unbiased=False)) > eps:
        rd = _rankdata(degree)
        rr = _rankdata(drift)
        rd = rd - rd.mean()
        rr = rr - rr.mean()
        spearman = _float((rd * rr).sum() / (torch.linalg.norm(rd) * torch.linalg.norm(rr)).clamp_min(float(eps)))
    else:
        spearman = None

    k = max(1, int(math.ceil(0.10 * max(1, int(num_nodes)))))
    order = torch.argsort(degree)
    bottom = drift.index_select(0, order[:k]).mean()
    top = drift.index_select(0, order[-k:]).mean()
    return {
        "event_degree_mean": _float(degree.mean()) if degree.numel() else 0.0,
        "event_degree_median": _median(degree),
        "event_degree_max": _float(degree.max()) if degree.numel() else 0.0,
        "node_drift_mean": _float(drift.mean()) if drift.numel() else 0.0,
        "node_drift_median": _median(drift),
        "node_drift_max": _float(drift.max()) if drift.numel() else 0.0,
        "event_degree_node_drift_spearman": spearman,
        "top10_degree_node_drift_mean": _float(top),
        "bottom10_degree_node_drift_mean": _float(bottom),
        "top_bottom_drift_ratio": _float(top / bottom.clamp_min(float(eps))),
    }


def cluster_head_param_l2(cluster_params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in cluster_params:
        total += float(param.detach().square().sum().cpu())
    return total ** 0.5


def _flatten_grads_with_alignment(grads, params):
    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            pieces.append(grad.detach().reshape(-1))
    if not pieces:
        return None
    return torch.cat(pieces)


def _grad_summary(grads, params, param_l2: float, eps: float = EPS, prefix: str = "") -> dict:
    none_count = sum(grad is None for grad in grads)
    total_sq = 0.0
    max_abs = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_detached = grad.detach()
        total_sq += float(grad_detached.square().sum().cpu())
        max_abs = max(max_abs, float(grad_detached.abs().max().cpu()) if grad_detached.numel() else 0.0)
    l2 = total_sq ** 0.5
    return {
        f"{prefix}_cluster_head_grad_l2": l2,
        f"{prefix}_cluster_head_grad_max_abs": max_abs,
        f"{prefix}_grad_none_param_count": int(none_count),
        f"{prefix}_grad_to_param_ratio": l2 / (float(param_l2) + float(eps)),
    }


def cluster_head_gradient_diagnostics(
    cut_loss: torch.Tensor,
    orth_loss: torch.Tensor,
    cluster_params: Iterable[torch.nn.Parameter],
    eps: float = EPS,
) -> dict:
    params = [param for param in cluster_params if param.requires_grad]
    param_l2 = cluster_head_param_l2(params)
    cut_grads = torch.autograd.grad(
        cut_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    orth_grads = torch.autograd.grad(
        orth_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    result = {"cluster_head_param_l2": param_l2}
    result.update(_grad_summary(cut_grads, params, param_l2, eps=eps, prefix="cut"))
    result.update(_grad_summary(orth_grads, params, param_l2, eps=eps, prefix="orth"))
    cut_vec = _flatten_grads_with_alignment(cut_grads, params)
    orth_vec = _flatten_grads_with_alignment(orth_grads, params)
    cosine = None
    if cut_vec is not None and orth_vec is not None:
        cut_norm = torch.linalg.norm(cut_vec)
        orth_norm = torch.linalg.norm(orth_vec)
        if _float(cut_norm) >= eps and _float(orth_norm) >= eps:
            cosine = _float(torch.dot(cut_vec, orth_vec) / (cut_norm * orth_norm).clamp_min(float(eps)))
    result["cut_orth_grad_cosine"] = cosine
    return result


def compute_uniform_collapse_stage(
    stage: str,
    Q_all: torch.Tensor,
    logits_all: torch.Tensor,
    src_all: torch.Tensor,
    dst_all: torch.Tensor,
    num_nodes: int,
    K: int,
    edge_repr_all: Optional[torch.Tensor] = None,
    cluster_input_all: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    cut_loss: Optional[torch.Tensor] = None,
    orth_loss: Optional[torch.Tensor] = None,
    grad_stats: Optional[dict] = None,
    extra_stats: Optional[dict] = None,
) -> dict:
    with torch.no_grad():
        Q_detached = Q_all.detach()
        logits_detached = logits_all.detach()
        S_all = project_edge_assignments_to_nodes_global(Q_detached, src_all, dst_all, int(num_nodes))
        node_labels = torch.argmax(S_all, dim=1)
        stats = {
            "stage": stage,
            "M": int(Q_detached.size(0)),
            "N": int(num_nodes),
            "K": int(K),
        }
        stats.update(hard_cluster_statistics(Q_detached, K, "edge"))
        stats.update(hard_cluster_statistics_from_labels(node_labels, K, "node"))
        stats.update(q_margin_statistics(Q_detached))
        stats.update(logits_statistics(logits_detached))
        stats.update(q_uniform_distance_statistics(Q_detached))
        stats.update(q_rank_statistics(Q_detached))
        if edge_repr_all is not None:
            stats.update(prefixed_feature_statistics(edge_repr_all.detach(), "before_norm"))
        if cluster_input_all is not None:
            stats.update(prefixed_feature_statistics(cluster_input_all.detach(), "after_norm"))
        if labels is not None:
            stats.update(evaluate_node_clustering(labels.detach().cpu().numpy(), node_labels.detach().cpu().numpy()))
        if cut_loss is not None:
            stats["cut_loss"] = _float(cut_loss)
        if orth_loss is not None:
            stats["orth_loss"] = _float(orth_loss)
        if grad_stats:
            stats.update(grad_stats)
        if extra_stats:
            stats.update(extra_stats)
        return stats


def uniform_delta(initial: dict, after: dict) -> dict:
    delta_keys = [
        "q_uniform_l2_mean",
        "q_uniform_l1_mean",
        "q_uniform_kl_mean",
        "q_entropy_gap",
        "logits_global_std",
        "q_margin_mean",
        "q_rank1_energy_ratio",
        "q_centered_energy",
        "num_active_edge_clusters",
        "num_active_node_clusters",
    ]
    result = {}
    for key in delta_keys:
        if key in initial and key in after and initial[key] is not None and after[key] is not None:
            result[key] = after[key] - initial[key]
    return result


SUMMARY_FIELDNAMES = [
    "dataset",
    "seed",
    "stage",
    "orth_type",
    "cluster_output_bias_mode",
    "cluster_input_norm",
    "cluster_init_mode",
    "M",
    "N",
    "K",
    "num_active_edge_clusters",
    "largest_edge_cluster_ratio",
    "smallest_nonempty_edge_cluster_ratio",
    "num_empty_hard_edge_clusters",
    "num_active_node_clusters",
    "largest_node_cluster_ratio",
    "smallest_nonempty_node_cluster_ratio",
    "num_empty_hard_node_clusters",
    "q_margin_mean",
    "q_margin_p50",
    "q_margin_p90",
    "q_margin_p99",
    "q_margin_lt_1e4_ratio",
    "q_margin_lt_1e3_ratio",
    "q_margin_lt_1e2_ratio",
    "logits_global_std",
    "logits_row_std_mean",
    "logits_cluster_mean_std",
    "logits_within_cluster_event_std",
    "logits_bias_to_event_variation_ratio",
    "cluster_volume_min_ratio",
    "cluster_volume_max_ratio",
    "cluster_volume_std",
    "cluster_volume_coefficient_of_variation",
    "hard_cluster_volume_min_ratio",
    "hard_cluster_volume_max_ratio",
    "hard_cluster_volume_cv",
    "mean_assignment_entropy",
    "mean_top1_probability",
    "mean_top2_probability",
    "mean_top1_top2_margin",
    "q_uniform_l2_mean",
    "q_uniform_l1_mean",
    "q_uniform_max_abs",
    "q_uniform_kl_mean",
    "q_entropy_mean",
    "q_entropy_gap",
    "q_rank1_energy_ratio",
    "q_second_energy_ratio",
    "q_effective_rank",
    "q_numerical_rank",
    "q_centered_energy",
    "q_centered_effective_rank",
    "q_centered_numerical_rank",
    "feature_common_to_variation_ratio_before_norm",
    "feature_common_to_variation_ratio_after_norm",
    "feature_global_mean_abs_before_norm",
    "feature_global_std_before_norm",
    "feature_event_mean_norm_before_norm",
    "feature_centered_event_rms_before_norm",
    "feature_global_mean_abs_after_norm",
    "feature_global_std_after_norm",
    "feature_event_mean_norm_after_norm",
    "feature_centered_event_rms_after_norm",
    "edge_encoder_mode",
    "event_repr_dim",
    "cluster_input_dim",
    "edge_mlp_trainable_parameter_count",
    "node_embedding_parameter_count",
    "cluster_head_parameter_count",
    "trainable_parameter_count",
    "node_embedding_norm_mean",
    "node_embedding_norm_std",
    "node_embedding_drift_fro_normalized",
    "node_embedding_relative_drift",
    "node_embedding_cosine_to_initial_mean",
    "edge_repr_norm_mean",
    "edge_repr_norm_std",
    "source_block_norm_mean",
    "destination_block_norm_mean",
    "time_block_norm_mean",
    "raw_time_block_norm_mean",
    "scaled_time_block_norm_mean",
    "scaled_time_to_node_ratio",
    "direct_time_scale",
    "output_bias_l2",
    "cluster_output_weight_l2",
    "prototype_pairwise_cosine_mean",
    "prototype_pairwise_cosine_min",
    "prototype_pairwise_cosine_max",
    "prototype_min_euclidean_distance",
    "prototype_max_euclidean_distance",
    "ACC",
    "NMI",
    "ARI",
    "Macro_F1",
    "cut_loss",
    "orth_loss",
    "orth_original_loss",
    "orthqa_loss",
    "selected_penalty_loss",
    "penalty_type",
    "penalty_weight",
    "cut_cluster_head_grad_l2",
    "orth_cluster_head_grad_l2",
    "orthqa_cluster_head_grad_l2",
    "node_grad_l2_from_prox",
    "node_grad_max_from_prox",
    "node_grad_l2_from_cut",
    "node_grad_max_from_cut",
    "node_grad_l2_from_penalty",
    "node_grad_max_from_penalty",
    "cluster_grad_l2_from_cut",
    "cluster_grad_l2_from_penalty",
    "cut_grad_to_param_ratio",
    "orth_grad_to_param_ratio",
    "cut_orth_grad_cosine",
    "cut_penalty_grad_cosine",
    "unweighted_proximity_loss",
    "weighted_proximity_loss",
    "node_anchor_loss",
    "weighted_node_anchor_loss",
    "node_update_from_prox",
    "node_update_from_global",
    "node_update_prox_global_ratio",
    "cluster_update_from_global",
    "event_degree_mean",
    "event_degree_median",
    "event_degree_max",
    "node_drift_mean",
    "node_drift_median",
    "node_drift_max",
    "event_degree_node_drift_spearman",
    "top10_degree_node_drift_mean",
    "bottom10_degree_node_drift_mean",
    "top_bottom_drift_ratio",
]


def write_diagnostic_outputs(
    output_dir: str,
    dataset: str,
    seed: int,
    config: dict,
    stages: dict,
    delta: Optional[dict] = None,
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "config": config,
        "after_model_initialization": stages.get("after_model_initialization"),
        "after_prototype_initialization": stages.get("after_prototype_initialization"),
        "after_cluster_initialization": stages.get("after_cluster_initialization"),
        "initial_before_training": stages.get("initial_before_training"),
        "before_first_global_update": stages.get("before_first_global_update"),
        "after_first_global_update": stages.get("after_first_global_update"),
        "final_epoch": stages.get("final_epoch"),
        "stages": stages,
        "delta_initial_to_after_first_global": delta or {},
    }
    with open(os.path.join(output_dir, "diagnostic.json"), "w", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, sort_keys=True, default=str)
    with open(os.path.join(output_dir, "diagnostic_summary.csv"), "w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=SUMMARY_FIELDNAMES)
        csv_writer.writeheader()
        ordered = [
            "after_model_initialization",
            "after_prototype_initialization",
            "after_cluster_initialization",
            "initial_before_training",
            "before_first_global_update",
            "after_first_global_update",
            "final_epoch",
        ]
        dynamic = sorted(
            [name for name in stages if name not in set(ordered)],
            key=lambda x: (0 if x.startswith("epoch_") else 1, x),
        )
        for stage_name in ordered + dynamic:
            stage = stages.get(stage_name)
            if not stage:
                continue
            row = {key: "" for key in SUMMARY_FIELDNAMES}
            row.update({key: stage.get(key, "") for key in SUMMARY_FIELDNAMES})
            row["dataset"] = dataset
            row["seed"] = int(seed)
            row["stage"] = stage_name
            row["orth_type"] = config.get("orth_type", "")
            row["cluster_output_bias_mode"] = config.get("cluster_output_bias_mode", "")
            row["cluster_input_norm"] = config.get("cluster_input_norm", "")
            row["cluster_init_mode"] = config.get("cluster_init_mode", "")
            csv_writer.writerow(row)


def print_stage_summary(stage_name: str, stats: dict, delta: Optional[dict] = None) -> None:
    print(
        f"[diagnostic][{stage_name}] "
        f"edge_active={stats.get('num_active_edge_clusters', 0)}/{stats.get('K', 0)} "
        f"edge_max_ratio={stats.get('largest_edge_cluster_ratio', 0.0):.4f} "
        f"edge_min_nonempty={stats.get('smallest_nonempty_edge_cluster_ratio', 0.0):.4f}"
    )
    print(
        f"[diagnostic][{stage_name}] "
        f"node_active={stats.get('num_active_node_clusters', 0)}/{stats.get('K', 0)} "
        f"node_max_ratio={stats.get('largest_node_cluster_ratio', 0.0):.4f} "
        f"node_min_nonempty={stats.get('smallest_nonempty_node_cluster_ratio', 0.0):.4f}"
    )
    print(
        f"[diagnostic][{stage_name}] "
        f"q_margin_mean={stats.get('q_margin_mean', 0.0):.6g} "
        f"q_margin_p99={stats.get('q_margin_p99', 0.0):.6g} "
        f"logits_global_std={stats.get('logits_global_std', 0.0):.6g} "
        f"logits_row_std_mean={stats.get('logits_row_std_mean', 0.0):.6g} "
        f"rank1={stats.get('q_rank1_energy_ratio', 0.0):.6g} "
        f"centered_energy={stats.get('q_centered_energy', 0.0):.6g} "
        f"uniform_l2={stats.get('q_uniform_l2_mean', 0.0):.6g} "
        f"entropy_gap={stats.get('q_entropy_gap', 0.0):.6g}"
    )
    if "cut_loss" in stats or "cut_cluster_head_grad_l2" in stats:
        cosine = stats.get("cut_orth_grad_cosine")
        cosine_text = "null" if cosine is None else f"{cosine:.6g}"
        print(
            f"[diagnostic][{stage_name}] "
            f"cut={stats.get('cut_loss', float('nan')):.6g} "
            f"orth={stats.get('orth_loss', float('nan')):.6g} "
            f"cut_grad_l2={stats.get('cut_cluster_head_grad_l2', float('nan')):.6g} "
            f"orth_grad_l2={stats.get('orth_cluster_head_grad_l2', float('nan')):.6g} "
            f"cut_orth_grad_cosine={cosine_text}"
        )
    if delta:
        print(
            f"[diagnostic][{stage_name}] "
            f"delta_uniform_l2={delta.get('q_uniform_l2_mean', 0.0):.6g} "
            f"delta_margin_mean={delta.get('q_margin_mean', 0.0):.6g} "
            f"delta_active_node={delta.get('num_active_node_clusters', 0)}"
        )
