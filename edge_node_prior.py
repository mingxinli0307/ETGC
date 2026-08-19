import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from edge_model import assign_all_to_centers, fit_kmeans_centers


def temporal_block_validation_auc(
    assignments: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    times: np.ndarray,
    K: int,
    seed: int = 20260819,
    train_ratio: float = 0.8,
) -> float:
    """Score a hard partition by future-link prediction without node labels."""
    assignments = np.asarray(assignments, dtype=np.int64).reshape(-1)
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    times = np.asarray(times).reshape(-1)
    if not (len(src) == len(dst) == len(times)) or len(src) < 2:
        raise ValueError("temporal block validation requires aligned nontrivial edge arrays")
    split = min(len(src) - 1, max(1, int(round(float(train_ratio) * len(src)))))
    order = np.argsort(times, kind="stable")
    train_ids, valid_ids = order[:split], order[split:]
    rng = np.random.RandomState(int(seed))

    def block_distribution(edge_ids: np.ndarray, shuffled: bool) -> np.ndarray:
        left = src[edge_ids]
        right = dst[edge_ids]
        if shuffled:
            right = right[rng.permutation(len(right))]
        result = np.zeros((int(K), int(K)), dtype=np.float64)
        np.add.at(result, (assignments[left], assignments[right]), 1.0)
        np.add.at(result, (assignments[right], assignments[left]), 1.0)
        return result / max(float(result.sum()), 1.0)

    positive = block_distribution(train_ids, shuffled=False)
    negative = block_distribution(train_ids, shuffled=True)
    block_logits = np.clip(np.log(positive + 1e-8) - np.log(negative + 1e-8), -8.0, 8.0)
    valid_src = src[valid_ids]
    valid_dst = dst[valid_ids]
    shuffled_dst = valid_dst[rng.permutation(len(valid_dst))]
    positive_scores = block_logits[assignments[valid_src], assignments[valid_dst]]
    negative_scores = block_logits[assignments[valid_src], assignments[shuffled_dst]]
    targets = np.concatenate(
        [np.ones(len(positive_scores), dtype=np.int64), np.zeros(len(negative_scores), dtype=np.int64)]
    )
    scores = np.concatenate([positive_scores, negative_scores])
    return float(roc_auc_score(targets, scores))


def build_adaptive_node_prior(
    node_features: torch.Tensor,
    src: np.ndarray,
    dst: np.ndarray,
    times: np.ndarray,
    K: int,
    restarts: int = 100,
    seed: int = 10000,
    seed_stride: int = 7919,
    lloyd_iters: int = 30,
    validation_seed: int = 20260819,
    auc_threshold: float = 0.9,
) -> tuple:
    """Build a deterministic label-free node partition for training guidance.

    Candidate KMeans partitions use L2-normalized initial node features. When
    their best held-out temporal link AUC is high-confidence, that temporally
    generalizing partition is selected; otherwise the minimum-inertia
    partition is used. No ground-truth labels enter either criterion.
    """
    if node_features.dim() != 2:
        raise ValueError(f"node_features must be 2D, got {tuple(node_features.shape)}")
    if int(restarts) <= 0:
        raise ValueError(f"restarts must be positive, got {restarts}")
    normalized = node_features.detach() / torch.linalg.norm(
        node_features.detach(), dim=1, keepdim=True
    ).clamp_min(1e-12)
    best_inertia = None
    best_auc = None
    candidates = []
    for restart in range(int(restarts)):
        candidate_seed = int(seed) + int(seed_stride) * restart
        centers, _ = fit_kmeans_centers(
            normalized,
            int(K),
            candidate_seed,
            sample_size=int(normalized.size(0)),
            max_iters=int(lloyd_iters),
        )
        labels_t = assign_all_to_centers(normalized, centers)
        inertia = float(
            (normalized - centers.index_select(0, labels_t)).square().sum().detach().cpu()
        )
        labels = labels_t.detach().cpu().numpy().astype(np.int64, copy=False)
        auc = temporal_block_validation_auc(
            labels,
            src,
            dst,
            times,
            int(K),
            seed=int(validation_seed),
        )
        candidate = {
            "labels": labels,
            "seed": candidate_seed,
            "inertia": inertia,
            "temporal_validation_auc": auc,
        }
        candidates.append(candidate)
        if best_inertia is None or inertia < best_inertia["inertia"]:
            best_inertia = candidate
        if best_auc is None or auc > best_auc["temporal_validation_auc"]:
            best_auc = candidate

    use_auc = float(best_auc["temporal_validation_auc"]) >= float(auc_threshold)
    selected = best_auc if use_auc else best_inertia
    info = {
        "node_prior_mode_effective": "temporal_validation_auc" if use_auc else "minimum_inertia",
        "node_prior_selected_seed": int(selected["seed"]),
        "node_prior_selected_inertia": float(selected["inertia"]),
        "node_prior_selected_temporal_validation_auc": float(selected["temporal_validation_auc"]),
        "node_prior_best_temporal_validation_auc": float(best_auc["temporal_validation_auc"]),
        "node_prior_minimum_inertia": float(best_inertia["inertia"]),
        "node_prior_restarts": int(restarts),
        "node_prior_auc_threshold": float(auc_threshold),
        "node_prior_active_clusters": int(np.unique(selected["labels"]).size),
    }
    labels_t = torch.from_numpy(selected["labels"].copy()).to(
        device=node_features.device, dtype=torch.long
    )
    return labels_t, info
