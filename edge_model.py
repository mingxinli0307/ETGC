import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_checksum(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_pretrained_node_features(
    path: str,
    num_nodes: int,
    fallback_dim: int,
    seed: int,
    require_existing: bool = False,
) -> np.ndarray:
    if not os.path.exists(path):
        if require_existing:
            raise FileNotFoundError(f"Missing required Node2Vec embedding file: {path}")
        print(f"Warning: pretrain embedding file not found: {path}; using random initialization.")
        rng = np.random.RandomState(seed)
        return rng.normal(0.0, 0.02, size=(num_nodes, fallback_dim)).astype(np.float32)

    rows = {}
    with open(path, "r", encoding="utf-8") as reader:
        first = reader.readline().lstrip("\ufeff").strip().split()
        has_header = len(first) == 2 and all(tok.lstrip("-").isdigit() for tok in first)
        if first and not has_header:
            arr = np.asarray(first, dtype=np.float32)
            rows[int(arr[0])] = arr[1:]
        for line in reader:
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, dtype=np.float32, sep=" ")
            if arr.size > 1:
                rows[int(arr[0])] = arr[1:]

    if not rows:
        raise ValueError(f"No embeddings could be parsed from {path}")
    dim = len(next(iter(rows.values())))
    if require_existing:
        missing = [nid for nid in range(int(num_nodes)) if nid not in rows or len(rows[nid]) != dim]
        if missing:
            preview = ",".join(str(x) for x in missing[:10])
            raise ValueError(
                f"Node2Vec embedding file is incomplete for contiguous ETGC node ids: "
                f"path={path}, missing_or_bad_count={len(missing)}, first_missing={preview}"
            )
    rng = np.random.RandomState(seed)
    features = rng.normal(0.0, 0.02, size=(num_nodes, dim)).astype(np.float32)
    for nid, vec in rows.items():
        if 0 <= nid < num_nodes and len(vec) == dim:
            features[nid] = vec
    return features


class EdgeHiNoSModel(nn.Module):
    def __init__(
        self,
        initial_node_features: np.ndarray,
        time_dim: int,
        edge_dim: int,
        edge_hidden_dim: int,
        cluster_hidden_dim: int,
        K: int,
        directed: bool,
        cluster_output_bias_mode: str = "default",
        cluster_input_norm: str = "none",
        edge_encoder_mode: str = "mlp",
        direct_time_scale: float = 1.0,
    ):
        super().__init__()
        initial_node_features = initial_node_features.astype(np.float32, copy=True)
        self.node_emb = nn.Parameter(torch.from_numpy(initial_node_features))
        self.register_buffer("node_emb_initial", torch.from_numpy(initial_node_features.copy()))
        self.directed = bool(directed)
        self.edge_encoder_mode = str(edge_encoder_mode).lower()
        self.direct_time_scale = float(direct_time_scale)
        self.cluster_output_bias_mode = str(cluster_output_bias_mode).lower()
        self.cluster_input_norm_mode = str(cluster_input_norm).lower()
        if self.edge_encoder_mode not in {"mlp", "direct_node_time"}:
            raise ValueError(f"Unsupported edge_encoder_mode: {edge_encoder_mode}")
        if self.cluster_output_bias_mode not in {"default", "zero", "none"}:
            raise ValueError(f"Unsupported cluster_output_bias_mode: {cluster_output_bias_mode}")
        if self.cluster_input_norm_mode not in {"none", "layernorm"}:
            raise ValueError(f"Unsupported cluster_input_norm: {cluster_input_norm}")
        node_dim = int(initial_node_features.shape[1])
        self.node_dim = node_dim
        self.time_dim = int(time_dim)
        if self.edge_encoder_mode == "mlp":
            pair_dim = 2 * node_dim if self.directed else 3 * node_dim
            in_dim = pair_dim + self.time_dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(in_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, edge_dim),
                nn.ReLU(),
            )
            cluster_input_dim = int(edge_dim)
        else:
            self.edge_mlp = None
            cluster_input_dim = 2 * node_dim + self.time_dim
        self.event_repr_dim = int(cluster_input_dim)
        self.cluster_input_dim = int(cluster_input_dim)
        if self.cluster_input_norm_mode == "layernorm":
            self.cluster_input_norm = nn.LayerNorm(self.cluster_input_dim, elementwise_affine=False)
        else:
            self.cluster_input_norm = nn.Identity()
        self.cluster_hidden = nn.Linear(self.cluster_input_dim, cluster_hidden_dim)
        self.cluster_activation = nn.ReLU()
        self.cluster_output = nn.Linear(
            cluster_hidden_dim,
            K,
            bias=self.cluster_output_bias_mode != "none",
        )
        if self.cluster_output_bias_mode == "zero" and self.cluster_output.bias is not None:
            nn.init.zeros_(self.cluster_output.bias)
        self.cluster_head = nn.Sequential(
            self.cluster_input_norm,
            self.cluster_hidden,
            self.cluster_activation,
            self.cluster_output,
        )

    def build_direct_node_time_event_repr(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> torch.Tensor:
        h_src = self.node_emb.index_select(0, src.long())
        h_dst = self.node_emb.index_select(0, dst.long())
        return torch.cat([h_src, h_dst, float(self.direct_time_scale) * time_feat], dim=-1)

    def encode_edge_events(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> torch.Tensor:
        h_u = self.node_emb.index_select(0, src.long())
        h_v = self.node_emb.index_select(0, dst.long())
        if self.edge_encoder_mode == "mlp":
            if self.directed:
                pair_feat = torch.cat([h_u, h_v], dim=-1)
            else:
                pair_feat = torch.cat([h_u + h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
            x_e = torch.cat([pair_feat, time_feat], dim=-1)
            return self.edge_mlp(x_e)
        if self.edge_encoder_mode == "direct_node_time":
            return self.build_direct_node_time_event_repr(src, dst, time_feat)
        raise ValueError(f"Unsupported edge_encoder_mode: {self.edge_encoder_mode}")

    def cluster_hidden_from_edge_repr(self, edge_repr: torch.Tensor):
        cluster_input = self.cluster_input_norm(edge_repr)
        cluster_hidden = self.cluster_activation(self.cluster_hidden(cluster_input))
        return cluster_input, cluster_hidden

    def cluster_logits_from_hidden(self, cluster_hidden: torch.Tensor) -> torch.Tensor:
        return self.cluster_output(cluster_hidden)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        time_feat: torch.Tensor,
        return_logits: bool = False,
        return_cluster_hidden: bool = False,
        return_edge_repr: bool = False,
    ):
        r_e = self.encode_edge_events(src, dst, time_feat)
        _cluster_input, cluster_hidden = self.cluster_hidden_from_edge_repr(r_e)
        logits = self.cluster_logits_from_hidden(cluster_hidden)
        q_e = F.softmax(logits, dim=-1)
        if return_logits and return_cluster_hidden:
            return r_e, q_e, logits, cluster_hidden
        if return_logits:
            return r_e, q_e, logits
        if return_cluster_hidden:
            return r_e, q_e, cluster_hidden
        return r_e, q_e


def feature_common_variation_statistics(features: torch.Tensor, eps: float = 1e-12) -> dict:
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got shape={tuple(features.shape)}")
    if features.numel() == 0:
        return {
            "feature_global_mean_abs": 0.0,
            "feature_global_std": 0.0,
            "feature_event_mean_norm": 0.0,
            "feature_centered_event_rms": 0.0,
            "feature_common_to_variation_ratio": 0.0,
        }
    feat = features.detach()
    mean_feature = feat.mean(dim=0)
    centered = feat - mean_feature
    centered_rms = torch.sqrt(centered.square().sum(dim=1).mean().clamp_min(0.0))
    ratio = torch.linalg.norm(mean_feature) / centered_rms.clamp_min(float(eps))
    return {
        "feature_global_mean_abs": float(feat.mean().abs().cpu()),
        "feature_global_std": float(feat.std(unbiased=False).cpu()),
        "feature_event_mean_norm": float(torch.linalg.norm(mean_feature).cpu()),
        "feature_centered_event_rms": float(centered_rms.cpu()),
        "feature_common_to_variation_ratio": float(ratio.cpu()),
    }


def fit_kmeans_centers(
    features: torch.Tensor,
    K: int,
    seed: int,
    sample_size: int = 20000,
    max_iters: int = 10,
    eps: float = 1e-12,
) -> tuple:
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got shape={tuple(features.shape)}")
    n = int(features.size(0))
    d = int(features.size(1))
    K = int(K)
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n < K:
        raise ValueError(f"Need at least K samples for prototype initialization, got n={n}, K={K}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    sample_size = min(max(K, int(sample_size)), n)
    if sample_size < n:
        sample_idx = torch.randperm(n, generator=gen)[:sample_size].to(device=features.device)
        X = features.index_select(0, sample_idx).detach()
    else:
        X = features.detach()
    n_sample = int(X.size(0))
    first = int(torch.randint(n_sample, (1,), generator=gen).item())
    selected = [first]
    centers = [X[first].clone()]
    min_dist = torch.cdist(X, centers[0].view(1, d)).square().squeeze(1)
    for _ in range(1, K):
        masked = min_dist.clone()
        masked[torch.as_tensor(selected, device=features.device)] = -1.0
        total = float(torch.clamp(masked, min=0.0).sum().cpu())
        if total <= eps:
            next_idx = int(torch.argmax(masked).item())
        else:
            probs = torch.clamp(masked, min=0.0)
            probs = (probs / probs.sum()).cpu()
            next_idx = int(torch.multinomial(probs, 1, generator=gen).item())
            while next_idx in selected:
                masked[next_idx] = -1.0
                if float(torch.clamp(masked, min=0.0).sum().cpu()) <= eps:
                    next_idx = int(torch.argmax(masked).item())
                    break
                probs = torch.clamp(masked, min=0.0)
                probs = (probs / probs.sum()).cpu()
                next_idx = int(torch.multinomial(probs, 1, generator=gen).item())
        selected.append(next_idx)
        centers.append(X[next_idx].clone())
        dist = torch.cdist(X, centers[-1].view(1, d)).square().squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
    C = torch.stack(centers, dim=0)
    max_iters = max(0, int(max_iters))
    lloyd_iters_run = 0
    for _ in range(max_iters):
        dist = torch.cdist(X, C).square()
        assign = torch.argmin(dist, dim=1)
        new_centers = []
        min_current = dist.gather(1, assign.view(-1, 1)).squeeze(1)
        for k in range(K):
            mask = assign == k
            if bool(mask.any()):
                new_centers.append(X[mask].mean(dim=0))
            else:
                farthest = int(torch.argmax(min_current).item())
                new_centers.append(X[farthest].clone())
                min_current[farthest] = -1.0
        new_C = torch.stack(new_centers, dim=0)
        lloyd_iters_run += 1
        if torch.allclose(new_C, C, atol=1e-6, rtol=1e-5):
            C = new_C
            break
        C = new_C
    if not torch.isfinite(C).all():
        raise FloatingPointError("KMeans centers contain NaN or Inf values")
    stats = {
        "kmeans_sample_size": int(n_sample),
        "kmeans_lloyd_iters_requested": int(max_iters),
        "kmeans_lloyd_iters_run": int(lloyd_iters_run),
        "kmeans_center_checksum": tensor_checksum(C),
    }
    return C, stats


def torch_kmeans_plus_plus(
    features: torch.Tensor,
    K: int,
    seed: int,
    sample_size: int = 20000,
    max_iters: int = 10,
    eps: float = 1e-12,
) -> torch.Tensor:
    centers, _ = fit_kmeans_centers(
        features,
        K=K,
        seed=seed,
        sample_size=sample_size,
        max_iters=max_iters,
        eps=eps,
    )
    return centers / torch.linalg.norm(centers, dim=1, keepdim=True).clamp_min(float(eps))


def prototype_pairwise_statistics(prototypes: torch.Tensor) -> dict:
    proto = prototypes.detach()
    K = int(proto.size(0))
    if K < 2:
        return {
            "prototype_pairwise_cosine_mean": 0.0,
            "prototype_pairwise_cosine_min": 0.0,
            "prototype_pairwise_cosine_max": 0.0,
            "prototype_min_euclidean_distance": 0.0,
            "prototype_max_euclidean_distance": 0.0,
        }
    normed = proto / torch.linalg.norm(proto, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = normed @ normed.t()
    upper = torch.triu_indices(K, K, offset=1, device=proto.device)
    cos_vals = cosine[upper[0], upper[1]]
    dist_vals = torch.cdist(proto, proto)[upper[0], upper[1]]
    return {
        "prototype_pairwise_cosine_mean": float(cos_vals.mean().cpu()),
        "prototype_pairwise_cosine_min": float(cos_vals.min().cpu()),
        "prototype_pairwise_cosine_max": float(cos_vals.max().cpu()),
        "prototype_min_euclidean_distance": float(dist_vals.min().cpu()),
        "prototype_max_euclidean_distance": float(dist_vals.max().cpu()),
    }


def initial_weight_pairwise_statistics(weight: torch.Tensor, prefix: str = "initial_weight") -> dict:
    stats = prototype_pairwise_statistics(weight)
    return {
        f"{prefix}_pairwise_cosine_mean": stats["prototype_pairwise_cosine_mean"],
        f"{prefix}_pairwise_cosine_min": stats["prototype_pairwise_cosine_min"],
        f"{prefix}_pairwise_cosine_max": stats["prototype_pairwise_cosine_max"],
    }


@torch.no_grad()
def _copy_cluster_output_weight(model: EdgeHiNoSModel, weight: torch.Tensor, zero_bias: bool = True) -> None:
    weight = weight.to(device=model.cluster_output.weight.device, dtype=model.cluster_output.weight.dtype)
    if tuple(model.cluster_output.weight.shape) != tuple(weight.shape):
        raise ValueError(
            f"Initial weight shape={tuple(weight.shape)} does not match cluster output weight "
            f"shape={tuple(model.cluster_output.weight.shape)}"
        )
    model.cluster_output.weight.copy_(weight)
    if zero_bias and model.cluster_output.bias is not None:
        nn.init.zeros_(model.cluster_output.bias)


@torch.no_grad()
def initialize_cluster_output_random_orthogonal(
    model: EdgeHiNoSModel,
    K: int,
    seed: int,
    eps: float = 1e-12,
) -> dict:
    K = int(K)
    d = int(model.cluster_output.weight.size(1))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if K <= d:
        base = torch.randn(d, K, generator=gen)
        q, _ = torch.linalg.qr(base, mode="reduced")
        weight = q.t().contiguous()
        strict = True
        note = "strict_row_orthogonal"
    else:
        pieces = []
        remaining = K
        while remaining > 0:
            block_rows = min(d, remaining)
            base = torch.randn(d, block_rows, generator=gen)
            q, _ = torch.linalg.qr(base, mode="reduced")
            pieces.append(q.t().contiguous())
            remaining -= block_rows
        weight = torch.cat(pieces, dim=0)[:K]
        strict = False
        note = "K_gt_dim_block_orthogonal_not_globally_strict"
    weight = weight / torch.linalg.norm(weight, dim=1, keepdim=True).clamp_min(float(eps))
    _copy_cluster_output_weight(model, weight, zero_bias=True)
    stats = initial_weight_pairwise_statistics(model.cluster_output.weight.detach())
    stats.update(
        {
            "cluster_init_mode_effective": "random_orthogonal",
            "strict_orthogonal_rows": bool(strict),
            "random_orthogonal_note": note,
            "initial_cluster_weight_checksum": tensor_checksum(model.cluster_output.weight.detach()),
        }
    )
    return stats


@torch.no_grad()
def initialize_cluster_output_random_event(
    model: EdgeHiNoSModel,
    cluster_hidden: torch.Tensor,
    K: int,
    seed: int,
    eps: float = 1e-12,
) -> dict:
    if cluster_hidden.dim() != 2:
        raise ValueError(f"cluster_hidden must be 2D, got shape={tuple(cluster_hidden.shape)}")
    K = int(K)
    n = int(cluster_hidden.size(0))
    if n < K:
        raise ValueError(f"random_event initialization needs at least K events, got M={n}, K={K}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    indices = torch.randperm(n, generator=gen)[:K].to(device=cluster_hidden.device)
    weight = cluster_hidden.index_select(0, indices).detach()
    weight = weight / torch.linalg.norm(weight, dim=1, keepdim=True).clamp_min(float(eps))
    _copy_cluster_output_weight(model, weight, zero_bias=True)
    stats = initial_weight_pairwise_statistics(model.cluster_output.weight.detach())
    stats.update(
        {
            "cluster_init_mode_effective": "random_event",
            "random_event_unique_count": int(torch.unique(indices).numel()),
            "random_event_index_checksum": tensor_checksum(indices.to(dtype=torch.int64)),
            "initial_cluster_weight_checksum": tensor_checksum(model.cluster_output.weight.detach()),
        }
    )
    return stats


@torch.no_grad()
def initialize_cluster_output_from_prototypes(
    model: EdgeHiNoSModel,
    cluster_hidden: torch.Tensor,
    K: int,
    seed: int,
    sample_size: int = 20000,
    lloyd_iters: int = 10,
) -> dict:
    raw_centers, fit_stats = fit_kmeans_centers(
        cluster_hidden,
        K=int(K),
        seed=int(seed),
        sample_size=int(sample_size),
        max_iters=int(lloyd_iters),
    )
    prototypes = raw_centers / torch.linalg.norm(raw_centers, dim=1, keepdim=True).clamp_min(1e-12)
    prototypes = prototypes.to(device=model.cluster_output.weight.device, dtype=model.cluster_output.weight.dtype)
    _copy_cluster_output_weight(model, prototypes, zero_bias=True)
    stats = prototype_pairwise_statistics(prototypes)
    stats.update(
        {
            "cluster_init_mode_effective": "prototype",
            "prototype_shape": [int(prototypes.size(0)), int(prototypes.size(1))],
            "prototype_sample_size": int(min(int(sample_size), int(cluster_hidden.size(0)))),
            "prototype_lloyd_iters": int(lloyd_iters),
            "prototype_weight_l2": float(torch.linalg.norm(model.cluster_output.weight.detach()).cpu()),
            "prototype_center_checksum": fit_stats["kmeans_center_checksum"],
            "initial_cluster_weight_checksum": tensor_checksum(model.cluster_output.weight.detach()),
        }
    )
    stats.update(fit_stats)
    return stats


@torch.no_grad()
def assign_all_to_centers(features: torch.Tensor, centers: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    if features.dim() != 2 or centers.dim() != 2:
        raise ValueError("features and centers must be 2D tensors")
    labels = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(features.size(0)), chunk_size):
        dist = torch.cdist(features[start : start + chunk_size], centers).square()
        labels.append(torch.argmin(dist, dim=1).detach())
    return torch.cat(labels, dim=0)
