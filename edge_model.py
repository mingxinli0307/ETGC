import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_pretrained_node_features(path: str, num_nodes: int, fallback_dim: int, seed: int) -> np.ndarray:
    if not os.path.exists(path):
        print(f"Warning: pretrain embedding file not found: {path}; using random initialization.")
        rng = np.random.RandomState(seed)
        return rng.normal(0.0, 0.02, size=(num_nodes, fallback_dim)).astype(np.float32)

    rows = {}
    with open(path, "r", encoding="utf-8") as reader:
        first = reader.readline().strip().split()
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
    ):
        super().__init__()
        self.node_emb = nn.Parameter(torch.from_numpy(initial_node_features.astype(np.float32)))
        self.directed = bool(directed)
        self.cluster_output_bias_mode = str(cluster_output_bias_mode).lower()
        self.cluster_input_norm_mode = str(cluster_input_norm).lower()
        if self.cluster_output_bias_mode not in {"default", "zero", "none"}:
            raise ValueError(f"Unsupported cluster_output_bias_mode: {cluster_output_bias_mode}")
        if self.cluster_input_norm_mode not in {"none", "layernorm"}:
            raise ValueError(f"Unsupported cluster_input_norm: {cluster_input_norm}")
        node_dim = int(initial_node_features.shape[1])
        pair_dim = 2 * node_dim if self.directed else 3 * node_dim
        in_dim = pair_dim + int(time_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_dim),
            nn.ReLU(),
        )
        if self.cluster_input_norm_mode == "layernorm":
            self.cluster_input_norm = nn.LayerNorm(edge_dim, elementwise_affine=False)
        else:
            self.cluster_input_norm = nn.Identity()
        self.cluster_hidden = nn.Linear(edge_dim, cluster_hidden_dim)
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
    ):
        h_u = self.node_emb.index_select(0, src.long())
        h_v = self.node_emb.index_select(0, dst.long())
        if self.directed:
            pair_feat = torch.cat([h_u, h_v], dim=-1)
        else:
            pair_feat = torch.cat([h_u + h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
        x_e = torch.cat([pair_feat, time_feat], dim=-1)
        r_e = self.edge_mlp(x_e)
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


def torch_kmeans_plus_plus(
    features: torch.Tensor,
    K: int,
    seed: int,
    sample_size: int = 20000,
    max_iters: int = 10,
    eps: float = 1e-12,
) -> torch.Tensor:
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
        if torch.allclose(new_C, C, atol=1e-6, rtol=1e-5):
            C = new_C
            break
        C = new_C
    return C / torch.linalg.norm(C, dim=1, keepdim=True).clamp_min(float(eps))


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


@torch.no_grad()
def initialize_cluster_output_from_prototypes(
    model: EdgeHiNoSModel,
    cluster_hidden: torch.Tensor,
    K: int,
    seed: int,
    sample_size: int = 20000,
    lloyd_iters: int = 10,
) -> dict:
    prototypes = torch_kmeans_plus_plus(
        cluster_hidden,
        K=int(K),
        seed=int(seed),
        sample_size=int(sample_size),
        max_iters=int(lloyd_iters),
    ).to(device=model.cluster_output.weight.device, dtype=model.cluster_output.weight.dtype)
    if tuple(model.cluster_output.weight.shape) != tuple(prototypes.shape):
        raise ValueError(
            f"Prototype shape={tuple(prototypes.shape)} does not match cluster output weight "
            f"shape={tuple(model.cluster_output.weight.shape)}"
        )
    model.cluster_output.weight.copy_(prototypes)
    if model.cluster_output.bias is not None:
        nn.init.zeros_(model.cluster_output.bias)
    stats = prototype_pairwise_statistics(prototypes)
    stats.update(
        {
            "prototype_shape": [int(prototypes.size(0)), int(prototypes.size(1))],
            "prototype_sample_size": int(min(int(sample_size), int(cluster_hidden.size(0)))),
            "prototype_lloyd_iters": int(lloyd_iters),
            "prototype_weight_l2": float(torch.linalg.norm(model.cluster_output.weight.detach()).cpu()),
        }
    )
    return stats
