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


def power_sharpen(q1: torch.Tensor, gamma: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable row-wise power sharpening, without frequency correction."""
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("sharpen_gamma must be finite and >= 1.0")
    q_power = q1.pow(gamma)
    return q_power / q_power.sum(dim=1, keepdim=True).clamp_min(eps)


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
        cluster_head_type: str = "legacy_mlp",
        prototype_temperature: float = 0.2,
        hier_ncut_h: int = -1,
        sharpen_gamma: float = 2.0,
    ):
        super().__init__()
        self.K = int(K)
        self.H = 2 * self.K if int(hier_ncut_h) <= 0 else int(hier_ncut_h)
        if self.K <= 0 or self.H < self.K:
            raise ValueError(f"Require H >= K > 0, got H={self.H}, K={self.K}")
        self.sharpen_gamma = float(sharpen_gamma)
        if not np.isfinite(self.sharpen_gamma) or self.sharpen_gamma < 1.0:
            raise ValueError("sharpen_gamma must be finite and >= 1.0")
        self.coarse_assignment_logits = nn.Parameter(torch.empty(self.H, self.K))
        nn.init.normal_(self.coarse_assignment_logits, mean=0.0, std=0.02)
        initial_node_features = initial_node_features.astype(np.float32, copy=True)
        self.node_emb = nn.Parameter(torch.from_numpy(initial_node_features))
        self.directed = bool(directed)
        self.edge_encoder_mode = str(edge_encoder_mode).lower()
        self.direct_time_scale = float(direct_time_scale)
        self.cluster_head_type = str(cluster_head_type).lower()
        self.prototype_temperature = float(prototype_temperature)
        self.cluster_output_bias_mode = str(cluster_output_bias_mode).lower()
        self.cluster_input_norm_mode = str(cluster_input_norm).lower()
        if self.edge_encoder_mode not in {"mlp", "direct_node_time"}:
            raise ValueError(f"Unsupported edge_encoder_mode: {edge_encoder_mode}")
        if self.cluster_head_type not in {"legacy_mlp", "cosine_prototype"}:
            raise ValueError(f"Unsupported cluster_head_type: {cluster_head_type}")
        if self.prototype_temperature <= 0.0:
            raise ValueError(f"prototype_temperature must be positive, got {prototype_temperature}")
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
        if self.cluster_head_type == "legacy_mlp" and self.cluster_input_norm_mode == "layernorm":
            self.cluster_input_norm = nn.LayerNorm(self.cluster_input_dim, elementwise_affine=False)
        else:
            self.cluster_input_norm = nn.Identity()
        if self.cluster_head_type == "legacy_mlp":
            self.cluster_hidden = nn.Linear(self.cluster_input_dim, cluster_hidden_dim)
            self.cluster_activation = nn.ReLU()
            self.cluster_output = nn.Linear(
                cluster_hidden_dim,
                self.H,
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
        else:
            self.cluster_hidden = None
            self.cluster_activation = None
            self.cluster_output = CosinePrototypeOutput(
                K=self.H,
                dim=self.cluster_input_dim,
                temperature=self.prototype_temperature,
            )
            self.cluster_head = self.cluster_output

    def build_direct_node_time_event_repr(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility helper for callers that provide event endpoint ids."""
        h_src = self.node_emb.index_select(0, src.long())
        h_dst = self.node_emb.index_select(0, dst.long())
        return self._direct_node_time_event_repr(h_src, h_dst, time_feat)

    def _direct_node_time_event_repr(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> torch.Tensor:
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
            return self._direct_node_time_event_repr(h_u, h_v, time_feat)
        raise ValueError(f"Unsupported edge_encoder_mode: {self.edge_encoder_mode}")

    def cluster_hidden_from_edge_repr(self, edge_repr: torch.Tensor):
        if self.cluster_head_type == "cosine_prototype":
            return edge_repr, edge_repr
        cluster_input = self.cluster_input_norm(edge_repr)
        cluster_hidden = self.cluster_activation(self.cluster_hidden(cluster_input))
        return cluster_input, cluster_hidden

    def cluster_logits_from_hidden(self, cluster_hidden: torch.Tensor) -> torch.Tensor:
        return self.cluster_output(cluster_hidden)

    def forward_hierarchical(self, src, dst, time_feat):
        edge_repr_all = self.encode_edge_events(src, dst, time_feat)
        _, cluster_hidden = self.cluster_hidden_from_edge_repr(edge_repr_all)
        z1_all = self.cluster_logits_from_hidden(cluster_hidden)
        q1_all = F.softmax(z1_all, dim=1)
        p1_all = power_sharpen(q1_all, self.sharpen_gamma)
        q2 = F.softmax(self.coarse_assignment_logits, dim=1)
        q_final_all = p1_all @ q2
        return dict(edge_repr_all=edge_repr_all, z1_all=z1_all, q1_all=q1_all,
                    p1_all=p1_all, q2=q2, q_final_all=q_final_all)

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
        q1 = F.softmax(logits, dim=1)
        p1 = power_sharpen(q1, self.sharpen_gamma)
        q2 = F.softmax(self.coarse_assignment_logits, dim=1)
        q_e = p1 @ q2
        if return_logits and return_cluster_hidden:
            return r_e, q_e, logits, cluster_hidden
        if return_logits:
            return r_e, q_e, logits
        if return_cluster_hidden:
            return r_e, q_e, cluster_hidden
        return r_e, q_e


class CosinePrototypeOutput(nn.Module):
    def __init__(self, K: int, dim: int, temperature: float = 0.2, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(K), int(dim)))
        self.bias = None
        self.temperature = float(temperature)
        self.eps = float(eps)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, edge_repr: torch.Tensor) -> torch.Tensor:
        r_hat = edge_repr / torch.linalg.norm(edge_repr, dim=1, keepdim=True).clamp_min(self.eps)
        c_hat = self.weight / torch.linalg.norm(self.weight, dim=1, keepdim=True).clamp_min(self.eps)
        return (r_hat @ c_hat.t()) / self.temperature


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
