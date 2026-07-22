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
    ):
        super().__init__()
        self.node_emb = nn.Parameter(torch.from_numpy(initial_node_features.astype(np.float32)))
        self.directed = bool(directed)
        node_dim = int(initial_node_features.shape[1])
        pair_dim = 2 * node_dim if self.directed else 3 * node_dim
        in_dim = pair_dim + int(time_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_dim),
            nn.ReLU(),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(edge_dim, cluster_hidden_dim),
            nn.ReLU(),
            nn.Linear(cluster_hidden_dim, K),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor, time_feat: torch.Tensor):
        h_u = self.node_emb.index_select(0, src.long())
        h_v = self.node_emb.index_select(0, dst.long())
        if self.directed:
            pair_feat = torch.cat([h_u, h_v], dim=-1)
        else:
            pair_feat = torch.cat([h_u + h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
        x_e = torch.cat([pair_feat, time_feat], dim=-1)
        r_e = self.edge_mlp(x_e)
        q_e = F.softmax(self.cluster_head(r_e), dim=-1)
        return r_e, q_e
