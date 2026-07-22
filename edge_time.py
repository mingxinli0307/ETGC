from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, max_frequency: float = 64.0):
        super().__init__()
        self.time_dim = int(time_dim)
        base_dim = max(1, (self.time_dim + 7) // 8)
        freqs = np.geomspace(1.0, max_frequency, num=base_dim).astype(np.float32)
        self.register_buffer("freqs", torch.from_numpy(freqs), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.unsqueeze(-1) * self.freqs.view(*([1] * values.dim()), -1)
        encoded = torch.cat([torch.sin(2.0 * torch.pi * x), torch.cos(2.0 * torch.pi * x)], dim=-1)
        encoded = encoded.flatten(start_dim=-2)
        if encoded.size(-1) >= self.time_dim:
            return encoded[..., : self.time_dim]
        pad = torch.zeros(*encoded.shape[:-1], self.time_dim - encoded.size(-1), device=encoded.device)
        return torch.cat([encoded, pad], dim=-1)


def build_edge_time_features(src: np.ndarray, dst: np.ndarray, times: np.ndarray, time_dim: int) -> np.ndarray:
    order = sorted(range(len(times)), key=lambda i: (float(times[i]), int(i)))
    raw = np.zeros((len(times), 4), dtype=np.float32)
    last_node_time: Dict[int, float] = {}
    last_pair_time: Dict[Tuple[int, int], float] = {}

    for eid in order:
        u = int(src[eid])
        v = int(dst[eid])
        t = float(times[eid])
        pair = (u, v)
        raw[eid, 0] = t
        raw[eid, 1] = 0.0 if u not in last_node_time else max(t - last_node_time[u], 0.0)
        raw[eid, 2] = 0.0 if v not in last_node_time else max(t - last_node_time[v], 0.0)
        raw[eid, 3] = 0.0 if pair not in last_pair_time else max(t - last_pair_time[pair], 0.0)
        last_node_time[u] = t
        last_node_time[v] = t
        last_pair_time[pair] = t

    encoder = TimeEncoder(time_dim)
    with torch.no_grad():
        features = encoder(torch.from_numpy(raw)).reshape(len(times), -1)
    return features.numpy().astype(np.float32, copy=False)
