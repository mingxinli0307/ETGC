import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EdgeEventData:
    src: np.ndarray
    dst: np.ndarray
    times: np.ndarray
    edge_ids: np.ndarray
    num_nodes: int
    num_events: int
    labels: np.ndarray
    K: int
    node_id_map: Dict[int, int]


def _read_raw_edges(path: str) -> List[Tuple[int, int, float, int]]:
    events = []
    with open(path, "r", encoding="utf-8") as reader:
        for eid, line in enumerate(reader):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                u = int(float(parts[0]))
                v = int(float(parts[1]))
                t = float(parts[2])
            except ValueError:
                continue
            if u == v:
                continue
            events.append((u, v, t, eid))
    return events


def _read_raw_labels(path: str) -> Dict[int, int]:
    labels = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            labels[int(float(parts[0]))] = int(float(parts[1]))
    return labels


def load_edge_event_data(data_root: str, dataset: str) -> EdgeEventData:
    edge_path = os.path.join(data_root, dataset, f"{dataset}.txt")
    label_path = os.path.join(data_root, dataset, "node2label.txt")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing temporal edge file: {edge_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")

    raw_edges = _read_raw_edges(edge_path)
    raw_labels = _read_raw_labels(label_path)
    if not raw_edges:
        raise ValueError(f"No valid temporal edge events found in {edge_path}")
    if not raw_labels:
        raise ValueError(f"No labels found in {label_path}")

    node_ids = sorted(set(raw_labels.keys()) | {u for u, _, _, _ in raw_edges} | {v for _, v, _, _ in raw_edges})
    is_contiguous = node_ids == list(range(len(node_ids)))
    node_id_map = {nid: nid for nid in node_ids} if is_contiguous else {nid: i for i, nid in enumerate(node_ids)}
    if not is_contiguous:
        print(f"Remapped non-contiguous node ids to 0..{len(node_ids) - 1}.")

    src = np.asarray([node_id_map[u] for u, _, _, _ in raw_edges], dtype=np.int64)
    dst = np.asarray([node_id_map[v] for _, v, _, _ in raw_edges], dtype=np.int64)
    times = np.asarray([t for _, _, t, _ in raw_edges], dtype=np.float32)
    edge_ids = np.arange(len(raw_edges), dtype=np.int64)

    labels = np.full((len(node_ids),), -1, dtype=np.int64)
    for old_id, label in raw_labels.items():
        if old_id in node_id_map:
            labels[node_id_map[old_id]] = label
    if np.any(labels < 0):
        missing = int(np.sum(labels < 0))
        raise ValueError(f"{missing} nodes are missing labels after id remapping.")

    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels.tolist())}
    labels = np.asarray([label_map[int(label)] for label in labels], dtype=np.int64)

    return EdgeEventData(
        src=src,
        dst=dst,
        times=times,
        edge_ids=edge_ids,
        num_nodes=len(node_ids),
        num_events=len(raw_edges),
        labels=labels,
        K=int(len(unique_labels)),
        node_id_map=node_id_map,
    )
