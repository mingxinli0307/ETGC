"""Shared inventory helpers for the standalone node-embedding baseline."""
import csv
import json
import os
from pathlib import Path
from edge_data import load_edge_event_data
from edge_time import build_edge_time_features
from utils import hash_cfg

FOREST_SEED = 20260725
COMMON_CONFIG = {'directed': 0, 'batch_size': 512, 'epoch': 30, 'learning_rate': 0.0001, 'edge_dim': 128, 'time_dim': 32, 'edge_hidden_dim': 128, 'cluster_hidden_dim': 64, 'alpha': 0.2, 'T': 4, 'beta': 5.0, 'edge_neighbor_k': -1, 'edge_ppr_topk': -1, 'edge_ppr_method': 'temporal_state_forest', 'forest_samples': 50, 'global_q_chunk_size': 8192, 'global_ncut_row_block_size': 65536, 'lambda_edge_ncut': 0.5, 'lambda_proj': 0.0, 'node_emb_lr': 1e-05, 'cluster_output_bias_mode': 'none', 'cluster_input_norm': 'layernorm', 'require_pretrained_node2vec': 1}

def resolve_node2vec_path(root: Path, dataset: str, feature_path: str = "") -> str:
    if feature_path:
        return str(Path(feature_path).resolve())
    candidates = [
        root / "pretrain" / f"{dataset}_feature.emb",
        root / "emb" / f"{dataset}_feature.emb",
        root / "emb" / dataset / f"{dataset}_feature.emb",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])

def inspect_node2vec_file(path: str, num_nodes: int) -> dict:
    payload = {
        "node2vec_path": path,
        "node2vec_exists": os.path.exists(path),
        "node2vec_shape": None,
        "node2vec_dtype": "",
        "node2vec_complete": False,
        "node2vec_missing_count": None,
    }
    if not payload["node2vec_exists"]:
        return payload
    rows = {}
    with open(path, "r", encoding="utf-8") as reader:
        first = reader.readline().lstrip("\ufeff").strip().split()
        has_header = len(first) == 2 and all(tok.lstrip("-").isdigit() for tok in first)
        if first and not has_header:
            arr = [float(x) for x in first]
            rows[int(arr[0])] = arr[1:]
        for line in reader:
            line = line.strip()
            if not line:
                continue
            arr = [float(x) for x in line.split()]
            if len(arr) > 1:
                rows[int(arr[0])] = arr[1:]
    dim = len(next(iter(rows.values()))) if rows else 0
    missing = [nid for nid in range(int(num_nodes)) if nid not in rows or len(rows[nid]) != dim]
    payload.update(
        {
            "node2vec_shape": [int(num_nodes), int(dim)] if not missing and dim > 0 else [len(rows), int(dim)],
            "node2vec_dtype": "float32",
            "node2vec_complete": len(missing) == 0 and dim > 0,
            "node2vec_missing_count": len(missing),
        }
    )
    return payload

def ppr_cache_paths(root: Path, dataset: str, data, common: dict) -> dict:
    method = str(common["edge_ppr_method"])
    forest_impl = {
        "temporal_state_forest": "state_expanded_temporal_subdivision_forest",
        "forest": "state_expanded_temporal_subdivision_forest_alias",
        "legacy_temporal_forest": "legacy_original_node_temporal_forest",
        "truncated": "temporal_edge_event_truncated_ppr",
    }[method]
    cfg = {
        "dataset": dataset,
        "method": method,
        "alpha": float(common["alpha"]),
        "T": int(common["T"]),
        "forest_samples": int(common["forest_samples"]),
        "edge_neighbor_k": int(common["edge_neighbor_k"]),
        "edge_ppr_topk": int(common["edge_ppr_topk"]),
        "beta": float(common["beta"]),
        "seed": int(FOREST_SEED),
        "num_events": int(data.num_events),
        "forest_impl": forest_impl,
    }
    cfg_hash = hash_cfg(cfg)
    ds_cache = root / "cache" / dataset
    pi_path = ds_cache / f"edge_ppr_{method}_{cfg_hash}.npz"
    w_path = ds_cache / f"edge_ncut_affinity_{method}_{cfg_hash}.npz"
    return {
        "proximity_cache_hash": cfg_hash,
        "proximity_cache_exists": pi_path.exists() and w_path.exists(),
        "pi_cache_path": str(pi_path),
        "w_cache_path": str(w_path),
    }

def discover_datasets(root: Path, common: dict) -> list:
    data_root = root / "dataset"
    rows = []
    if not data_root.exists():
        return [
            {
                "dataset_name": "",
                "dataset_path": str(data_root),
                "usable": False,
                "failure_reason": "dataset root does not exist",
            }
        ]
    for child in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        name = child.name
        row = {
            "dataset_name": name,
            "dataset_path": str(child),
            "node_count": "",
            "event_count": "",
            "timestamp_count": "",
            "class_count": "",
            "node2vec_path": resolve_node2vec_path(root, name),
            "node2vec_exists": False,
            "node2vec_shape": None,
            "node2vec_dtype": "",
            "time_feature_shape": None,
            "proximity_cache_exists": False,
            "usable": False,
            "failure_reason": "",
        }
        if not (child / f"{name}.txt").exists():
            row["failure_reason"] = f"missing edge file: {child / (name + '.txt')}"
            rows.append(row)
            continue
        if not (child / "node2label.txt").exists():
            row["failure_reason"] = f"missing label file: {child / 'node2label.txt'}"
            rows.append(row)
            continue
        try:
            data = load_edge_event_data(str(data_root), name)
            time_feat = build_edge_time_features(data.src, data.dst, data.times, int(common["time_dim"]))
            row.update(
                {
                    "node_count": int(data.num_nodes),
                    "event_count": int(data.num_events),
                    "timestamp_count": int(len(set(float(x) for x in data.times.tolist()))),
                    "class_count": int(data.K),
                    "time_feature_shape": [int(x) for x in time_feat.shape],
                }
            )
            row.update(inspect_node2vec_file(row["node2vec_path"], data.num_nodes))
            row.update(ppr_cache_paths(root, name, data, common))
            if not row["node2vec_exists"]:
                row["failure_reason"] = f"missing Node2Vec embedding file: {row['node2vec_path']}"
            elif not row["node2vec_complete"]:
                row["failure_reason"] = (
                    f"incomplete Node2Vec embedding file: missing_or_bad_count={row['node2vec_missing_count']}"
                )
            else:
                row["usable"] = True
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    rows.sort(key=lambda x: (not bool(x.get("usable")), int(x.get("event_count") or 10**18), str(x.get("dataset_name"))))
    return rows

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

def write_csv(path: Path, rows: list, fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({key: row.get(key, "") for key in fieldnames})
