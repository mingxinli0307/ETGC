import hashlib
import os

import numpy as np
import torch


DEFAULT_NEG_TABLE_SIZE = int(1e6)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def hash_cfg(cfg: dict) -> str:
    items = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg.keys()))
    return hashlib.md5(items.encode("utf-8")).hexdigest()[:10]


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def choose_device(device: str) -> torch.device:
    device = str(device).strip().lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return torch.device("cpu")

    if device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU.")
            return torch.device("cpu")

        if device == "cuda":
            device = "cuda:0"

        resolved = torch.device(device)
        visible_gpu_count = torch.cuda.device_count()
        if resolved.index is None or resolved.index < 0 or resolved.index >= visible_gpu_count:
            raise ValueError(
                f"Invalid CUDA device {device}. "
                f"Visible CUDA device count is {visible_gpu_count}."
            )

        torch.cuda.set_device(resolved.index)
        return resolved

    return torch.device(device)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
