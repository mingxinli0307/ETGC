import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from edge_data import load_edge_event_data
from edge_losses import (
    edge_ppr_proximity_loss, edge_ppr_proximity_loss_preindexed,
    edge_expected_structural_score_gain_loss_global, hierarchical_ncut_terms,
    trace_mincut_orthogonality_loss, projection_loss_global,
    scipy_csr_to_torch_sparse_coo,
)
from edge_metrics import evaluate_node_clustering
from edge_model import (
    EdgeHiNoSModel, initialize_cluster_output_from_prototypes,
    initialize_cluster_output_random_event, initialize_cluster_output_random_orthogonal,
    load_pretrained_node_features, tensor_checksum,
)
from edge_proximity import compute_edge_ppr_cached
from edge_time import build_edge_time_features
from utils import choose_device, ensure_dir


def _parameter_count(params) -> int:
    return int(sum(int(p.numel()) for p in params))


def _optimizer_group_summary(optimizer) -> list:
    result = []
    seen = set()
    duplicate_count = 0
    for index, group in enumerate(optimizer.param_groups):
        params = list(group.get("params", []))
        ids = [id(p) for p in params]
        duplicate_count += len(ids) - len(set(ids))
        for pid in ids:
            if pid in seen:
                duplicate_count += 1
            seen.add(pid)
        result.append(
            {
                "optimizer_group_name": group.get("name", f"group_{index}"),
                "parameter_count": _parameter_count(params),
                "learning_rate": float(group.get("lr", 0.0)),
            }
        )
    if duplicate_count:
        raise ValueError(f"Optimizer parameter groups contain duplicated parameters: duplicate_count={duplicate_count}")
    return result


def build_optimizer_for_node_emb_mode(model: EdgeHiNoSModel, lr: float, node_emb_mode: str, node_emb_lr: float):
    mode = str(node_emb_mode).lower()
    if mode not in {"frozen", "small_lr", "full"}:
        raise ValueError(f"Unsupported node_emb_mode: {node_emb_mode}")

    lr = float(lr)
    node_emb_lr = float(node_emb_lr)
    node_param = model.node_emb
    if mode == "frozen":
        node_param.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters remain after freezing node_emb.")
        optimizer = Adam([{"params": params, "lr": lr, "name": "non_node_parameters"}])
        info = {
            "node_emb_mode": mode,
            "node_emb_trainable": False,
            "node_emb_lr": 0.0,
            "other_lr": lr,
            "main_learning_rate": lr,
            "node_embedding_learning_rate": 0.0,
            "node_lr_ratio": 0.0,
            "param_group_lrs": [lr],
            "optimizer_groups": _optimizer_group_summary(optimizer),
        }
        return optimizer, info

    node_param.requires_grad_(True)
    node_id = id(node_param)
    other_params = [p for p in model.parameters() if id(p) != node_id and p.requires_grad]
    if mode == "small_lr":
        optimizer = Adam(
            [
                {"params": [node_param], "lr": node_emb_lr, "name": "node_emb"},
                {"params": other_params, "lr": lr, "name": "non_node_parameters"},
            ]
        )
        info = {
            "node_emb_mode": mode,
            "node_emb_trainable": True,
            "node_emb_lr": node_emb_lr,
            "other_lr": lr,
            "main_learning_rate": lr,
            "node_embedding_learning_rate": node_emb_lr,
            "node_lr_ratio": node_emb_lr / lr if lr != 0.0 else 0.0,
            "param_group_lrs": [lr, node_emb_lr],
            "optimizer_groups": _optimizer_group_summary(optimizer),
        }
        return optimizer, info

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam([{"params": params, "lr": lr, "name": "all_trainable_parameters"}])
    info = {
        "node_emb_mode": mode,
        "node_emb_trainable": True,
        "node_emb_lr": lr,
        "other_lr": lr,
        "main_learning_rate": lr,
        "node_embedding_learning_rate": lr,
        "node_lr_ratio": 1.0 if lr != 0.0 else 0.0,
        "param_group_lrs": [lr],
        "optimizer_groups": _optimizer_group_summary(optimizer),
    }
    return optimizer, info


class EdgeHiNoSTrainer:
    def __init__(self, args):
        self.args = args
        self.device = choose_device(args.device)
        self.model_seed = int(getattr(args, "model_seed", args.seed))
        self.prototype_seed = int(getattr(args, "prototype_seed", self.model_seed))
        self.forest_seed = int(getattr(args, "forest_seed", args.seed))
        self.rng = np.random.RandomState(self.model_seed)
        self.ncut_scope = "global"
        self.cluster_loss_type = "matrix_ncut"
        if args.ncut_scope != "global" or args.cluster_loss_type != "matrix_ncut":
            raise ValueError("Hierarchical C-form requires global matrix_ncut")
        if args.batch_size <= 0 or args.epoch <= 0:
            raise ValueError("batch_size and epoch must be positive")
        for name in ("lambda_fine_ncut", "lambda_coarse_ncut", "lambda_edge_ncut",
                     "lambda_orth", "lambda_esg", "lambda_proj", "lambda_prox"):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.data = load_edge_event_data(args.data_root, args.dataset)
        self.K = int(self.data.K)
        self.H = 2 * self.K if args.hier_ncut_h <= 0 else args.hier_ncut_h
        if self.H < self.K:
            raise ValueError(f"H must be >= K, got H={self.H}, K={self.K}")
        if not np.isfinite(args.sharpen_gamma) or args.sharpen_gamma < 1:
            raise ValueError("sharpen_gamma must be finite and >= 1")
        self.time_feat_np = build_edge_time_features(
            self.data.src,
            self.data.dst,
            self.data.times,
            int(args.time_dim),
            mode=str(getattr(args, "time_feature_mode", "current")),
        )

        self.P_E, self.Pi_E, self.Pi_cut, self.prox_stats = compute_edge_ppr_cached(
            dataset=args.dataset,
            src=self.data.src,
            dst=self.data.dst,
            times=self.data.times,
            num_nodes=self.data.num_nodes,
            cache_dir=args.cache_dir,
            method=args.edge_ppr_method,
            alpha=args.alpha,
            T=args.T,
            forest_samples=args.forest_samples,
            edge_neighbor_k=args.edge_neighbor_k,
            edge_ppr_topk=args.edge_ppr_topk,
            beta=args.beta,
            seed=self.forest_seed,
            affinity_sparsify=str(getattr(args, "affinity_sparsify", "symmetric_union_knn")),
            quiet=bool(int(getattr(args, "quiet", 0))),
        )
        self.Pi_E.sort_indices()
        self.Pi_cut.sort_indices()
        # Fixed CSR arrays used by the exact vectorized proximity batch
        # builder.  Pair selection still follows the shuffled anchor batches
        # and the original Pi_E row order.
        self._prox_indptr_np = np.asarray(self.Pi_E.indptr, dtype=np.int64)
        self._prox_indices_np = np.asarray(self.Pi_E.indices, dtype=np.int64)
        self._prox_data_np = np.asarray(self.Pi_E.data, dtype=np.float32)
        self._prox_global_to_local_np = np.full(self.data.num_events, -1, dtype=np.int64)
        # Pi_E is raw temporal edge PPR. Pi_cut is its symmetric, zero-diagonal
        # Ncut affinity. D_Pi is stored only as the matching row-sum vector.
        self.D_Pi_degree_np = np.asarray(self.Pi_cut.sum(axis=1)).ravel().astype(np.float32)
        self.Pi_cut_sparse_torch = None
        self.Pi_cut_sparse_mode = "scipy_row_block"
        if self.Pi_cut.nnz > 0:
            try:
                self.Pi_cut_sparse_torch = scipy_csr_to_torch_sparse_coo(self.Pi_cut, self.device, torch.float32)
                self.Pi_cut_sparse_mode = "torch_sparse_coo"
            except RuntimeError as exc:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    "Pi_cut_sparse_conversion=failed "
                    f"fallback=row_block_sparse_mm reason={type(exc).__name__}: {str(exc).splitlines()[0]}"
                )
        # Compatibility attributes for historical diagnostics/result readers.
        self.W_E = self.Pi_cut
        self.W_E_degree_np = self.D_Pi_degree_np
        self.W_E_sparse_torch = self.Pi_cut_sparse_torch
        self.W_E_sparse_mode = self.Pi_cut_sparse_mode

        self.edge_encoder_mode = str(getattr(args, "edge_encoder_mode", "mlp")).lower()
        if self.edge_encoder_mode not in {"mlp", "direct_node_time"}:
            raise ValueError(f"Unsupported edge_encoder_mode: {self.edge_encoder_mode}")
        self.cluster_head_type = str(getattr(args, "cluster_head_type", "legacy_mlp")).lower()
        self.prox_similarity_mode = str(getattr(args, "prox_similarity_mode", "event_dot")).lower()
        if self.prox_similarity_mode not in {"event_dot", "cosine", "role_aware"}:
            raise ValueError(f"Unsupported prox_similarity_mode: {self.prox_similarity_mode}")
        if self.prox_similarity_mode == "role_aware" and self.edge_encoder_mode != "direct_node_time":
            raise ValueError("prox_similarity_mode=role_aware requires edge_encoder_mode=direct_node_time")
        self.direct_time_scale = float(getattr(args, "direct_time_scale", 1.0))
        self.require_pretrained_node2vec = bool(int(getattr(args, "require_pretrained_node2vec", 0)))
        feature_path = self._resolve_feature_path()
        self.node_embedding_path = feature_path
        node_features = load_pretrained_node_features(
            feature_path,
            self.data.num_nodes,
            int(args.edge_dim),
            self.model_seed,
            require_existing=self.require_pretrained_node2vec,
        )
        self.node_embedding_source = "node2vec" if os.path.exists(feature_path) else "random_fallback"
        self.node_dim = int(node_features.shape[1])
        self.model = EdgeHiNoSModel(
            initial_node_features=node_features,
            time_dim=int(args.time_dim),
            edge_dim=int(args.edge_dim),
            edge_hidden_dim=int(args.edge_hidden_dim),
            cluster_hidden_dim=int(args.cluster_hidden_dim),
            K=self.K,
            hier_ncut_h=self.H,
            sharpen_gamma=args.sharpen_gamma,
            directed=bool(args.directed),
            cluster_output_bias_mode=getattr(args, "cluster_output_bias_mode", "default"),
            cluster_input_norm=getattr(args, "cluster_input_norm", "none"),
            edge_encoder_mode=self.edge_encoder_mode,
            direct_time_scale=self.direct_time_scale,
            cluster_head_type=getattr(args, "cluster_head_type", "legacy_mlp"),
            prototype_temperature=float(getattr(args, "prototype_temperature", 0.2)),
        ).to(self.device)
        self.event_repr_dim = int(self.model.event_repr_dim)
        self.cluster_input_dim = int(self.model.cluster_input_dim)

        self.src_t = torch.from_numpy(self.data.src).long().to(self.device)
        self.dst_t = torch.from_numpy(self.data.dst).long().to(self.device)
        self.time_feat_t = torch.from_numpy(self.time_feat_np).float().to(self.device)
        self.output_dir = str(args.output_dir or "")
        self.metrics_csv_path = os.path.join(self.output_dir, "metrics.csv") if self.output_dir else ""
        self.epoch_records = []
        self.model_init_info = self._current_cluster_output_stats()
        self.model_init_info.update(self._model_parameter_info())
        self._apply_node_embedding_requires_grad_state()
        self.prototype_initialization_pending = False
        if args.initialization_state_in:
            self._load_initialization_state(args.initialization_state_in)
            if args.apply_cluster_initialization_after_state_load:
                self._apply_cluster_initialization()
        elif self._should_defer_prototype_initialization():
            self.prototype_initialization_pending = True
        else:
            self._apply_cluster_initialization()
        if args.initialization_state_out:
            self._save_initialization_state(args.initialization_state_out)
        self.optimizer, self.node_emb_optimizer_info = build_optimizer_for_node_emb_mode(
            self.model, args.learning_rate, args.node_emb_mode, args.node_emb_lr)

    def _model_state_checksum(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()


    def _pi_cut_checksum(self) -> str:
        matrix = self.Pi_cut.tocsr()
        digest = hashlib.sha256()
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix.data, dtype=np.float64).tobytes())
        return digest.hexdigest()


    def _initialization_metadata(self) -> dict:
        return {
            "dataset": str(self.args.dataset),
            "M": int(self.data.num_events),
            "N": int(self.data.num_nodes),
            "K": int(self.K),
            "H": int(self.H),
            "sharpen_gamma": self.model.sharpen_gamma,
            "cluster_head_type": self.cluster_head_type,
            "edge_encoder_mode": self.edge_encoder_mode,
            "model_seed": int(self.model_seed),
            "prototype_seed": int(self.prototype_seed),
            "forest_seed": int(self.forest_seed),
            "model_state_checksum": self._model_state_checksum(),
            "Pi_cut_checksum": self._pi_cut_checksum(),
        }


    def _save_initialization_state(self, path: str) -> None:
        path = os.path.abspath(path)
        ensure_dir(os.path.dirname(path))
        metadata = self._initialization_metadata()
        self.model_init_info.update(
            {
                "initialization_state_source": "generated",
                "initialization_state_path": path,
                **metadata,
            }
        )
        payload = {
            "format": "etgc_hierarchical_initialization_state_v1",
            "metadata": metadata,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in self.model.state_dict().items()
            },
            "numpy_rng_state": self.rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self.device).cpu()
                if self.device.type == "cuda"
                else None
            ),
            "model_init_info": self.model_init_info,
        }
        torch.save(payload, path)
        print(
            "initialization_state_saved="
            f"{path} model_checksum={metadata['model_state_checksum']} "
            f"Pi_cut_checksum={metadata['Pi_cut_checksum']}"
        )


    def _load_initialization_state(self, path: str) -> None:
        path = os.path.abspath(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "etgc_hierarchical_initialization_state_v1":
            raise ValueError(f"Unsupported ETGC initialization snapshot: {path}")
        metadata = payload.get("metadata", {})
        expected = {
            "dataset": str(self.args.dataset),
            "M": int(self.data.num_events),
            "N": int(self.data.num_nodes),
            "K": int(self.K),
            "H": int(self.H),
            "sharpen_gamma": self.model.sharpen_gamma,
            "cluster_head_type": self.cluster_head_type,
            "edge_encoder_mode": self.edge_encoder_mode,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Initialization snapshot metadata mismatch: {mismatches}")
        affinity_checksum = self._pi_cut_checksum()
        if metadata.get("Pi_cut_checksum") != affinity_checksum:
            raise ValueError(
                "Initialization snapshot Pi_cut mismatch: "
                f"snapshot={metadata.get('Pi_cut_checksum')} current={affinity_checksum}"
            )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.rng.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
        loaded_checksum = self._model_state_checksum()
        if metadata.get("model_state_checksum") != loaded_checksum:
            raise ValueError(
                "Initialization snapshot model mismatch after load: "
                f"snapshot={metadata.get('model_state_checksum')} loaded={loaded_checksum}"
            )
        source_info = dict(payload.get("model_init_info") or {})
        self.model_init_info.update(source_info)
        self.model_init_info.update(
            {
                "initialization_state_source": "loaded",
                "initialization_state_path": path,
                "model_state_checksum": loaded_checksum,
                "Pi_cut_checksum": affinity_checksum,
            }
        )
        print(
            "initialization_state_loaded="
            f"{path} model_checksum={loaded_checksum} Pi_cut_checksum={affinity_checksum}"
        )


    def _resolve_feature_path(self) -> str:
        if getattr(self.args, "feature_path", ""):
            return self.args.feature_path
        candidates = [
            os.path.join(self.args.pretrain_emb_dir, f"{self.args.dataset}_feature.emb"),
            os.path.join(self.args.emb_root, f"{self.args.dataset}_feature.emb"),
            os.path.join(self.args.emb_root, self.args.dataset, f"{self.args.dataset}_feature.emb"),
        ]
        return next((path for path in candidates if os.path.exists(path)), candidates[0])


    def _apply_node_embedding_requires_grad_state(self) -> None:
        mode = str(getattr(self.args, "node_emb_mode", "full")).lower()
        if mode == "frozen":
            self.model.node_emb.requires_grad_(False)
        elif mode in {"small_lr", "full"}:
            self.model.node_emb.requires_grad_(True)
        else:
            raise ValueError(f"Unsupported node_emb_mode: {mode}")


    def _edge_mlp_parameters(self) -> list:
        if getattr(self.model, "edge_mlp", None) is None:
            return []
        return list(self.model.edge_mlp.parameters())


    def _model_parameter_info(self) -> dict:
        edge_mlp_params = self._edge_mlp_parameters()
        cluster_params = list(self.model.cluster_head.parameters())
        all_params = list(self.model.parameters())
        trainable_params = [p for p in all_params if p.requires_grad]
        return {
            "edge_encoder_mode": self.edge_encoder_mode,
            "cluster_head_type": self.cluster_head_type,
            "node_dim": int(self.node_dim),
            "time_dim": int(getattr(self.args, "time_dim", 0)),
            "event_repr_dim": int(self.event_repr_dim),
            "cluster_input_dim": int(self.cluster_input_dim),
            "edge_mlp_parameter_count": _parameter_count(edge_mlp_params),
            "edge_mlp_trainable_parameter_count": _parameter_count([p for p in edge_mlp_params if p.requires_grad]),
            "node_embedding_parameter_count": int(self.model.node_emb.numel()),
            "cluster_head_parameter_count": _parameter_count(cluster_params),
            "trainable_parameter_count": _parameter_count(trainable_params),
            "node_emb_trainable": bool(self.model.node_emb.requires_grad),
        }


    def _batch_union_ids(self, batch_ids: np.ndarray) -> np.ndarray:
        ids = set(int(i) for i in batch_ids.tolist())
        for eid in batch_ids.tolist():
            start, end = self.Pi_E.indptr[eid], self.Pi_E.indptr[eid + 1]
            ids.update(int(x) for x in self.Pi_E.indices[start:end].tolist())
        return np.asarray(sorted(ids), dtype=np.int64)


    def _proximity_cpu_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor


    def _prepare_proximity_batch(
        self,
        batch_ids: np.ndarray,
        rng: np.random.RandomState,
    ) -> dict:
        """Build the historical proximity pairs with vectorized CSR gathers."""
        started = time.perf_counter()
        batch_ids = np.asarray(batch_ids, dtype=np.int64)
        starts = self._prox_indptr_np[batch_ids]
        counts = self._prox_indptr_np[batch_ids + 1] - starts
        pair_positions = np.empty(0, dtype=np.int64)
        anchors_all = np.empty(0, dtype=np.int64)
        neighbors_all = np.empty(0, dtype=np.int64)
        total = int(counts.sum())
        if total:
            flat_offsets = np.arange(total, dtype=np.int64)
            segment_offsets = np.repeat(np.cumsum(counts, dtype=np.int64) - counts, counts)
            pair_positions = np.repeat(starts, counts) + flat_offsets - segment_offsets
            anchors_all = np.repeat(batch_ids, counts)
            neighbors_all = self._prox_indices_np[pair_positions]

        # np.unique preserves the sorted union produced by the old set/sorted
        # implementation, including neighbors later removed as self-pairs.
        union_ids = np.unique(np.concatenate([batch_ids, neighbors_all])).astype(
            np.int64,
            copy=False,
        )
        nonself = neighbors_all != anchors_all
        anchors_global = anchors_all[nonself]
        positives_global = neighbors_all[nonself]
        weights = self._prox_data_np[pair_positions[nonself]]

        mapping = self._prox_global_to_local_np
        mapping[union_ids] = np.arange(union_ids.size, dtype=np.int64)
        try:
            anchors_local = mapping[anchors_global].copy()
            positives_local = mapping[positives_global].copy()
            pair_count = int(anchors_local.size)
            if pair_count:
                neg_global = rng.randint(0, self.data.num_events, size=pair_count)
                # dict.get evaluates its default argument eagerly.  The old
                # list comprehension therefore consumed one fallback draw for
                # every pair, including negatives already inside the union.
                fallback_local = rng.randint(0, union_ids.size, size=pair_count)
                mapped_negative = mapping[neg_global]
                negatives_local = np.where(
                    mapped_negative >= 0,
                    mapped_negative,
                    fallback_local,
                ).astype(np.int64, copy=False)
            else:
                negatives_local = np.empty(0, dtype=np.int64)
        finally:
            mapping[union_ids] = -1

        prepared = {
            "union_ids": self._proximity_cpu_tensor(union_ids),
            "anchors": self._proximity_cpu_tensor(anchors_local),
            "positives": self._proximity_cpu_tensor(positives_local),
            "negatives": self._proximity_cpu_tensor(negatives_local),
            "weights": self._proximity_cpu_tensor(weights.astype(np.float32, copy=False)),
            "pair_count": int(anchors_local.size),
        }
        prepared["pair_build_seconds"] = time.perf_counter() - started
        return prepared


    def _proximity_batch_to_device(self, prepared: dict) -> dict:
        non_blocking = self.device.type == "cuda"
        return {
            key: prepared[key].to(self.device, non_blocking=non_blocking)
            for key in ("union_ids", "anchors", "positives", "negatives", "weights")
        }


    def _encode_ids(self, ids: np.ndarray) -> torch.Tensor:
        """Encode event ids without evaluating the cluster head or assignments."""
        ids_t = torch.from_numpy(ids).long().to(self.device)
        return self._encode_id_tensor(ids_t)


    def _encode_id_tensor(self, ids_t: torch.Tensor) -> torch.Tensor:
        """Encode an existing device index tensor without another host copy."""
        return self.model.encode_edge_events(
            self.src_t.index_select(0, ids_t),
            self.dst_t.index_select(0, ids_t),
            self.time_feat_t.index_select(0, ids_t),
        )


    def _proximity_loss_for_union(
        self,
        r_union: torch.Tensor,
        union_ids: np.ndarray,
        local_index: dict,
        batch_ids: np.ndarray,
        rng: np.random.RandomState,
    ) -> torch.Tensor:
        role_kwargs = {}
        if self.prox_similarity_mode == "role_aware":
            ids_t = torch.from_numpy(union_ids).long().to(self.device)
            role_kwargs = {
                "node_emb": self.model.node_emb,
                "src_union": self.src_t.index_select(0, ids_t),
                "dst_union": self.dst_t.index_select(0, ids_t),
                "time_feat_union": self.time_feat_t.index_select(0, ids_t),
            }
        return edge_ppr_proximity_loss(
            r_union,
            local_index,
            batch_ids,
            self.Pi_E,
            self.data.num_events,
            rng,
            self.device,
            similarity_mode=self.prox_similarity_mode,
            **role_kwargs,
            prox_role_ss_weight=float(getattr(self.args, "prox_role_ss_weight", 0.25)),
            prox_role_dd_weight=float(getattr(self.args, "prox_role_dd_weight", 0.25)),
            prox_role_ds_weight=float(getattr(self.args, "prox_role_ds_weight", 1.0)),
            prox_role_sd_weight=float(getattr(self.args, "prox_role_sd_weight", 0.0)),
            prox_role_time_weight=float(getattr(self.args, "prox_role_time_weight", 0.25)),
            prox_temperature=float(getattr(self.args, "prox_temperature", 0.2)),
        )


    def _proximity_loss_for_prepared(
        self,
        r_union: torch.Tensor,
        prepared: dict,
    ) -> torch.Tensor:
        role_kwargs = {}
        if self.prox_similarity_mode == "role_aware":
            ids_t = prepared["union_ids"]
            role_kwargs = {
                "node_emb": self.model.node_emb,
                "src_union": self.src_t.index_select(0, ids_t),
                "dst_union": self.dst_t.index_select(0, ids_t),
                "time_feat_union": self.time_feat_t.index_select(0, ids_t),
            }
        return edge_ppr_proximity_loss_preindexed(
            r_union,
            prepared["anchors"],
            prepared["positives"],
            prepared["negatives"],
            prepared["weights"],
            similarity_mode=self.prox_similarity_mode,
            **role_kwargs,
            prox_role_ss_weight=float(getattr(self.args, "prox_role_ss_weight", 0.25)),
            prox_role_dd_weight=float(getattr(self.args, "prox_role_dd_weight", 0.25)),
            prox_role_ds_weight=float(getattr(self.args, "prox_role_ds_weight", 1.0)),
            prox_role_sd_weight=float(getattr(self.args, "prox_role_sd_weight", 0.0)),
            prox_role_time_weight=float(getattr(self.args, "prox_role_time_weight", 0.25)),
            prox_temperature=float(getattr(self.args, "prox_temperature", 0.2)),
        )


    @torch.no_grad()
    def _forward_all_cluster_hidden_no_grad(self, chunk_size: int) -> torch.Tensor:
        self.model.eval()
        hidden_chunks = []
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            _, _, cluster_hidden = self._forward_event_tensors(
                self.src_t[start : start + chunk_size],
                self.dst_t[start : start + chunk_size],
                self.time_feat_t[start : start + chunk_size],
                return_cluster_hidden=True,
            )
            hidden_chunks.append(cluster_hidden.detach())
        return torch.cat(hidden_chunks, dim=0)


    @torch.no_grad()
    def _forward_all_edge_repr_no_grad(self, chunk_size: int) -> torch.Tensor:
        self.model.eval()
        repr_chunks = []
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            r, _ = self._forward_range(start, min(self.data.num_events, start + chunk_size))
            repr_chunks.append(r.detach())
        return torch.cat(repr_chunks, dim=0)


    def _current_cluster_output_stats(self) -> dict:
        bias = self.model.cluster_output.bias
        return {
            "output_bias_l2": 0.0 if bias is None else float(torch.linalg.norm(bias.detach()).cpu()),
            "cluster_output_weight_l2": float(torch.linalg.norm(self.model.cluster_output.weight.detach()).cpu()),
        }


    def _should_defer_prototype_initialization(self) -> bool:
        if str(getattr(self.args, "cluster_head_type", "legacy_mlp")).lower() != "cosine_prototype":
            return False
        if str(getattr(self.args, "prototype_init_mode", "kmeans_plus_plus")).lower() != "kmeans_plus_plus":
            return False
        return int(getattr(self.args, "prox_warmup_epochs", 0)) > 0


    def _apply_cluster_initialization(self) -> None:
        if self.cluster_head_type == "cosine_prototype":
            mode = str(getattr(self.args, "prototype_init_mode", "kmeans_plus_plus")).lower()
        else:
            mode = str(getattr(self.args, "cluster_init_mode", "random")).lower()
        if mode == "random":
            self.model_init_info["cluster_init_mode_effective"] = "random"
            self.model_init_info["initial_cluster_weight_checksum"] = tensor_checksum(
                self.model.cluster_output.weight.detach()
            )
            return
        if mode == "random_orthogonal":
            self.model_init_info.update(
                initialize_cluster_output_random_orthogonal(self.model, self.H, seed=self.prototype_seed)
            )
            return
        hidden = self._prototype_initialization_features()
        self.model_init_info["prototype_feature_checksum"] = tensor_checksum(hidden)
        self.model_init_info["prototype_feature_source"] = (
            "normalized_edge_repr_R" if self.cluster_head_type == "cosine_prototype" else "legacy_cluster_hidden"
        )
        if mode == "random_event":
            self.model_init_info.update(
                initialize_cluster_output_random_event(
                    self.model,
                    hidden,
                    K=self.H,
                    seed=self.prototype_seed,
                )
            )
            return
        if mode == "kmeans_plus_plus":
            self.model_init_info.update(
                initialize_cluster_output_from_prototypes(
                    self.model,
                    hidden,
                    K=self.H,
                    seed=self.prototype_seed,
                    sample_size=int(getattr(self.args, "prototype_sample_size", 20000)),
                    lloyd_iters=0,
                )
            )
            self.model_init_info["prototype_init_executed"] = True
            self.model_init_info["cluster_init_mode_effective"] = "kmeans_plus_plus"
            self.prototype_initialization_pending = False
            return
        if mode == "prototype":
            self.model_init_info.update(self._initialize_cluster_output_prototypes(hidden=hidden))
            self.model_init_info["prototype_init_executed"] = True
            self.prototype_initialization_pending = False
            return
        raise ValueError(f"Unsupported cluster_init_mode: {getattr(self.args, 'cluster_init_mode', None)}")


    def _initialize_cluster_output_prototypes(self, hidden=None) -> dict:
        if hidden is None:
            hidden = self._prototype_initialization_features()
        stats = initialize_cluster_output_from_prototypes(
            self.model,
            hidden,
            K=self.H,
            seed=self.prototype_seed,
            sample_size=int(getattr(self.args, "prototype_sample_size", 20000)),
            lloyd_iters=int(getattr(self.args, "prototype_lloyd_iters", 10)),
        )
        return stats


    def _prototype_initialization_features(self) -> torch.Tensor:
        if self.cluster_head_type == "cosine_prototype":
            edge_repr = self._forward_all_edge_repr_no_grad(int(self.args.global_q_chunk_size))
            return edge_repr / torch.linalg.norm(edge_repr, dim=1, keepdim=True).clamp_min(1e-12)
        return self._forward_all_cluster_hidden_no_grad(int(self.args.global_q_chunk_size))


    def _ensure_prototypes_initialized_after_warmup(self) -> None:
        if not getattr(self, "prototype_initialization_pending", False):
            return
        self._apply_cluster_initialization()


    @staticmethod
    def _q_distribution_stats(Q: np.ndarray, eps: float = 1e-12) -> dict:
        if Q.size == 0:
            return {
                "Q_mean_entropy": 0.0,
                "max_cluster_ratio": 0.0,
                "min_cluster_ratio": 0.0,
                "empty_cluster_count": 0,
            }
        Q = np.asarray(Q, dtype=np.float64)
        p = Q.mean(axis=0)
        entropy = -float(np.sum(Q * np.log(Q + float(eps))) / max(1, Q.shape[0]))
        return {
            "Q_mean_entropy": entropy,
            "max_cluster_ratio": float(p.max()) if p.size else 0.0,
            "min_cluster_ratio": float(p.min()) if p.size else 0.0,
            "empty_cluster_count": int(np.sum(p < 1e-4)),
        }


    def _evaluate_from_Q(self, Q: np.ndarray):
        S = np.zeros((self.data.num_nodes, self.K), dtype=np.float32)
        np.add.at(S, self.data.src, Q)
        np.add.at(S, self.data.dst, Q)
        S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-8)
        pred_y = S.argmax(axis=1)
        metrics = evaluate_node_clustering(self.data.labels, pred_y)
        metrics.update(self._q_distribution_stats(Q))
        edge_labels = Q.argmax(axis=1).astype(np.int64)
        edge_counts = np.bincount(edge_labels, minlength=self.K)[: self.K]
        node_counts = np.bincount(pred_y.astype(np.int64), minlength=self.K)[: self.K]
        metrics.update(
            {
                "edge_hard_active_clusters": int(np.sum(edge_counts > 0)),
                "edge_hard_largest_ratio": float(edge_counts.max() / max(1, int(edge_counts.sum()))),
                "node_hard_active_clusters": int(np.sum(node_counts > 0)),
                "node_hard_largest_ratio": float(node_counts.max() / max(1, int(node_counts.sum()))),
            }
        )
        return metrics, S


    def _evaluate_stage(self) -> dict:
        Q = self.infer_Q()
        metrics, _ = self._evaluate_from_Q(Q)
        return metrics


    def _init_metrics_csv(self) -> None:
        if not self.metrics_csv_path:
            return
        ensure_dir(self.output_dir)
        fieldnames = self._metrics_fieldnames()
        with open(self.metrics_csv_path, "w", encoding="utf-8", newline="") as writer:
            csv.DictWriter(writer, fieldnames=fieldnames).writeheader()


    def _append_epoch_record(self, record: dict) -> None:
        self.epoch_records.append(record)
        if not self.metrics_csv_path:
            return
        fieldnames = self._metrics_fieldnames()
        row = {key: record.get(key, "") for key in fieldnames}
        with open(self.metrics_csv_path, "a", encoding="utf-8", newline="") as writer:
            csv.DictWriter(writer, fieldnames=fieldnames).writerow(row)


    @torch.no_grad()
    def infer_Q(self) -> np.ndarray:
        self.model.eval()
        chunks = []
        chunk_size = max(1, int(self.args.batch_size) * 4)
        for start in range(0, self.data.num_events, chunk_size):
            _, q = self._forward_range(start, min(self.data.num_events, start + chunk_size))
            chunks.append(q.detach().cpu().numpy())
        return np.vstack(chunks)


    @torch.no_grad()
    def evaluate(self):
        Q = self.infer_Q()
        return self._evaluate_from_Q(Q)


    def _forward_event_tensors(self, src, dst, time_feat, **kwargs):
        return self.model(src, dst, time_feat, **kwargs)

    def _forward_range(self, start, end):
        return self.model(self.src_t[start:end], self.dst_t[start:end], self.time_feat_t[start:end])

    def forward_hierarchical(self):
        chunk_size = max(1, int(self.args.global_q_chunk_size))
        names = ("edge_repr_all", "z1_all", "q1_all", "p1_all")
        chunks = {name: [] for name in names}
        for start in range(0, self.data.num_events, chunk_size):
            end = min(start + chunk_size, self.data.num_events)
            out = self.model.forward_hierarchical(self.src_t[start:end], self.dst_t[start:end], self.time_feat_t[start:end])
            for name in names:
                chunks[name].append(out[name])
        result = {name: torch.cat(values, dim=0) for name, values in chunks.items()}
        result["q2"] = torch.softmax(self.model.coarse_assignment_logits, dim=1)
        result["q_final_all"] = result["p1_all"] @ result["q2"]
        return result

    def compute_global_objective(self, hierarchical):
        args = self.args
        q1_all = hierarchical["q1_all"]
        p1_all = hierarchical["p1_all"]
        q2 = hierarchical["q2"]
        q_final_all = hierarchical["q_final_all"]
        affinity = self.Pi_cut_sparse_torch if self.Pi_cut_sparse_torch is not None else self.Pi_cut
        fine_ncut_loss, coarse_ncut_loss, Pi_H, degree_H, diagnostics = hierarchical_ncut_terms(
            q1_all, p1_all, q2, affinity, self.D_Pi_degree_np,
            row_block_size=args.global_ncut_row_block_size)
        fine_orth_loss = trace_mincut_orthogonality_loss(p1_all, self.H)
        coarse_orth_loss = trace_mincut_orthogonality_loss(q2, self.K)
        esg_loss, _ = edge_expected_structural_score_gain_loss_global(
            p1_all, affinity, self.D_Pi_degree_np, tau=0.5, eps=1e-8,
            row_block_size=args.global_ncut_row_block_size)
        projection_loss = projection_loss_global(q_final_all, self.src_t, self.dst_t, self.data.num_nodes)
        hierarchical_cut_loss = (
            args.lambda_fine_ncut * fine_ncut_loss
            + args.lambda_coarse_ncut * coarse_ncut_loss
        )
        orth_loss = (
            fine_orth_loss
            + coarse_orth_loss
        )
        global_loss = (
            args.lambda_edge_ncut * hierarchical_cut_loss
            + args.lambda_orth * orth_loss
            + args.lambda_esg * esg_loss
            + args.lambda_proj * projection_loss
        )
        losses = dict(fine_ncut_loss=fine_ncut_loss, coarse_ncut_loss=coarse_ncut_loss,
                      hierarchical_cut_loss=hierarchical_cut_loss, fine_orth_loss=fine_orth_loss,
                      coarse_orth_loss=coarse_orth_loss, orth_loss=orth_loss,
                      esg_loss=esg_loss, projection_loss=projection_loss, global_loss=global_loss)
        for name, loss in losses.items():
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} is not finite")
        return global_loss, losses, diagnostics

    def _train_proximity_epoch(self, epoch):
        m = self.data.num_events
        batch_size = int(self.args.batch_size)
        lambda_prox = float(self.args.lambda_prox)
        quiet = bool(self.args.quiet)
        def sync_cuda():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        def scalar_value(value):
            return float(value.detach().cpu())
        self.model.train()
        prox_total_t = torch.zeros((), dtype=self.model.node_emb.dtype, device=self.device)
        weighted_prox_total_t = torch.zeros((), dtype=self.model.node_emb.dtype, device=self.device)
        prox_steps = 0
        prox_optimizer_steps = 0
        prox_pair_count = 0
        prox_pair_build_seconds = 0.0
        prox_pair_build_wait_seconds = 0.0
        prox_stage_seconds = {
            "h2d": 0.0,
            "encode": 0.0,
            "loss_forward": 0.0,
            "backward": 0.0,
            "optimizer_step": 0.0,
        }
        prox_cuda_events = {key: [] for key in prox_stage_seconds}

        def prox_stage_start():
            if self.device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event
            return time.perf_counter()

        def prox_stage_end(name: str, started) -> None:
            if self.device.type == "cuda":
                ended = torch.cuda.Event(enable_timing=True)
                ended.record()
                prox_cuda_events[name].append((started, ended))
            else:
                prox_stage_seconds[name] += time.perf_counter() - started

        sync_cuda()
        prox_start = time.time()
        if lambda_prox > 0.0:
            order = self.rng.permutation(m)
            batches = [order[start : start + batch_size] for start in range(0, m, batch_size)]
            with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
                prepared_future = prefetch_pool.submit(
                    self._prepare_proximity_batch,
                    batches[0],
                    self.rng,
                )
                iterator = tqdm(
                    range(len(batches)),
                    desc=f"ETGC epoch {epoch}",
                    leave=False,
                    disable=quiet,
                )
                for batch_index in iterator:
                    wait_started = time.perf_counter()
                    prepared_cpu = prepared_future.result()
                    prox_pair_build_wait_seconds += time.perf_counter() - wait_started
                    prox_pair_build_seconds += float(prepared_cpu["pair_build_seconds"])
                    prox_pair_count += int(prepared_cpu["pair_count"])
                    if batch_index + 1 < len(batches):
                        prepared_future = prefetch_pool.submit(
                            self._prepare_proximity_batch,
                            batches[batch_index + 1],
                            self.rng,
                        )

                    stage_started = prox_stage_start()
                    prepared = self._proximity_batch_to_device(prepared_cpu)
                    prox_stage_end("h2d", stage_started)

                    stage_started = prox_stage_start()
                    r_union = self._encode_id_tensor(prepared["union_ids"])
                    prox_stage_end("encode", stage_started)

                    stage_started = prox_stage_start()
                    l_prox = self._proximity_loss_for_prepared(r_union, prepared)
                    loss = lambda_prox * l_prox
                    prox_stage_end("loss_forward", stage_started)

                    self.optimizer.zero_grad(set_to_none=True)
                    if loss.requires_grad:
                        stage_started = prox_stage_start()
                        loss.backward()
                        prox_stage_end("backward", stage_started)

                        stage_started = prox_stage_start()
                        self.optimizer.step()
                        prox_stage_end("optimizer_step", stage_started)
                        prox_optimizer_steps += 1
                    prox_total_t += l_prox.detach()
                    weighted_prox_total_t += loss.detach()
                    prox_steps += 1
        sync_cuda()
        if self.device.type == "cuda":
            for name, event_pairs in prox_cuda_events.items():
                prox_stage_seconds[name] = sum(
                    started.elapsed_time(ended) for started, ended in event_pairs
                ) / 1000.0
        prox_forward_backward_seconds = time.time() - prox_start
        prox_total = scalar_value(prox_total_t)
        weighted_prox_total = scalar_value(weighted_prox_total_t)
        return dict(prox_loss=prox_total / max(1, prox_steps),
                    weighted_proximity_loss=weighted_prox_total / max(1, prox_steps),
                    prox_pair_count=prox_pair_count, prox_optimizer_steps=prox_optimizer_steps,
                    prox_forward_backward_seconds=prox_forward_backward_seconds,
                    prox_pair_build_seconds=prox_pair_build_seconds,
                    prox_pair_build_wait_seconds=prox_pair_build_wait_seconds,
                    **{f"prox_{key}_seconds": value for key, value in prox_stage_seconds.items()})

    def train(self):
        self._init_metrics_csv()
        best_epoch, best_metrics, best_key = -1, {}, None
        for epoch in range(1, int(self.args.epoch) + 1):
            started = time.time()
            record = dict(epoch=epoch, **self._train_proximity_epoch(epoch))
            if epoch > self.args.prox_warmup_epochs:
                self._ensure_prototypes_initialized_after_warmup()
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                hierarchical = self.forward_hierarchical()
                global_loss, losses, diagnostics = self.compute_global_objective(hierarchical)
                global_loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient: {name}")
                self.optimizer.step()
                record.update({name: float(value.detach().cpu()) for name, value in losses.items()})
                record.update(diagnostics)
                del global_loss, losses, hierarchical
            metrics = self._evaluate_stage()
            record.update({key: metrics[key] for key in ("ACC", "Macro_F1", "NMI", "ARI")})
            record["epoch_seconds"] = time.time() - started
            self._append_epoch_record(record)
            key = metrics["Macro_F1"]
            if best_key is None or key > best_key:
                best_epoch, best_metrics, best_key = epoch, metrics.copy(), key
            self.final_metrics = metrics.copy()
            print(" ".join(f"{key}={value:.8g}" if isinstance(value, float) else f"{key}={value}"
                           for key, value in record.items()), flush=True)
        if self.args.save_embeddings:
            self.save_outputs()
        if self.output_dir:
            with open(os.path.join(self.output_dir, "diagnostic.json"), "w", encoding="utf-8") as writer:
                json.dump({"M": self.data.num_events, "H": self.H, "K": self.K,
                           "epochs": self.epoch_records}, writer, indent=2)
        return best_epoch, best_metrics

    @staticmethod
    def _metrics_fieldnames():
        return ["epoch", "ACC", "Macro_F1", "NMI", "ARI", "fine_ncut_loss", "coarse_ncut_loss",
                "hierarchical_cut_loss", "fine_orth_loss", "coarse_orth_loss", "orth_loss",
                "esg_loss", "projection_loss", "global_loss", "q1_softness", "p1_softness",
                "coarse_discrepancy_trace", "prox_loss", "weighted_proximity_loss",
                "prox_pair_count", "prox_optimizer_steps", "prox_forward_backward_seconds",
                "prox_pair_build_seconds", "prox_pair_build_wait_seconds", "prox_h2d_seconds",
                "prox_encode_seconds", "prox_loss_forward_seconds", "prox_backward_seconds",
                "prox_optimizer_step_seconds", "epoch_seconds"]

    def write_config_json(self):
        if not self.output_dir:
            return
        ensure_dir(self.output_dir)
        config = dict(vars(self.args), M=self.data.num_events, N=self.data.num_nodes, H=self.H, K=self.K,
                      model_init_info=self.model_init_info, prox_stats=self.prox_stats,
                      node_emb_optimizer_info=self.node_emb_optimizer_info)
        with open(os.path.join(self.output_dir, "config.json"), "w", encoding="utf-8") as writer:
            json.dump(config, writer, indent=2, default=str)

    def write_result_json(self, best_epoch, best_metrics, final_metrics, runtime_seconds):
        if not self.output_dir:
            return
        result = dict(status="success", dataset=self.args.dataset, seed=self.model_seed,
                      M=self.data.num_events, H=self.H, K=self.K, best_epoch=best_epoch,
                      best_metrics=best_metrics, final_metrics=final_metrics,
                      runtime_seconds=runtime_seconds)
        with open(os.path.join(self.output_dir, "result.json"), "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=2)

    def save_outputs(self):
        if not self.output_dir:
            raise ValueError("save_embeddings requires a fresh output_dir")
        np.save(os.path.join(self.output_dir, "etgc_Q_final.npy"), self.infer_Q())
        _, S_final = self.evaluate()
        np.save(os.path.join(self.output_dir, "etgc_S_final.npy"), S_final)
