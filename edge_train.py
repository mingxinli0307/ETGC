import csv
import json
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from edge_data import load_edge_event_data
from edge_losses import (
    balance_loss,
    edge_orthqa_penalty_global,
    edge_ncut_loss,
    edge_ncut_loss_global,
    edge_ppr_proximity_loss,
    edge_trace_mincut_loss_global,
    node_embedding_anchor_loss,
    projection_loss,
    projection_loss_global,
    scipy_csr_to_torch_sparse_coo,
)
from edge_metrics import evaluate_node_clustering
from edge_model import (
    EdgeHiNoSModel,
    assign_all_to_centers,
    fit_kmeans_centers,
    initialize_cluster_output_from_prototypes,
    initialize_cluster_output_random_event,
    initialize_cluster_output_random_orthogonal,
    load_pretrained_node_features,
    tensor_checksum,
)
from edge_proximity import compute_edge_ppr_cached
from edge_time import build_edge_time_features
from edge_uniform_diagnostic import (
    cluster_head_gradient_diagnostics,
    cluster_volume_statistics,
    compute_uniform_collapse_stage,
    edge_repr_block_norm_statistics,
    node_event_degree_drift_statistics,
    node_embedding_drift_statistics,
    print_stage_summary,
    uniform_delta,
    write_diagnostic_outputs,
)
from utils import choose_device, ensure_dir


def edge_hard_labels_to_node_predictions(edge_labels: np.ndarray, src: np.ndarray, dst: np.ndarray, num_nodes: int, K: int) -> np.ndarray:
    edge_labels = np.asarray(edge_labels, dtype=np.int64).reshape(-1)
    q_hard = np.zeros((int(edge_labels.shape[0]), int(K)), dtype=np.float32)
    q_hard[np.arange(edge_labels.shape[0]), edge_labels] = 1.0
    S = np.zeros((int(num_nodes), int(K)), dtype=np.float32)
    np.add.at(S, np.asarray(src, dtype=np.int64), q_hard)
    np.add.at(S, np.asarray(dst, dtype=np.int64), q_hard)
    S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-8)
    return S.argmax(axis=1).astype(np.int64)


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
        self.ncut_scope = str(getattr(args, "ncut_scope", "batch")).lower()
        if self.ncut_scope not in {"batch", "global"}:
            raise ValueError(f"Unsupported ncut_scope: {self.ncut_scope}")
        self.cluster_loss_type = str(getattr(args, "cluster_loss_type", "trace_mincut")).lower()
        if self.cluster_loss_type not in {"trace_mincut", "legacy_ncut"}:
            raise ValueError(f"Unsupported cluster_loss_type: {self.cluster_loss_type}")
        if self.cluster_loss_type == "trace_mincut" and self.ncut_scope != "global":
            raise ValueError("cluster_loss_type=trace_mincut requires ncut_scope=global because Q_all and W_E are global.")

        self.data = load_edge_event_data(args.data_root, args.dataset)
        self.K = int(self.data.K)
        self.time_feat_np = build_edge_time_features(
            self.data.src, self.data.dst, self.data.times, int(args.time_dim)
        )

        self.P_E, self.Pi_E, self.W_E, self.prox_stats = compute_edge_ppr_cached(
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
            quiet=bool(int(getattr(args, "quiet", 0))),
        )
        self.Pi_E.sort_indices()
        self.W_E.sort_indices()
        self.W_E_degree_np = np.asarray(self.W_E.sum(axis=1)).ravel().astype(np.float32)
        self.W_E_sparse_torch = None
        self.W_E_sparse_mode = "scipy_row_block"
        if self.ncut_scope == "global" and self.cluster_loss_type == "trace_mincut" and self.W_E.nnz > 0:
            try:
                self.W_E_sparse_torch = scipy_csr_to_torch_sparse_coo(self.W_E, self.device, torch.float32)
                self.W_E_sparse_mode = "torch_sparse_coo"
            except RuntimeError as exc:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    "trace_mincut_sparse_conversion=failed "
                    f"fallback=row_block_sparse_mm reason={type(exc).__name__}: {str(exc).splitlines()[0]}"
                )

        self.edge_encoder_mode = str(getattr(args, "edge_encoder_mode", "mlp")).lower()
        if self.edge_encoder_mode not in {"mlp", "direct_node_time"}:
            raise ValueError(f"Unsupported edge_encoder_mode: {self.edge_encoder_mode}")
        self.prox_similarity_mode = str(getattr(args, "prox_similarity_mode", "event_dot")).lower()
        if self.prox_similarity_mode not in {"event_dot", "role_aware"}:
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
            directed=bool(args.directed),
            cluster_output_bias_mode=getattr(args, "cluster_output_bias_mode", "default"),
            cluster_input_norm=getattr(args, "cluster_input_norm", "none"),
            edge_encoder_mode=self.edge_encoder_mode,
            direct_time_scale=self.direct_time_scale,
        ).to(self.device)
        self.event_repr_dim = int(self.model.event_repr_dim)
        self.cluster_input_dim = int(self.model.cluster_input_dim)

        self.src_t = torch.from_numpy(self.data.src).long().to(self.device)
        self.dst_t = torch.from_numpy(self.data.dst).long().to(self.device)
        self.time_feat_t = torch.from_numpy(self.time_feat_np).float().to(self.device)
        self.output_dir = str(getattr(args, "output_dir", "") or "")
        self.metrics_csv_path = os.path.join(self.output_dir, "metrics.csv") if self.output_dir else ""
        self.epoch_records = []
        self.best_epoch_record = None
        self.overnight_diagnostic = bool(int(getattr(args, "overnight_diagnostic", 0)))
        self.uniform_collapse_diagnostic = bool(int(getattr(args, "uniform_collapse_diagnostic", 0))) or self.overnight_diagnostic
        self.diagnostic_only_first_epoch = bool(int(getattr(args, "diagnostic_only_first_epoch", 1)))
        self.uniform_diag_dir = self._resolve_uniform_diag_dir()
        self.uniform_diag_stages = {}
        self.uniform_diag_delta = {}
        self.uniform_diag_after_first_done = False
        self.diagnostic_epochs = self._parse_diagnostic_epochs(getattr(args, "diagnostic_epochs", "1,5,10,20,30"))
        self.init_reference = None
        self.init_reference_weight = None
        self.init_reference_metrics = None
        self.node_emb_initial_cpu = None
        if (
            self.edge_encoder_mode == "direct_node_time"
            and str(getattr(args, "node_emb_mode", "full")).lower() != "frozen"
        ) or self.uniform_collapse_diagnostic or self.overnight_diagnostic or float(getattr(args, "lambda_node_anchor", 0.0)) > 0.0:
            self.node_emb_initial_cpu = self.model.node_emb.detach().cpu().clone()
        self._apply_node_embedding_requires_grad_state()
        self.model_init_info = self._current_cluster_output_stats()
        self.model_init_info.update(self._model_parameter_info())
        self.model_init_info["node_embedding_source"] = self.node_embedding_source
        self.model_init_info["node2vec_path"] = self.node_embedding_path
        self.model_init_info["node2vec_shape"] = [int(x) for x in node_features.shape]
        self.model_init_info["cluster_output_bias_mode"] = str(getattr(args, "cluster_output_bias_mode", "default"))
        self.model_init_info["cluster_input_norm"] = str(getattr(args, "cluster_input_norm", "none"))
        self.model_init_info["cluster_init_mode"] = str(getattr(args, "cluster_init_mode", "random"))
        self.model_init_info["edge_encoder_mode"] = self.edge_encoder_mode
        self.model_init_info["cluster_output_bias_l2_initial"] = self.model_init_info["output_bias_l2"]
        self.model_init_info["cluster_output_weight_l2_initial"] = self.model_init_info["cluster_output_weight_l2"]
        self.model_init_info["prototype_init_executed"] = False
        self._record_uniform_stage_no_grad("after_model_initialization")
        self._apply_cluster_initialization()
        self.model_init_info["cluster_output_bias_l2_after_cluster_initialization"] = self._current_cluster_output_stats()["output_bias_l2"]
        self.model_init_info["cluster_output_weight_l2_after_cluster_initialization"] = self._current_cluster_output_stats()["cluster_output_weight_l2"]
        self._record_uniform_stage_no_grad("after_cluster_initialization")
        self._capture_init_reference()
        if bool(int(getattr(args, "direct_kmeans_eval", 0))):
            self.optimizer = None
            self.node_emb_optimizer_info = {
                "node_emb_mode": str(getattr(args, "node_emb_mode", "full")),
                "node_emb_trainable": bool(self.model.node_emb.requires_grad),
                "node_emb_lr": 0.0,
                "other_lr": 0.0,
                "param_group_lrs": [],
                "optimizer_created": False,
            }
        else:
            self.optimizer, self.node_emb_optimizer_info = build_optimizer_for_node_emb_mode(
                self.model,
                lr=float(args.learning_rate),
                node_emb_mode=getattr(args, "node_emb_mode", "full"),
                node_emb_lr=float(getattr(args, "node_emb_lr", 1e-5)),
            )
        self.model_init_info.update(self._model_parameter_info())

    @staticmethod
    def _parse_diagnostic_epochs(value) -> set:
        if value is None:
            return set()
        result = set()
        for part in str(value).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            result.add(int(part))
        return result

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

    def _forward_ids(self, ids: np.ndarray):
        ids_t = torch.from_numpy(ids).long().to(self.device)
        return self.model(
            self.src_t.index_select(0, ids_t),
            self.dst_t.index_select(0, ids_t),
            self.time_feat_t.index_select(0, ids_t),
        )

    def _forward_ids_with_logits(self, ids: np.ndarray):
        ids_t = torch.from_numpy(ids).long().to(self.device)
        return self.model(
            self.src_t.index_select(0, ids_t),
            self.dst_t.index_select(0, ids_t),
            self.time_feat_t.index_select(0, ids_t),
            return_logits=True,
        )

    def _forward_ids_diagnostic(self, ids: np.ndarray):
        ids_t = torch.from_numpy(ids).long().to(self.device)
        r, q, logits, _cluster_hidden = self.model(
            self.src_t.index_select(0, ids_t),
            self.dst_t.index_select(0, ids_t),
            self.time_feat_t.index_select(0, ids_t),
            return_logits=True,
            return_cluster_hidden=True,
        )
        cluster_input = self.model.cluster_input_norm(r)
        return r, q, logits, cluster_input

    def _proximity_loss_for_union(
        self,
        r_union: torch.Tensor,
        union_ids: np.ndarray,
        local_index: dict,
        batch_ids: np.ndarray,
        rng: np.random.RandomState,
    ) -> torch.Tensor:
        ids_t = torch.from_numpy(union_ids).long().to(self.device)
        return edge_ppr_proximity_loss(
            r_union,
            local_index,
            batch_ids,
            self.Pi_E,
            self.data.num_events,
            rng,
            self.device,
            similarity_mode=self.prox_similarity_mode,
            node_emb=self.model.node_emb,
            src_union=self.src_t.index_select(0, ids_t),
            dst_union=self.dst_t.index_select(0, ids_t),
            time_feat_union=self.time_feat_t.index_select(0, ids_t),
            prox_role_ss_weight=float(getattr(self.args, "prox_role_ss_weight", 0.25)),
            prox_role_dd_weight=float(getattr(self.args, "prox_role_dd_weight", 0.25)),
            prox_role_ds_weight=float(getattr(self.args, "prox_role_ds_weight", 1.0)),
            prox_role_sd_weight=float(getattr(self.args, "prox_role_sd_weight", 0.0)),
            prox_role_time_weight=float(getattr(self.args, "prox_role_time_weight", 0.25)),
            prox_temperature=float(getattr(self.args, "prox_temperature", 0.2)),
        )

    def _node_emb_snapshot_cpu(self):
        return self.model.node_emb.detach().cpu().clone()

    def _node_update_from_snapshot(self, snapshot_cpu) -> float:
        if snapshot_cpu is None:
            return 0.0
        current = self.model.node_emb.detach().cpu()
        diff = current - snapshot_cpu.to(dtype=current.dtype)
        return float(torch.linalg.norm(diff, ord="fro").item() / (max(1, int(current.size(0))) ** 0.5))

    def _cluster_head_snapshot_cpu(self) -> list:
        return [p.detach().cpu().clone() for p in self.model.cluster_head.parameters()]

    def _cluster_update_from_snapshot(self, snapshot_cpu: list) -> float:
        if not snapshot_cpu:
            return 0.0
        total_sq = 0.0
        for param, before in zip(self.model.cluster_head.parameters(), snapshot_cpu):
            diff = param.detach().cpu() - before.to(dtype=param.detach().cpu().dtype)
            total_sq += float(diff.square().sum().item())
        return total_sq ** 0.5

    def _node_anchor_loss(self) -> torch.Tensor:
        initial = getattr(self.model, "node_emb_initial", None)
        if initial is None:
            return self._zero_scalar()
        return node_embedding_anchor_loss(self.model.node_emb, initial)

    def _forward_all_q_with_grad(self, chunk_size: int) -> torch.Tensor:
        chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            _, q = self._forward_ids(ids[start : start + chunk_size])
            chunks.append(q)
        return torch.cat(chunks, dim=0)

    def _forward_all_q_logits_with_grad(self, chunk_size: int) -> tuple:
        r_chunks = []
        q_chunks = []
        logits_chunks = []
        cluster_input_chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            r, q, logits, cluster_input = self._forward_ids_diagnostic(ids[start : start + chunk_size])
            r_chunks.append(r)
            q_chunks.append(q)
            logits_chunks.append(logits)
            cluster_input_chunks.append(cluster_input)
        return (
            torch.cat(q_chunks, dim=0),
            torch.cat(logits_chunks, dim=0),
            torch.cat(r_chunks, dim=0),
            torch.cat(cluster_input_chunks, dim=0),
        )

    @torch.no_grad()
    def _forward_all_q_logits_no_grad(self, chunk_size: int) -> tuple:
        self.model.eval()
        r_chunks = []
        q_chunks = []
        logits_chunks = []
        cluster_input_chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            r, q, logits, cluster_input = self._forward_ids_diagnostic(ids[start : start + chunk_size])
            r_chunks.append(r.detach())
            q_chunks.append(q.detach())
            logits_chunks.append(logits.detach())
            cluster_input_chunks.append(cluster_input.detach())
        return (
            torch.cat(q_chunks, dim=0),
            torch.cat(logits_chunks, dim=0),
            torch.cat(r_chunks, dim=0),
            torch.cat(cluster_input_chunks, dim=0),
        )

    @torch.no_grad()
    def _forward_all_cluster_hidden_no_grad(self, chunk_size: int) -> torch.Tensor:
        self.model.eval()
        hidden_chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            _, _, cluster_hidden = self.model(
                self.src_t.index_select(0, torch.from_numpy(ids[start : start + chunk_size]).long().to(self.device)),
                self.dst_t.index_select(0, torch.from_numpy(ids[start : start + chunk_size]).long().to(self.device)),
                self.time_feat_t.index_select(
                    0, torch.from_numpy(ids[start : start + chunk_size]).long().to(self.device)
                ),
                return_cluster_hidden=True,
            )
            hidden_chunks.append(cluster_hidden.detach())
        return torch.cat(hidden_chunks, dim=0)

    def _zero_scalar(self) -> torch.Tensor:
        return next(self.model.parameters()).sum() * 0.0

    def _current_cluster_output_stats(self) -> dict:
        bias = self.model.cluster_output.bias
        return {
            "output_bias_l2": 0.0 if bias is None else float(torch.linalg.norm(bias.detach()).cpu()),
            "cluster_output_weight_l2": float(torch.linalg.norm(self.model.cluster_output.weight.detach()).cpu()),
        }

    def _apply_cluster_initialization(self) -> None:
        mode = str(getattr(self.args, "cluster_init_mode", "random")).lower()
        if mode == "random":
            self.model_init_info["cluster_init_mode_effective"] = "random"
            self.model_init_info["initial_cluster_weight_checksum"] = tensor_checksum(
                self.model.cluster_output.weight.detach()
            )
            return
        if mode == "random_orthogonal":
            self.model_init_info.update(
                initialize_cluster_output_random_orthogonal(self.model, self.K, seed=self.prototype_seed)
            )
            return
        hidden = self._forward_all_cluster_hidden_no_grad(int(self.args.global_q_chunk_size))
        self.model_init_info["cluster_hidden_checksum"] = tensor_checksum(hidden)
        if mode == "random_event":
            self.model_init_info.update(
                initialize_cluster_output_random_event(
                    self.model,
                    hidden,
                    K=self.K,
                    seed=self.prototype_seed,
                )
            )
            return
        if mode == "kmeans_plus_plus":
            self.model_init_info.update(
                initialize_cluster_output_from_prototypes(
                    self.model,
                    hidden,
                    K=self.K,
                    seed=self.prototype_seed,
                    sample_size=int(getattr(self.args, "prototype_sample_size", 20000)),
                    lloyd_iters=0,
                )
            )
            self.model_init_info["prototype_init_executed"] = True
            self.model_init_info["cluster_init_mode_effective"] = "kmeans_plus_plus"
            self._record_uniform_stage_no_grad("after_prototype_initialization")
            return
        if mode == "prototype":
            self.model_init_info.update(self._initialize_cluster_output_prototypes(hidden=hidden))
            self.model_init_info["prototype_init_executed"] = True
            self._record_uniform_stage_no_grad("after_prototype_initialization")
            return
        raise ValueError(f"Unsupported cluster_init_mode: {getattr(self.args, 'cluster_init_mode', None)}")

    def _initialize_cluster_output_prototypes(self, hidden=None) -> dict:
        if hidden is None:
            hidden = self._forward_all_cluster_hidden_no_grad(int(self.args.global_q_chunk_size))
        stats = initialize_cluster_output_from_prototypes(
            self.model,
            hidden,
            K=self.K,
            seed=self.prototype_seed,
            sample_size=int(getattr(self.args, "prototype_sample_size", 20000)),
            lloyd_iters=int(getattr(self.args, "prototype_lloyd_iters", 10)),
        )
        return stats

    def _predict_nodes_from_edge_labels(self, edge_labels: np.ndarray) -> np.ndarray:
        return edge_hard_labels_to_node_predictions(edge_labels, self.data.src, self.data.dst, self.data.num_nodes, self.K)

    def _direct_kmeans_metrics_from_hidden(self, hidden: torch.Tensor, lloyd_iters: int) -> dict:
        raw_centers, fit_stats = fit_kmeans_centers(
            hidden,
            K=self.K,
            seed=self.prototype_seed,
            sample_size=int(getattr(self.args, "prototype_sample_size", 20000)),
            max_iters=int(lloyd_iters),
        )
        labels_t = assign_all_to_centers(
            hidden,
            raw_centers.to(device=hidden.device, dtype=hidden.dtype),
            chunk_size=int(getattr(self.args, "global_q_chunk_size", 8192)),
        )
        edge_labels = labels_t.detach().cpu().numpy().astype(np.int64)
        pred_node = self._predict_nodes_from_edge_labels(edge_labels)
        metrics = evaluate_node_clustering(self.data.labels, pred_node)
        counts = np.bincount(edge_labels, minlength=self.K)[: self.K]
        ratios = counts.astype(np.float64) / max(1, counts.sum())
        inertia = 0.0
        chunk_size = max(1, int(getattr(self.args, "global_q_chunk_size", 8192)))
        for start in range(0, int(hidden.size(0)), chunk_size):
            h = hidden[start : start + chunk_size]
            c = raw_centers.index_select(0, labels_t[start : start + chunk_size].to(raw_centers.device))
            inertia += float((h.cpu() - c.cpu()).square().sum())
        degree = np.asarray(self.W_E_degree_np, dtype=np.float64)
        volumes = np.bincount(edge_labels, weights=degree, minlength=self.K)[: self.K]
        volume_ratios = volumes / max(float(volumes.sum()), 1e-12)
        metrics.update(
            {
                "direct_kmeans_inertia": float(inertia),
                "edge_hard_active_clusters": int(np.sum(counts > 0)),
                "edge_hard_largest_ratio": float(ratios.max()) if ratios.size else 0.0,
                "node_hard_active_clusters": int(len(np.unique(pred_node))),
                "node_hard_largest_ratio": float(
                    np.bincount(pred_node, minlength=self.K).max() / max(1, int(pred_node.shape[0]))
                ),
                "cluster_volume_min_ratio": float(volume_ratios.min()) if volume_ratios.size else 0.0,
                "cluster_volume_max_ratio": float(volume_ratios.max()) if volume_ratios.size else 0.0,
                "cluster_volume_std": float(volume_ratios.std()) if volume_ratios.size else 0.0,
                "kmeans_center_checksum": fit_stats["kmeans_center_checksum"],
                "prototype_center_checksum": fit_stats["kmeans_center_checksum"],
                "kmeans_lloyd_iters_run": fit_stats["kmeans_lloyd_iters_run"],
            }
        )
        return metrics

    def _resolve_uniform_diag_dir(self) -> str:
        if not self.uniform_collapse_diagnostic:
            return ""
        if self.output_dir:
            return self.output_dir
        base = str(getattr(self.args, "diagnostic_output_dir", "diagnostics/uniform_collapse") or "")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(base, str(self.args.dataset), str(int(self.args.seed)), timestamp)

    def _uniform_diag_config(self) -> dict:
        cfg = vars(self.args).copy()
        cfg.update(
            {
                "M": int(self.data.num_events),
                "N": int(self.data.num_nodes),
                "K": int(self.K),
                "diagnostic_dir": self.uniform_diag_dir,
                "logits_std_unbiased": False,
                "edge_encoder_mode": self.edge_encoder_mode,
                "direct_time_scale": float(self.direct_time_scale),
                "prox_similarity_mode": self.prox_similarity_mode,
                "lambda_node_anchor": float(getattr(self.args, "lambda_node_anchor", 0.0)),
                "node_emb_mode": str(getattr(self.args, "node_emb_mode", "full")),
                "require_pretrained_node2vec": int(self.require_pretrained_node2vec),
                "node_dim": int(self.node_dim),
                "time_dim": int(getattr(self.args, "time_dim", 0)),
                "event_repr_dim": int(self.event_repr_dim),
                "cluster_input_dim": int(self.cluster_input_dim),
                "node_embedding_source": self.node_embedding_source,
                "node2vec_path": self.node_embedding_path,
                "model_parameter_info": self._model_parameter_info(),
                "projection": "S=RowNorm(BQ) using index_add over src/dst events",
                "model_init_info": self.model_init_info,
            }
        )
        return cfg

    def _overnight_stage_extra_stats(self, q_all: torch.Tensor, edge_repr_all: torch.Tensor = None) -> dict:
        degree_t = torch.from_numpy(self.W_E_degree_np).to(device=q_all.device, dtype=q_all.dtype)
        extra = self._current_cluster_output_stats()
        extra.update(self._model_parameter_info())
        if self.node_emb_initial_cpu is not None:
            extra.update(
                node_embedding_drift_statistics(
                    self.model.node_emb.detach(),
                    self.node_emb_initial_cpu.to(device=self.model.node_emb.device, dtype=self.model.node_emb.dtype),
                )
            )
        if edge_repr_all is not None:
            edge_detached = edge_repr_all.detach()
            if edge_detached.numel():
                edge_norm = torch.linalg.norm(edge_detached, dim=1)
                extra.update(
                    {
                        "edge_repr_norm_mean": float(edge_norm.mean().cpu()),
                        "edge_repr_norm_std": float(edge_norm.std(unbiased=False).cpu()),
                    }
                )
            else:
                extra.update({"edge_repr_norm_mean": 0.0, "edge_repr_norm_std": 0.0})
            extra.update(
                edge_repr_block_norm_statistics(
                    edge_detached,
                    node_dim=self.node_dim,
                    time_dim=int(getattr(self.args, "time_dim", 0)),
                    edge_encoder_mode=self.edge_encoder_mode,
                    raw_time_feat=self.time_feat_t.detach(),
                    direct_time_scale=float(self.direct_time_scale),
                )
            )
        extra.update(cluster_volume_statistics(q_all, degree_t))
        selected = str(getattr(self.args, "orth_type", "orth")).lower()
        penalty_weight = float(getattr(self.args, "lambda_orth", 1.0))
        extra["penalty_type"] = selected
        extra["penalty_weight"] = penalty_weight
        extra["direct_time_scale"] = float(self.direct_time_scale)
        extra["prox_similarity_mode"] = self.prox_similarity_mode
        extra["lambda_node_anchor"] = float(getattr(self.args, "lambda_node_anchor", 0.0))
        with torch.no_grad():
            _, cut_loss, orth_original = edge_trace_mincut_loss_global(
                q_all,
                self.W_E_sparse_torch if self.W_E_sparse_torch is not None else self.W_E,
                self.W_E_degree_np,
                self.K,
                lambda_orth=1.0,
                row_block_size=int(self.args.global_ncut_row_block_size),
                orth_type="orth",
            )
            orthqa_loss = edge_orthqa_penalty_global(q_all, degree_t)
        extra["cut_loss"] = float(cut_loss.detach().cpu())
        extra["orth_original_loss"] = float(orth_original.detach().cpu())
        extra["orthqa_loss"] = float(orthqa_loss.detach().cpu())
        extra["selected_penalty_loss"] = (
            extra["orthqa_loss"] if selected == "orthqa" else extra["orth_original_loss"]
        )
        extra["orth_loss"] = extra["selected_penalty_loss"]
        return extra

    @staticmethod
    def _grad_l2_max(grads) -> tuple:
        total_sq = 0.0
        max_abs = 0.0
        any_grad = False
        for grad in grads:
            if grad is None:
                continue
            grad_detached = grad.detach()
            if not torch.isfinite(grad_detached).all():
                raise FloatingPointError("Gradient diagnostic encountered NaN or Inf")
            total_sq += float(grad_detached.square().sum().cpu())
            max_abs = max(max_abs, float(grad_detached.abs().max().cpu()) if grad_detached.numel() else 0.0)
            any_grad = True
        if not any_grad:
            return None, None
        return total_sq ** 0.5, max_abs

    @staticmethod
    def _flat_grad_vector(grads, params) -> torch.Tensor:
        pieces = []
        for grad, param in zip(grads, params):
            if grad is None:
                pieces.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
            else:
                pieces.append(grad.detach().reshape(-1))
        if not pieces:
            return None
        return torch.cat(pieces)

    def _independent_gradient_diagnostics(self, stage_name: str) -> dict:
        if not self.overnight_diagnostic:
            return {}
        if int(self.model_seed) != 42:
            return {}
        allowed = {
            "after_cluster_initialization",
            "before_first_global_update",
            "before_global_epoch_1",
            "epoch_1",
            "final_epoch",
        }
        if stage_name not in allowed:
            return {}

        was_training = self.model.training
        self.model.train()
        result = {}
        node_params = [self.model.node_emb] if self.model.node_emb.requires_grad else []
        cluster_params = [p for p in self.model.cluster_head.parameters() if p.requires_grad]

        if node_params and float(getattr(self.args, "lambda_prox", 0.0)) > 0.0:
            batch_ids = np.arange(min(int(self.args.batch_size), self.data.num_events), dtype=np.int64)
            union_ids = self._batch_union_ids(batch_ids)
            local_index = {int(eid): i for i, eid in enumerate(union_ids.tolist())}
            r_union, _ = self._forward_ids(union_ids)
            diag_rng = np.random.RandomState(int(self.model_seed) + 104729)
            prox_loss = self._proximity_loss_for_union(r_union, union_ids, local_index, batch_ids, diag_rng)
            prox_grads = torch.autograd.grad(
                prox_loss,
                node_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            prox_l2, prox_max = self._grad_l2_max(prox_grads)
            result["node_grad_l2_from_prox"] = prox_l2
            result["node_grad_max_from_prox"] = prox_max
        else:
            result["node_grad_l2_from_prox"] = None
            result["node_grad_max_from_prox"] = None

        q_all = self._forward_all_q_with_grad(int(self.args.global_q_chunk_size))
        _, cut_loss, penalty_loss = edge_trace_mincut_loss_global(
            q_all,
            self.W_E_sparse_torch if self.W_E_sparse_torch is not None else self.W_E,
            self.W_E_degree_np,
            self.K,
            lambda_orth=float(getattr(self.args, "lambda_orth", 1.0)),
            row_block_size=int(self.args.global_ncut_row_block_size),
            orth_type=str(getattr(self.args, "orth_type", "orth")),
        )
        if node_params:
            cut_node_grads = torch.autograd.grad(
                cut_loss,
                node_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            penalty_node_grads = torch.autograd.grad(
                penalty_loss,
                node_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            cut_node_l2, cut_node_max = self._grad_l2_max(cut_node_grads)
            penalty_node_l2, penalty_node_max = self._grad_l2_max(penalty_node_grads)
            result["node_grad_l2_from_cut"] = cut_node_l2
            result["node_grad_max_from_cut"] = cut_node_max
            result["node_grad_l2_from_penalty"] = penalty_node_l2
            result["node_grad_max_from_penalty"] = penalty_node_max
        else:
            result["node_grad_l2_from_cut"] = None
            result["node_grad_max_from_cut"] = None
            result["node_grad_l2_from_penalty"] = None
            result["node_grad_max_from_penalty"] = None

        cut_cluster_grads = torch.autograd.grad(
            cut_loss,
            cluster_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        penalty_cluster_grads = torch.autograd.grad(
            penalty_loss,
            cluster_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        cut_cluster_l2, _ = self._grad_l2_max(cut_cluster_grads)
        penalty_cluster_l2, _ = self._grad_l2_max(penalty_cluster_grads)
        result["cluster_grad_l2_from_cut"] = cut_cluster_l2
        result["cluster_grad_l2_from_penalty"] = penalty_cluster_l2
        cut_vec = self._flat_grad_vector(cut_cluster_grads, cluster_params)
        penalty_vec = self._flat_grad_vector(penalty_cluster_grads, cluster_params)
        cosine = None
        if cut_vec is not None and penalty_vec is not None:
            cut_norm = torch.linalg.norm(cut_vec)
            penalty_norm = torch.linalg.norm(penalty_vec)
            if float(cut_norm.detach().cpu()) > 1e-12 and float(penalty_norm.detach().cpu()) > 1e-12:
                cosine = float(torch.dot(cut_vec, penalty_vec).detach().cpu() / (cut_norm * penalty_norm).clamp_min(1e-12))
        result["cut_penalty_gradient_cosine"] = cosine
        if not was_training:
            self.model.eval()
        return result

    def _write_uniform_diagnostics(self) -> None:
        if not self.uniform_collapse_diagnostic or not self.uniform_diag_dir:
            return
        write_diagnostic_outputs(
            self.uniform_diag_dir,
            self.args.dataset,
            int(self.args.seed),
            self._uniform_diag_config(),
            self.uniform_diag_stages,
            self.uniform_diag_delta,
        )

    def _record_uniform_stage_no_grad(self, stage_name: str) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        q_all, logits_all, edge_repr_all, cluster_input_all = self._forward_all_q_logits_no_grad(
            int(self.args.global_q_chunk_size)
        )
        stats = compute_uniform_collapse_stage(
            stage_name,
            q_all,
            logits_all,
            self.src_t,
            self.dst_t,
            self.data.num_nodes,
            self.K,
            edge_repr_all=edge_repr_all,
            cluster_input_all=cluster_input_all,
            labels=torch.from_numpy(self.data.labels).long().to(self.device),
            extra_stats=self._overnight_stage_extra_stats(q_all, edge_repr_all=edge_repr_all) if self.overnight_diagnostic else self._current_cluster_output_stats(),
        )
        stats.update(self._independent_gradient_diagnostics(stage_name))
        self.uniform_diag_stages[stage_name] = stats
        if "initial_before_training" not in self.uniform_diag_stages:
            self.uniform_diag_stages["initial_before_training"] = stats
        self._write_uniform_diagnostics()
        print_stage_summary(stage_name, stats)

    def _record_uniform_before_global(
        self,
        q_all,
        logits_all,
        edge_repr_all,
        cluster_input_all,
        cut_loss,
        orth_loss,
        stage_name: str = "before_first_global_update",
    ) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        grad_stats = cluster_head_gradient_diagnostics(
            cut_loss,
            orth_loss,
            [p for p in self.model.cluster_head.parameters() if p.requires_grad],
        )
        grad_stats["cut_penalty_grad_cosine"] = grad_stats.get("cut_orth_grad_cosine")
        if str(getattr(self.args, "orth_type", "orth")).lower() == "orthqa":
            grad_stats["orthqa_cluster_head_grad_l2"] = grad_stats.get("orth_cluster_head_grad_l2", 0.0)
        stats = compute_uniform_collapse_stage(
            stage_name,
            q_all,
            logits_all,
            self.src_t,
            self.dst_t,
            self.data.num_nodes,
            self.K,
            edge_repr_all=edge_repr_all,
            cluster_input_all=cluster_input_all,
            labels=torch.from_numpy(self.data.labels).long().to(self.device),
            cut_loss=cut_loss,
            orth_loss=orth_loss,
            grad_stats=grad_stats,
            extra_stats=self._overnight_stage_extra_stats(q_all, edge_repr_all=edge_repr_all) if self.overnight_diagnostic else self._current_cluster_output_stats(),
        )
        stats.update(self._independent_gradient_diagnostics(stage_name))
        self.uniform_diag_stages[stage_name] = stats
        self._write_uniform_diagnostics()
        print_stage_summary("before_global" if stage_name == "before_first_global_update" else stage_name, stats)

    def _record_uniform_after_global(self) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        q_all, logits_all, edge_repr_all, cluster_input_all = self._forward_all_q_logits_no_grad(
            int(self.args.global_q_chunk_size)
        )
        stats = compute_uniform_collapse_stage(
            "after_first_global_update",
            q_all,
            logits_all,
            self.src_t,
            self.dst_t,
            self.data.num_nodes,
            self.K,
            edge_repr_all=edge_repr_all,
            cluster_input_all=cluster_input_all,
            labels=torch.from_numpy(self.data.labels).long().to(self.device),
            extra_stats=self._overnight_stage_extra_stats(q_all, edge_repr_all=edge_repr_all) if self.overnight_diagnostic else self._current_cluster_output_stats(),
        )
        stats.update(self._independent_gradient_diagnostics("after_first_global_update"))
        self.uniform_diag_stages["after_first_global_update"] = stats
        initial = self.uniform_diag_stages.get("after_model_initialization", self.uniform_diag_stages.get("initial_before_training", {}))
        self.uniform_diag_delta = uniform_delta(initial, stats) if initial else {}
        self._write_uniform_diagnostics()
        print_stage_summary("after_global", stats, self.uniform_diag_delta)

    def _record_uniform_final_epoch(self) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        self._record_uniform_stage_no_grad("final_epoch")

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
        return metrics, S

    def _capture_init_reference(self) -> None:
        Q = self.infer_Q()
        metrics, S = self._evaluate_from_Q(Q)
        self.init_reference = {
            "Q": Q.astype(np.float32, copy=True),
            "edge_labels": Q.argmax(axis=1).astype(np.int64),
            "node_labels": S.argmax(axis=1).astype(np.int64),
        }
        self.init_reference_weight = self.model.cluster_output.weight.detach().cpu().clone()
        self.init_reference_metrics = metrics.copy()
        self.model_init_info["initial_Q_summary_checksum"] = self._q_summary_checksum(Q)
        self.model_init_info["initial_cluster_weight_checksum"] = tensor_checksum(self.init_reference_weight)

    @staticmethod
    def _q_summary_checksum(Q: np.ndarray) -> str:
        summary = np.asarray(
            [
                float(np.mean(Q)),
                float(np.std(Q)),
                float(np.min(Q)),
                float(np.max(Q)),
                float(np.sum(Q * Q)),
            ],
            dtype=np.float64,
        )
        import hashlib

        return hashlib.sha256(summary.tobytes()).hexdigest()

    def _compute_init_final_change_metrics(self, final_metrics: dict) -> dict:
        if not self.init_reference:
            return {}
        Q_final = self.infer_Q()
        metrics_final, S_final = self._evaluate_from_Q(Q_final)
        node_final = S_final.argmax(axis=1).astype(np.int64)
        edge_final = Q_final.argmax(axis=1).astype(np.int64)
        Q_init = self.init_reference["Q"]
        weight_final = self.model.cluster_output.weight.detach().cpu()
        weight_init = self.init_reference_weight
        q_drift = float(np.linalg.norm(Q_final.astype(np.float64) - Q_init.astype(np.float64)) / (max(1, Q_init.shape[0]) ** 0.5))
        weight_drift = float(
            torch.linalg.norm(weight_final - weight_init).item()
            / (torch.linalg.norm(weight_init).item() + 1e-12)
        )
        cut_init = self.uniform_diag_stages.get("after_cluster_initialization", {}).get("cut_loss")
        cut_final = self.uniform_diag_stages.get("final_epoch", {}).get("cut_loss")
        result = {
            "node_prediction_ari_init_final": float(adjusted_rand_score(self.init_reference["node_labels"], node_final)),
            "edge_prediction_ari_init_final": float(adjusted_rand_score(self.init_reference["edge_labels"], edge_final)),
            "q_drift_fro_normalized": q_drift,
            "cluster_weight_drift_relative": weight_drift,
            "macro_f1_delta_final_init": float(metrics_final.get("Macro_F1", final_metrics.get("Macro_F1", 0.0)) - self.init_reference_metrics.get("Macro_F1", 0.0)),
            "macro_f1_delta_best_init": float((getattr(self, "best_metrics_for_result", {}) or {}).get("Macro_F1", 0.0) - self.init_reference_metrics.get("Macro_F1", 0.0)),
            "MacroF1_init": float(self.init_reference_metrics.get("Macro_F1", 0.0)),
            "MacroF1_final": float(metrics_final.get("Macro_F1", final_metrics.get("Macro_F1", 0.0))),
        }
        if cut_init is not None and cut_final is not None:
            result["cut_delta_final_init"] = float(cut_final) - float(cut_init)
        if self.node_emb_initial_cpu is not None:
            result.update(
                node_embedding_drift_statistics(
                    self.model.node_emb.detach(),
                    self.node_emb_initial_cpu.to(device=self.model.node_emb.device, dtype=self.model.node_emb.dtype),
                )
            )
            result.update(
                node_event_degree_drift_statistics(
                    self.data.src,
                    self.data.dst,
                    self.data.num_nodes,
                    self.model.node_emb.detach(),
                    self.node_emb_initial_cpu,
                )
            )
        return result

    def run_init_only_eval(self) -> dict:
        metrics = (self.init_reference_metrics or self._evaluate_stage()).copy()
        self._init_metrics_csv()
        init_stage = self.uniform_diag_stages.get("after_cluster_initialization", {})
        self._append_epoch_record(
            {
                "epoch": 0,
                "ACC": metrics.get("ACC", 0.0),
                "NMI": metrics.get("NMI", 0.0),
                "ARI": metrics.get("ARI", 0.0),
                "Macro_F1": metrics.get("Macro_F1", 0.0),
                "penalty_type": str(getattr(self.args, "orth_type", "orth")).lower(),
                "penalty_weight": float(getattr(self.args, "lambda_orth", 1.0)),
                "edge_hard_active_clusters": init_stage.get("num_active_edge_clusters", ""),
                "edge_hard_largest_ratio": init_stage.get("largest_edge_cluster_ratio", ""),
                "node_hard_active_clusters": init_stage.get("num_active_node_clusters", ""),
                "node_hard_largest_ratio": init_stage.get("largest_node_cluster_ratio", ""),
                "q_rank1_energy_ratio": init_stage.get("q_rank1_energy_ratio", ""),
                "q_effective_rank": init_stage.get("q_effective_rank", ""),
                "q_centered_energy": init_stage.get("q_centered_energy", ""),
            }
        )
        self.final_metrics = metrics.copy()
        self.training_change_metrics = self._compute_init_final_change_metrics(metrics)
        return metrics

    def run_direct_kmeans_eval(self) -> dict:
        hidden = self._forward_all_cluster_hidden_no_grad(int(self.args.global_q_chunk_size))
        metrics = self._direct_kmeans_metrics_from_hidden(
            hidden,
            lloyd_iters=int(getattr(self.args, "prototype_lloyd_iters", 10)),
        )
        self._init_metrics_csv()
        self._append_epoch_record(
            {
                "epoch": 0,
                "ACC": metrics.get("ACC", 0.0),
                "NMI": metrics.get("NMI", 0.0),
                "ARI": metrics.get("ARI", 0.0),
                "Macro_F1": metrics.get("Macro_F1", 0.0),
                "edge_hard_active_clusters": metrics.get("edge_hard_active_clusters", ""),
                "edge_hard_largest_ratio": metrics.get("edge_hard_largest_ratio", ""),
                "node_hard_active_clusters": metrics.get("node_hard_active_clusters", ""),
                "node_hard_largest_ratio": metrics.get("node_hard_largest_ratio", ""),
            }
        )
        if self.output_dir:
            ensure_dir(self.output_dir)
            with open(os.path.join(self.output_dir, "diagnostic.json"), "w", encoding="utf-8") as writer:
                json.dump(
                    {
                        "config": self._uniform_diag_config(),
                        "direct_kmeans": metrics,
                        "after_model_initialization": self.uniform_diag_stages.get("after_model_initialization"),
                    },
                    writer,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
        self.final_metrics = metrics.copy()
        self.training_change_metrics = {}
        return metrics

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

    @staticmethod
    def _metrics_fieldnames():
        return [
            "epoch",
            "ACC",
            "NMI",
            "ARI",
            "Macro_F1",
            "cut_loss",
            "orth_loss",
            "orth_original_loss",
            "orthqa_loss",
            "selected_penalty_loss",
            "penalty_type",
            "penalty_weight",
            "cluster_loss",
            "projection_loss",
            "global_total_loss",
            "prox_loss",
            "unweighted_proximity_loss",
            "weighted_proximity_loss",
            "node_anchor_loss",
            "weighted_node_anchor_loss",
            "node_update_from_prox",
            "node_update_from_global",
            "node_update_prox_global_ratio",
            "cluster_update_from_global",
            "Q_mean_entropy",
            "max_cluster_ratio",
            "min_cluster_ratio",
            "empty_cluster_count",
            "edge_hard_active_clusters",
            "edge_hard_largest_ratio",
            "node_hard_active_clusters",
            "node_hard_largest_ratio",
            "q_rank1_energy_ratio",
            "q_effective_rank",
            "q_centered_energy",
            "q_centered_effective_rank",
            "q_margin_mean",
            "q_margin_p99",
            "cluster_volume_coefficient_of_variation",
            "hard_cluster_volume_cv",
            "logits_bias_to_event_variation_ratio",
            "edge_encoder_mode",
            "direct_time_scale",
            "prox_similarity_mode",
            "node_embedding_norm_mean",
            "node_embedding_norm_std",
            "node_embedding_drift_fro_normalized",
            "node_embedding_relative_drift",
            "node_embedding_cosine_to_initial_mean",
            "edge_repr_norm_mean",
            "edge_repr_norm_std",
            "feature_common_to_variation_ratio_before_norm",
            "feature_common_to_variation_ratio_after_norm",
            "source_block_norm_mean",
            "destination_block_norm_mean",
            "time_block_norm_mean",
            "raw_time_block_norm_mean",
            "scaled_time_block_norm_mean",
            "scaled_time_to_node_ratio",
            "node_grad_l2_from_prox",
            "node_grad_max_from_prox",
            "node_grad_l2_from_cut",
            "node_grad_max_from_cut",
            "node_grad_l2_from_penalty",
            "node_grad_max_from_penalty",
            "cluster_grad_l2_from_cut",
            "cluster_grad_l2_from_penalty",
            "cut_penalty_gradient_cosine",
            "event_degree_mean",
            "event_degree_median",
            "event_degree_max",
            "node_drift_mean",
            "node_drift_median",
            "node_drift_max",
            "event_degree_node_drift_spearman",
            "top10_degree_node_drift_mean",
            "bottom10_degree_node_drift_mean",
            "top_bottom_drift_ratio",
            "cluster_forward_seconds",
            "cluster_backward_seconds",
            "epoch_seconds",
            "total_runtime_seconds",
            "peak_gpu_memory_mb",
            "before_ACC",
            "before_NMI",
            "before_ARI",
            "before_Macro_F1",
            "before_Q_mean_entropy",
            "before_max_cluster_ratio",
            "before_min_cluster_ratio",
            "before_empty_cluster_count",
            "after_prox_ACC",
            "after_prox_NMI",
            "after_prox_ARI",
            "after_prox_Macro_F1",
            "after_prox_Q_mean_entropy",
            "after_prox_max_cluster_ratio",
            "after_prox_min_cluster_ratio",
            "after_prox_empty_cluster_count",
            "after_global_ACC",
            "after_global_NMI",
            "after_global_ARI",
            "after_global_Macro_F1",
            "after_global_Q_mean_entropy",
            "after_global_max_cluster_ratio",
            "after_global_min_cluster_ratio",
            "after_global_empty_cluster_count",
        ]

    def _append_epoch_record(self, record: dict) -> None:
        self.epoch_records.append(record)
        if not self.metrics_csv_path:
            return
        fieldnames = self._metrics_fieldnames()
        row = {key: record.get(key, "") for key in fieldnames}
        with open(self.metrics_csv_path, "a", encoding="utf-8", newline="") as writer:
            csv.DictWriter(writer, fieldnames=fieldnames).writerow(row)

    def write_config_json(self) -> None:
        if not self.output_dir:
            return
        ensure_dir(self.output_dir)
        cfg = vars(self.args).copy()
        cfg.update(
            {
                "M": int(self.data.num_events),
                "N": int(self.data.num_nodes),
                "K": int(self.K),
                "edge_encoder_mode": self.edge_encoder_mode,
                "direct_time_scale": float(self.direct_time_scale),
                "prox_similarity_mode": self.prox_similarity_mode,
                "prox_role_ss_weight": float(getattr(self.args, "prox_role_ss_weight", 0.25)),
                "prox_role_dd_weight": float(getattr(self.args, "prox_role_dd_weight", 0.25)),
                "prox_role_ds_weight": float(getattr(self.args, "prox_role_ds_weight", 1.0)),
                "prox_role_sd_weight": float(getattr(self.args, "prox_role_sd_weight", 0.0)),
                "prox_role_time_weight": float(getattr(self.args, "prox_role_time_weight", 0.25)),
                "prox_temperature": float(getattr(self.args, "prox_temperature", 0.2)),
                "lambda_node_anchor": float(getattr(self.args, "lambda_node_anchor", 0.0)),
                "node_emb_mode": str(getattr(self.args, "node_emb_mode", "full")),
                "require_pretrained_node2vec": int(self.require_pretrained_node2vec),
                "node_dim": int(self.node_dim),
                "time_dim": int(getattr(self.args, "time_dim", 0)),
                "event_repr_dim": int(self.event_repr_dim),
                "cluster_input_dim": int(self.cluster_input_dim),
                "node_embedding_source": self.node_embedding_source,
                "node2vec_path": self.node_embedding_path,
                "model_seed": self.model_seed,
                "prototype_seed": self.prototype_seed,
                "forest_seed": self.forest_seed,
                "Pi_E_nnz": int(self.Pi_E.nnz),
                "W_E_nnz": int(self.W_E.nnz),
                "W_E_avg_nnz_per_row": float(self.W_E.nnz / max(1, self.W_E.shape[0])),
                "W_E_sparse_mode": self.W_E_sparse_mode,
                "legacy_balance_disabled": self.cluster_loss_type == "trace_mincut",
                "trace_mincut_complexity": "O(nnz(W_E) K + M K^2)",
                "node_emb_optimizer_info": self.node_emb_optimizer_info,
                "main_learning_rate": float(self.node_emb_optimizer_info.get("main_learning_rate", 0.0)),
                "node_embedding_learning_rate": float(self.node_emb_optimizer_info.get("node_embedding_learning_rate", 0.0)),
                "node_lr_ratio": float(self.node_emb_optimizer_info.get("node_lr_ratio", 0.0)),
                "model_parameter_info": self._model_parameter_info(),
                "model_init_info": self.model_init_info,
                "prox_stats": self.prox_stats,
            }
        )
        with open(os.path.join(self.output_dir, "config.json"), "w", encoding="utf-8") as writer:
            json.dump(cfg, writer, indent=2, sort_keys=True, default=str)

    def write_result_json(self, best_epoch: int, best_metrics: dict, final_metrics: dict, runtime_seconds: float) -> None:
        if not self.output_dir:
            return
        result = {
            "status": "success",
            "dataset": self.args.dataset,
            "seed": int(self.args.seed),
            "model_seed": self.model_seed,
            "prototype_seed": self.prototype_seed,
            "forest_seed": self.forest_seed,
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
            "runtime_seconds": float(runtime_seconds),
            "M": int(self.data.num_events),
            "N": int(self.data.num_nodes),
            "K": int(self.K),
            "edge_encoder_mode": self.edge_encoder_mode,
            "direct_time_scale": float(self.direct_time_scale),
            "prox_similarity_mode": self.prox_similarity_mode,
            "prox_role_ss_weight": float(getattr(self.args, "prox_role_ss_weight", 0.25)),
            "prox_role_dd_weight": float(getattr(self.args, "prox_role_dd_weight", 0.25)),
            "prox_role_ds_weight": float(getattr(self.args, "prox_role_ds_weight", 1.0)),
            "prox_role_sd_weight": float(getattr(self.args, "prox_role_sd_weight", 0.0)),
            "prox_role_time_weight": float(getattr(self.args, "prox_role_time_weight", 0.25)),
            "prox_temperature": float(getattr(self.args, "prox_temperature", 0.2)),
            "lambda_node_anchor": float(getattr(self.args, "lambda_node_anchor", 0.0)),
            "node_emb_mode": str(getattr(self.args, "node_emb_mode", "full")),
            "node_dim": int(self.node_dim),
            "time_dim": int(getattr(self.args, "time_dim", 0)),
            "event_repr_dim": int(self.event_repr_dim),
            "cluster_input_dim": int(self.cluster_input_dim),
            "node_embedding_source": self.node_embedding_source,
            "node2vec_path": self.node_embedding_path,
            "Pi_E_nnz": int(self.Pi_E.nnz),
            "W_E_nnz": int(self.W_E.nnz),
            "W_E_avg_nnz_per_row": float(self.W_E.nnz / max(1, self.W_E.shape[0])),
            "W_E_sparse_mode": self.W_E_sparse_mode,
            "trace_mincut_complexity": "O(nnz(W_E) K + M K^2)",
            "node_emb_optimizer_info": self.node_emb_optimizer_info,
            "main_learning_rate": float(self.node_emb_optimizer_info.get("main_learning_rate", 0.0)),
            "node_embedding_learning_rate": float(self.node_emb_optimizer_info.get("node_embedding_learning_rate", 0.0)),
            "node_lr_ratio": float(self.node_emb_optimizer_info.get("node_lr_ratio", 0.0)),
            "model_parameter_info": self._model_parameter_info(),
            "model_init_info": self.model_init_info,
            "training_change_metrics": getattr(self, "training_change_metrics", {}),
        }
        with open(os.path.join(self.output_dir, "result.json"), "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=2, sort_keys=True)

    def train(self) -> Tuple[int, dict]:
        best_epoch = -1
        best_metrics = None
        best_key = None
        final_metrics = {}
        m = self.data.num_events
        batch_size = int(self.args.batch_size)
        quiet = bool(int(getattr(self.args, "quiet", 0)))
        diagnostic = bool(int(getattr(self.args, "diagnostic_stages", 0)))
        warmup_epochs = int(getattr(self.args, "global_warmup_epochs", 0))
        lambda_prox = float(self.args.lambda_prox)
        lambda_edge_ncut = float(self.args.lambda_edge_ncut)
        lambda_proj = float(self.args.lambda_proj)
        lambda_bal = float(self.args.lambda_bal)
        lambda_node_anchor = float(getattr(self.args, "lambda_node_anchor", 0.0))
        total_start = time.time()
        self._init_metrics_csv()

        def sync_cuda() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def scalar_value(value) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        for epoch in range(1, int(self.args.epoch) + 1):
            epoch_start = time.time()
            before_metrics = self._evaluate_stage() if diagnostic else {}
            self.model.train()
            prox_total = 0.0
            weighted_prox_total = 0.0
            prox_steps = 0
            h_before_prox = self._node_emb_snapshot_cpu()
            if self.ncut_scope == "global":
                if lambda_prox > 0.0:
                    order = self.rng.permutation(m)
                    iterator = range(0, m, batch_size)
                    for start in tqdm(iterator, desc=f"ETGC epoch {epoch}", leave=False, disable=quiet):
                        batch_ids = order[start : start + batch_size]
                        union_ids = self._batch_union_ids(batch_ids)
                        local_index = {int(eid): i for i, eid in enumerate(union_ids.tolist())}
                        r_union, _ = self._forward_ids(union_ids)
                        l_prox = self._proximity_loss_for_union(r_union, union_ids, local_index, batch_ids, self.rng)
                        loss = lambda_prox * l_prox
                        self.optimizer.zero_grad()
                        if loss.requires_grad:
                            loss.backward()
                            self.optimizer.step()
                        prox_total += scalar_value(l_prox)
                        weighted_prox_total += scalar_value(loss)
                        prox_steps += 1
            else:
                order = self.rng.permutation(m)
                iterator = range(0, m, batch_size)
                for start in tqdm(iterator, desc=f"ETGC epoch {epoch}", leave=False, disable=quiet):
                    batch_ids = order[start : start + batch_size]
                    union_ids = self._batch_union_ids(batch_ids)
                    local_index = {int(eid): i for i, eid in enumerate(union_ids.tolist())}
                    r_union, q_union = self._forward_ids(union_ids)
                    batch_local = torch.as_tensor(
                        [local_index[int(e)] for e in batch_ids],
                        dtype=torch.long,
                        device=self.device,
                    )
                    q_batch = q_union.index_select(0, batch_local)
                    batch_t = torch.from_numpy(batch_ids).long().to(self.device)

                    l_prox = self._proximity_loss_for_union(r_union, union_ids, local_index, batch_ids, self.rng)
                    l_proj = projection_loss(
                        q_batch,
                        self.src_t.index_select(0, batch_t),
                        self.dst_t.index_select(0, batch_t),
                        self.data.num_nodes,
                    )
                    l_bal = balance_loss(q_union, self.K)
                    loss = lambda_prox * l_prox + lambda_proj * l_proj + lambda_bal * l_bal
                    if self.cluster_loss_type == "legacy_ncut":
                        l_ncut = edge_ncut_loss(q_union, union_ids, self.W_E, self.K)
                        loss = loss + lambda_edge_ncut * l_ncut
                    self.optimizer.zero_grad()
                    if loss.requires_grad:
                        loss.backward()
                        self.optimizer.step()
                    prox_total += scalar_value(l_prox)
                    weighted_prox_total += scalar_value(lambda_prox * l_prox)
                    prox_steps += 1
            node_update_from_prox = self._node_update_from_snapshot(h_before_prox)
            del h_before_prox

            after_prox_metrics = self._evaluate_stage() if diagnostic else {}

            cut_loss_value = float("nan")
            orth_loss_value = float("nan")
            orth_original_loss_value = float("nan")
            orthqa_loss_value = float("nan")
            cluster_loss_value = float("nan")
            projection_loss_value = 0.0
            node_anchor_loss_value = 0.0
            weighted_node_anchor_loss_value = 0.0
            global_total_loss_value = 0.0
            node_update_from_global = 0.0
            cluster_update_from_global = 0.0
            cluster_forward_seconds = 0.0
            cluster_backward_seconds = 0.0
            global_q_forwards = 0
            if self.ncut_scope == "global" and epoch > warmup_epochs:
                self.model.train()
                h_before_global = self._node_emb_snapshot_cpu()
                cluster_before_global = self._cluster_head_snapshot_cpu()
                self.optimizer.zero_grad(set_to_none=True)
                should_uniform_diag = (
                    self.uniform_collapse_diagnostic
                    and not self.uniform_diag_after_first_done
                    and (not self.diagnostic_only_first_epoch or epoch == 1)
                )
                should_grad_epoch_diag = (
                    self.overnight_diagnostic
                    and epoch in {1, int(self.args.epoch)}
                    and epoch > warmup_epochs
                )
                if should_uniform_diag or should_grad_epoch_diag:
                    q_all, logits_all, edge_repr_all, cluster_input_all = self._forward_all_q_logits_with_grad(
                        int(self.args.global_q_chunk_size)
                    )
                else:
                    q_all = self._forward_all_q_with_grad(int(self.args.global_q_chunk_size))
                    logits_all = None
                    edge_repr_all = None
                    cluster_input_all = None
                global_q_forwards = 1
                sync_cuda()
                cluster_start = time.time()
                if self.cluster_loss_type == "trace_mincut":
                    cluster_loss, cut_loss, orth_loss = edge_trace_mincut_loss_global(
                        q_all,
                        self.W_E_sparse_torch if self.W_E_sparse_torch is not None else self.W_E,
                        self.W_E_degree_np,
                        self.K,
                        lambda_orth=float(getattr(self.args, "lambda_orth", 1.0)),
                        row_block_size=int(self.args.global_ncut_row_block_size),
                        orth_type=str(getattr(self.args, "orth_type", "orth")),
                    )
                else:
                    cut_loss = edge_ncut_loss_global(
                        q_all,
                        self.W_E,
                        self.K,
                        row_block_size=int(self.args.global_ncut_row_block_size),
                    )
                    orth_loss = balance_loss(q_all, self.K)
                    cluster_loss = cut_loss
                sync_cuda()
                cluster_forward_seconds = time.time() - cluster_start

                if lambda_proj > 0.0:
                    proj_loss = projection_loss_global(
                        q_all,
                        self.src_t,
                        self.dst_t,
                        self.data.num_nodes,
                    )
                else:
                    proj_loss = self._zero_scalar()
                if not torch.isfinite(proj_loss).all():
                    raise FloatingPointError(f"projection_loss_global is not finite: value={scalar_value(proj_loss)}")
                if lambda_node_anchor > 0.0:
                    anchor_loss = self._node_anchor_loss()
                else:
                    anchor_loss = self._zero_scalar()
                if not torch.isfinite(anchor_loss).all():
                    raise FloatingPointError(f"node_anchor_loss is not finite: value={scalar_value(anchor_loss)}")

                if self.cluster_loss_type == "trace_mincut":
                    global_loss = (
                        lambda_edge_ncut * cluster_loss
                        + lambda_proj * proj_loss
                        + lambda_node_anchor * anchor_loss
                    )
                else:
                    global_loss = (
                        lambda_edge_ncut * cluster_loss
                        + lambda_proj * proj_loss
                        + lambda_bal * orth_loss
                        + lambda_node_anchor * anchor_loss
                    )

                cut_loss_value = scalar_value(cut_loss)
                orth_loss_value = scalar_value(orth_loss)
                if str(getattr(self.args, "orth_type", "orth")).lower() == "orth":
                    orth_original_loss_value = orth_loss_value
                elif str(getattr(self.args, "orth_type", "orth")).lower() == "orthqa":
                    orthqa_loss_value = orth_loss_value
                cluster_loss_value = scalar_value(cluster_loss)
                projection_loss_value = scalar_value(proj_loss)
                node_anchor_loss_value = scalar_value(anchor_loss)
                weighted_node_anchor_loss_value = scalar_value(lambda_node_anchor * anchor_loss)
                global_total_loss_value = scalar_value(global_loss)

                if should_uniform_diag:
                    self._record_uniform_before_global(
                        q_all, logits_all, edge_repr_all, cluster_input_all, cut_loss, orth_loss
                    )
                elif should_grad_epoch_diag:
                    self._record_uniform_before_global(
                        q_all,
                        logits_all,
                        edge_repr_all,
                        cluster_input_all,
                        cut_loss,
                        orth_loss,
                        stage_name=f"before_global_epoch_{epoch}",
                    )

                if global_loss.requires_grad and (
                    lambda_edge_ncut != 0.0
                    or lambda_proj != 0.0
                    or lambda_node_anchor != 0.0
                    or (self.cluster_loss_type == "legacy_ncut" and lambda_bal != 0.0)
                ):
                    sync_cuda()
                    backward_start = time.time()
                    global_loss.backward()
                    sync_cuda()
                    cluster_backward_seconds = time.time() - backward_start
                    self.optimizer.step()
                    node_update_from_global = self._node_update_from_snapshot(h_before_global)
                    cluster_update_from_global = self._cluster_update_from_snapshot(cluster_before_global)
                    if should_uniform_diag:
                        self._record_uniform_after_global()
                        self.uniform_diag_after_first_done = True
                else:
                    node_update_from_global = self._node_update_from_snapshot(h_before_global)
                    cluster_update_from_global = self._cluster_update_from_snapshot(cluster_before_global)
                del h_before_global
                del cluster_before_global

            after_global_metrics = self._evaluate_stage()
            metrics = after_global_metrics
            key = float(metrics.get("Macro_F1", 0.0))
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_metrics = metrics.copy()
                self.best_epoch_record = epoch
            final_metrics = metrics.copy()
            if self.device.type == "cuda":
                sync_cuda()
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            else:
                peak_gpu_memory_mb = 0.0
            epoch_seconds = time.time() - epoch_start
            total_runtime_seconds = time.time() - total_start
            stage_stats_for_epoch = {}
            if self.overnight_diagnostic and epoch in self.diagnostic_epochs:
                self._record_uniform_stage_no_grad(f"epoch_{epoch}")
                stage_stats_for_epoch = self.uniform_diag_stages.get(f"epoch_{epoch}", {})
            record = {
                "epoch": epoch,
                "ACC": metrics.get("ACC", 0.0),
                "NMI": metrics.get("NMI", 0.0),
                "ARI": metrics.get("ARI", 0.0),
                "Macro_F1": metrics.get("Macro_F1", 0.0),
                "cut_loss": cut_loss_value,
                "orth_loss": orth_loss_value,
                "orth_original_loss": orth_original_loss_value,
                "orthqa_loss": orthqa_loss_value,
                "selected_penalty_loss": orth_loss_value,
                "penalty_type": str(getattr(self.args, "orth_type", "orth")).lower(),
                "penalty_weight": float(getattr(self.args, "lambda_orth", 1.0)),
                "cluster_loss": cluster_loss_value,
                "projection_loss": projection_loss_value,
                "global_total_loss": global_total_loss_value,
                "prox_loss": prox_total / max(1, prox_steps),
                "unweighted_proximity_loss": prox_total / max(1, prox_steps),
                "weighted_proximity_loss": weighted_prox_total / max(1, prox_steps),
                "node_anchor_loss": node_anchor_loss_value,
                "weighted_node_anchor_loss": weighted_node_anchor_loss_value,
                "node_update_from_prox": node_update_from_prox,
                "node_update_from_global": node_update_from_global,
                "node_update_prox_global_ratio": node_update_from_prox / (node_update_from_global + 1e-12),
                "cluster_update_from_global": cluster_update_from_global,
                "Q_mean_entropy": metrics.get("Q_mean_entropy", 0.0),
                "max_cluster_ratio": metrics.get("max_cluster_ratio", 0.0),
                "min_cluster_ratio": metrics.get("min_cluster_ratio", 0.0),
                "empty_cluster_count": metrics.get("empty_cluster_count", 0),
                "edge_hard_active_clusters": stage_stats_for_epoch.get("num_active_edge_clusters", ""),
                "edge_hard_largest_ratio": stage_stats_for_epoch.get("largest_edge_cluster_ratio", ""),
                "node_hard_active_clusters": stage_stats_for_epoch.get("num_active_node_clusters", ""),
                "node_hard_largest_ratio": stage_stats_for_epoch.get("largest_node_cluster_ratio", ""),
                "q_rank1_energy_ratio": stage_stats_for_epoch.get("q_rank1_energy_ratio", ""),
                "q_effective_rank": stage_stats_for_epoch.get("q_effective_rank", ""),
                "q_centered_energy": stage_stats_for_epoch.get("q_centered_energy", ""),
                "q_centered_effective_rank": stage_stats_for_epoch.get("q_centered_effective_rank", ""),
                "q_margin_mean": stage_stats_for_epoch.get("q_margin_mean", ""),
                "q_margin_p99": stage_stats_for_epoch.get("q_margin_p99", ""),
                "cluster_volume_coefficient_of_variation": stage_stats_for_epoch.get("cluster_volume_coefficient_of_variation", ""),
                "hard_cluster_volume_cv": stage_stats_for_epoch.get("hard_cluster_volume_cv", ""),
                "logits_bias_to_event_variation_ratio": stage_stats_for_epoch.get("logits_bias_to_event_variation_ratio", ""),
                "edge_encoder_mode": self.edge_encoder_mode,
                "direct_time_scale": float(self.direct_time_scale),
                "prox_similarity_mode": self.prox_similarity_mode,
                "node_embedding_norm_mean": stage_stats_for_epoch.get("node_embedding_norm_mean", ""),
                "node_embedding_norm_std": stage_stats_for_epoch.get("node_embedding_norm_std", ""),
                "node_embedding_drift_fro_normalized": stage_stats_for_epoch.get("node_embedding_drift_fro_normalized", ""),
                "node_embedding_relative_drift": stage_stats_for_epoch.get("node_embedding_relative_drift", ""),
                "node_embedding_cosine_to_initial_mean": stage_stats_for_epoch.get("node_embedding_cosine_to_initial_mean", ""),
                "edge_repr_norm_mean": stage_stats_for_epoch.get("edge_repr_norm_mean", ""),
                "edge_repr_norm_std": stage_stats_for_epoch.get("edge_repr_norm_std", ""),
                "feature_common_to_variation_ratio_before_norm": stage_stats_for_epoch.get("feature_common_to_variation_ratio_before_norm", ""),
                "feature_common_to_variation_ratio_after_norm": stage_stats_for_epoch.get("feature_common_to_variation_ratio_after_norm", ""),
                "source_block_norm_mean": stage_stats_for_epoch.get("source_block_norm_mean", ""),
                "destination_block_norm_mean": stage_stats_for_epoch.get("destination_block_norm_mean", ""),
                "time_block_norm_mean": stage_stats_for_epoch.get("time_block_norm_mean", ""),
                "raw_time_block_norm_mean": stage_stats_for_epoch.get("raw_time_block_norm_mean", ""),
                "scaled_time_block_norm_mean": stage_stats_for_epoch.get("scaled_time_block_norm_mean", ""),
                "scaled_time_to_node_ratio": stage_stats_for_epoch.get("scaled_time_to_node_ratio", ""),
                "node_grad_l2_from_prox": stage_stats_for_epoch.get("node_grad_l2_from_prox", ""),
                "node_grad_max_from_prox": stage_stats_for_epoch.get("node_grad_max_from_prox", ""),
                "node_grad_l2_from_cut": stage_stats_for_epoch.get("node_grad_l2_from_cut", ""),
                "node_grad_max_from_cut": stage_stats_for_epoch.get("node_grad_max_from_cut", ""),
                "node_grad_l2_from_penalty": stage_stats_for_epoch.get("node_grad_l2_from_penalty", ""),
                "node_grad_max_from_penalty": stage_stats_for_epoch.get("node_grad_max_from_penalty", ""),
                "cluster_grad_l2_from_cut": stage_stats_for_epoch.get("cluster_grad_l2_from_cut", ""),
                "cluster_grad_l2_from_penalty": stage_stats_for_epoch.get("cluster_grad_l2_from_penalty", ""),
                "cut_penalty_gradient_cosine": stage_stats_for_epoch.get("cut_penalty_gradient_cosine", ""),
                "cluster_forward_seconds": cluster_forward_seconds,
                "cluster_backward_seconds": cluster_backward_seconds,
                "epoch_seconds": epoch_seconds,
                "total_runtime_seconds": total_runtime_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            }
            for prefix, stage_metrics in (
                ("before", before_metrics),
                ("after_prox", after_prox_metrics),
                ("after_global", after_global_metrics),
            ):
                for key_name in [
                    "ACC",
                    "NMI",
                    "ARI",
                    "Macro_F1",
                    "Q_mean_entropy",
                    "max_cluster_ratio",
                    "min_cluster_ratio",
                    "empty_cluster_count",
                ]:
                    record[f"{prefix}_{key_name}"] = stage_metrics.get(key_name, "")
            self._append_epoch_record(record)

            if diagnostic:
                print(
                    f"epoch={epoch} before_f1={record.get('before_Macro_F1', 0.0):.4f} "
                    f"after_prox_f1={record.get('after_prox_Macro_F1', 0.0):.4f} "
                    f"after_global_f1={record.get('after_global_Macro_F1', 0.0):.4f} "
                    f"cut={cut_loss_value:.4f} orth={orth_loss_value:.4f} "
                    f"proj={projection_loss_value:.4f} cluster={cluster_loss_value:.4f} "
                    f"global={global_total_loss_value:.4f} prox={record['prox_loss']:.4f} "
                    f"wprox={record['weighted_proximity_loss']:.4f} anchor={node_anchor_loss_value:.6g} "
                    f"node_up_prox={node_update_from_prox:.6g} node_up_global={node_update_from_global:.6g} "
                    f"entropy={record['Q_mean_entropy']:.4f} max_ratio={record['max_cluster_ratio']:.4f} "
                    f"min_ratio={record['min_cluster_ratio']:.4f} empty={int(record['empty_cluster_count'])} "
                    f"cluster_forward_seconds={cluster_forward_seconds:.4f} "
                    f"cluster_backward_seconds={cluster_backward_seconds:.4f} "
                    f"epoch_seconds={epoch_seconds:.4f} peak_gpu_memory_mb={peak_gpu_memory_mb:.2f} "
                    f"global_q_forwards={global_q_forwards}"
                )
            else:
                print(
                    f"epoch={epoch} ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f} "
                    f"ARI={metrics['ARI']:.4f} Macro_F1={metrics['Macro_F1']:.4f} "
                    f"cut={cut_loss_value:.4f} orth={orth_loss_value:.4f} "
                    f"penalty_type={str(getattr(self.args, 'orth_type', 'orth')).lower()} "
                    f"proj={projection_loss_value:.4f} cluster={cluster_loss_value:.4f} "
                    f"global={global_total_loss_value:.4f} prox={record['prox_loss']:.4f} "
                    f"wprox={record['weighted_proximity_loss']:.4f} anchor={node_anchor_loss_value:.6g} "
                    f"node_up_prox={node_update_from_prox:.6g} node_up_global={node_update_from_global:.6g} "
                    f"entropy={record['Q_mean_entropy']:.4f} max_ratio={record['max_cluster_ratio']:.4f} "
                    f"min_ratio={record['min_cluster_ratio']:.4f} empty={int(record['empty_cluster_count'])} "
                    f"cluster_forward_seconds={cluster_forward_seconds:.4f} "
                    f"cluster_backward_seconds={cluster_backward_seconds:.4f} "
                    f"epoch_seconds={epoch_seconds:.4f} peak_gpu_memory_mb={peak_gpu_memory_mb:.2f} "
                    f"global_q_forwards={global_q_forwards}"
                )
        self._record_uniform_final_epoch()
        self.best_metrics_for_result = best_metrics or {}
        self.training_change_metrics = self._compute_init_final_change_metrics(final_metrics)
        if int(self.args.save_embeddings):
            self.save_outputs()
        self.final_metrics = final_metrics
        return best_epoch, best_metrics or {}

    @torch.no_grad()
    def infer_Q(self) -> np.ndarray:
        self.model.eval()
        chunks = []
        chunk_size = max(1, int(self.args.batch_size) * 4)
        ids = np.arange(self.data.num_events, dtype=np.int64)
        for start in range(0, self.data.num_events, chunk_size):
            _, q = self._forward_ids(ids[start : start + chunk_size])
            chunks.append(q.detach().cpu().numpy())
        return np.vstack(chunks)

    @torch.no_grad()
    def evaluate(self):
        Q = self.infer_Q()
        return self._evaluate_from_Q(Q)

    def save_outputs(self) -> None:
        out_dir = os.path.join(self.args.emb_root, self.args.dataset)
        ensure_dir(out_dir)
        Q = self.infer_Q()
        np.save(os.path.join(out_dir, "edge_hinos_Q.npy"), Q)
        _, S = self.evaluate()
        np.save(os.path.join(out_dir, "edge_hinos_S.npy"), S)
