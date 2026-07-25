import csv
import json
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from edge_data import load_edge_event_data
from edge_losses import (
    balance_loss,
    edge_ncut_loss,
    edge_ncut_loss_global,
    edge_ppr_proximity_loss,
    edge_trace_mincut_loss_global,
    projection_loss,
    projection_loss_global,
    scipy_csr_to_torch_sparse_coo,
)
from edge_metrics import evaluate_node_clustering
from edge_model import EdgeHiNoSModel, load_pretrained_node_features
from edge_proximity import compute_edge_ppr_cached
from edge_time import build_edge_time_features
from edge_uniform_diagnostic import (
    cluster_head_gradient_diagnostics,
    compute_uniform_collapse_stage,
    print_stage_summary,
    uniform_delta,
    write_diagnostic_outputs,
)
from utils import choose_device, ensure_dir


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
        optimizer = Adam(params, lr=lr)
        info = {
            "node_emb_mode": mode,
            "node_emb_trainable": False,
            "node_emb_lr": 0.0,
            "other_lr": lr,
            "param_group_lrs": [lr],
        }
        return optimizer, info

    node_param.requires_grad_(True)
    if mode == "small_lr":
        node_id = id(node_param)
        other_params = [p for p in model.parameters() if id(p) != node_id and p.requires_grad]
        optimizer = Adam(
            [
                {"params": other_params, "lr": lr},
                {"params": [node_param], "lr": node_emb_lr},
            ]
        )
        info = {
            "node_emb_mode": mode,
            "node_emb_trainable": True,
            "node_emb_lr": node_emb_lr,
            "other_lr": lr,
            "param_group_lrs": [lr, node_emb_lr],
        }
        return optimizer, info

    optimizer = Adam(model.parameters(), lr=lr)
    info = {
        "node_emb_mode": mode,
        "node_emb_trainable": True,
        "node_emb_lr": lr,
        "other_lr": lr,
        "param_group_lrs": [lr],
    }
    return optimizer, info


class EdgeHiNoSTrainer:
    def __init__(self, args):
        self.args = args
        self.device = choose_device(args.device)
        self.rng = np.random.RandomState(int(args.seed))
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
            seed=args.seed,
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

        feature_path = self._resolve_feature_path()
        node_features = load_pretrained_node_features(
            feature_path, self.data.num_nodes, int(args.edge_dim), int(args.seed)
        )
        self.model = EdgeHiNoSModel(
            initial_node_features=node_features,
            time_dim=int(args.time_dim),
            edge_dim=int(args.edge_dim),
            edge_hidden_dim=int(args.edge_hidden_dim),
            cluster_hidden_dim=int(args.cluster_hidden_dim),
            K=self.K,
            directed=bool(args.directed),
        ).to(self.device)
        self.optimizer, self.node_emb_optimizer_info = build_optimizer_for_node_emb_mode(
            self.model,
            lr=float(args.learning_rate),
            node_emb_mode=getattr(args, "node_emb_mode", "full"),
            node_emb_lr=float(getattr(args, "node_emb_lr", 1e-5)),
        )

        self.src_t = torch.from_numpy(self.data.src).long().to(self.device)
        self.dst_t = torch.from_numpy(self.data.dst).long().to(self.device)
        self.time_feat_t = torch.from_numpy(self.time_feat_np).float().to(self.device)
        self.output_dir = str(getattr(args, "output_dir", "") or "")
        self.metrics_csv_path = os.path.join(self.output_dir, "metrics.csv") if self.output_dir else ""
        self.epoch_records = []
        self.best_epoch_record = None
        self.uniform_collapse_diagnostic = bool(int(getattr(args, "uniform_collapse_diagnostic", 0)))
        self.diagnostic_only_first_epoch = bool(int(getattr(args, "diagnostic_only_first_epoch", 1)))
        self.uniform_diag_dir = self._resolve_uniform_diag_dir()
        self.uniform_diag_stages = {}
        self.uniform_diag_delta = {}
        self.uniform_diag_after_first_done = False

    def _resolve_feature_path(self) -> str:
        if getattr(self.args, "feature_path", ""):
            return self.args.feature_path
        candidates = [
            os.path.join(self.args.pretrain_emb_dir, f"{self.args.dataset}_feature.emb"),
            os.path.join(self.args.emb_root, f"{self.args.dataset}_feature.emb"),
            os.path.join(self.args.emb_root, self.args.dataset, f"{self.args.dataset}_feature.emb"),
        ]
        return next((path for path in candidates if os.path.exists(path)), candidates[0])

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

    def _forward_all_q_with_grad(self, chunk_size: int) -> torch.Tensor:
        chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            _, q = self._forward_ids(ids[start : start + chunk_size])
            chunks.append(q)
        return torch.cat(chunks, dim=0)

    def _forward_all_q_logits_with_grad(self, chunk_size: int) -> tuple:
        q_chunks = []
        logits_chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            _, q, logits = self._forward_ids_with_logits(ids[start : start + chunk_size])
            q_chunks.append(q)
            logits_chunks.append(logits)
        return torch.cat(q_chunks, dim=0), torch.cat(logits_chunks, dim=0)

    @torch.no_grad()
    def _forward_all_q_logits_no_grad(self, chunk_size: int) -> tuple:
        self.model.eval()
        q_chunks = []
        logits_chunks = []
        ids = np.arange(self.data.num_events, dtype=np.int64)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, self.data.num_events, chunk_size):
            _, q, logits = self._forward_ids_with_logits(ids[start : start + chunk_size])
            q_chunks.append(q.detach())
            logits_chunks.append(logits.detach())
        return torch.cat(q_chunks, dim=0), torch.cat(logits_chunks, dim=0)

    def _zero_scalar(self) -> torch.Tensor:
        return next(self.model.parameters()).sum() * 0.0

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
                "projection": "S=RowNorm(BQ) using index_add over src/dst events",
            }
        )
        return cfg

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

    def _record_uniform_initial(self) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        q_all, logits_all = self._forward_all_q_logits_no_grad(int(self.args.global_q_chunk_size))
        stats = compute_uniform_collapse_stage(
            "initial_before_training",
            q_all,
            logits_all,
            self.src_t,
            self.dst_t,
            self.data.num_nodes,
            self.K,
        )
        self.uniform_diag_stages["initial_before_training"] = stats
        self._write_uniform_diagnostics()
        print_stage_summary("initial", stats)

    def _record_uniform_before_global(self, q_all, logits_all, cut_loss, orth_loss) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        grad_stats = cluster_head_gradient_diagnostics(
            cut_loss,
            orth_loss,
            [p for p in self.model.cluster_head.parameters() if p.requires_grad],
        )
        stats = compute_uniform_collapse_stage(
            "before_first_global_update",
            q_all,
            logits_all,
            self.src_t,
            self.dst_t,
            self.data.num_nodes,
            self.K,
            cut_loss=cut_loss,
            orth_loss=orth_loss,
            grad_stats=grad_stats,
        )
        self.uniform_diag_stages["before_first_global_update"] = stats
        self._write_uniform_diagnostics()
        print_stage_summary("before_global", stats)

    def _record_uniform_after_global(self) -> None:
        if not self.uniform_collapse_diagnostic:
            return
        q_all, logits_all = self._forward_all_q_logits_no_grad(int(self.args.global_q_chunk_size))
        stats = compute_uniform_collapse_stage(
            "after_first_global_update",
            q_all,
            logits_all,
            self.src_t,
            self.dst_t,
            self.data.num_nodes,
            self.K,
        )
        self.uniform_diag_stages["after_first_global_update"] = stats
        initial = self.uniform_diag_stages.get("initial_before_training", {})
        self.uniform_diag_delta = uniform_delta(initial, stats) if initial else {}
        self._write_uniform_diagnostics()
        print_stage_summary("after_global", stats, self.uniform_diag_delta)

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
            "cluster_loss",
            "projection_loss",
            "global_total_loss",
            "prox_loss",
            "Q_mean_entropy",
            "max_cluster_ratio",
            "min_cluster_ratio",
            "empty_cluster_count",
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
                "Pi_E_nnz": int(self.Pi_E.nnz),
                "W_E_nnz": int(self.W_E.nnz),
                "W_E_avg_nnz_per_row": float(self.W_E.nnz / max(1, self.W_E.shape[0])),
                "W_E_sparse_mode": self.W_E_sparse_mode,
                "legacy_balance_disabled": self.cluster_loss_type == "trace_mincut",
                "trace_mincut_complexity": "O(nnz(W_E) K + M K^2)",
                "node_emb_optimizer_info": self.node_emb_optimizer_info,
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
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
            "runtime_seconds": float(runtime_seconds),
            "M": int(self.data.num_events),
            "N": int(self.data.num_nodes),
            "K": int(self.K),
            "Pi_E_nnz": int(self.Pi_E.nnz),
            "W_E_nnz": int(self.W_E.nnz),
            "W_E_avg_nnz_per_row": float(self.W_E.nnz / max(1, self.W_E.shape[0])),
            "W_E_sparse_mode": self.W_E_sparse_mode,
            "trace_mincut_complexity": "O(nnz(W_E) K + M K^2)",
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
        total_start = time.time()
        self._init_metrics_csv()
        self._record_uniform_initial()

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
            prox_steps = 0
            if self.ncut_scope == "global":
                if lambda_prox > 0.0:
                    order = self.rng.permutation(m)
                    iterator = range(0, m, batch_size)
                    for start in tqdm(iterator, desc=f"Edge-HiNoS epoch {epoch}", leave=False, disable=quiet):
                        batch_ids = order[start : start + batch_size]
                        union_ids = self._batch_union_ids(batch_ids)
                        local_index = {int(eid): i for i, eid in enumerate(union_ids.tolist())}
                        r_union, _ = self._forward_ids(union_ids)
                        l_prox = edge_ppr_proximity_loss(
                            r_union, local_index, batch_ids, self.Pi_E, m, self.rng, self.device
                        )
                        loss = lambda_prox * l_prox
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        prox_total += scalar_value(l_prox)
                        prox_steps += 1
            else:
                order = self.rng.permutation(m)
                iterator = range(0, m, batch_size)
                for start in tqdm(iterator, desc=f"Edge-HiNoS epoch {epoch}", leave=False, disable=quiet):
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

                    l_prox = edge_ppr_proximity_loss(
                        r_union, local_index, batch_ids, self.Pi_E, m, self.rng, self.device
                    )
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
                    loss.backward()
                    self.optimizer.step()
                    prox_total += scalar_value(l_prox)
                    prox_steps += 1

            after_prox_metrics = self._evaluate_stage() if diagnostic else {}

            cut_loss_value = float("nan")
            orth_loss_value = float("nan")
            cluster_loss_value = float("nan")
            projection_loss_value = 0.0
            global_total_loss_value = 0.0
            cluster_forward_seconds = 0.0
            cluster_backward_seconds = 0.0
            global_q_forwards = 0
            if self.ncut_scope == "global" and epoch >= warmup_epochs:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                should_uniform_diag = (
                    self.uniform_collapse_diagnostic
                    and not self.uniform_diag_after_first_done
                    and (not self.diagnostic_only_first_epoch or epoch == 1)
                )
                if should_uniform_diag:
                    q_all, logits_all = self._forward_all_q_logits_with_grad(int(self.args.global_q_chunk_size))
                else:
                    q_all = self._forward_all_q_with_grad(int(self.args.global_q_chunk_size))
                    logits_all = None
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

                if self.cluster_loss_type == "trace_mincut":
                    global_loss = lambda_edge_ncut * cluster_loss + lambda_proj * proj_loss
                else:
                    global_loss = lambda_edge_ncut * cluster_loss + lambda_proj * proj_loss + lambda_bal * orth_loss

                cut_loss_value = scalar_value(cut_loss)
                orth_loss_value = scalar_value(orth_loss)
                cluster_loss_value = scalar_value(cluster_loss)
                projection_loss_value = scalar_value(proj_loss)
                global_total_loss_value = scalar_value(global_loss)

                if should_uniform_diag:
                    self._record_uniform_before_global(q_all, logits_all, cut_loss, orth_loss)

                if global_loss.requires_grad and (
                    lambda_edge_ncut != 0.0 or lambda_proj != 0.0 or (self.cluster_loss_type == "legacy_ncut" and lambda_bal != 0.0)
                ):
                    sync_cuda()
                    backward_start = time.time()
                    global_loss.backward()
                    sync_cuda()
                    cluster_backward_seconds = time.time() - backward_start
                    self.optimizer.step()
                    if should_uniform_diag:
                        self._record_uniform_after_global()
                        self.uniform_diag_after_first_done = True

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
            record = {
                "epoch": epoch,
                "ACC": metrics.get("ACC", 0.0),
                "NMI": metrics.get("NMI", 0.0),
                "ARI": metrics.get("ARI", 0.0),
                "Macro_F1": metrics.get("Macro_F1", 0.0),
                "cut_loss": cut_loss_value,
                "orth_loss": orth_loss_value,
                "cluster_loss": cluster_loss_value,
                "projection_loss": projection_loss_value,
                "global_total_loss": global_total_loss_value,
                "prox_loss": prox_total / max(1, prox_steps),
                "Q_mean_entropy": metrics.get("Q_mean_entropy", 0.0),
                "max_cluster_ratio": metrics.get("max_cluster_ratio", 0.0),
                "min_cluster_ratio": metrics.get("min_cluster_ratio", 0.0),
                "empty_cluster_count": metrics.get("empty_cluster_count", 0),
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
                    f"proj={projection_loss_value:.4f} cluster={cluster_loss_value:.4f} "
                    f"global={global_total_loss_value:.4f} prox={record['prox_loss']:.4f} "
                    f"entropy={record['Q_mean_entropy']:.4f} max_ratio={record['max_cluster_ratio']:.4f} "
                    f"min_ratio={record['min_cluster_ratio']:.4f} empty={int(record['empty_cluster_count'])} "
                    f"cluster_forward_seconds={cluster_forward_seconds:.4f} "
                    f"cluster_backward_seconds={cluster_backward_seconds:.4f} "
                    f"epoch_seconds={epoch_seconds:.4f} peak_gpu_memory_mb={peak_gpu_memory_mb:.2f} "
                    f"global_q_forwards={global_q_forwards}"
                )
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
