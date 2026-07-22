import os
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from edge_data import load_edge_event_data
from edge_losses import balance_loss, edge_ncut_loss, edge_ppr_proximity_loss, projection_loss
from edge_metrics import evaluate_node_clustering
from edge_model import EdgeHiNoSModel, load_pretrained_node_features
from edge_proximity import compute_edge_ppr_cached
from edge_time import build_edge_time_features
from utils import choose_device, ensure_dir


class EdgeHiNoSTrainer:
    def __init__(self, args):
        self.args = args
        self.device = choose_device(args.device)
        self.rng = np.random.RandomState(int(args.seed))

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
        )
        self.Pi_E.sort_indices()
        self.W_E.sort_indices()

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
        self.optimizer = Adam(self.model.parameters(), lr=float(args.learning_rate))

        self.src_t = torch.from_numpy(self.data.src).long().to(self.device)
        self.dst_t = torch.from_numpy(self.data.dst).long().to(self.device)
        self.time_feat_t = torch.from_numpy(self.time_feat_np).float().to(self.device)

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

    def train(self) -> Tuple[int, dict]:
        best_epoch = -1
        best_metrics = None
        best_key = None
        m = self.data.num_events
        batch_size = int(self.args.batch_size)
        for epoch in range(1, int(self.args.epoch) + 1):
            self.model.train()
            order = self.rng.permutation(m)
            total_loss = 0.0
            steps = 0
            iterator = range(0, m, batch_size)
            for start in tqdm(iterator, desc=f"Edge-HiNoS epoch {epoch}", leave=False):
                batch_ids = order[start : start + batch_size]
                union_ids = self._batch_union_ids(batch_ids)
                local_index = {int(eid): i for i, eid in enumerate(union_ids.tolist())}
                r_union, q_union = self._forward_ids(union_ids)
                batch_local = torch.as_tensor([local_index[int(e)] for e in batch_ids], dtype=torch.long, device=self.device)
                q_batch = q_union.index_select(0, batch_local)
                batch_t = torch.from_numpy(batch_ids).long().to(self.device)

                l_prox = edge_ppr_proximity_loss(
                    r_union, local_index, batch_ids, self.Pi_E, m, self.rng, self.device
                )
                l_ncut = edge_ncut_loss(q_union, union_ids, self.W_E, self.K)
                l_proj = projection_loss(
                    q_batch,
                    self.src_t.index_select(0, batch_t),
                    self.dst_t.index_select(0, batch_t),
                    self.data.num_nodes,
                )
                l_bal = balance_loss(q_union, self.K)
                loss = (
                    float(self.args.lambda_prox) * l_prox
                    + float(self.args.lambda_edge_ncut) * l_ncut
                    + float(self.args.lambda_proj) * l_proj
                    + float(self.args.lambda_bal) * l_bal
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().cpu())
                steps += 1

            if epoch % int(self.args.eval_every) == 0 or epoch == int(self.args.epoch):
                metrics, _ = self.evaluate()
                key = (float(metrics.get("ACC", 0.0)), float(metrics.get("F1", 0.0)))
                if best_key is None or key > best_key:
                    best_key = key
                    best_epoch = epoch
                    best_metrics = metrics
                print(
                    f"epoch={epoch} loss={total_loss / max(1, steps):.4f} "
                    f"ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f} "
                    f"ARI={metrics['ARI']:.4f} F1={metrics['F1']:.4f}"
                )
        if int(self.args.save_embeddings):
            self.save_outputs()
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
        S = np.zeros((self.data.num_nodes, self.K), dtype=np.float32)
        np.add.at(S, self.data.src, Q)
        np.add.at(S, self.data.dst, Q)
        S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-8)
        pred_y = S.argmax(axis=1)
        return evaluate_node_clustering(self.data.labels, pred_y), S

    def save_outputs(self) -> None:
        out_dir = os.path.join(self.args.emb_root, self.args.dataset)
        ensure_dir(out_dir)
        Q = self.infer_Q()
        np.save(os.path.join(out_dir, "edge_hinos_Q.npy"), Q)
        _, S = self.evaluate()
        np.save(os.path.join(out_dir, "edge_hinos_S.npy"), S)
