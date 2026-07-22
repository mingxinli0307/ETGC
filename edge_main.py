import argparse
import os
import time

import torch

from edge_train import EdgeHiNoSTrainer
from utils import resolve_path, set_random_seed


def build_parser():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Edge-HiNoS over temporal edge events.")
    parser.add_argument("--dataset", default="school")
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default=os.path.join(cur_dir, "dataset"))
    parser.add_argument("--emb_root", default=os.path.join(cur_dir, "emb"))
    parser.add_argument("--pretrain_emb_dir", default=os.path.join(cur_dir, "pretrain"))
    parser.add_argument("--feature_path", default="", help="Optional explicit node embedding file.")
    parser.add_argument("--cache_dir", default=os.path.join(cur_dir, "cache"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--edge_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--edge_hidden_dim", type=int, default=128)
    parser.add_argument("--cluster_hidden_dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument(
        "--edge_neighbor_k",
        type=int,
        default=-1,
        help="Maximum number of future temporal successor edge events per endpoint-history state. <=0 means using all successors.",
    )
    parser.add_argument("--edge_ppr_topk", type=int, default=20)
    parser.add_argument(
        "--edge_ppr_method",
        choices=["temporal_state_forest", "forest", "legacy_temporal_forest", "truncated"],
        default="temporal_state_forest",
    )
    parser.add_argument("--forest_samples", type=int, default=5)
    parser.add_argument("--ncut_scope", choices=["batch", "global"], default="batch")
    parser.add_argument("--global_q_chunk_size", type=int, default=8192)
    parser.add_argument("--global_ncut_row_block_size", type=int, default=65536)
    parser.add_argument("--quiet", type=int, default=0)
    parser.add_argument("--lambda_prox", type=float, default=1.0)
    parser.add_argument("--lambda_edge_ncut", type=float, default=0.5)
    parser.add_argument("--lambda_proj", type=float, default=0.2)
    parser.add_argument("--lambda_bal", type=float, default=50.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_embeddings", type=int, default=0)
    return parser


def get_args():
    return build_parser().parse_args()


def print_config(args, K=None):
    print("\n========== Edge-HiNoS Run ==========")
    print("mode=Edge-HiNoS")
    print("object=temporal edge event")
    print(f"dataset={args.dataset}, device={args.device}, directed={bool(args.directed)}")
    print(f"edge_ppr_method={args.edge_ppr_method}")
    print("default_proximity=state-expanded temporal subdivision forest")
    print(f"successor_limit={args.edge_neighbor_k}, with <=0 meaning all successors")
    print("time_usage=raw_normalized_float_timestamp")
    print(f"K_edge=K_node={K if K is not None else 'from node2label unique labels'}")
    print("projection=S=RowNorm(B_T Q)")
    print(
        f"alpha={args.alpha}, T={args.T}, beta={args.beta}, edge_neighbor_k={args.edge_neighbor_k}, "
        f"edge_ppr_topk={args.edge_ppr_topk}, forest_samples={args.forest_samples}"
    )
    print(
        f"ncut_scope={args.ncut_scope}, global_q_chunk_size={args.global_q_chunk_size}, "
        f"global_ncut_row_block_size={args.global_ncut_row_block_size}, F1_type=macro"
    )
    print(
        f"lambda_prox={args.lambda_prox}, lambda_edge_ncut={args.lambda_edge_ncut}, "
        f"lambda_proj={args.lambda_proj}, lambda_bal={args.lambda_bal}"
    )
    print("====================================\n")


def main(args):
    start_time = time.time()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.emb_root = resolve_path(cur_dir, args.emb_root)
    args.pretrain_emb_dir = resolve_path(cur_dir, args.pretrain_emb_dir)
    if args.feature_path:
        args.feature_path = resolve_path(cur_dir, args.feature_path)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    set_random_seed(args.seed)
    trainer = EdgeHiNoSTrainer(args)
    print_config(args, trainer.K)
    stats = trainer.prox_stats
    print(f"seed={args.seed}")
    print(f"resolved_device={trainer.device}")
    print(f"num_nodes={trainer.data.num_nodes} num_events={trainer.data.num_events} K={trainer.K}")
    print(f"P_E shape={stats['P_shape']} nnz={stats['P_nnz']} avg_outdegree={stats['P_avg_outdegree']:.4f}")
    print(f"Pi_E shape={stats['Pi_shape']} nnz={stats['Pi_nnz']} avg_row_nnz={stats['Pi_avg_row_nnz']:.4f}")
    print(f"W_E shape={stats['W_shape']} nnz={stats['W_nnz']}")
    print(f"edge_ppr_topk={args.edge_ppr_topk}")
    print(f"Pi_E nnz={stats['Pi_nnz']}")
    print(f"W_E nnz={stats['W_nnz']}")
    print(f"ncut_scope={args.ncut_scope}")
    print("F1_type=macro")
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    best_epoch, metrics = trainer.train()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(trainer.device) / (1024.0 * 1024.0)
    else:
        peak_gpu_memory_mb = 0.0
    runtime_seconds = time.time() - start_time
    print("\nFinal Results:")
    print(f"best_epoch={best_epoch}")
    for key in ["ACC", "NMI", "ARI", "Macro_F1"]:
        print(f"{key}={metrics.get(key, 0.0):.4f}")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(f"peak_gpu_memory_mb={peak_gpu_memory_mb:.2f}")
    print("status=success")


if __name__ == "__main__":
    main(get_args())
