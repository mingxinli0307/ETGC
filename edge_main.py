import argparse
import os
import time

import torch

from edge_train import EdgeHiNoSTrainer
from utils import resolve_path, set_random_seed


def build_parser():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="ETGC over temporal edge events.")
    parser.add_argument("--dataset", default="school")
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_seed", type=int, default=None)
    parser.add_argument("--forest_seed", type=int, default=None)
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
    parser.add_argument("--time_feature_mode", choices=["current", "history"], default="current")
    parser.add_argument("--edge_encoder_mode", choices=["mlp", "direct_node_time"], default="direct_node_time")
    parser.add_argument("--direct_time_scale", type=float, default=1.0)
    parser.add_argument("--cluster_head_type", choices=["legacy_mlp", "cosine_prototype"], default="cosine_prototype")
    parser.add_argument("--prototype_temperature", type=float, default=0.2)
    parser.add_argument("--require_pretrained_node2vec", type=int, choices=[0, 1], default=0)
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
        "--affinity_sparsify",
        choices=["symmetric_union_knn", "none"],
        default="symmetric_union_knn",
    )
    parser.add_argument(
        "--edge_ppr_method",
        choices=["temporal_state_forest", "forest", "legacy_temporal_forest", "truncated"],
        default="temporal_state_forest",
    )
    parser.add_argument("--forest_samples", type=int, default=50)
    parser.add_argument("--ncut_scope", choices=["global"], default="global")
    parser.add_argument(
        "--cluster_loss_type",
        choices=["matrix_ncut"],
        default="matrix_ncut",
    )
    parser.add_argument("--global_q_chunk_size", type=int, default=8192)
    parser.add_argument("--global_ncut_row_block_size", type=int, default=65536)
    parser.add_argument("--prox_warmup_epochs", type=int, default=5)
    parser.add_argument("--quiet", type=int, default=0)
    parser.add_argument("--lambda_prox", type=float, default=1.0)
    parser.add_argument("--lambda_edge_ncut", type=float, default=0.5)
    parser.add_argument("--lambda_orth", type=float, default=1.0)
    parser.add_argument("--lambda_esg", type=float, default=1.0)
    parser.add_argument("--lambda_proj", type=float, default=0.0)
    parser.add_argument("--node_emb_mode", choices=["frozen", "small_lr", "full"], default="small_lr")
    parser.add_argument("--node_emb_lr", type=float, default=1e-5)
    parser.add_argument("--prox_similarity_mode", choices=["event_dot", "cosine", "role_aware"], default="cosine")
    parser.add_argument("--prox_role_ss_weight", type=float, default=0.25)
    parser.add_argument("--prox_role_dd_weight", type=float, default=0.25)
    parser.add_argument("--prox_role_ds_weight", type=float, default=1.0)
    parser.add_argument("--prox_role_sd_weight", type=float, default=0.0)
    parser.add_argument("--prox_role_time_weight", type=float, default=0.25)
    parser.add_argument("--prox_temperature", type=float, default=0.2)
    parser.add_argument("--cluster_output_bias_mode", choices=["default", "zero", "none"], default="default")
    parser.add_argument("--cluster_input_norm", choices=["none", "layernorm"], default="none")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_embeddings", type=int, default=0)
    parser.add_argument("--hier_ncut_h", type=int, default=-1)
    parser.add_argument("--sharpen_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_fine_ncut", type=float, default=1.0)
    parser.add_argument("--lambda_coarse_ncut", type=float, default=1.0)
    return parser



def get_args():
    return build_parser().parse_args()


def main(args):
    started = time.time()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model_seed is None:
        args.model_seed = args.seed
    if args.forest_seed is None:
        args.forest_seed = args.seed
    args.seed = args.model_seed
    for name in ("data_root", "emb_root", "pretrain_emb_dir", "feature_path", "cache_dir",
                 "output_dir"):
        if getattr(args, name):
            setattr(args, name, resolve_path(cur_dir, getattr(args, name)))
    # Never overwrite a previous experiment's output.
    if args.output_dir and os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(f"output_dir must be new or empty: {args.output_dir}")
    set_random_seed(args.model_seed)
    trainer = EdgeHiNoSTrainer(args)
    trainer.write_config_json()
    print(f"ETGC hierarchical C-form: M={trainer.data.num_events} -> H={trainer.H} -> K={trainer.K}")
    print(f"gamma={args.sharpen_gamma}; projection=RowNorm(incidence_NxM @ Q_final)")
    print(f"device={trainer.device}; F1=Hungarian-matched Macro_F1", flush=True)
    if args.cluster_head_type == "cosine_prototype":
        print("Cosine prototypes are trainable parameters initialized randomly; no KMeans seeding is used.", flush=True)
    best_epoch, best_metrics = trainer.train()
    trainer.write_result_json(best_epoch, best_metrics, trainer.final_metrics, time.time() - started)
    print(f"best_epoch={best_epoch} final_metrics={trainer.final_metrics} status=success", flush=True)


if __name__ == "__main__":
    main(get_args())
