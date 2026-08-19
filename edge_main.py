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
    parser.add_argument("--prototype_seed", type=int, default=None)
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
        choices=["row_topk", "symmetric_union_knn", "none"],
        default="symmetric_union_knn",
    )
    parser.add_argument(
        "--edge_ppr_method",
        choices=["temporal_state_forest", "forest", "legacy_temporal_forest", "truncated"],
        default="temporal_state_forest",
    )
    parser.add_argument("--forest_samples", type=int, default=50)
    parser.add_argument("--ncut_scope", choices=["batch", "global"], default="global")
    parser.add_argument(
        "--cluster_loss_type",
        choices=["matrix_ncut", "legacy_trace_ratio", "trace_mincut", "legacy_ncut"],
        default="matrix_ncut",
    )
    parser.add_argument("--orth_type", choices=["orth", "orthqa"], default="orthqa")
    parser.add_argument("--global_q_chunk_size", type=int, default=8192)
    parser.add_argument("--global_ncut_row_block_size", type=int, default=65536)
    parser.add_argument("--global_warmup_epochs", type=int, default=0)
    parser.add_argument("--prox_warmup_epochs", type=int, default=5)
    parser.add_argument("--quiet", type=int, default=0)
    parser.add_argument("--lambda_prox", type=float, default=1.0)
    parser.add_argument("--lambda_edge_ncut", type=float, default=0.5)
    parser.add_argument("--lambda_orth", type=float, default=1.0)
    parser.add_argument("--lambda_proj", type=float, default=0.0)
    parser.add_argument("--lambda_bal", type=float, default=50.0)
    parser.add_argument("--lambda_node_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_node_sbm", type=float, default=0.0)
    parser.add_argument("--node_sbm_negative_ratio", type=float, default=1.0)
    parser.add_argument("--lambda_node_prior", type=float, default=0.0)
    parser.add_argument(
        "--node_prior_mode",
        choices=["none", "adaptive_temporal_kmeans"],
        default="none",
    )
    parser.add_argument("--node_prior_restarts", type=int, default=100)
    parser.add_argument("--node_prior_seed", type=int, default=10000)
    parser.add_argument("--node_prior_lloyd_iters", type=int, default=30)
    parser.add_argument("--node_prior_auc_threshold", type=float, default=0.9)
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
    parser.add_argument(
        "--cluster_init_mode",
        choices=["random", "random_orthogonal", "random_event", "kmeans_plus_plus", "prototype"],
        default="random",
    )
    parser.add_argument("--prototype_init_mode", choices=["random", "random_orthogonal", "kmeans_plus_plus"], default="kmeans_plus_plus")
    parser.add_argument("--prototype_sample_size", type=int, default=20000)
    parser.add_argument("--prototype_lloyd_iters", type=int, default=10)
    parser.add_argument("--direct_kmeans_eval", type=int, choices=[0, 1], default=0)
    parser.add_argument("--init_only", type=int, choices=[0, 1], default=0)
    parser.add_argument("--overnight_diagnostic", type=int, choices=[0, 1], default=0)
    parser.add_argument("--loss_formulation_diagnostic", type=int, choices=[0, 1], default=0)
    parser.add_argument("--diagnostic_epochs", default="1,5,10,20,30")
    parser.add_argument("--diagnostic_stages", type=int, choices=[0, 1], default=0)
    parser.add_argument("--uniform_collapse_diagnostic", type=int, choices=[0, 1], default=0)
    parser.add_argument("--diagnostic_output_dir", default="diagnostics/uniform_collapse")
    parser.add_argument("--diagnostic_only_first_epoch", type=int, default=1)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_embeddings", type=int, default=0)
    return parser


def get_args():
    return build_parser().parse_args()


def print_config(args, K=None):
    print("\n========== ETGC Run ==========")
    print("mode=ETGC")
    print("object=temporal edge event")
    print(f"dataset={args.dataset}, device={args.device}, directed={bool(args.directed)}")
    print(f"seed={args.seed}, model_seed={args.model_seed}, prototype_seed={args.prototype_seed}, forest_seed={args.forest_seed}")
    print(f"edge_ppr_method={args.edge_ppr_method}")
    print("default_proximity=state-expanded temporal subdivision forest")
    print(f"successor_limit={args.edge_neighbor_k}, with <=0 meaning all successors")
    print(f"time_feature_mode={args.time_feature_mode}")
    print("time_usage=current_timestamp_only" if args.time_feature_mode == "current" else "time_usage=history_compatibility")
    print(f"K_edge=K_node={K if K is not None else 'from node2label unique labels'}")
    print("projection=S=RowNorm(B_T Q)")
    if args.cluster_loss_type == "matrix_ncut":
        print("cluster_objective=global matrix Ncut")
        print("cluster_loss_formula=Tr[(QTDQ)^-1 QT(D-Pi)Q]")
        print("cut_affinity_source=temporal_edge_ppr")
        print("cut_affinity_symmetrization=0.5*(Pi_E+Pi_E.T)")
        print("cut_affinity_diagonal=zero")
        print("matrix_ncut_complexity=O(nnz(Pi_cut) K + M K^2 + K^3)")
    elif args.cluster_loss_type in {"legacy_trace_ratio", "trace_mincut"}:
        print("cluster_objective=legacy scalar trace ratio")
        print("legacy_trace_ratio_complexity=O(nnz(W_E) K + M K^2)")
    else:
        print("cluster_objective=legacy ncut")
    print(
        f"alpha={args.alpha}, T={args.T}, beta={args.beta}, edge_neighbor_k={args.edge_neighbor_k}, "
        f"forest_samples={args.forest_samples}"
    )
    print(f"edge_ppr_topk={args.edge_ppr_topk}")
    print(f"affinity_sparsify={args.affinity_sparsify}")
    print(
        f"ncut_scope={args.ncut_scope}, global_q_chunk_size={args.global_q_chunk_size}, "
        f"global_ncut_row_block_size={args.global_ncut_row_block_size}, "
        f"global_warmup_epochs={args.global_warmup_epochs}, prox_warmup_epochs={args.prox_warmup_epochs}, "
        f"F1_type=macro"
    )
    print(f"cluster_loss_type={args.cluster_loss_type}, orth_type={args.orth_type}, lambda_orth={args.lambda_orth}")
    print(
        f"lambda_prox={args.lambda_prox}, lambda_edge_ncut={args.lambda_edge_ncut}, "
        f"lambda_proj={args.lambda_proj}, lambda_bal={args.lambda_bal}"
    )
    print(f"legacy_balance_disabled={str(args.cluster_loss_type != 'legacy_ncut').lower()}")
    print(f"node_emb_mode={args.node_emb_mode}, node_emb_lr={args.node_emb_lr}")
    print(f"edge_encoder_mode={args.edge_encoder_mode}, direct_time_scale={args.direct_time_scale}")
    print(f"cluster_head_type={args.cluster_head_type}, prototype_temperature={args.prototype_temperature}")
    print(f"require_pretrained_node2vec={args.require_pretrained_node2vec}")
    print(
        f"prox_similarity_mode={args.prox_similarity_mode}, prox_temperature={args.prox_temperature}, "
        f"prox_role_weights=ss:{args.prox_role_ss_weight},dd:{args.prox_role_dd_weight},"
        f"ds:{args.prox_role_ds_weight},sd:{args.prox_role_sd_weight},time:{args.prox_role_time_weight}"
    )
    print(f"lambda_node_anchor={args.lambda_node_anchor}")
    print(
        f"lambda_node_sbm={args.lambda_node_sbm}, "
        f"node_sbm_negative_ratio={args.node_sbm_negative_ratio}"
    )
    print(
        f"lambda_node_prior={args.lambda_node_prior}, node_prior_mode={args.node_prior_mode}, "
        f"node_prior_restarts={args.node_prior_restarts}"
    )
    print(f"cluster_output_bias_mode={args.cluster_output_bias_mode}")
    print(f"cluster_input_norm={args.cluster_input_norm}")
    print(f"cluster_init_mode={args.cluster_init_mode}")
    print(f"prototype_init_mode={args.prototype_init_mode}")
    print(f"prototype_sample_size={args.prototype_sample_size}, prototype_lloyd_iters={args.prototype_lloyd_iters}")
    print(f"direct_kmeans_eval={args.direct_kmeans_eval}, init_only={args.init_only}")
    print(f"overnight_diagnostic={args.overnight_diagnostic}, diagnostic_epochs={args.diagnostic_epochs}")
    print(f"loss_formulation_diagnostic={args.loss_formulation_diagnostic}")
    print(f"diagnostic_stages={args.diagnostic_stages}")
    print(f"uniform_collapse_diagnostic={args.uniform_collapse_diagnostic}")
    print(f"diagnostic_output_dir={args.diagnostic_output_dir}")
    print(f"diagnostic_only_first_epoch={args.diagnostic_only_first_epoch}")
    print("====================================\n")


def main(args):
    start_time = time.time()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model_seed is None:
        args.model_seed = int(args.seed)
    if args.prototype_seed is None:
        args.prototype_seed = int(args.model_seed)
    if args.forest_seed is None:
        args.forest_seed = int(args.seed)
    args.seed = int(args.model_seed)
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.emb_root = resolve_path(cur_dir, args.emb_root)
    args.pretrain_emb_dir = resolve_path(cur_dir, args.pretrain_emb_dir)
    if args.feature_path:
        args.feature_path = resolve_path(cur_dir, args.feature_path)
    args.cache_dir = resolve_path(cur_dir, args.cache_dir)
    args.diagnostic_output_dir = resolve_path(cur_dir, args.diagnostic_output_dir)
    set_random_seed(args.model_seed)
    trainer = EdgeHiNoSTrainer(args)
    # Persist/log the canonical name even when an old trace_mincut command is replayed.
    args.cluster_loss_type = trainer.cluster_loss_type
    print_config(args, trainer.K)
    stats = trainer.prox_stats
    trainer.write_config_json()
    print(f"seed={args.seed}")
    print(f"model_seed={args.model_seed}")
    print(f"prototype_seed={args.prototype_seed}")
    print(f"forest_seed={args.forest_seed}")
    print(f"resolved_device={trainer.device}")
    print(f"num_nodes={trainer.data.num_nodes} num_events={trainer.data.num_events} K={trainer.K}")
    print(f"P_E shape={stats['P_shape']} nnz={stats['P_nnz']} avg_outdegree={stats['P_avg_outdegree']:.4f}")
    print(f"Pi_E shape={stats['Pi_shape']} nnz={stats['Pi_nnz']} avg_row_nnz={stats['Pi_avg_row_nnz']:.4f}")
    print(f"Pi_cut shape={stats['Pi_cut_shape']} nnz={stats['Pi_cut_nnz']}")
    print(f"Pi_cut avg_row_nnz={stats['Pi_cut_avg_row_nnz']:.4f}")
    print(f"Pi_symmetry_error={stats.get('Pi_cut_symmetry_error', 0.0):.8g}")
    print(f"edge_neighbor_k={args.edge_neighbor_k}")
    print(f"forest_samples={args.forest_samples}")
    print(f"Pi_cut isolated event count={stats.get('Pi_cut_isolated_event_count', 0)}")
    print(f"D_Pi degree min={stats.get('D_Pi_degree_min', 0.0):.8g}")
    print(f"D_Pi degree max={stats.get('D_Pi_degree_max', 0.0):.8g}")
    print(f"D_Pi degree mean={stats.get('D_Pi_degree_mean', 0.0):.8g}")
    print(f"affinity_sparsify_effective={stats.get('affinity_sparsify_effective', '')}")
    print(f"ncut_scope={args.ncut_scope}")
    print(f"orth_type={args.orth_type}")
    print(f"lambda_orth={args.lambda_orth}")
    print(f"legacy_balance_disabled={str(args.cluster_loss_type != 'legacy_ncut').lower()}")
    print(f"Pi_cut_sparse_mode={trainer.Pi_cut_sparse_mode}")
    print(f"node_emb_mode_effective={trainer.node_emb_optimizer_info['node_emb_mode']}")
    print(f"node_emb_lr_effective={trainer.node_emb_optimizer_info['node_emb_lr']}")
    print(f"other_lr_effective={trainer.node_emb_optimizer_info['other_lr']}")
    print("[model]")
    print(f"time_feature_mode={args.time_feature_mode}")
    print(f"edge_encoder_mode={trainer.edge_encoder_mode}")
    print(f"direct_time_scale={args.direct_time_scale}")
    print(f"cluster_head_type={args.cluster_head_type}")
    print(f"prototype_temperature={args.prototype_temperature}")
    print(f"node_emb_mode={args.node_emb_mode}")
    print(f"node_embedding_source={trainer.node_embedding_source}")
    print(f"node2vec_path={trainer.node_embedding_path}")
    print(f"node_dim={trainer.node_dim}")
    print(f"time_dim={args.time_dim}")
    print(f"event_repr_dim={trainer.event_repr_dim}")
    print(f"cluster_input_dim={trainer.cluster_input_dim}")
    print(f"edge_mlp_trainable_params={trainer.model_init_info.get('edge_mlp_trainable_parameter_count')}")
    print(f"node_emb_trainable={trainer.model.node_emb.requires_grad}")
    print(f"node_emb_lr={trainer.node_emb_optimizer_info['node_emb_lr']}")
    print(f"main_learning_rate={trainer.node_emb_optimizer_info.get('main_learning_rate')}")
    print(f"node_embedding_learning_rate={trainer.node_emb_optimizer_info.get('node_embedding_learning_rate')}")
    print(f"node_lr_ratio={trainer.node_emb_optimizer_info.get('node_lr_ratio')}")
    print(f"prox_similarity_mode={args.prox_similarity_mode}")
    print(f"lambda_node_anchor={args.lambda_node_anchor}")
    print(f"lambda_node_sbm={args.lambda_node_sbm}")
    print(f"lambda_node_prior={args.lambda_node_prior}")
    if trainer.node_prior_info:
        print(f"node_prior_info={trainer.node_prior_info}")
    for group in trainer.node_emb_optimizer_info.get("optimizer_groups", []):
        print(
            f"optimizer_group_name={group.get('optimizer_group_name')} "
            f"parameter_count={group.get('parameter_count')} "
            f"learning_rate={group.get('learning_rate')}"
        )
    print(f"cluster_output_bias_mode={args.cluster_output_bias_mode}")
    print(f"cluster_output_bias_l2_initial={trainer.model_init_info.get('cluster_output_bias_l2_initial')}")
    print(f"cluster_output_weight_l2_initial={trainer.model_init_info.get('cluster_output_weight_l2_initial')}")
    print(f"cluster_input_norm={args.cluster_input_norm}")
    print(f"cluster_init_mode={args.cluster_init_mode}")
    print(f"prototype_init_mode={args.prototype_init_mode}")
    print(f"prototype_init_executed={trainer.model_init_info.get('prototype_init_executed')}")
    print("F1_type=macro")
    if int(getattr(args, "direct_kmeans_eval", 0)):
        metrics = trainer.run_direct_kmeans_eval()
        runtime_seconds = time.time() - start_time
        trainer.write_result_json(
            best_epoch=0,
            best_metrics=metrics,
            final_metrics=metrics,
            runtime_seconds=runtime_seconds,
        )
        print("\nFinal Results:")
        print("best_epoch=0")
        for key in ["ACC", "NMI", "ARI", "Macro_F1"]:
            print(f"{key}={metrics.get(key, 0.0):.4f}")
        print(f"runtime_seconds={runtime_seconds:.2f}")
        print("peak_gpu_memory_mb=0.00")
        print("status=success")
        return
    if int(getattr(args, "init_only", 0)):
        metrics = trainer.run_init_only_eval()
        runtime_seconds = time.time() - start_time
        trainer.write_result_json(
            best_epoch=0,
            best_metrics=metrics,
            final_metrics=metrics,
            runtime_seconds=runtime_seconds,
        )
        print("\nFinal Results:")
        print("best_epoch=0")
        for key in ["ACC", "NMI", "ARI", "Macro_F1"]:
            print(f"{key}={metrics.get(key, 0.0):.4f}")
        print(f"runtime_seconds={runtime_seconds:.2f}")
        print("peak_gpu_memory_mb=0.00")
        print("status=success")
        return
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    best_epoch, metrics = trainer.train()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(trainer.device) / (1024.0 * 1024.0)
    else:
        peak_gpu_memory_mb = 0.0
    runtime_seconds = time.time() - start_time
    trainer.write_result_json(
        best_epoch=best_epoch,
        best_metrics=metrics,
        final_metrics=getattr(trainer, "final_metrics", metrics),
        runtime_seconds=runtime_seconds,
    )
    print("\nFinal Results:")
    print(f"best_epoch={best_epoch}")
    for key in ["ACC", "NMI", "ARI", "Macro_F1"]:
        print(f"{key}={metrics.get(key, 0.0):.4f}")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(f"peak_gpu_memory_mb={peak_gpu_memory_mb:.2f}")
    print("status=success")


if __name__ == "__main__":
    main(get_args())
