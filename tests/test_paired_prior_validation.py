import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from edge_main import build_parser
from edge_train import EdgeHiNoSTrainer
from run_component_prior_decomposition import (
    CONFIGS,
    Case,
    build_edge_command,
    make_cases,
    run_dir_for as decomposition_run_dir,
    snapshot_for as decomposition_snapshot,
)
from run_paired_objective_validation import (
    BRANCHES,
    Pair,
    build_command as paired_command,
    make_pairs,
    run_dir_for as paired_run_dir,
    snapshot_for as paired_snapshot,
)


def _toy_args(tmp_path, *extra):
    data = tmp_path / "dataset" / "toy"
    data.mkdir(parents=True, exist_ok=True)
    (data / "toy.txt").write_text(
        "0 1 0.0\n1 2 0.2\n2 3 0.4\n3 0 0.6\n0 2 0.8\n1 3 1.0\n",
        encoding="utf-8",
    )
    (data / "node2label.txt").write_text("0 0\n1 0\n2 1\n3 1\n", encoding="utf-8")
    values = [
        "--dataset", "toy", "--data_root", str(tmp_path / "dataset"),
        "--cache_dir", str(tmp_path / "cache"), "--device", "cpu",
        "--edge_ppr_method", "truncated", "--edge_ppr_topk", "-1",
        "--time_dim", "5", "--edge_dim", "4", "--edge_hidden_dim", "6",
        "--cluster_hidden_dim", "4", "--time_feature_mode", "history",
        "--edge_encoder_mode", "mlp", "--cluster_head_type", "legacy_mlp",
        "--cluster_output_bias_mode", "zero", "--cluster_input_norm", "layernorm",
        "--cluster_init_mode", "prototype", "--prototype_sample_size", "6",
        "--prototype_lloyd_iters", "1", "--node_emb_mode", "frozen",
        "--lambda_prox", "0", "--prox_warmup_epochs", "0", "--quiet", "1",
    ]
    args = build_parser().parse_args(values + list(extra))
    args.model_seed = args.seed; args.prototype_seed = args.seed; args.forest_seed = args.seed
    return args


def test_initialization_snapshot_restores_exact_model_q_and_affinity(tmp_path):
    snapshot = tmp_path / "states" / "toy.pt"
    source = EdgeHiNoSTrainer(_toy_args(tmp_path, "--initialization_state_out", str(snapshot)))
    source_q = source.infer_Q()
    source_checksum = source.model_init_info["model_state_checksum"]
    torch.manual_seed(98765)
    loaded = EdgeHiNoSTrainer(_toy_args(tmp_path, "--initialization_state_in", str(snapshot)))
    loaded_q = loaded.infer_Q()
    assert snapshot.exists()
    assert loaded.model_init_info["initialization_state_source"] == "loaded"
    assert loaded.model_init_info["model_state_checksum"] == source_checksum
    assert loaded.model_init_info["Pi_cut_checksum"] == source.model_init_info["Pi_cut_checksum"]
    assert np.array_equal(source_q, loaded_q)


def test_direct_node_prior_eval_bypasses_q_projection():
    trainer = object.__new__(EdgeHiNoSTrainer)
    trainer.node_prior_t = torch.tensor([0, 0, 1, 1])
    trainer.data = SimpleNamespace(labels=np.asarray([1, 1, 0, 0]), num_nodes=4)
    trainer.K = 2
    trainer.args = SimpleNamespace(orth_type="orth", lambda_orth=1.0)
    trainer.metrics_csv_path = ""
    trainer.epoch_records = []
    metrics = trainer.run_direct_node_prior_eval()
    assert metrics["ACC"] == pytest.approx(1.0)
    assert metrics["Macro_F1"] == pytest.approx(1.0)
    assert metrics["uses_event_assignment_Q"] is False
    assert metrics["uses_incidence_projection"] is False


def test_paired_plan_and_commands_use_shared_snapshot(tmp_path):
    assert len(make_pairs()) == 10
    assert set(BRANCHES) == {
        "P0_init_only", "P1_matrix_only", "P2_orth_only",
        "P3_matrix_orth_prior_off", "P4_current_full",
    }
    args = SimpleNamespace(device="cuda:0", asset_root=tmp_path / "assets", python_bin="python", epochs=20)
    pair = Pair("dblp", 42); snapshot = paired_snapshot(tmp_path / "out", pair)
    p0 = " ".join(paired_command(args, pair, "P0_init_only", paired_run_dir(tmp_path / "out", "P0_init_only", pair), snapshot))
    p3 = " ".join(paired_command(args, pair, "P3_matrix_orth_prior_off", paired_run_dir(tmp_path / "out", "P3_matrix_orth_prior_off", pair), snapshot))
    assert "--initialization_state_out" in p0 and "--init_only 1" in p0
    assert "--initialization_state_in" in p3
    assert "--node_prior_training_logit_strength 0.0" in p3
    assert "--cluster_loss_type matrix_ncut" in p3 and "--forest_samples 50" in p3


def test_decomposition_plan_has_four_stages_and_reuses_prior_snapshot(tmp_path):
    assert len(make_cases()) == 20
    assert len(CONFIGS) == 4
    args = SimpleNamespace(device="cuda:0", asset_root=tmp_path / "assets", python_bin="python")
    case = Case("patent", 43); snapshot = decomposition_snapshot(tmp_path / "out", case)
    b1 = " ".join(build_edge_command(args, "B1_component_prior_direct", case, decomposition_run_dir(tmp_path / "out", "B1_component_prior_direct", case), snapshot))
    b2 = " ".join(build_edge_command(args, "B2_component_prior_q_random", case, decomposition_run_dir(tmp_path / "out", "B2_component_prior_q_random", case), snapshot))
    b3 = " ".join(build_edge_command(args, "B3_component_prior_q_prototype", case, decomposition_run_dir(tmp_path / "out", "B3_component_prior_q_prototype", case), snapshot))
    assert "--direct_node_prior_eval 1" in b1 and "--initialization_state_in" in b1
    assert "--cluster_init_mode random" in b2 and "--initialization_state_out" in b2
    assert "--cluster_init_mode prototype" in b3
    assert "--apply_cluster_initialization_after_state_load 1" in b3
