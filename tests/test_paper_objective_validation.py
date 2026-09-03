import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_paper_objective_validation import (  # noqa: E402
    DATASETS,
    OBJECTIVES,
    SEEDS,
    aggregate_rows,
    build_task_command,
    make_tasks,
    run_dir_for,
    validate_devices,
)


def _args(tmp_path):
    return SimpleNamespace(
        root_dir=ROOT,
        asset_root=tmp_path / "assets",
        output_dir=tmp_path / "out",
        python_bin=sys.executable,
        epochs=20,
        no_resume=False,
    )


def test_objective_matrix_is_four_datasets_by_five_seeds():
    assert set(OBJECTIVES) == {"O11_full", "O10_cut_only", "O01_orth_only", "O00_prior_only"}
    assert len(DATASETS) == 4
    assert SEEDS == (42, 43, 44, 45, 46)
    assert len(make_tasks()) == 4 * 4 * 5


def test_objective_commands_use_matrix_ncut_forest50_and_independent_scales(tmp_path):
    args = _args(tmp_path)
    task = next(task for task in make_tasks() if task.config == "O01_orth_only" and task.dataset == "dblp")
    command = " ".join(build_task_command(args, task, "cuda:0", run_dir_for(args.output_dir, task)))
    for fragment in (
        "--cluster_loss_type matrix_ncut",
        "--orth_type orth",
        "--forest_samples 50",
        "--global_cut_scale 0.0",
        "--global_orth_scale 1.0",
        "--lambda_edge_ncut 0.5",
        "--node_prior_logit_strength 4.0",
    ):
        assert fragment in command


def test_aggregate_rows_keeps_acc_and_reports_population_std():
    rows = []
    for config in OBJECTIVES:
        for dataset in DATASETS:
            for seed, value in ((42, 0.4), (43, 0.6)):
                rows.append(
                    {
                        "config": config,
                        "dataset": dataset,
                        "seed": seed,
                        "status": "success",
                        "final_acc": value,
                        "final_macro_f1": value - 0.1,
                    }
                )
    aggregate = aggregate_rows(rows)[0]
    assert aggregate["final_acc_mean"] == 0.5
    assert aggregate["final_acc_std"] == pytest.approx(0.1)
    assert aggregate["final_macro_f1_mean"] == pytest.approx(0.4)


def test_formal_runner_rejects_non_a100_device():
    validate_devices(["cuda:0"], ["NVIDIA A100 80GB PCIe"], "A100")
    with pytest.raises(RuntimeError, match="does not contain"):
        validate_devices(["cuda:0"], ["NVIDIA GeForce RTX 4060 Ti"], "A100")
