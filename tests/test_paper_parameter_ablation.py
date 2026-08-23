import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_paper_parameter_ablation import (  # noqa: E402
    ABLATIONS,
    PARAMETER_AXES,
    build_task_command,
    make_ablation_tasks,
    make_parameter_tasks,
    partition_tasks,
    task_run_dir,
)


def _args(tmp_path):
    return SimpleNamespace(
        root_dir=ROOT,
        asset_root=tmp_path / "assets",
        output_dir=tmp_path / "out",
        python_bin=sys.executable,
        device="cuda:0",
        devices="cuda:0",
        epochs=20,
        no_resume=False,
    )


def test_only_three_parameter_axes_are_defined():
    assert set(PARAMETER_AXES) == {"gamma", "alpha", "lambda_orth"}
    assert all(len(spec["values"]) == 5 for spec in PARAMETER_AXES.values())
    tasks = make_parameter_tasks()
    assert len(tasks) == 3 * 5 * 2 * 1


def test_ablation_matrix_covers_final_modules_and_only_patent_global_prior():
    assert set(ABLATIONS) == {
        "A0_full",
        "A1_no_structural_prior",
        "A2_current_time_only",
        "A3_random_cluster_init",
        "A4_no_layernorm",
        "A5_no_global_cluster_objective",
        "A6_no_orth",
        "A7_global_prior",
    }
    tasks = make_ablation_tasks()
    global_tasks = [task for task in tasks if task.name == "A7_global_prior"]
    assert {(task.dataset, task.seed) for task in global_tasks} == {("patent", 42), ("patent", 43)}
    assert len(tasks) == 7 * 4 * 2 + 2


def test_no_global_cluster_objective_disables_cut_and_orth_independently(tmp_path):
    args = _args(tmp_path)
    task = next(
        task
        for task in make_ablation_tasks()
        if task.name == "A5_no_global_cluster_objective" and task.dataset == "patent" and task.seed == 42
    )
    cmd = " ".join(build_task_command(args, task, "cuda:0", task_run_dir(args.output_dir, task)))
    assert "--global_cut_scale 0.0" in cmd
    assert "--global_orth_scale 0.0" in cmd
    assert "--lambda_edge_ncut 0.5" in cmd


def test_commands_keep_paper_base_and_apply_ablation(tmp_path):
    args = _args(tmp_path)
    task = next(task for task in make_ablation_tasks() if task.name == "A4_no_layernorm" and task.dataset == "dblp" and task.seed == 43)
    run_dir = task_run_dir(args.output_dir, task)
    cmd = build_task_command(args, task, "cuda:1", run_dir)
    joined = " ".join(cmd)
    for fragment in (
        "--cluster_loss_type matrix_ncut",
        "--orth_type orth",
        "--forest_samples 50",
        "--node_prior_logit_strength 4.0",
        "--cluster_input_norm none",
        "--model_seed 43",
        "--prototype_seed 43",
        "--node_prior_seed 43",
        "--device cuda:1",
    ):
        assert fragment in joined


def test_runtime_partition_keeps_each_task_once_and_balances_large_datasets():
    tasks = make_ablation_tasks()
    queues = partition_tasks(tasks, 4)
    assert sorted((task.family, task.name, task.dataset, task.seed) for queue in queues for task in queue) == sorted(
        (task.family, task.name, task.dataset, task.seed) for task in tasks
    )
    large_counts = [sum(task.dataset in {"arXivAI", "dblp"} for task in queue) for queue in queues]
    assert max(large_counts) - min(large_counts) <= 1
