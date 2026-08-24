import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_component_prior_init_eval import Task, build_task_command, make_tasks


def test_component_prior_init_plan_is_four_datasets_five_seeds():
    tasks = make_tasks()
    assert len(tasks) == 20
    assert {task.dataset for task in tasks} == {"school", "dblp", "patent", "arXivAI"}
    assert {task.seed for task in tasks} == {42, 43, 44, 45, 46}


def test_component_prior_init_command_executes_no_training_epoch(tmp_path):
    args = SimpleNamespace(
        device="cuda:0",
        asset_root=tmp_path / "assets",
        python_bin="python",
    )
    cmd = build_task_command(args, Task("dblp", 42), tmp_path / "run")
    rendered = " ".join(cmd)
    assert "--dataset dblp" in rendered
    assert "--init_only 1" in rendered
    assert "--epoch 1" in rendered
    assert "--node_prior_mode component_structural" in rendered
    assert "--node_prior_logit_strength 4.0" in rendered
    assert "--node_prior_restarts 500" in rendered
    assert "--node_prior_bisecting_restarts 50" in rendered
    assert "--forest_samples 50" in rendered
    assert "--cluster_loss_type matrix_ncut" in rendered
