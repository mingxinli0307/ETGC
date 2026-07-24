#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python or python3 executable found." >&2
  exit 127
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-1}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${ROOT_DIR}/logs/trace_mincut_global/${RUN_TIMESTAMP}"
mkdir -p "${OUT_DIR}/code_info" "${OUT_DIR}/phase1_diagnosis" "${OUT_DIR}/phase2_all_datasets"

{
  git rev-parse HEAD 2>/dev/null || true
  git log -1 --oneline 2>/dev/null || true
} > "${OUT_DIR}/code_info/commit.txt"
git diff > "${OUT_DIR}/code_info/git_diff.patch" 2>/dev/null || true

{
  echo "which_python=$(command -v "${PYTHON_BIN}" || true)"
  "${PYTHON_BIN}" --version 2>&1
  "${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print("torch_version=" + str(torch.__version__))
    print("cuda_available=" + str(torch.cuda.is_available()))
    print("cuda_device=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
except Exception as exc:
    print("torch_probe_error=" + repr(exc))
PY
  env | sort | grep -E '^(DEVICE|CUDA_VISIBLE_DEVICES|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|TEMPORAL_FOREST_)=' || true
} > "${OUT_DIR}/code_info/environment.txt"

cat > "${OUT_DIR}/code_info/common_config.json" <<EOF
{
  "cluster_loss_type": "trace_mincut",
  "ncut_scope": "global",
  "edge_ppr_method": "temporal_state_forest",
  "edge_ppr_topk": -1,
  "lambda_orth": 1.0,
  "lambda_bal": 0.0,
  "trace_mincut_complexity": "O(nnz(W_E) K + M K^2)",
  "forest_samples": "not overridden by this script",
  "edge_neighbor_k": "not overridden by this script"
}
EOF

"${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR}" "${PYTHON_BIN}" "${DEVICE:-}" <<'PY'
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
python_bin = sys.argv[3]
device = sys.argv[4]

phase1_dir = out_dir / "phase1_diagnosis"
phase2_dir = out_dir / "phase2_all_datasets"
seeds = [42, 43]


def dataset_exists(name):
    ds = root / "dataset" / name
    return (ds / f"{name}.txt").exists() and (ds / "node2label.txt").exists()


def resolve_dataset(alias):
    if dataset_exists(alias):
        return alias
    dataset_root = root / "dataset"
    if not dataset_root.exists():
        return None
    for ds in dataset_root.iterdir():
        if ds.is_dir() and ds.name.lower() == alias.lower() and dataset_exists(ds.name):
            return ds.name
    return None


def count_events(dataset):
    path = root / "dataset" / dataset / f"{dataset}.txt"
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as reader:
            return sum(1 for line in reader if line.strip())
    except OSError:
        return 10**18


def all_existing_preferred():
    preferred = ["School", "Patent", "DBLP", "arXivAI", "arXivCS"]
    found = []
    for alias in preferred:
        ds = resolve_dataset(alias)
        found.append((alias, ds))
    return found


def diagnosis_dataset():
    school = resolve_dataset("School") or resolve_dataset("school")
    if school:
        return school
    candidates = [ds for _alias, ds in all_existing_preferred() if ds]
    dataset_root = root / "dataset"
    if not candidates and dataset_root.exists():
        for ds in dataset_root.iterdir():
            if ds.is_dir() and dataset_exists(ds.name):
                candidates.append(ds.name)
    if not candidates:
        return None
    return min(candidates, key=count_events)


def is_complete_result(run_dir):
    path = run_dir / "result.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "status" in data


def write_failure(run_dir, config, status, error_summary, exit_code=None, attempts=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "status": status,
        "error_summary": error_summary,
        "exit_code": exit_code,
        "attempts": attempts or [],
        "dataset": config.get("dataset"),
        "seed": config.get("seed"),
        "config_name": config.get("config_name"),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")


def command_for(config):
    cmd = [
        python_bin,
        "-u",
        "edge_main.py",
        "--dataset",
        str(config["dataset"]),
        "--cluster_loss_type",
        "trace_mincut",
        "--ncut_scope",
        "global",
        "--edge_ppr_method",
        "temporal_state_forest",
        "--edge_ppr_topk",
        "-1",
        "--lambda_bal",
        "0",
        "--lambda_orth",
        "1.0",
        "--quiet",
        "1",
        "--output_dir",
        str(config["run_dir"]),
    ]
    if device:
        cmd.extend(["--device", device])
    for key, value in config["args"].items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_one(config, oom_retries=False):
    run_dir = Path(config["run_dir"])
    if is_complete_result(run_dir):
        return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
    attempts = []
    q_chunks = [int(config["args"].get("global_q_chunk_size", 8192))]
    row_blocks = [int(config["args"].get("global_ncut_row_block_size", 65536))]
    if oom_retries:
        q_chunks = [8192, 4096, 2048, 1024]
        row_blocks = [65536, 32768, 16384, 8192, 4096]
    for q in q_chunks:
        for block in row_blocks:
            cfg = json.loads(json.dumps(config, default=str))
            cfg["run_dir"] = str(run_dir)
            cfg["args"]["global_q_chunk_size"] = q
            cfg["args"]["global_ncut_row_block_size"] = block
            log_path = run_dir / "train.log"
            cmd = command_for(cfg)
            start = time.time()
            with log_path.open("a", encoding="utf-8", errors="ignore") as log:
                log.write(f"attempt_global_q_chunk_size={q} attempt_global_ncut_row_block_size={block}\n")
                proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            attempt = {"global_q_chunk_size": q, "global_ncut_row_block_size": block, "exit_code": proc.returncode, "seconds": elapsed}
            attempts.append(attempt)
            if proc.returncode == 0 and (run_dir / "result.json").exists():
                result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                result["attempts"] = attempts
                result["exit_code"] = 0
                result["config_name"] = config.get("config_name")
                result["source_config"] = config.get("source_config", config.get("config_name"))
                result["run_dir"] = str(run_dir)
                (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                return result
            oom = re.search(r"out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA out of memory", log_text, re.I)
            if not oom:
                tail = "\n".join(log_text.splitlines()[-5:])[:400]
                write_failure(run_dir, config, "failed", tail, proc.returncode, attempts)
                return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            try:
                subprocess.run([python_bin, "-c", "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"], cwd=root)
            except Exception:
                pass
    write_failure(run_dir, config, "failed", "OOM after allowed chunk/block retries", 1, attempts)
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def read_metrics(run_dir):
    path = Path(run_dir) / "metrics.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def finite_float(value, default=math.nan):
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def longest_run(values, predicate):
    best = cur = 0
    for value in values:
        if predicate(value):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_quality(run_dir):
    result_path = Path(run_dir) / "result.json"
    metrics = read_metrics(run_dir)
    if not result_path.exists():
        return {"status": "failed", "exclude_reason": "missing result.json"}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "success":
        return {"status": result.get("status", "failed"), "exclude_reason": result.get("error_summary", "failed")}
    macro = [finite_float(row.get("Macro_F1")) for row in metrics]
    final_macro = macro[-1] if macro else finite_float(result.get("final_metrics", {}).get("Macro_F1"))
    best_macro = max([x for x in macro if math.isfinite(x)] or [finite_float(result.get("best_metrics", {}).get("Macro_F1"), 0.0)])
    empty = [finite_float(row.get("empty_cluster_count"), 0.0) for row in metrics]
    max_ratio = [finite_float(row.get("max_cluster_ratio"), 0.0) for row in metrics]
    if longest_run(empty, lambda x: x > 0) > 5:
        return {"status": "excluded", "exclude_reason": "empty clusters persisted >5 epochs", "best_macro": best_macro, "final_macro": final_macro}
    if longest_run(max_ratio, lambda x: x >= 0.95) > 5:
        return {"status": "excluded", "exclude_reason": "max cluster ratio persisted near 1", "best_macro": best_macro, "final_macro": final_macro}
    return {
        "status": "success",
        "best_macro": best_macro,
        "final_macro": final_macro,
        "gap": best_macro - final_macro,
        "runtime": finite_float(result.get("runtime_seconds"), 0.0),
        "peak_memory": max([finite_float(row.get("peak_gpu_memory_mb"), 0.0) for row in metrics] or [0.0]),
        "empty_last": empty[-1] if empty else 0.0,
    }


def aggregate_config(config_name):
    dirs = sorted(phase1_dir.glob(f"{config_name}_seed*"))
    qualities = [run_quality(path) for path in dirs]
    usable = [q for q in qualities if q.get("status") == "success"]
    if not usable:
        return {
            "Config": config_name,
            "Status": "failed",
            "Mean Best Macro-F1": math.nan,
            "Mean Final Macro-F1": math.nan,
            "Gap": math.nan,
            "Std Best Macro-F1": math.nan,
            "Mean Runtime": math.nan,
            "Empty Clusters": "",
        }
    bests = [q["best_macro"] for q in usable]
    finals = [q["final_macro"] for q in usable]
    gaps = [q["gap"] for q in usable]
    runtimes = [q["runtime"] for q in usable]
    return {
        "Config": config_name,
        "Status": "success" if len(usable) == len(dirs) else "partial",
        "Mean Best Macro-F1": sum(bests) / len(bests),
        "Mean Final Macro-F1": sum(finals) / len(finals),
        "Gap": sum(gaps) / len(gaps),
        "Std Best Macro-F1": (sum((x - sum(bests) / len(bests)) ** 2 for x in bests) / len(bests)) ** 0.5,
        "Mean Runtime": sum(runtimes) / len(runtimes),
        "Empty Clusters": max(q["empty_last"] for q in usable),
    }


def rank_configs(config_names):
    rows = [aggregate_config(name) for name in config_names]
    usable = [row for row in rows if row["Status"] in {"success", "partial"} and math.isfinite(row["Mean Best Macro-F1"])]
    if not usable:
        return None, rows
    usable.sort(key=lambda r: (-r["Mean Best Macro-F1"], r["Gap"], r["Std Best Macro-F1"], r["Mean Runtime"]))
    best = usable[0]
    tied = [row for row in usable if abs(row["Mean Best Macro-F1"] - best["Mean Best Macro-F1"]) < 0.005]
    tied.sort(key=lambda r: (r["Gap"], r["Std Best Macro-F1"], r["Mean Runtime"]))
    return tied[0]["Config"], rows


def phase1_config(name, dataset, seed, prox, proj, node_mode="frozen", warmup=0, node_emb_lr=None):
    args = {
        "epoch": 30,
        "seed": seed,
        "lambda_prox": prox,
        "lambda_edge_ncut": 0.5,
        "lambda_proj": proj,
        "node_emb_mode": node_mode,
        "global_warmup_epochs": warmup,
        "diagnostic_stages": 1,
        "global_q_chunk_size": 8192,
        "global_ncut_row_block_size": 65536,
    }
    if node_emb_lr is not None:
        args["node_emb_lr"] = node_emb_lr
    run_dir = phase1_dir / f"{name}_seed{seed}"
    return {"config_name": name, "dataset": dataset, "seed": seed, "run_dir": str(run_dir), "args": args}


def run_phase1():
    dataset = diagnosis_dataset()
    if not dataset:
        return [], None, None
    print(f"phase1_dataset={dataset}", flush=True)
    executed = []
    first = [
        ("D1_trace_only", 0.0, 0.0),
        ("D2_prox_trace", 1.0, 0.0),
    ]
    for name, prox, proj in first:
        for seed in seeds:
            run_one(phase1_config(name, dataset, seed, prox, proj), oom_retries=False)
        executed.append(name)
    best_base, rows = rank_configs(executed)
    base_prox = 0.0 if best_base == "D1_trace_only" else 1.0
    d3_names = []
    for proj in [0.01, 0.05, 0.1, 0.2]:
        tag = str(proj).replace(".", "p")
        name = f"D3_proj_{tag}"
        for seed in seeds:
            run_one(phase1_config(name, dataset, seed, base_prox, proj), oom_retries=False)
        d3_names.append(name)
    best_loss, _ = rank_configs(executed + d3_names)
    loss_cfg = next((name for name in executed + d3_names if name == best_loss), best_base or "D1_trace_only")
    if loss_cfg.startswith("D3_proj_"):
        proj_map = {"D3_proj_0p01": 0.01, "D3_proj_0p05": 0.05, "D3_proj_0p1": 0.1, "D3_proj_0p2": 0.2}
        best_proj = proj_map[loss_cfg]
        best_prox = base_prox
    elif loss_cfg == "D2_prox_trace":
        best_proj = 0.0
        best_prox = 1.0
    else:
        best_proj = 0.0
        best_prox = 0.0
    d4_names = []
    for mode, node_lr in [("frozen", None), ("small_lr", 1e-5), ("full", None)]:
        name = f"D4_node_{mode}"
        for seed in seeds:
            run_one(phase1_config(name, dataset, seed, best_prox, best_proj, node_mode=mode, node_emb_lr=node_lr), oom_retries=False)
        d4_names.append(name)
    best_node, _ = rank_configs(d4_names)
    if not best_node:
        best_node = "D4_node_frozen"
    mode = best_node.replace("D4_node_", "")
    d5_names = []
    for warmup in [0, 5]:
        name = f"D5_warmup_{warmup}"
        for seed in seeds:
            run_one(
                phase1_config(
                    name,
                    dataset,
                    seed,
                    best_prox,
                    best_proj,
                    node_mode=mode,
                    warmup=warmup,
                    node_emb_lr=1e-5 if mode == "small_lr" else None,
                ),
                oom_retries=False,
            )
        d5_names.append(name)
    all_names = executed + d3_names + d4_names + d5_names
    best_final, rows = rank_configs(all_names)
    no_proj_names = [name for name in all_names if name in {"D1_trace_only", "D2_prox_trace"}]
    stable_baseline, _ = rank_configs(no_proj_names)
    return all_names, best_final, stable_baseline


def write_phase1_summary(config_names):
    fieldnames = [
        "Config",
        "Prox",
        "Trace Mincut",
        "Projection",
        "Node Mode",
        "Warmup",
        "Mean Best Macro-F1",
        "Mean Final Macro-F1",
        "Gap",
        "Std Best Macro-F1",
        "Mean Runtime",
        "Empty Clusters",
        "Status",
    ]
    path = out_dir / "summary_phase1.csv"
    with path.open("w", encoding="utf-8", newline="") as writer:
        w = csv.DictWriter(writer, fieldnames=fieldnames)
        w.writeheader()
        for name in config_names:
            row = aggregate_config(name)
            cfg_files = sorted(phase1_dir.glob(f"{name}_seed*/config.json"))
            args = {}
            if cfg_files:
                try:
                    args = json.loads(cfg_files[0].read_text(encoding="utf-8")).get("args", {})
                except Exception:
                    args = {}
            prox = str(args.get("lambda_prox", ""))
            projection = str(args.get("lambda_proj", ""))
            node_mode = str(args.get("node_emb_mode", ""))
            warmup = str(args.get("global_warmup_epochs", ""))
            w.writerow(
                {
                    "Config": name,
                    "Prox": prox,
                    "Trace Mincut": "0.5",
                    "Projection": projection,
                    "Node Mode": node_mode,
                    "Warmup": warmup,
                    "Mean Best Macro-F1": row["Mean Best Macro-F1"],
                    "Mean Final Macro-F1": row["Mean Final Macro-F1"],
                    "Gap": row["Gap"],
                    "Std Best Macro-F1": row["Std Best Macro-F1"],
                    "Mean Runtime": row["Mean Runtime"],
                    "Empty Clusters": row["Empty Clusters"],
                    "Status": row["Status"],
                }
            )


def config_values_from_name(name, fallback_no_proj=False):
    prox = 0.0
    proj = 0.0
    node_mode = "frozen"
    warmup = 0
    if name in {"D2_prox_trace"} or name.startswith("D3_") or name.startswith("D4_") or name.startswith("D5_"):
        prox = 1.0
    proj_match = re.search(r"proj_([0-9p]+)", name)
    if proj_match and not fallback_no_proj:
        proj = float(proj_match.group(1).replace("p", "."))
    if name.startswith("D4_node_") and not fallback_no_proj:
        node_mode = name.replace("D4_node_", "")
    if name.startswith("D5_warmup_") and not fallback_no_proj:
        warmup = int(name.replace("D5_warmup_", ""))
    return prox, proj, node_mode, warmup


def phase1_args_for_config(name):
    cfg_files = sorted(phase1_dir.glob(f"{name}_seed*/config.json"))
    if not cfg_files:
        prox, proj, node_mode, warmup = config_values_from_name(name)
        return {"lambda_prox": prox, "lambda_proj": proj, "node_emb_mode": node_mode, "global_warmup_epochs": warmup}
    try:
        return json.loads(cfg_files[0].read_text(encoding="utf-8")).get("args", {})
    except Exception:
        prox, proj, node_mode, warmup = config_values_from_name(name)
        return {"lambda_prox": prox, "lambda_proj": proj, "node_emb_mode": node_mode, "global_warmup_epochs": warmup}


def formal_epochs(dataset):
    key = dataset.lower()
    if key == "patent":
        return 25
    if key in {"arxivai", "arxivcs"}:
        return 20
    return 30


def phase2_config(config_label, source_name, dataset, seed, q_chunk=8192, row_block=65536, no_proj=False):
    source_args = phase1_args_for_config(source_name)
    prox = float(source_args.get("lambda_prox", 0.0))
    proj = float(source_args.get("lambda_proj", 0.0))
    node_mode = str(source_args.get("node_emb_mode", "frozen"))
    warmup = int(source_args.get("global_warmup_epochs", 0))
    args = {
        "epoch": formal_epochs(dataset),
        "seed": seed,
        "lambda_prox": prox,
        "lambda_edge_ncut": 0.5,
        "lambda_proj": 0.0 if no_proj else proj,
        "node_emb_mode": node_mode,
        "global_warmup_epochs": warmup,
        "diagnostic_stages": 0,
        "global_q_chunk_size": q_chunk,
        "global_ncut_row_block_size": row_block,
    }
    if node_mode == "small_lr":
        args["node_emb_lr"] = 1e-5
    run_dir = phase2_dir / f"{dataset}_{config_label}_seed{seed}"
    return {"config_name": config_label, "source_config": source_name, "dataset": dataset, "seed": seed, "run_dir": str(run_dir), "args": args}


def write_phase2_missing(alias):
    run_dir = phase2_dir / f"{alias}_missing"
    write_failure(run_dir, {"config_name": "missing", "dataset": alias, "seed": 42, "run_dir": str(run_dir), "args": {}}, "missing", "dataset directory or required files missing")


def run_phase2(best_config, baseline_config):
    datasets = all_existing_preferred()
    if not best_config:
        best_config = "D1_trace_only"
    if not baseline_config:
        baseline_config = "D1_trace_only"
    for alias, dataset in datasets:
        if not dataset:
            write_phase2_missing(alias)
            continue
        run_one(phase2_config("best", best_config, dataset, 42), oom_retries=True)
        run_one(phase2_config("no_projection_baseline", baseline_config, dataset, 42, no_proj=True), oom_retries=True)


def summarize_phase2():
    fieldnames = ["Dataset", "Config", "ACC", "NMI", "ARI", "Best Macro-F1", "Final Macro-F1", "Best Epoch", "Runtime", "Peak Memory", "Status", "Error Summary"]
    rows = []
    for run_dir in sorted(phase2_dir.iterdir()) if phase2_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = read_metrics(run_dir)
        macros = [finite_float(row.get("Macro_F1")) for row in metrics]
        best_macro = max([x for x in macros if math.isfinite(x)] or [finite_float(result.get("best_metrics", {}).get("Macro_F1"))])
        final_macro = macros[-1] if macros else finite_float(result.get("final_metrics", {}).get("Macro_F1"))
        peak = max([finite_float(row.get("peak_gpu_memory_mb"), 0.0) for row in metrics] or [0.0])
        rows.append(
            {
                "Dataset": result.get("dataset", ""),
                "Config": result.get("config_name", run_dir.name),
                "ACC": result.get("final_metrics", {}).get("ACC", ""),
                "NMI": result.get("final_metrics", {}).get("NMI", ""),
                "ARI": result.get("final_metrics", {}).get("ARI", ""),
                "Best Macro-F1": best_macro,
                "Final Macro-F1": final_macro,
                "Best Epoch": result.get("best_epoch", ""),
                "Runtime": result.get("runtime_seconds", ""),
                "Peak Memory": peak,
                "Status": result.get("status", "failed"),
                "Error Summary": result.get("error_summary", ""),
            }
        )
    path = out_dir / "summary_phase2.csv"
    with path.open("w", encoding="utf-8", newline="") as writer:
        w = csv.DictWriter(writer, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return rows


def write_all_runs():
    fieldnames = ["phase", "run", "dataset", "config", "seed", "status", "best_macro_f1", "final_macro_f1", "runtime_seconds", "error_summary"]
    rows = []
    for phase, base in [("phase1", phase1_dir), ("phase2", phase2_dir)]:
        if not base.exists():
            continue
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            result_path = run_dir / "result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = read_metrics(run_dir)
            macros = [finite_float(row.get("Macro_F1")) for row in metrics]
            best_macro = max([x for x in macros if math.isfinite(x)] or [finite_float(result.get("best_metrics", {}).get("Macro_F1"))])
            final_macro = macros[-1] if macros else finite_float(result.get("final_metrics", {}).get("Macro_F1"))
            rows.append(
                {
                    "phase": phase,
                    "run": run_dir.name,
                    "dataset": result.get("dataset", ""),
                    "config": result.get("config_name", ""),
                    "seed": result.get("seed", ""),
                    "status": result.get("status", ""),
                    "best_macro_f1": best_macro,
                    "final_macro_f1": final_macro,
                    "runtime_seconds": result.get("runtime_seconds", ""),
                    "error_summary": result.get("error_summary", ""),
                }
            )
    path = out_dir / "all_runs.csv"
    with path.open("w", encoding="utf-8", newline="") as writer:
        w = csv.DictWriter(writer, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def diagnose_drop_location():
    deltas = {"after_prox": [], "after_global": []}
    evidence = []
    for run_dir in sorted(phase1_dir.glob("*_seed*")):
        metrics = read_metrics(run_dir)
        for row in metrics:
            b = finite_float(row.get("before_Macro_F1"))
            p = finite_float(row.get("after_prox_Macro_F1"))
            g = finite_float(row.get("after_global_Macro_F1"))
            if all(math.isfinite(x) for x in [b, p, g]):
                deltas["after_prox"].append(p - b)
                deltas["after_global"].append(g - p)
        if metrics:
            row = metrics[-1]
            evidence.append(
                f"{run_dir.name} epoch={row.get('epoch')} before={row.get('before_Macro_F1')} after_prox={row.get('after_prox_Macro_F1')} after_global={row.get('after_global_Macro_F1')}"
            )
    avg = {k: (sum(v) / len(v) if v else math.nan) for k, v in deltas.items()}
    if math.isfinite(avg["after_prox"]) and math.isfinite(avg["after_global"]):
        location = "after_prox" if avg["after_prox"] < avg["after_global"] else "after_global"
    else:
        location = "unknown"
    return location, avg, evidence[:5]


def write_final_report(best_config, baseline_config, phase2_rows):
    drop_location, drop_avg, evidence = diagnose_drop_location()
    lines = [
        "# Trace Mincut Global Experiments",
        "",
        f"Output directory: {out_dir}",
        f"Best phase1 config: {best_config}",
        f"No-projection baseline: {baseline_config}",
        f"Macro-F1 drop location estimate: {drop_location}",
        f"Mean delta after_prox-before: {drop_avg.get('after_prox')}",
        f"Mean delta after_global-after_prox: {drop_avg.get('after_global')}",
        "",
        "## Evidence",
    ]
    lines.extend(f"- {item}" for item in evidence)
    lines.extend(["", "## Phase 2"])
    for row in phase2_rows:
        lines.append(
            f"- {row['Dataset']} {row['Config']} status={row['Status']} best_macro_f1={row['Best Macro-F1']} final_macro_f1={row['Final Macro-F1']} runtime={row['Runtime']} peak_mb={row['Peak Memory']}"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- summary_phase1.csv: {out_dir / 'summary_phase1.csv'}",
            f"- summary_phase2.csv: {out_dir / 'summary_phase2.csv'}",
            f"- all_runs.csv: {out_dir / 'all_runs.csv'}",
        ]
    )
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation():
    validation_log = out_dir / "code_info" / "validation.log"
    with validation_log.open("w", encoding="utf-8", errors="ignore") as log:
        pyc = subprocess.run(
            [python_bin, "-m", "py_compile", "edge_main.py", "edge_train.py", "edge_losses.py", "edge_metrics.py", "edge_model.py"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"py_compile_exit_code={pyc.returncode}\n")
        tests = subprocess.run([python_bin, "-m", "pytest", "-q", "tests"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"pytest_exit_code={tests.returncode}\n")
        ds = diagnosis_dataset()
        if ds:
            smoke_dir = out_dir / "code_info" / "smoke_output"
            smoke = subprocess.run(
                [
                    python_bin,
                    "-u",
                    "edge_main.py",
                    "--dataset",
                    ds,
                    "--epoch",
                    "2",
                    "--cluster_loss_type",
                    "trace_mincut",
                    "--ncut_scope",
                    "global",
                    "--edge_ppr_topk",
                    "-1",
                    "--lambda_prox",
                    "1.0",
                    "--lambda_edge_ncut",
                    "0.5",
                    "--lambda_orth",
                    "1.0",
                    "--lambda_proj",
                    "0",
                    "--lambda_bal",
                    "0",
                    "--node_emb_mode",
                    "frozen",
                    "--global_warmup_epochs",
                    "0",
                    "--diagnostic_stages",
                    "1",
                    "--quiet",
                    "1",
                    "--output_dir",
                    str(smoke_dir),
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.write(f"smoke_dataset={ds}\n")
            log.write(f"smoke_exit_code={smoke.returncode}\n")
    return pyc.returncode == 0 and tests.returncode == 0


run_validation()
phase1_names, best_config, baseline_config = run_phase1()
write_phase1_summary(phase1_names)
run_phase2(best_config, baseline_config)
phase2_rows = summarize_phase2()
write_all_runs()
write_final_report(best_config, baseline_config, phase2_rows)
print(f"OUT_DIR={out_dir}", flush=True)
print(f"SUMMARY_PHASE1={out_dir / 'summary_phase1.csv'}", flush=True)
print(f"SUMMARY_PHASE2={out_dir / 'summary_phase2.csv'}", flush=True)
print(f"FINAL_REPORT={out_dir / 'final_report.md'}", flush=True)
PY
