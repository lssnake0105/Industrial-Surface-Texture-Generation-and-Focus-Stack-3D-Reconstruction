"""Evaluate matched ablation checkpoints on the fixed synthetic test split.

This runner is protected: it reads checkpoints by tag, writes only under
tmp/ablation_results, and keeps claim eligibility disabled until a separate
eligibility audit is added and passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays, metrics  # noqa: E402
from train_focus_resunet_loss_experiment import FocusResUNet, augment_features, predict_tiled_upgraded  # noqa: E402

from run_ablation_variant_training import ABL_ROOT, RUN_SPECS, apply_zero_channels, safe_tag  # noqa: E402


DEFAULT_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]
DEFAULT_EVAL_TAG = "2026-06-19_matched_evaluator_smoke"
DEFAULT_CHECKPOINT_TAG = "2026-06-19_matched_training_smoke"
DEFAULT_TRAINING_SCOPE = "matched_smoke_training"
DEFAULT_EVALUATION_SCOPE = "matched_full_split_evaluator_smoke"
CHECKPOINT_CONFIG_KEYS = [
    "matched_full_candidate_training",
    "matched_smoke_training",
    "pilot_training",
    "debug_training",
]


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_config(run_id: str) -> dict[str, Any] | None:
    return read_json(ABL_ROOT / run_id / "run_config.json")


def checkpoint_for_run(run_id: str, checkpoint_tag: str) -> Path:
    cfg = run_config(run_id) or {}
    for key in CHECKPOINT_CONFIG_KEYS:
        entry = cfg.get(key, {})
        if isinstance(entry, dict) and entry.get("tag") == checkpoint_tag and entry.get("checkpoint"):
            return Path(str(entry["checkpoint"]))
    return ABL_ROOT / run_id / "checkpoints" / f"{checkpoint_tag}.pt"


def load_model_for_run(device: str, checkpoint: Path) -> FocusResUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    eval_tag = safe_tag(args.tag)
    checkpoint_tag = safe_tag(args.checkpoint_tag)
    start = time.time()
    run_ids = args.run_id or DEFAULT_RUN_IDS
    test_items = build_dataset()["test"]
    if args.max_samples:
        test_items = test_items[: args.max_samples]
    expected_samples = min(args.max_samples or 7, 7)
    smoke_evaluation = bool(args.max_samples)

    checks: list[dict[str, Any]] = []
    checkpoints: dict[str, Path] = {}
    models: dict[str, FocusResUNet] = {}
    for run_id in run_ids:
        cfg = run_config(run_id)
        checks.append(check(f"{run_id} run_config exists", cfg is not None, str(ABL_ROOT / run_id / "run_config.json")))
        if cfg is not None:
            checks.append(check(f"{run_id} claim_eligible false before eval", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
            checks.append(check(f"{run_id} main_table_eligible false before eval", cfg.get("main_table_eligible") is False, str(cfg.get("main_table_eligible"))))
        checkpoint = checkpoint_for_run(run_id, checkpoint_tag)
        checkpoints[run_id] = checkpoint
        checks.append(check(f"{run_id} checkpoint exists", checkpoint.exists(), str(checkpoint)))
        if checkpoint.exists():
            models[run_id] = load_model_for_run(args.device, checkpoint)

    per_sample_rows: list[dict[str, Any]] = []
    sample_count = 0
    for category, scenario in test_items:
        sample_count += 1
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        base = np.asarray(arrays["features"], dtype=np.float32)
        truth = np.asarray(arrays["truth"], dtype=np.float32)
        risk = np.asarray(arrays["risk"], dtype=np.float32)
        dff = np.asarray(arrays["dff"], dtype=np.float32)
        gadff = np.asarray(arrays["gadff"], dtype=np.float32)
        upgraded = augment_features(base)
        dff_metrics = metrics(dff, truth, risk, scenario.depth_range_um)
        gadff_metrics = metrics(gadff, truth, risk, scenario.depth_range_um)
        for run_id, model in models.items():
            spec = RUN_SPECS[run_id]
            masked = apply_zero_channels(upgraded, list(spec["zero_channels"]))
            pred = predict_tiled_upgraded(model, masked, args.device, tile=args.tile, overlap=args.overlap)
            row_metrics = metrics(pred, truth, risk, scenario.depth_range_um)
            per_sample_rows.append(
                {
                    "run_id": run_id,
                    "variant": spec["variant"],
                    "split": "test",
                    "category": category,
                    "sample": scenario.name,
                    "resolution": f"{scenario.width}x{scenario.height}",
                    "depth_range_um": scenario.depth_range_um,
                    "z_step_um": scenario.depth_range_um / max(DEFAULT_STACK_LAYERS - 1, 1),
                    "mae_norm": row_metrics["mae_norm"],
                    "rmse_norm": row_metrics["rmse_norm"],
                    "p90_norm": row_metrics["p90_norm"],
                    "mae_um": row_metrics["mae_um"],
                    "edge_mae_um": row_metrics["edge_mae_um"],
                    "high_risk_mae_um": row_metrics["high_risk_mae_um"],
                    "dff_mae_um": dff_metrics["mae_um"],
                    "gadff_mae_um": gadff_metrics["mae_um"],
                    "checkpoint": str(checkpoints[run_id]),
                    "checkpoint_tag": checkpoint_tag,
                    "evaluation_tag": eval_tag,
                    "training_scope": args.training_scope,
                    "evaluation_scope": args.evaluation_scope,
                    "smoke_evaluation": smoke_evaluation,
                    "claim_eligible": False,
                    "main_table_eligible": False,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        rows = [row for row in per_sample_rows if row["run_id"] == run_id]
        if not rows:
            continue
        summary_rows.append(
            {
                "run_id": run_id,
                "variant": RUN_SPECS[run_id]["variant"],
                "sample_count": len(rows),
                "mean_mae_um": float(np.mean([float(row["mae_um"]) for row in rows])),
                "mean_edge_mae_um": float(np.mean([float(row["edge_mae_um"]) for row in rows])),
                "mean_high_risk_mae_um": float(np.mean([float(row["high_risk_mae_um"]) for row in rows])),
                "mean_p90_norm": float(np.mean([float(row["p90_norm"]) for row in rows])),
                "checkpoint_tag": checkpoint_tag,
                "evaluation_tag": eval_tag,
                "training_scope": args.training_scope,
                "evaluation_scope": args.evaluation_scope,
                "smoke_evaluation": smoke_evaluation,
                "claim_eligible": False,
                "main_table_eligible": False,
            }
        )

    out_dir = ABL_ROOT / "matched_full_split_eval" / eval_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / f"{eval_tag}_per_sample_metrics.csv"
    summary_csv = out_dir / f"{eval_tag}_method_summary_metrics.csv"
    per_sample_json = out_dir / f"{eval_tag}_per_sample_metrics.json"
    summary_json = out_dir / f"{eval_tag}_method_summary_metrics.json"
    write_csv(per_sample_csv, per_sample_rows)
    write_csv(summary_csv, summary_rows)
    per_sample_json.write_text(json.dumps(per_sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for run_id in run_ids:
        run_rows = [row for row in per_sample_rows if row["run_id"] == run_id]
        run_summary = [row for row in summary_rows if row["run_id"] == run_id]
        metrics_dir = ABL_ROOT / run_id / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        run_csv = metrics_dir / f"{eval_tag}_matched_full_split_metrics.csv"
        run_json = metrics_dir / f"{eval_tag}_matched_full_split_metrics.json"
        if run_rows:
            write_csv(run_csv, run_rows)
            run_json.write_text(json.dumps({"per_sample": run_rows, "summary": run_summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg = run_config(run_id)
        if cfg is not None:
            cfg["claim_eligible"] = False
            cfg["main_table_eligible"] = False
            evaluations = cfg.setdefault("matched_full_split_evaluations", {})
            evaluations[eval_tag] = {
                "date": "2026-06-19",
                "tag": eval_tag,
                "checkpoint_tag": checkpoint_tag,
                "checkpoint": str(checkpoints[run_id]),
                "per_run_metrics_csv": str(run_csv),
                "global_per_sample_csv": str(per_sample_csv),
                "global_summary_csv": str(summary_csv),
                "sample_count": len(run_rows),
                "training_scope": args.training_scope,
                "evaluation_scope": args.evaluation_scope,
                "smoke_evaluation": smoke_evaluation,
                "claim_eligible": False,
                "main_table_eligible": False,
                "interpretation": "Matched checkpoint full-split evaluator output remains outside manuscript tables until eligibility audit passes.",
            }
            (ABL_ROOT / run_id / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    checks.append(check("matched evaluator sample count", sample_count == expected_samples, f"sample_count={sample_count}, expected={expected_samples}"))
    checks.append(check("matched evaluator per-sample rows", len(per_sample_rows) == len(models) * sample_count, f"rows={len(per_sample_rows)}"))
    checks.append(check("matched evaluator summary rows", len(summary_rows) == len(models), f"rows={len(summary_rows)}"))
    checks.append(check("matched evaluator per-sample CSV exists", per_sample_csv.exists(), str(per_sample_csv)))
    checks.append(check("matched evaluator summary CSV exists", summary_csv.exists(), str(summary_csv)))
    checks.append(check("matched evaluator claim false", True, "claim_eligible=false"))
    checks.append(check("matched evaluator main table false", True, "main_table_eligible=false"))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    report = {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "tag": eval_tag,
        "checkpoint_tag": checkpoint_tag,
        "device": args.device,
        "run_ids": run_ids,
        "sample_count": sample_count,
        "smoke_evaluation": smoke_evaluation,
        "training_scope": args.training_scope,
        "evaluation_scope": args.evaluation_scope,
        "per_sample_metrics_csv": str(per_sample_csv),
        "method_summary_metrics_csv": str(summary_csv),
        "per_sample_metrics_json": str(per_sample_json),
        "method_summary_metrics_json": str(summary_json),
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "elapsed_s": time.time() - start,
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": "Matched checkpoint evaluation under tmp only. Smoke outputs are not manuscript ablation evidence.",
        "checks": checks,
        "summary": summary_rows,
    }
    report_json = out_dir / f"{eval_tag}_summary.json"
    report_md = out_dir / f"{eval_tag}_summary.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(report_md, report)
    return report


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Matched Full-Split Evaluation Summary",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Evaluation tag: {report['tag']}",
        f"- Checkpoint tag: {report['checkpoint_tag']}",
        f"- Device: {report['device']}",
        f"- Runs: {report['run_ids']}",
        f"- Test samples: {report['sample_count']}",
        f"- Smoke evaluation: {str(report['smoke_evaluation']).lower()}",
        f"- Training scope: `{report['training_scope']}`",
        f"- Evaluation scope: `{report['evaluation_scope']}`",
        f"- Claim eligible: {str(report['claim_eligible']).lower()}",
        f"- Main table eligible: {str(report['main_table_eligible']).lower()}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "| Run | Variant | Samples | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['run_id']} | {row['variant']} | {row['sample_count']} | "
            f"{row['mean_mae_um']:.4f} | {row['mean_edge_mae_um']:.4f} | {row['mean_high_risk_mae_um']:.4f} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=sorted(RUN_SPECS), help="Run id to evaluate. May be repeated.")
    parser.add_argument("--tag", default=DEFAULT_EVAL_TAG, help="Safe evaluation artifact tag.")
    parser.add_argument("--checkpoint-tag", default=DEFAULT_CHECKPOINT_TAG, help="Checkpoint tag without .pt suffix.")
    parser.add_argument("--training-scope", default=DEFAULT_TRAINING_SCOPE)
    parser.add_argument("--evaluation-scope", default=DEFAULT_EVALUATION_SCOPE)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=1, help="Smoke-test sample limit. Use 0 for the full 7-sample test split.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    report = evaluate(args)
    print(f"Ablation matched full-split evaluation: {report['status']}")
    print(f"Runs: {', '.join(report['run_ids'])}")
    print(f"Samples: {report['sample_count']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {report['per_sample_metrics_csv']}")
    print(f"Wrote {report['method_summary_metrics_csv']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
