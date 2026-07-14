"""Evaluate controlled-pilot ablation checkpoints on the fixed synthetic test split.

This runner is diagnostic: it computes full-test-split metrics for the tiny
controlled-pilot checkpoints, keeps all outputs under tmp/ablation_results, and
keeps claim eligibility disabled.
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
DEFAULT_TAG = "2026-06-19_full_split_debug_eval"
PILOT_TAG = "2026-06-19_controlled_pilot"


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


def load_model_for_run(run_id: str, device: str, checkpoint: Path) -> FocusResUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def checkpoint_for_run(run_id: str, tag: str) -> Path:
    cfg = read_json(ABL_ROOT / run_id / "run_config.json") or {}
    pilot = cfg.get("pilot_training", {})
    if pilot.get("tag") == PILOT_TAG and pilot.get("checkpoint"):
        return Path(str(pilot["checkpoint"]))
    return ABL_ROOT / run_id / "checkpoints" / f"{tag}.pt"


def run_config(run_id: str) -> dict[str, Any] | None:
    return read_json(ABL_ROOT / run_id / "run_config.json")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    tag = safe_tag(args.tag)
    start = time.time()
    run_ids = args.run_id or DEFAULT_RUN_IDS
    test_items = build_dataset()["test"]
    if args.max_samples:
        test_items = test_items[: args.max_samples]

    checks: list[dict[str, Any]] = []
    checkpoints: dict[str, Path] = {}
    models: dict[str, FocusResUNet] = {}
    for run_id in run_ids:
        cfg = run_config(run_id)
        checks.append(check(f"{run_id} run_config exists", cfg is not None, str(ABL_ROOT / run_id / "run_config.json")))
        if cfg is not None:
            checks.append(check(f"{run_id} claim_eligible false before eval", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
            checks.append(check(f"{run_id} main_table_eligible false before eval", cfg.get("main_table_eligible") is False, str(cfg.get("main_table_eligible"))))
        checkpoint = checkpoint_for_run(run_id, PILOT_TAG)
        checkpoints[run_id] = checkpoint
        checks.append(check(f"{run_id} pilot checkpoint exists", checkpoint.exists(), str(checkpoint)))
        if checkpoint.exists():
            models[run_id] = load_model_for_run(run_id, args.device, checkpoint)

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
                    "training_scope": "controlled_pilot_p10_debug",
                    "evaluation_scope": "full_test_split_debug_eval",
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
                "training_scope": "controlled_pilot_p10_debug",
                "evaluation_scope": "full_test_split_debug_eval",
                "claim_eligible": False,
                "main_table_eligible": False,
            }
        )

    out_dir = ABL_ROOT / "full_split_debug_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / f"{tag}_per_sample_metrics.csv"
    summary_csv = out_dir / f"{tag}_method_summary_metrics.csv"
    per_sample_json = out_dir / f"{tag}_per_sample_metrics.json"
    summary_json = out_dir / f"{tag}_method_summary_metrics.json"
    write_csv(per_sample_csv, per_sample_rows)
    write_csv(summary_csv, summary_rows)
    per_sample_json.write_text(json.dumps(per_sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for run_id in run_ids:
        run_rows = [row for row in per_sample_rows if row["run_id"] == run_id]
        run_summary = [row for row in summary_rows if row["run_id"] == run_id]
        metrics_dir = ABL_ROOT / run_id / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        run_csv = metrics_dir / f"{tag}_metrics.csv"
        run_json = metrics_dir / f"{tag}_metrics.json"
        write_csv(run_csv, run_rows)
        run_json.write_text(json.dumps({"per_sample": run_rows, "summary": run_summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg = run_config(run_id)
        if cfg is not None:
            cfg["claim_eligible"] = False
            cfg["main_table_eligible"] = False
            cfg["full_split_debug_evaluation"] = {
                "date": "2026-06-19",
                "tag": tag,
                "debug_only": True,
                "checkpoint": str(checkpoints[run_id]),
                "per_run_metrics_csv": str(run_csv),
                "global_per_sample_csv": str(per_sample_csv),
                "global_summary_csv": str(summary_csv),
                "sample_count": len(run_rows),
                "training_scope": "controlled_pilot_p10_debug",
                "evaluation_scope": "full_test_split_debug_eval",
                "interpretation": "Full test split diagnostic evaluation of controlled-pilot checkpoints; not manuscript evidence.",
            }
            (ABL_ROOT / run_id / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    checks.append(check("full split sample count", sample_count == 7 or bool(args.max_samples), f"sample_count={sample_count}"))
    checks.append(check("per-sample rows exist", len(per_sample_rows) == len(models) * sample_count, f"rows={len(per_sample_rows)}"))
    checks.append(check("summary rows exist", len(summary_rows) == len(models), f"rows={len(summary_rows)}"))
    checks.append(check("per-sample metrics CSV exists", per_sample_csv.exists(), str(per_sample_csv)))
    checks.append(check("summary metrics CSV exists", summary_csv.exists(), str(summary_csv)))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    report = {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "tag": tag,
        "device": args.device,
        "run_ids": run_ids,
        "sample_count": sample_count,
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
        "interpretation": "Full-split diagnostic evaluation for controlled-pilot checkpoints only; not manuscript ablation evidence.",
        "checks": checks,
        "summary": summary_rows,
    }
    report_json = out_dir / f"{tag}_summary.json"
    report_md = out_dir / f"{tag}_summary.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(report_md, report)
    return report


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Full-Split Debug Evaluation Summary",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Tag: {report['tag']}",
        f"- Device: {report['device']}",
        f"- Runs: {report['run_ids']}",
        f"- Test samples: {report['sample_count']}",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=sorted(RUN_SPECS), help="Run id to evaluate. May be repeated.")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional smoke-test limit.")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    report = evaluate(args)
    print(f"Ablation full-split debug evaluation: {report['status']}")
    print(f"Runs: {', '.join(report['run_ids'])}")
    print(f"Samples: {report['sample_count']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {report['per_sample_metrics_csv']}")
    print(f"Wrote {report['method_summary_metrics_csv']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
