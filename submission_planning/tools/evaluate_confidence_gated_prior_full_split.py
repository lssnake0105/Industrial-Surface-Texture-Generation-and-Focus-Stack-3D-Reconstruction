"""Evaluate ABL-07 confidence-gated prior checkpoints on the fixed test split."""

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
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays, metrics  # noqa: E402
from train_focus_resunet_loss_experiment import FocusResUNet, augment_features, predict_tiled_upgraded  # noqa: E402
from run_confidence_weighted_loss_training import ABL_ROOT, DATE, RUN_ID, safe_tag, write_json  # noqa: E402


VARIANT = "Confidence-gated DFF/GADFF prior loss"
DEFAULT_TAG = "2026-06-22_confidence_gated_prior_evaluator_smoke"
DEFAULT_CHECKPOINT_TAG = "2026-06-22_confidence_gated_prior_smoke"
STRATA = ["high_risk", "low_confidence", "normal"]


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_for_tag(checkpoint_tag: str) -> Path:
    cfg_path = ABL_ROOT / RUN_ID / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("latest_smoke", "latest_paired_loss_smoke"):
            entry = cfg.get(key)
            if isinstance(entry, dict) and entry.get("tag") == checkpoint_tag and entry.get("checkpoint"):
                return Path(str(entry["checkpoint"]))
        latest = cfg.get("latest_smoke")
        if isinstance(latest, dict) and latest.get("tag") == checkpoint_tag and latest.get("checkpoint"):
            return Path(str(latest["checkpoint"]))
    return ABL_ROOT / RUN_ID / "checkpoints" / f"{checkpoint_tag}.pt"


def load_model(checkpoint: Path, device: str) -> FocusResUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def combined_focus_confidence(base_features: np.ndarray) -> np.ndarray:
    prior_offset = DEFAULT_STACK_LAYERS
    conf = np.clip(base_features[prior_offset + 2], 0, 1)
    ga_conf = np.clip(base_features[prior_offset + 4], 0, 1)
    return np.clip(0.65 * conf + 0.35 * ga_conf, 0, 1).astype(np.float32)


def masked_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, depth_range_um: float) -> dict[str, float]:
    if not np.any(mask):
        return {
            "pixel_count": 0,
            "mae_norm": float("nan"),
            "rmse_norm": float("nan"),
            "p90_norm": float("nan"),
            "mae_um": float("nan"),
            "rmse_um": float("nan"),
            "p90_um": float("nan"),
        }
    err = np.abs(pred[mask] - truth[mask])
    sq = (pred[mask] - truth[mask]) ** 2
    return {
        "pixel_count": int(np.sum(mask)),
        "mae_norm": float(np.mean(err)),
        "rmse_norm": float(np.sqrt(np.mean(sq))),
        "p90_norm": float(np.percentile(err, 90)),
        "mae_um": float(np.mean(err) * depth_range_um),
        "rmse_um": float(np.sqrt(np.mean(sq)) * depth_range_um),
        "p90_um": float(np.percentile(err, 90) * depth_range_um),
    }


def stratum_masks(risk: np.ndarray, focus_conf: np.ndarray) -> dict[str, np.ndarray]:
    risk_thr = max(float(np.percentile(risk, 90)), 0.35)
    conf_thr = float(np.percentile(focus_conf, 25))
    high_risk = risk >= risk_thr
    low_conf = focus_conf <= conf_thr
    normal = (risk <= float(np.percentile(risk, 40))) & (focus_conf >= float(np.percentile(focus_conf, 50)))
    return {"high_risk": high_risk, "low_confidence": low_conf, "normal": normal}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    eval_tag = safe_tag(args.tag)
    checkpoint_tag = safe_tag(args.checkpoint_tag)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    checkpoint = checkpoint_for_tag(checkpoint_tag)
    checks = [check("checkpoint exists", checkpoint.exists(), str(checkpoint))]
    if not checkpoint.exists():
        report = {
            "status": "fail",
            "date": DATE,
            "tag": eval_tag,
            "checkpoint_tag": checkpoint_tag,
            "checks": checks,
            "error_count": 1,
            "claim_eligible": False,
            "main_table_eligible": False,
        }
        return report
    model = load_model(checkpoint, args.device)
    test_items = build_dataset()["test"]
    if args.max_samples:
        test_items = test_items[: args.max_samples]
    sample_count = len(test_items)
    smoke_evaluation = bool(args.max_samples)
    expected_samples = min(args.max_samples or 7, 7)

    per_sample_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for category, scenario in test_items:
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        base = np.asarray(arrays["features"], dtype=np.float32)
        truth = np.asarray(arrays["truth"], dtype=np.float32)
        risk = np.asarray(arrays["risk"], dtype=np.float32)
        dff = np.asarray(arrays["dff"], dtype=np.float32)
        gadff = np.asarray(arrays["gadff"], dtype=np.float32)
        focus_conf = combined_focus_confidence(base)
        pred = predict_tiled_upgraded(model, augment_features(base), args.device, tile=args.tile, overlap=args.overlap)
        row_metrics = metrics(pred, truth, risk, scenario.depth_range_um)
        dff_metrics = metrics(dff, truth, risk, scenario.depth_range_um)
        gadff_metrics = metrics(gadff, truth, risk, scenario.depth_range_um)
        per_sample_rows.append(
            {
                "run_id": RUN_ID,
                "variant": VARIANT,
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
                "risk_mean": float(np.mean(risk)),
                "focus_conf_mean": float(np.mean(focus_conf)),
                "checkpoint": str(checkpoint),
                "checkpoint_tag": checkpoint_tag,
                "evaluation_tag": eval_tag,
                "smoke_evaluation": smoke_evaluation,
                "claim_eligible": False,
                "main_table_eligible": False,
            }
        )
        masks = stratum_masks(risk, focus_conf)
        for stratum in STRATA:
            model_m = masked_metrics(pred, truth, masks[stratum], scenario.depth_range_um)
            dff_m = masked_metrics(dff, truth, masks[stratum], scenario.depth_range_um)
            gadff_m = masked_metrics(gadff, truth, masks[stratum], scenario.depth_range_um)
            stratum_rows.append(
                {
                    "run_id": RUN_ID,
                    "variant": VARIANT,
                    "split": "test",
                    "category": category,
                    "sample": scenario.name,
                    "stratum": stratum,
                    "pixel_count": model_m["pixel_count"],
                    "risk_mean": float(np.mean(risk[masks[stratum]])) if np.any(masks[stratum]) else float("nan"),
                    "focus_conf_mean": float(np.mean(focus_conf[masks[stratum]])) if np.any(masks[stratum]) else float("nan"),
                    "model_mae_um": model_m["mae_um"],
                    "model_rmse_um": model_m["rmse_um"],
                    "model_p90_um": model_m["p90_um"],
                    "dff_mae_um": dff_m["mae_um"],
                    "gadff_mae_um": gadff_m["mae_um"],
                    "model_vs_dff_gain_percent": (dff_m["mae_um"] - model_m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100
                    if np.isfinite(dff_m["mae_um"])
                    else float("nan"),
                    "checkpoint_tag": checkpoint_tag,
                    "evaluation_tag": eval_tag,
                    "smoke_evaluation": smoke_evaluation,
                    "claim_eligible": False,
                    "main_table_eligible": False,
                }
            )

    summary_rows = [
        {
            "run_id": RUN_ID,
            "variant": VARIANT,
            "sample_count": len(per_sample_rows),
            "mean_mae_um": float(np.mean([row["mae_um"] for row in per_sample_rows])),
            "mean_rmse_norm": float(np.mean([row["rmse_norm"] for row in per_sample_rows])),
            "mean_edge_mae_um": float(np.mean([row["edge_mae_um"] for row in per_sample_rows])),
            "mean_high_risk_mae_um": float(np.mean([row["high_risk_mae_um"] for row in per_sample_rows])),
            "mean_dff_mae_um": float(np.mean([row["dff_mae_um"] for row in per_sample_rows])),
            "mean_gadff_mae_um": float(np.mean([row["gadff_mae_um"] for row in per_sample_rows])),
            "model_vs_dff_gain_ratio_of_means_percent": (
                float(np.mean([row["dff_mae_um"] for row in per_sample_rows]))
                - float(np.mean([row["mae_um"] for row in per_sample_rows]))
            )
            / max(float(np.mean([row["dff_mae_um"] for row in per_sample_rows])), 1e-6)
            * 100,
            "model_vs_dff_win_rate": float(np.mean([row["mae_um"] < row["dff_mae_um"] for row in per_sample_rows])),
            "checkpoint_tag": checkpoint_tag,
            "evaluation_tag": eval_tag,
            "smoke_evaluation": smoke_evaluation,
            "claim_eligible": False,
            "main_table_eligible": False,
        }
    ]
    stratum_summary_rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        rows = [row for row in stratum_rows if row["stratum"] == stratum and np.isfinite(row["model_mae_um"])]
        stratum_summary_rows.append(
            {
                "run_id": RUN_ID,
                "variant": VARIANT,
                "stratum": stratum,
                "sample_count": len(rows),
                "mean_pixel_count": float(np.mean([row["pixel_count"] for row in rows])),
                "mean_risk": float(np.mean([row["risk_mean"] for row in rows])),
                "mean_focus_conf": float(np.mean([row["focus_conf_mean"] for row in rows])),
                "mean_model_mae_um": float(np.mean([row["model_mae_um"] for row in rows])),
                "mean_model_rmse_um": float(np.mean([row["model_rmse_um"] for row in rows])),
                "mean_model_p90_um": float(np.mean([row["model_p90_um"] for row in rows])),
                "mean_dff_mae_um": float(np.mean([row["dff_mae_um"] for row in rows])),
                "mean_gadff_mae_um": float(np.mean([row["gadff_mae_um"] for row in rows])),
                "mean_model_vs_dff_gain_percent_per_sample": float(np.mean([row["model_vs_dff_gain_percent"] for row in rows])),
                "model_vs_dff_gain_ratio_of_means_percent": (
                    float(np.mean([row["dff_mae_um"] for row in rows]))
                    - float(np.mean([row["model_mae_um"] for row in rows]))
                )
                / max(float(np.mean([row["dff_mae_um"] for row in rows])), 1e-6)
                * 100,
                "model_vs_dff_win_rate": float(np.mean([row["model_mae_um"] < row["dff_mae_um"] for row in rows])),
                "checkpoint_tag": checkpoint_tag,
                "evaluation_tag": eval_tag,
                "smoke_evaluation": smoke_evaluation,
                "claim_eligible": False,
                "main_table_eligible": False,
            }
        )

    out_dir = ABL_ROOT / "confidence_gated_full_split_eval" / eval_tag
    per_sample_csv = out_dir / f"{eval_tag}_per_sample_metrics.csv"
    stratum_csv = out_dir / f"{eval_tag}_stratum_metrics.csv"
    summary_csv = out_dir / f"{eval_tag}_method_summary_metrics.csv"
    stratum_summary_csv = out_dir / f"{eval_tag}_stratum_summary_metrics.csv"
    write_csv(per_sample_csv, per_sample_rows)
    write_csv(stratum_csv, stratum_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(stratum_summary_csv, stratum_summary_rows)
    write_json(out_dir / f"{eval_tag}_per_sample_metrics.json", per_sample_rows)
    write_json(out_dir / f"{eval_tag}_stratum_metrics.json", stratum_rows)
    write_json(out_dir / f"{eval_tag}_method_summary_metrics.json", summary_rows)
    write_json(out_dir / f"{eval_tag}_stratum_summary_metrics.json", stratum_summary_rows)

    checks.extend(
        [
            check("test sample count", sample_count == expected_samples, f"sample_count={sample_count}, expected={expected_samples}"),
            check("per-sample rows", len(per_sample_rows) == sample_count, f"rows={len(per_sample_rows)}"),
            check("stratum rows", len(stratum_rows) == sample_count * len(STRATA), f"rows={len(stratum_rows)}"),
            check("per-sample CSV exists", per_sample_csv.exists(), str(per_sample_csv)),
            check("stratum summary CSV exists", stratum_summary_csv.exists(), str(stratum_summary_csv)),
            check("claim_eligible false", True, "claim_eligible=false"),
            check("main_table_eligible false", True, "main_table_eligible=false"),
        ]
    )
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    report = {
        "status": "pass" if not errors else "fail",
        "date": DATE,
        "tag": eval_tag,
        "checkpoint_tag": checkpoint_tag,
        "checkpoint": str(checkpoint),
        "device": args.device,
        "sample_count": sample_count,
        "smoke_evaluation": smoke_evaluation,
        "tile": args.tile,
        "overlap": args.overlap,
        "per_sample_metrics_csv": str(per_sample_csv),
        "stratum_metrics_csv": str(stratum_csv),
        "method_summary_metrics_csv": str(summary_csv),
        "stratum_summary_metrics_csv": str(stratum_summary_csv),
        "summary": summary_rows,
        "stratum_summary": stratum_summary_rows,
        "checks": checks,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len([row for row in checks if not row["passed"] and row["severity"] == "warning"]),
        "elapsed_s": time.time() - start,
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": "ABL-07 evaluation under tmp only. Results remain outside manuscript tables until full training, repeat evaluation, and eligibility audit pass.",
    }
    report_json = out_dir / f"{eval_tag}_summary.json"
    report_md = out_dir / f"{eval_tag}_summary.md"
    write_json(report_json, report)
    write_markdown_summary(report_md, report)
    update_run_config(report)
    return report


def update_run_config(report: dict[str, Any]) -> None:
    cfg_path = ABL_ROOT / RUN_ID / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"run_id": RUN_ID}
    cfg["claim_eligible"] = False
    cfg["main_table_eligible"] = False
    evaluations = cfg.setdefault("confidence_gated_full_split_evaluations", {})
    evaluations[report["tag"]] = {
        "date": report["date"],
        "tag": report["tag"],
        "checkpoint_tag": report["checkpoint_tag"],
        "checkpoint": report["checkpoint"],
        "per_sample_metrics_csv": report["per_sample_metrics_csv"],
        "stratum_summary_metrics_csv": report["stratum_summary_metrics_csv"],
        "sample_count": report["sample_count"],
        "smoke_evaluation": report["smoke_evaluation"],
        "summary": report["summary"],
        "stratum_summary": report["stratum_summary"],
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": report["interpretation"],
    }
    write_json(cfg_path, cfg)


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ABL-07 Confidence-Gated Prior Full-Split Evaluation",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Evaluation tag: `{report['tag']}`",
        f"- Checkpoint tag: `{report['checkpoint_tag']}`",
        f"- Device: `{report['device']}`",
        f"- Test samples: `{report['sample_count']}`",
        f"- Smoke evaluation: `{str(report['smoke_evaluation']).lower()}`",
        f"- Claim eligible: `{str(report['claim_eligible']).lower()}`",
        f"- Main table eligible: `{str(report['main_table_eligible']).lower()}`",
        f"- Errors: `{report['error_count']}`",
        f"- Warnings: `{report['warning_count']}`",
        "",
        report["interpretation"],
        "",
        "## Overall Summary",
        "",
        "| Run | Samples | Mean MAE um | Mean High-Risk MAE um | Mean DFF MAE um | Gain vs DFF | Win Rate vs DFF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['run_id']} | {row['sample_count']} | {row['mean_mae_um']:.4f} | "
            f"{row['mean_high_risk_mae_um']:.4f} | {row['mean_dff_mae_um']:.4f} | "
            f"{row['model_vs_dff_gain_ratio_of_means_percent']:.2f}% | {row['model_vs_dff_win_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Stratum Summary",
            "",
            "| Stratum | Samples | Risk | Focus Conf | Model MAE um | DFF MAE um | Gain vs DFF | Win Rate vs DFF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["stratum_summary"]:
        lines.append(
            f"| {row['stratum']} | {row['sample_count']} | {row['mean_risk']:.4f} | {row['mean_focus_conf']:.4f} | "
            f"{row['mean_model_mae_um']:.4f} | {row['mean_dff_mae_um']:.4f} | "
            f"{row['model_vs_dff_gain_ratio_of_means_percent']:.2f}% | {row['model_vs_dff_win_rate']:.2f} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--checkpoint-tag", default=DEFAULT_CHECKPOINT_TAG)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=1, help="Smoke sample limit. Use 0 for the full 7-sample test split.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    print(f"ABL-07 full-split evaluator: {report['status']}")
    print(f"Samples: {report.get('sample_count')}")
    print(f"Errors: {report.get('error_count')}")
    if report.get("method_summary_metrics_csv"):
        print(f"Wrote {report['method_summary_metrics_csv']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
