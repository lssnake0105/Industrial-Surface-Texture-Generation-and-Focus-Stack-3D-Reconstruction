"""Stratified patch diagnostic for ABL-07 paired loss smoke checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from train_focus_resunet_loss_experiment import FocusResUNet  # noqa: E402
from compare_confidence_gated_prior_loss import VARIANTS  # noqa: E402
from run_confidence_weighted_loss_training import (  # noqa: E402
    ABL_ROOT,
    DATE,
    REPORT_DIR,
    RUN_ID,
    STACK_LAYERS,
    check,
    prepare_samples,
    safe_tag,
    write_json,
)


STRATA = {
    "high_risk": "highest glare/risk mean patches",
    "low_confidence": "lowest combined DFF/GADFF focus confidence patches",
    "normal": "lowest risk with above-median focus confidence patches",
}


def patch_candidates(sample: dict[str, Any], patch_size: int, stride: int) -> list[dict[str, Any]]:
    features = np.asarray(sample["model_features"], dtype=np.float32)
    truth = np.asarray(sample["truth"], dtype=np.float32)
    prior_offset = STACK_LAYERS + (STACK_LAYERS - 1)
    risk = np.clip(features[prior_offset + 0], 0, 1)
    conf = np.clip(features[prior_offset + 2], 0, 1)
    ga_conf = np.clip(features[prior_offset + 4], 0, 1)
    focus_conf = np.clip(0.65 * conf + 0.35 * ga_conf, 0, 1)
    _, height, width = features.shape
    rows: list[dict[str, Any]] = []
    for y0 in range(0, height - patch_size + 1, stride):
        for x0 in range(0, width - patch_size + 1, stride):
            r = risk[y0 : y0 + patch_size, x0 : x0 + patch_size]
            c = focus_conf[y0 : y0 + patch_size, x0 : x0 + patch_size]
            rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "x0": x0,
                    "y0": y0,
                    "risk_mean": float(np.mean(r)),
                    "risk_p90": float(np.percentile(r, 90)),
                    "focus_conf_mean": float(np.mean(c)),
                    "focus_conf_p10": float(np.percentile(c, 10)),
                    "features": features[:, y0 : y0 + patch_size, x0 : x0 + patch_size],
                    "truth": truth[None, y0 : y0 + patch_size, x0 : x0 + patch_size],
                }
            )
    return rows


def select_strata(candidates: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    focus_median = float(np.median([row["focus_conf_mean"] for row in candidates]))
    high_risk = sorted(candidates, key=lambda row: (row["risk_mean"], row["risk_p90"]), reverse=True)[:top_k]
    low_conf = sorted(candidates, key=lambda row: (row["focus_conf_mean"], row["focus_conf_p10"]))[:top_k]
    normal_pool = [row for row in candidates if row["focus_conf_mean"] >= focus_median]
    if len(normal_pool) < top_k:
        normal_pool = candidates
    normal = sorted(normal_pool, key=lambda row: (row["risk_mean"], -row["focus_conf_mean"]))[:top_k]
    return {"high_risk": high_risk, "low_confidence": low_conf, "normal": normal}


def load_model(checkpoint: Path, device: str) -> FocusResUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def eval_model(model: FocusResUNet, patches: dict[str, list[dict[str, Any]]], device: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for stratum, items in patches.items():
            for item in items:
                x = torch.from_numpy(item["features"][None].astype(np.float32)).to(device)
                y = torch.from_numpy(item["truth"][None].astype(np.float32)).to(device)
                pred = model(x)
                err = torch.abs(pred - y)
                rows.append(
                    {
                        "stratum": stratum,
                        "sample_id": item["sample_id"],
                        "x0": item["x0"],
                        "y0": item["y0"],
                        "risk_mean": item["risk_mean"],
                        "risk_p90": item["risk_p90"],
                        "focus_conf_mean": item["focus_conf_mean"],
                        "focus_conf_p10": item["focus_conf_p10"],
                        "mae_norm": float(torch.mean(err).detach().cpu()),
                        "p90_abs_error_norm": float(torch.quantile(err.flatten(), 0.90).detach().cpu()),
                    }
                )
    return rows


def summarize(rows_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for variant_id, rows in rows_by_variant.items():
        for stratum in STRATA:
            part = [row for row in rows if row["stratum"] == stratum]
            summary_rows.append(
                {
                    "variant_id": variant_id,
                    "variant_label": VARIANTS[variant_id]["label"],
                    "stratum": stratum,
                    "n_patches": len(part),
                    "risk_mean": float(np.mean([row["risk_mean"] for row in part])),
                    "focus_conf_mean": float(np.mean([row["focus_conf_mean"] for row in part])),
                    "mae_norm_mean": float(np.mean([row["mae_norm"] for row in part])),
                    "p90_abs_error_norm_mean": float(np.mean([row["p90_abs_error_norm"] for row in part])),
                }
            )
    deltas: list[dict[str, Any]] = []
    for stratum in STRATA:
        base = next(row for row in summary_rows if row["variant_id"] == "baseline_hybrid" and row["stratum"] == stratum)
        gated = next(row for row in summary_rows if row["variant_id"] == "confidence_gated" and row["stratum"] == stratum)
        delta = gated["mae_norm_mean"] - base["mae_norm_mean"]
        rel = 100.0 * delta / max(abs(base["mae_norm_mean"]), 1e-8)
        deltas.append(
            {
                "stratum": stratum,
                "baseline_mae_norm_mean": base["mae_norm_mean"],
                "confidence_gated_mae_norm_mean": gated["mae_norm_mean"],
                "delta_gated_minus_baseline": delta,
                "relative_delta_percent": rel,
                "preferred": "confidence_gated" if delta < 0 else "baseline_hybrid" if delta > 0 else "tie",
            }
        )
    return {"summary_rows": summary_rows, "deltas": deltas}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any], artifacts: dict[str, str]) -> None:
    lines = [
        "# ABL-07 Stratified Diagnostic: Paired Smoke Checkpoints",
        "",
        f"- Date: {payload['date']}",
        f"- Run id: `{payload['run_id']}`",
        f"- Source tag: `{payload['source_tag']}`",
        f"- Artifact level: `{payload['artifact_level']}`",
        "",
        "## Stratum Summary",
        "",
        "| Variant | Stratum | N | Risk Mean | Focus Conf Mean | MAE Norm | P90 Abs Error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['variant_label']} | {row['stratum']} | {row['n_patches']} | "
            f"{row['risk_mean']:.6f} | {row['focus_conf_mean']:.6f} | "
            f"{row['mae_norm_mean']:.8f} | {row['p90_abs_error_norm_mean']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Delta: Confidence-Gated Minus Baseline",
            "",
            "| Stratum | Baseline MAE | Confidence-Gated MAE | Delta | Relative Delta | Preferred |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["deltas"]:
        lines.append(
            f"| {row['stratum']} | {row['baseline_mae_norm_mean']:.8f} | "
            f"{row['confidence_gated_mae_norm_mean']:.8f} | {row['delta_gated_minus_baseline']:.8f} | "
            f"{row['relative_delta_percent']:.2f}% | {row['preferred']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This diagnostic evaluates only two smoke checkpoints trained on a tiny budget. It is useful for deciding whether the proposed mechanism deserves a full candidate run, but it remains outside manuscript evidence.",
            "",
            "## Artifacts",
            "",
            "| Artifact | Path |",
            "|---|---|",
        ]
    )
    for name, artifact_path in artifacts.items():
        lines.append(f"| {name} | `{artifact_path}` |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dataset = build_dataset()
    samples = prepare_samples(dataset["validation"], args.max_val_samples)
    candidates: list[dict[str, Any]] = []
    for sample in samples:
        candidates.extend(patch_candidates(sample, args.patch_size, args.stride))
    patches = select_strata(candidates, args.top_k)
    checkpoints = {
        "baseline_hybrid": ABL_ROOT / RUN_ID / "checkpoints" / f"{args.source_tag}_baseline_hybrid.pt",
        "confidence_gated": ABL_ROOT / RUN_ID / "checkpoints" / f"{args.source_tag}_confidence_gated.pt",
    }
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_id, checkpoint in checkpoints.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_model(checkpoint, args.device)
        rows_by_variant[variant_id] = eval_model(model, patches, args.device)
    flat_rows = []
    for variant_id, rows in rows_by_variant.items():
        for row in rows:
            flat_rows.append({"variant_id": variant_id, "variant_label": VARIANTS[variant_id]["label"], **row})
    stats = summarize(rows_by_variant)
    payload = {
        "status": "stratified_diagnostic_completed",
        "date": DATE,
        "run_id": RUN_ID,
        "source_tag": args.source_tag,
        "artifact_level": "smoke_checkpoint_diagnostic_claim_ineligible",
        "strata": STRATA,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "top_k": args.top_k,
        "candidate_count": len(candidates),
        "summary_rows": stats["summary_rows"],
        "deltas": stats["deltas"],
        "claim_eligible": False,
        "main_table_eligible": False,
    }
    out_dir = ABL_ROOT / RUN_ID / "stratified_diagnostics"
    detail_csv = out_dir / f"{args.output_tag}_detail.csv"
    summary_csv = out_dir / f"{args.output_tag}_summary.csv"
    summary_json = out_dir / f"{args.output_tag}_summary.json"
    report_md = REPORT_DIR / "confidence_gated_prior_loss_stratified_diagnostic_report.md"
    artifacts = {
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "mechanism_report": str(report_md),
        "baseline_checkpoint": str(checkpoints["baseline_hybrid"]),
        "confidence_gated_checkpoint": str(checkpoints["confidence_gated"]),
    }
    checks = [
        check("candidate patches available", len(candidates) > 0, str(len(candidates))),
        check("all strata nonempty", all(len(items) == args.top_k for items in patches.values()), str({k: len(v) for k, v in patches.items()})),
        check("claim_eligible false", payload["claim_eligible"] is False, str(payload["claim_eligible"])),
    ]
    payload["checks"] = checks
    payload["error_count"] = len([row for row in checks if not row["passed"] and row["severity"] == "error"])
    write_csv(detail_csv, flat_rows)
    write_csv(summary_csv, stats["summary_rows"])
    write_json(summary_json, payload)
    write_report(report_md, payload, artifacts)
    config_path = ABL_ROOT / RUN_ID / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["claim_eligible"] = False
    config["main_table_eligible"] = False
    config["latest_stratified_diagnostic"] = {
        "date": DATE,
        "source_tag": args.source_tag,
        "output_tag": args.output_tag,
        "artifact_level": payload["artifact_level"],
        "deltas": payload["deltas"],
        "artifacts": artifacts,
    }
    write_json(config_path, config)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tag", default="2026-06-22_paired_loss_smoke")
    parser.add_argument("--output-tag", default="2026-06-22_stratified_diagnostic")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source_tag = safe_tag(args.source_tag)
    args.output_tag = safe_tag(args.output_tag)
    payload = run(args)
    print(json.dumps({"status": payload["status"], "deltas": payload["deltas"]}, ensure_ascii=False, indent=2))
    return 0 if payload["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
