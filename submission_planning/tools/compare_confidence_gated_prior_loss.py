"""Paired smoke comparison for baseline and confidence-gated prior losses.

The comparison is designed for mechanism preflight only. It writes under
tmp/ablation_results/ABL-07 and keeps all outputs claim-ineligible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from train_focus_resunet_loss_experiment import FocusResUNet, HybridDFFLoss  # noqa: E402
from run_confidence_weighted_loss_training import (  # noqa: E402
    ABL_ROOT,
    CHANNEL_MAP,
    DATE,
    REPORT_DIR,
    RUN_ID,
    SEED,
    STACK_LAYERS,
    ConfidenceGatedPriorLoss,
    check,
    prepare_samples,
    safe_tag,
    split_patch_batch,
    write_json,
)


VARIANTS: dict[str, dict[str, str]] = {
    "baseline_hybrid": {
        "label": "Baseline HybridDFFLoss",
        "loss": "src/train_focus_resunet_loss_experiment.py::HybridDFFLoss",
    },
    "confidence_gated": {
        "label": "Confidence-gated prior loss",
        "loss": "submission_planning/tools/run_confidence_weighted_loss_training.py::ConfidenceGatedPriorLoss",
    },
}


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def context_stats(features: torch.Tensor, actual_mode: str) -> dict[str, float]:
    prior_offset = STACK_LAYERS + (STACK_LAYERS - 1)
    risk = torch.clamp(features[:, prior_offset + 0 : prior_offset + 1], 0, 1)
    conf = torch.clamp(features[:, prior_offset + 2 : prior_offset + 3], 0, 1)
    ga_conf = torch.clamp(features[:, prior_offset + 4 : prior_offset + 5], 0, 1)
    focus_conf = torch.clamp(0.65 * conf + 0.35 * ga_conf, 0, 1)
    baseline_prior_weight = torch.clamp(focus_conf * (1.0 - risk).pow(1.5), 0, 1)
    confidence_prior_weight = torch.clamp(focus_conf.pow(1.5) * (1.0 - 0.45 * risk), min=0.02, max=1.0)
    glare_data_weight = 1.0 + 0.80 * risk
    if actual_mode == "baseline_hybrid":
        actual_prior_weight = baseline_prior_weight
        actual_data_weight = glare_data_weight
    else:
        actual_prior_weight = confidence_prior_weight
        actual_data_weight = torch.ones_like(glare_data_weight)
    return {
        "risk_mean": float(torch.mean(risk).detach().cpu()),
        "focus_conf_mean": float(torch.mean(focus_conf).detach().cpu()),
        "baseline_prior_weight_mean": float(torch.mean(baseline_prior_weight).detach().cpu()),
        "confidence_prior_weight_mean": float(torch.mean(confidence_prior_weight).detach().cpu()),
        "glare_data_weight_mean": float(torch.mean(glare_data_weight).detach().cpu()),
        "actual_prior_weight_mean": float(torch.mean(actual_prior_weight).detach().cpu()),
        "actual_data_weight_mean": float(torch.mean(actual_data_weight).detach().cpu()),
    }


def merged_parts(parts: dict[str, float], features: torch.Tensor, variant_id: str) -> dict[str, float]:
    stats = context_stats(features, variant_id)
    return {
        "total": float(parts["total"]),
        "data": float(parts["data"]),
        "focus_prior": float(parts["focus_prior"]),
        "gradient": float(parts["gradient"]),
        "curvature": float(parts["curvature"]),
        "normal": float(parts["normal"]),
        **stats,
    }


def run_variant(
    variant_id: str,
    loss_factory: Callable[[], nn.Module],
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(SEED + 17)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED + 17)
    rng = np.random.default_rng(SEED + 17)
    model = FocusResUNet().to(args.device)
    criterion = loss_factory().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.5e-4)

    train_steps = max(1, int(np.ceil(args.train_patches / args.batch_size)))
    val_steps = max(1, int(np.ceil(args.val_patches / args.batch_size)))
    history: list[dict[str, float]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_rows: list[dict[str, float]] = []
        for _ in range(train_steps):
            xb, yb = split_patch_batch(train_samples, rng, args.batch_size, args.patch_size)
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            pred = model(xb)
            loss, parts = criterion(pred, yb, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_rows.append(merged_parts(parts, xb, variant_id))

        model.eval()
        val_rows: list[dict[str, float]] = []
        val_mae: list[float] = []
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = split_patch_batch(val_samples, rng, args.batch_size, args.patch_size)
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                pred = model(xb)
                _, parts = criterion(pred, yb, xb)
                val_rows.append(merged_parts(parts, xb, variant_id))
                val_mae.append(float(torch.mean(torch.abs(pred - yb)).detach().cpu()))

        train_mean = mean_rows(train_rows)
        val_mean = mean_rows(val_rows)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_debug": train_mean["total"],
                "val_loss_debug": val_mean["total"],
                "val_mae_norm_debug": float(np.mean(val_mae)),
                "train_data_debug": train_mean["data"],
                "val_data_debug": val_mean["data"],
                "train_focus_prior_debug": train_mean["focus_prior"],
                "val_focus_prior_debug": val_mean["focus_prior"],
                "train_risk_mean": train_mean["risk_mean"],
                "val_risk_mean": val_mean["risk_mean"],
                "train_focus_conf_mean": train_mean["focus_conf_mean"],
                "val_focus_conf_mean": val_mean["focus_conf_mean"],
                "train_actual_prior_weight_mean": train_mean["actual_prior_weight_mean"],
                "val_actual_prior_weight_mean": val_mean["actual_prior_weight_mean"],
                "train_actual_data_weight_mean": train_mean["actual_data_weight_mean"],
                "val_actual_data_weight_mean": val_mean["actual_data_weight_mean"],
            }
        )
    return model, history


def write_comparison_csv(path: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    rows: list[dict[str, Any]] = []
    for variant_id, history in histories.items():
        for row in history:
            rows.append({"variant_id": variant_id, "variant_label": VARIANTS[variant_id]["label"], **row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(histories: dict[str, list[dict[str, float]]], args: argparse.Namespace, elapsed_s: float) -> dict[str, Any]:
    baseline = histories["baseline_hybrid"][-1]
    gated = histories["confidence_gated"][-1]
    delta_mae = gated["val_mae_norm_debug"] - baseline["val_mae_norm_debug"]
    delta_loss = gated["val_loss_debug"] - baseline["val_loss_debug"]
    rel_mae = 100.0 * delta_mae / max(abs(baseline["val_mae_norm_debug"]), 1e-8)
    rel_loss = 100.0 * delta_loss / max(abs(baseline["val_loss_debug"]), 1e-8)
    if delta_mae < 0:
        preferred_by_smoke = "confidence_gated"
    elif delta_mae > 0:
        preferred_by_smoke = "baseline_hybrid"
    else:
        preferred_by_smoke = "tie"
    return {
        "status": "paired_loss_smoke_completed",
        "date": DATE,
        "run_id": RUN_ID,
        "tag": args.tag,
        "device": args.device,
        "artifact_level": "smoke_only_claim_ineligible",
        "prepared_train_samples": args.max_train_samples,
        "prepared_validation_samples": args.max_val_samples,
        "training_budget": {
            "max_epochs": args.max_epochs,
            "train_patches": args.train_patches,
            "val_patches": args.val_patches,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "learning_rate": args.learning_rate,
        },
        "paired_design": "same model seed, same patch RNG seed, same train/validation samples, same optimizer budget",
        "channel_map": CHANNEL_MAP,
        "variants": VARIANTS,
        "histories": histories,
        "last_epoch_comparison": {
            "baseline_val_mae_norm": baseline["val_mae_norm_debug"],
            "confidence_gated_val_mae_norm": gated["val_mae_norm_debug"],
            "delta_val_mae_norm_gated_minus_baseline": delta_mae,
            "relative_delta_val_mae_percent": rel_mae,
            "baseline_val_loss": baseline["val_loss_debug"],
            "confidence_gated_val_loss": gated["val_loss_debug"],
            "delta_val_loss_gated_minus_baseline": delta_loss,
            "relative_delta_val_loss_percent": rel_loss,
            "preferred_by_this_smoke": preferred_by_smoke,
        },
        "elapsed_s": elapsed_s,
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": "Single-sample paired smoke only. It checks loss behavior under identical tiny-budget conditions and is not manuscript evidence.",
    }


def write_markdown_report(path: Path, summary: dict[str, Any], artifacts: dict[str, str]) -> None:
    cmp = summary["last_epoch_comparison"]
    rows = []
    for variant_id, history in summary["histories"].items():
        last = history[-1]
        rows.append(
            "| {label} | {mae:.8f} | {loss:.8f} | {prior:.8f} | {data_w:.8f} | {prior_w:.8f} |".format(
                label=VARIANTS[variant_id]["label"],
                mae=last["val_mae_norm_debug"],
                loss=last["val_loss_debug"],
                prior=last["val_focus_prior_debug"],
                data_w=last["val_actual_data_weight_mean"],
                prior_w=last["val_actual_prior_weight_mean"],
            )
        )
    lines = [
        "# Paired Smoke: Baseline vs Confidence-Gated Prior Loss",
        "",
        f"- Date: {summary['date']}",
        f"- Run id: `{summary['run_id']}`",
        f"- Tag: `{summary['tag']}`",
        f"- Artifact level: `{summary['artifact_level']}`",
        f"- Paired design: {summary['paired_design']}",
        "",
        "## Last-Epoch Smoke Metrics",
        "",
        "| Variant | Val MAE Norm | Val Loss | Val Prior Loss | Val Data Weight | Val Prior Weight |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Paired Delta",
        "",
        f"- Delta val MAE norm, gated minus baseline: `{cmp['delta_val_mae_norm_gated_minus_baseline']:.8f}`",
        f"- Relative delta val MAE: `{cmp['relative_delta_val_mae_percent']:.2f}%`",
        f"- Delta val loss, gated minus baseline: `{cmp['delta_val_loss_gated_minus_baseline']:.8f}`",
        f"- Preferred by this smoke: `{cmp['preferred_by_this_smoke']}`",
        "",
        "## Mechanism Interpretation",
        "",
        "The paired smoke isolates loss behavior rather than model capacity. The confidence-gated design removes direct glare-based upweighting from the supervised data term and moves the mechanism emphasis to DFF/GADFF prior reliability. A full candidate run still needs high-risk, low-confidence, and normal-region stratified evaluation before the result can support a manuscript claim.",
        "",
        "## Artifacts",
        "",
        "| Artifact | Path |",
        "|---|---|",
    ]
    for name, artifact_path in artifacts.items():
        lines.append(f"| {name} | `{artifact_path}` |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def update_run_config(path: Path, summary: dict[str, Any], artifacts: dict[str, str]) -> None:
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        config = {"run_id": RUN_ID, "claim_eligible": False, "main_table_eligible": False}
    config["claim_eligible"] = False
    config["main_table_eligible"] = False
    config["latest_paired_loss_smoke"] = {
        "date": DATE,
        "tag": summary["tag"],
        "artifact_level": summary["artifact_level"],
        "last_epoch_comparison": summary["last_epoch_comparison"],
        "artifacts": artifacts,
        "interpretation": summary["interpretation"],
    }
    write_json(path, config)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dataset = build_dataset()
    train_samples = prepare_samples(dataset["train"], args.max_train_samples)
    val_samples = prepare_samples(dataset["validation"], args.max_val_samples)
    histories: dict[str, list[dict[str, float]]] = {}
    checkpoints: dict[str, str] = {}
    for variant_id, loss_factory in (
        ("baseline_hybrid", HybridDFFLoss),
        ("confidence_gated", ConfidenceGatedPriorLoss),
    ):
        model, history = run_variant(variant_id, loss_factory, train_samples, val_samples, args)
        histories[variant_id] = history
        checkpoint_path = ABL_ROOT / RUN_ID / "checkpoints" / f"{args.tag}_{variant_id}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "variant_label": VARIANTS[variant_id]["label"],
                "debug_only": True,
                "claim_eligible": False,
                "tag": args.tag,
                "history": history,
            },
            checkpoint_path,
        )
        checkpoints[variant_id] = str(checkpoint_path)

    summary = build_summary(histories, args, time.time() - start)
    comparison_dir = ABL_ROOT / RUN_ID / "comparisons"
    summary_json = comparison_dir / f"{args.tag}_summary.json"
    summary_md = comparison_dir / f"{args.tag}_summary.md"
    metrics_csv = comparison_dir / f"{args.tag}_metrics.csv"
    mechanism_report = REPORT_DIR / "confidence_gated_prior_loss_paired_smoke_report.md"
    run_config_path = ABL_ROOT / RUN_ID / "run_config.json"
    artifacts = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "metrics_csv": str(metrics_csv),
        "mechanism_report": str(mechanism_report),
        "baseline_checkpoint": checkpoints["baseline_hybrid"],
        "confidence_gated_checkpoint": checkpoints["confidence_gated"],
        "run_config": str(run_config_path),
    }
    checks = [
        check("baseline history rows", len(histories["baseline_hybrid"]) == args.max_epochs, str(len(histories["baseline_hybrid"]))),
        check("confidence history rows", len(histories["confidence_gated"]) == args.max_epochs, str(len(histories["confidence_gated"]))),
        check("finite histories", all(np.isfinite(float(v)) for hist in histories.values() for row in hist for v in row.values()), "finite"),
    ]
    summary["checks"] = checks
    summary["error_count"] = len([row for row in checks if not row["passed"] and row["severity"] == "error"])
    write_json(summary_json, summary)
    write_comparison_csv(metrics_csv, histories)
    write_markdown_report(summary_md, summary, artifacts)
    write_markdown_report(mechanism_report, summary, artifacts)
    update_run_config(run_config_path, summary, artifacts)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="2026-06-22_paired_loss_smoke")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-patches", type=int, default=2)
    parser.add_argument("--val-patches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-train-samples", type=int, default=1)
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.tag = safe_tag(args.tag)
    summary = run_comparison(args)
    print(json.dumps(summary["last_epoch_comparison"], ensure_ascii=False, indent=2))
    return 0 if summary.get("error_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
