"""Smoke runner for a confidence-gated DFF/GADFF prior loss.

This script is intentionally isolated from src/. It writes only under
tmp/ablation_results and marks all outputs as claim-ineligible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402
from train_focus_resunet_loss_experiment import (  # noqa: E402
    FocusResUNet,
    augment_features,
    charbonnier,
    grad_xy,
    laplacian,
    normal_loss,
    upgraded_channel_count,
)


DATE = "2026-06-22"
RUN_ID = "ABL-07"
VARIANT = "Confidence-gated DFF/GADFF prior loss"
RUNNER_MODE = "focus_resunet_confidence_gated_prior_loss"
ABL_ROOT = ROOT / "tmp" / "ablation_results"
REPORT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis"
SUMMARY_DIR = ABL_ROOT / "training_runner_confidence_gated_prior_smoke"
PATCH_SIZE = 64
SEED = 20260622
STACK_LAYERS = DEFAULT_STACK_LAYERS

CHANNEL_MAP = {
    "0-16": "focus stack intensity layers",
    "17-32": "adjacent focal-difference layers",
    "33": "glare/risk prior",
    "34": "DFF depth prior",
    "35": "DFF focus confidence",
    "36": "GADFF depth prior",
    "37": "GADFF focus confidence",
}


def safe_tag(value: str) -> str:
    tag = value.strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not tag or any(char not in allowed for char in tag):
        raise ValueError("Tag must contain only letters, numbers, underscores, and hyphens.")
    return tag


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


class ConfidenceGatedPriorLoss(nn.Module):
    """Uniform supervised data term with confidence-gated DFF/GADFF consistency."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        prior_offset = STACK_LAYERS + (STACK_LAYERS - 1)
        risk = torch.clamp(features[:, prior_offset + 0 : prior_offset + 1], 0, 1)
        dff = features[:, prior_offset + 1 : prior_offset + 2]
        conf = torch.clamp(features[:, prior_offset + 2 : prior_offset + 3], 0, 1)
        gadff = features[:, prior_offset + 3 : prior_offset + 4]
        ga_conf = torch.clamp(features[:, prior_offset + 4 : prior_offset + 5], 0, 1)

        data = torch.mean(charbonnier(pred - target))

        pdx, pdy = grad_xy(pred)
        tdx, tdy = grad_xy(target)
        grad = torch.mean(charbonnier(pdx - tdx)) + torch.mean(charbonnier(pdy - tdy))
        curv = torch.mean(charbonnier(laplacian(pred) - laplacian(target)))
        nrm = normal_loss(pred, target)

        focus_conf = torch.clamp(0.65 * conf + 0.35 * ga_conf, 0, 1)
        prior_weight = torch.clamp(focus_conf.pow(1.5) * (1.0 - 0.45 * risk), min=0.02, max=1.0)
        prior_target = 0.45 * dff + 0.55 * gadff
        prior = torch.sum(prior_weight * charbonnier(pred - prior_target)) / torch.clamp(torch.sum(prior_weight), min=1.0)

        loss = data + 0.22 * grad + 0.055 * curv + 0.035 * nrm + 0.045 * prior
        parts = {
            "data": float(data.detach().cpu()),
            "gradient": float(grad.detach().cpu()),
            "curvature": float(curv.detach().cpu()),
            "normal": float(nrm.detach().cpu()),
            "focus_prior": float(prior.detach().cpu()),
            "focus_conf_mean": float(torch.mean(focus_conf).detach().cpu()),
            "prior_weight_mean": float(torch.mean(prior_weight).detach().cpu()),
            "risk_mean": float(torch.mean(risk).detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
        return loss, parts


def prepare_samples(split_items: list[tuple[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    selected = split_items[: max_samples or None]
    samples: list[dict[str, Any]] = []
    for category, scenario in selected:
        arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
        base = np.asarray(arrays["features"], dtype=np.float32)
        arrays["model_features"] = augment_features(base)
        arrays["truth"] = np.asarray(arrays["truth"], dtype=np.float32)
        arrays["category"] = category
        arrays["sample_id"] = scenario.name
        samples.append(arrays)
    return samples


def split_patch_batch(
    samples: list[dict[str, Any]],
    rng: np.random.Generator,
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _ in range(batch_size):
        item = samples[int(rng.integers(0, len(samples)))]
        features = np.asarray(item["model_features"], dtype=np.float32)
        truth = np.asarray(item["truth"], dtype=np.float32)
        _, height, width = features.shape
        y0 = int(rng.integers(0, height - patch_size + 1))
        x0 = int(rng.integers(0, width - patch_size + 1))
        x = features[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        y = truth[None, y0 : y0 + patch_size, x0 : x0 + patch_size]
        if rng.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        if rng.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def mean_parts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_run_log(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {RUN_ID} Confidence-Gated Prior Loss Smoke",
        "",
        f"- Status: {report['status']}",
        f"- Variant: {VARIANT}",
        f"- Runner mode: {RUNNER_MODE}",
        f"- Tag: `{report['tag']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Metrics CSV: `{report['metrics_csv']}`",
        "",
        "This is a smoke-level runner-continuity artifact. It is not manuscript evidence.",
        "",
        "## Loss Design",
        "",
        "- Supervised data term: uniform Charbonnier loss against synthetic ground truth.",
        "- DFF/GADFF prior term: weighted by combined focus confidence.",
        "- Glare/risk channel: used only to soften unreliable prior consistency, not to upweight the supervised data term.",
        "",
        "## History",
        "",
        "| Epoch | Train Loss | Val Loss | Val MAE Norm | Val Focus Conf | Val Prior Weight |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["history"]:
        lines.append(
            f"| {int(row['epoch'])} | {row['train_loss_debug']:.8f} | "
            f"{row['val_loss_debug']:.8f} | {row['val_mae_norm_debug']:.8f} | "
            f"{row['val_focus_conf_mean']:.8f} | {row['val_prior_weight_mean']:.8f} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, report: dict[str, Any]) -> None:
    last = report["history"][-1]
    lines = [
        "# Confidence-Gated Prior Loss Smoke Summary",
        "",
        f"- Status: {report['status']}",
        f"- Date: {DATE}",
        f"- Run: {RUN_ID}",
        f"- Variant: {VARIANT}",
        f"- Device: {report['device']}",
        f"- Tag: `{report['tag']}`",
        f"- Seed: `{report['seed']}`",
        f"- Last val MAE norm: `{last['val_mae_norm_debug']:.8f}`",
        f"- Last val prior weight mean: `{last['val_prior_weight_mean']:.8f}`",
        f"- Errors: `{report['error_count']}`",
        f"- Warnings: `{report['warning_count']}`",
        "",
        "The run verifies that the confidence-gated prior loss can train through the existing FocusResUNet feature interface. It remains a smoke artifact only.",
        "",
        "| Artifact | Path |",
        "|---|---|",
        f"| Checkpoint | `{report['checkpoint']}` |",
        f"| Metrics CSV | `{report['metrics_csv']}` |",
        f"| Metrics JSON | `{report['metrics_json']}` |",
        f"| Run config | `{report['run_config']}` |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_mechanism_report(path: Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    last = report["history"][-1]
    full_command = (
        "python -B submission_planning/tools/run_confidence_weighted_loss_training.py "
        "--tag 2026-06-22_confidence_gated_prior_full_candidate "
        "--max-epochs 12 --train-patches 384 --val-patches 96 --batch-size 6 "
        "--max-train-samples 27 --max-val-samples 10 --device cpu"
    )
    lines = [
        "# Confidence-Gated Prior Loss Runner Report",
        "",
        f"- Date: {DATE}",
        f"- Run id: `{RUN_ID}`",
        f"- Variant: {VARIANT}",
        f"- Current artifact level: smoke only, claim-ineligible.",
        f"- Seed: `{report['seed']}`",
        "",
        "## Rationale",
        "",
        "Earlier mechanism probes show that DFF reliability varies with focus-curve confidence and that glare/risk alone is not a stable universal quality signal. This runner keeps the supervised ground-truth term uniform and uses confidence to gate only the DFF/GADFF consistency term.",
        "",
        "## Channel and Loss Mapping",
        "",
        "| Channel | Meaning | Loss role |",
        "|---|---|---|",
        "| 33 | glare/risk prior | softens prior consistency in high-risk regions |",
        "| 34 | DFF depth prior | fused prior target |",
        "| 35 | DFF confidence | dominant prior gate |",
        "| 36 | GADFF depth prior | fused prior target |",
        "| 37 | GADFF confidence | secondary prior gate |",
        "",
        "## Smoke Result",
        "",
        f"- Epochs: `{args.max_epochs}`",
        f"- Prepared train samples: `{report['prepared_train_samples']}`",
        f"- Prepared validation samples: `{report['prepared_validation_samples']}`",
        f"- Last validation MAE norm: `{last['val_mae_norm_debug']:.8f}`",
        f"- Last validation focus confidence mean: `{last['val_focus_conf_mean']:.8f}`",
        f"- Last validation prior weight mean: `{last['val_prior_weight_mean']:.8f}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Metrics: `{report['metrics_csv']}`",
        "",
        "## Claim Boundary",
        "",
        "This result only proves that the new loss passes data generation, feature augmentation, forward, backward, checkpoint, and logging. It should not be used as a paper table value until a full matched-split run, external evaluation, seed repeat, and eligibility audit pass.",
        "",
        "## Next Full Candidate Command",
        "",
        "```powershell",
        full_command,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_run_config(report: dict[str, Any]) -> dict[str, Any]:
    cfg_path = ABL_ROOT / RUN_ID / "run_config.json"
    existing = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    preserved = {
        key: existing[key]
        for key in (
            "latest_paired_loss_smoke",
            "latest_stratified_diagnostic",
            "confidence_gated_full_split_evaluations",
        )
        if key in existing
    }
    config = {
        "run_id": RUN_ID,
        "variant": VARIANT,
        "workspace": str(ABL_ROOT / RUN_ID),
        "feature_switches": {
            "dff_gadff_prior": True,
            "focal_difference": True,
            "glare_cue": True,
            "domain_randomization": True,
            "residual_bounded": True,
        },
        "loss_switches": {
            "uniform_supervised_data_term": True,
            "confidence_gated_prior_consistency": True,
            "direct_glare_data_upweight": False,
        },
        "channel_map": CHANNEL_MAP,
        "status": "confidence_gated_prior_smoke_run",
        "main_table_eligible": False,
        "claim_eligible": False,
        "source_training_entry": "submission_planning/tools/run_confidence_weighted_loss_training.py",
        "source_model_entry": "src/train_focus_resunet_loss_experiment.py::FocusResUNet",
        "latest_smoke": {
            "date": DATE,
            "tag": report["tag"],
            "seed": report["seed"],
            "checkpoint": report["checkpoint"],
            "metrics_csv": report["metrics_csv"],
            "metrics_json": report["metrics_json"],
            "prepared_train_samples": report["prepared_train_samples"],
            "prepared_validation_samples": report["prepared_validation_samples"],
            "history": report["history"],
            "interpretation": "Smoke-level continuity artifact only. Not manuscript evidence.",
        },
    }
    config.update(preserved)
    return config


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    tag = safe_tag(args.tag)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)

    dataset = build_dataset()
    split_counts = {name: len(items) for name, items in dataset.items()}
    train_samples = prepare_samples(dataset["train"], args.max_train_samples)
    val_samples = prepare_samples(dataset["validation"], args.max_val_samples)

    model = FocusResUNet().to(args.device)
    criterion = ConfidenceGatedPriorLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.5e-4)

    history: list[dict[str, float]] = []
    train_steps = max(1, int(math.ceil(args.train_patches / args.batch_size)))
    val_steps = max(1, int(math.ceil(args.val_patches / args.batch_size)))

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_parts_rows: list[dict[str, float]] = []
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
            train_parts_rows.append(parts)

        model.eval()
        val_parts_rows: list[dict[str, float]] = []
        val_mae_rows: list[float] = []
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = split_patch_batch(val_samples, rng, args.batch_size, args.patch_size)
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                pred = model(xb)
                _, parts = criterion(pred, yb, xb)
                val_parts_rows.append(parts)
                val_mae_rows.append(float(torch.mean(torch.abs(pred - yb)).detach().cpu()))

        train_parts = mean_parts(train_parts_rows)
        val_parts = mean_parts(val_parts_rows)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_debug": train_parts["total"],
                "val_loss_debug": val_parts["total"],
                "val_mae_norm_debug": float(np.mean(val_mae_rows)),
                "train_data_debug": train_parts["data"],
                "val_data_debug": val_parts["data"],
                "train_focus_prior_debug": train_parts["focus_prior"],
                "val_focus_prior_debug": val_parts["focus_prior"],
                "train_focus_conf_mean": train_parts["focus_conf_mean"],
                "val_focus_conf_mean": val_parts["focus_conf_mean"],
                "train_prior_weight_mean": train_parts["prior_weight_mean"],
                "val_prior_weight_mean": val_parts["prior_weight_mean"],
                "train_risk_mean": train_parts["risk_mean"],
                "val_risk_mean": val_parts["risk_mean"],
            }
        )

    run_root = ABL_ROOT / RUN_ID
    checkpoint_path = run_root / "checkpoints" / f"{tag}.pt"
    metrics_csv = run_root / "metrics" / f"{tag}_metrics.csv"
    metrics_json = run_root / "metrics" / f"{tag}_metrics.json"
    run_config_path = run_root / "run_config.json"
    log_json = run_root / "logs" / f"{tag}.json"
    log_md = run_root / "logs" / f"{tag}.md"
    summary_json = SUMMARY_DIR / f"{tag}_summary.json"
    summary_md = SUMMARY_DIR / f"{tag}_summary.md"
    mechanism_report = REPORT_DIR / "confidence_gated_prior_loss_runner_report.md"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_id": RUN_ID,
            "variant": VARIANT,
            "runner_mode": RUNNER_MODE,
            "loss": "ConfidenceGatedPriorLoss",
            "debug_only": True,
            "claim_eligible": False,
            "tag": tag,
            "channel_map": CHANNEL_MAP,
            "history": history,
            "training": {
                "seed": seed,
                "max_epochs": args.max_epochs,
                "train_patches": args.train_patches,
                "val_patches": args.val_patches,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "patch_size": args.patch_size,
                "device": args.device,
                "split_counts": split_counts,
                "prepared_train_samples": len(train_samples),
                "prepared_validation_samples": len(val_samples),
            },
        },
        checkpoint_path,
    )
    write_metrics_csv(metrics_csv, history)
    write_json(metrics_json, history)

    checks = [
        check("run root under tmp/ablation_results", str(run_root).startswith(str(ABL_ROOT)), str(run_root)),
        check("upgraded feature count", train_samples[0]["model_features"].shape[0] == upgraded_channel_count(), str(train_samples[0]["model_features"].shape)),
        check("train split count", split_counts.get("train") == 27, str(split_counts.get("train"))),
        check("validation split count", split_counts.get("validation") == 10, str(split_counts.get("validation"))),
        check("test split count", split_counts.get("test") == 7, str(split_counts.get("test"))),
        check("prepared train samples", len(train_samples) > 0, str(len(train_samples))),
        check("prepared validation samples", len(val_samples) > 0, str(len(val_samples))),
        check("history rows", len(history) == args.max_epochs, str(len(history))),
        check("history finite", all(np.isfinite(float(value)) for row in history for value in row.values()), "finite"),
        check("checkpoint exists", checkpoint_path.exists(), str(checkpoint_path)),
        check("metrics CSV exists", metrics_csv.exists(), str(metrics_csv)),
        check("metrics JSON exists", metrics_json.exists(), str(metrics_json)),
    ]
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    status = "confidence_gated_prior_smoke_completed" if not errors else "confidence_gated_prior_smoke_failed"

    report = {
        "run_id": RUN_ID,
        "variant": VARIANT,
        "runner_mode": RUNNER_MODE,
        "status": status,
        "date": DATE,
        "device": args.device,
        "tag": tag,
        "seed": seed,
        "split_counts": split_counts,
        "prepared_train_samples": len(train_samples),
        "prepared_validation_samples": len(val_samples),
        "checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "run_config": str(run_config_path),
        "history": history,
        "checks": checks,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "elapsed_s": time.time() - start,
        "interpretation": "Smoke-level continuity artifact only. Not manuscript evidence.",
    }

    run_config = build_run_config(report)
    write_json(run_config_path, run_config)
    report["checks"].append(check("claim_eligible remains false", run_config["claim_eligible"] is False, str(run_config["claim_eligible"])))
    write_json(log_json, report)
    write_run_log(log_md, report)
    write_json(summary_json, report)
    write_summary(summary_md, report)
    write_mechanism_report(mechanism_report, report, args)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="2026-06-22_confidence_gated_prior_smoke")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-patches", type=int, default=4)
    parser.add_argument("--val-patches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.tag = safe_tag(args.tag)
    report = run_smoke(args)
    print(json.dumps({"status": report["status"], "tag": report["tag"], "last_history": report["history"][-1]}, ensure_ascii=False, indent=2))
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
