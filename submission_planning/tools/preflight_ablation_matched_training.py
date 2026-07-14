"""Preflight the matched full-split ablation training plan.

The script is read-only with respect to project delivery assets. It verifies
dataset splits, feature masks, shared training settings, and output locations
for a future matched ablation training runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402
from train_focus_resunet_loss_experiment import FocusResUNet, HybridDFFLoss, augment_features, upgraded_channel_count  # noqa: E402

from run_ablation_variant_training import ABL_ROOT, RUN_SPECS, apply_zero_channels, safe_tag  # noqa: E402


DEFAULT_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]
DEFAULT_OUT_DIR = ABL_ROOT / "matched_training_preflight"
PATCH_SIZE = 64
SEED = 20260619


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sample_patch(features: np.ndarray, truth: np.ndarray, y0: int = 0, x0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return features[:, y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE], truth[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]


def build_variant_plan(run_id: str, train_sample: dict[str, Any], val_sample: dict[str, Any], device: str) -> dict[str, Any]:
    spec = RUN_SPECS[run_id]
    checks: list[dict[str, Any]] = []
    run_root = ABL_ROOT / run_id
    cfg_path = run_root / "run_config.json"
    cfg = read_json(cfg_path)
    checks.append(check(f"{run_id} run_config exists", cfg is not None, str(cfg_path)))
    if cfg:
        checks.append(check(f"{run_id} claim_eligible false", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
        checks.append(check(f"{run_id} main_table_eligible false", cfg.get("main_table_eligible") is False, str(cfg.get("main_table_eligible"))))
    checks.append(check(f"{run_id} trainable in upgraded runner", spec["status"] == "dry_run_supported", str(spec["status"])))

    train_features = np.asarray(train_sample["model_features"], dtype=np.float32)
    train_truth = np.asarray(train_sample["truth"], dtype=np.float32)
    val_features = np.asarray(val_sample["model_features"], dtype=np.float32)
    val_truth = np.asarray(val_sample["truth"], dtype=np.float32)
    masked_train = apply_zero_channels(train_features, list(spec["zero_channels"]))
    masked_val = apply_zero_channels(val_features, list(spec["zero_channels"]))
    target = masked_train[spec["zero_channels"], :, :] if spec["zero_channels"] else np.empty((0,), dtype=np.float32)
    checks.append(check(f"{run_id} upgraded train feature count", masked_train.shape[0] == upgraded_channel_count(), str(masked_train.shape)))
    checks.append(check(f"{run_id} upgraded val feature count", masked_val.shape[0] == upgraded_channel_count(), str(masked_val.shape)))
    checks.append(check(f"{run_id} target channels zeroed", bool(target.size == 0 or np.max(np.abs(target)) == 0.0), f"zero_channels={spec['zero_channels']}"))

    patch_x, patch_y = sample_patch(masked_train, train_truth)
    val_patch_x, val_patch_y = sample_patch(masked_val, val_truth)
    model = FocusResUNet().to(device)
    criterion = HybridDFFLoss()
    model.train()
    x = torch.from_numpy(patch_x[None].astype(np.float32)).to(device)
    y = torch.from_numpy(patch_y[None, None].astype(np.float32)).to(device)
    out = model(x)
    loss, parts = criterion(out, y, x)
    checks.append(check(f"{run_id} train forward shape", list(out.shape) == [1, 1, PATCH_SIZE, PATCH_SIZE], str(list(out.shape))))
    checks.append(check(f"{run_id} train loss finite", np.isfinite(float(loss.detach().cpu())), f"{float(loss.detach().cpu()):.8f}"))
    model.eval()
    vx = torch.from_numpy(val_patch_x[None].astype(np.float32)).to(device)
    vy = torch.from_numpy(val_patch_y[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        vout = model(vx)
        vloss, _ = criterion(vout, vy, vx)
    checks.append(check(f"{run_id} val forward shape", list(vout.shape) == [1, 1, PATCH_SIZE, PATCH_SIZE], str(list(vout.shape))))
    checks.append(check(f"{run_id} val loss finite", np.isfinite(float(vloss.detach().cpu())), f"{float(vloss.detach().cpu()):.8f}"))

    return {
        "run_id": run_id,
        "variant": spec["variant"],
        "runner_mode": spec["runner_mode"],
        "zero_channels": spec["zero_channels"],
        "trainable_now": spec["status"] == "dry_run_supported",
        "planned_output_root": str(run_root),
        "planned_checkpoint": str(run_root / "checkpoints" / "2026-06-19_matched_training_smoke.pt"),
        "planned_history_csv": str(run_root / "metrics" / "2026-06-19_matched_training_history.csv"),
        "planned_test_metrics_csv": str(run_root / "metrics" / "2026-06-19_matched_training_test_metrics.csv"),
        "diagnostic_train_loss": float(loss.detach().cpu()),
        "diagnostic_val_loss": float(vloss.detach().cpu()),
        "checks": checks,
    }


def prepare_split(split_items: list[tuple[str, Any]], max_samples: int = 0) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for category, scenario in split_items[: max_samples or None]:
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        base = np.asarray(arrays["features"], dtype=np.float32)
        arrays["model_features"] = augment_features(base)
        arrays["category"] = category
        prepared.append(arrays)
    return prepared


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tag = safe_tag(args.tag)
    checks: list[dict[str, Any]] = []
    dataset = build_dataset()
    split_counts = {name: len(items) for name, items in dataset.items()}
    checks.append(check("train split has samples", split_counts.get("train", 0) >= 12, str(split_counts.get("train", 0))))
    checks.append(check("validation split has samples", split_counts.get("validation", 0) >= 5, str(split_counts.get("validation", 0))))
    checks.append(check("test split has seven samples", split_counts.get("test", 0) == 7, str(split_counts.get("test", 0))))
    checks.append(check("tag is safe", tag == args.tag, tag))

    train_samples = prepare_split(dataset["train"], max_samples=args.max_train_samples)
    val_samples = prepare_split(dataset["validation"], max_samples=args.max_val_samples)
    checks.append(check("prepared train samples", bool(train_samples), f"prepared={len(train_samples)}"))
    checks.append(check("prepared validation samples", bool(val_samples), f"prepared={len(val_samples)}"))

    variant_plans = [
        build_variant_plan(run_id, train_samples[0], val_samples[0], args.device)
        for run_id in (args.run_id or DEFAULT_RUN_IDS)
    ]
    for plan in variant_plans:
        checks.extend(plan["checks"])

    matched_plan = {
        "tag": tag,
        "seed": SEED,
        "patch_size": PATCH_SIZE,
        "default_epochs": args.epochs,
        "default_train_patches_per_epoch": args.train_patches,
        "default_val_patches_per_epoch": args.val_patches,
        "default_batch_size": args.batch_size,
        "default_learning_rate": args.learning_rate,
        "train_split_count": split_counts["train"],
        "validation_split_count": split_counts["validation"],
        "test_split_count": split_counts["test"],
        "training_scope": "matched_train_validation_split",
        "output_scope": "tmp/ablation_results/<run_id>/",
        "claim_eligible_after_preflight": False,
        "main_table_eligible_after_preflight": False,
    }
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "tag": tag,
        "device": args.device,
        "split_counts": split_counts,
        "matched_training_plan": matched_plan,
        "variant_plans": variant_plans,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": "Preflight only. No optimizer step, checkpoint, test prediction, or manuscript-eligible result was produced.",
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    plan = report["matched_training_plan"]
    lines = [
        "# Ablation Matched Training Preflight",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Tag: {report['tag']}",
        f"- Device: {report['device']}",
        f"- Split counts: {report['split_counts']}",
        f"- Claim eligible: {str(report['claim_eligible']).lower()}",
        f"- Main table eligible: {str(report['main_table_eligible']).lower()}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Matched Training Plan",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in plan.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Variant Plans",
            "",
            "| Run | Variant | Trainable | Zero Channels | Train Loss Diagnostic | Val Loss Diagnostic |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in report["variant_plans"]:
        lines.append(
            f"| {row['run_id']} | {row['variant']} | {row['trainable_now']} | "
            f"`{row['zero_channels']}` | {row['diagnostic_train_loss']:.8f} | {row['diagnostic_val_loss']:.8f} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=sorted(RUN_SPECS), help="Run id to preflight. May be repeated.")
    parser.add_argument("--tag", default="2026-06-19_matched_training_preflight")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-patches", type=int, default=32)
    parser.add_argument("--val-patches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{report['tag']}.json"
    md_path = args.out_dir / f"{report['tag']}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation matched training preflight: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
