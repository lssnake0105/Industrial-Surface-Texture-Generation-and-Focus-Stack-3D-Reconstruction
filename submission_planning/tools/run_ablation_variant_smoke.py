"""Smoke-test the minimal ablation runner interface without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_dataset_training import surface_scenario  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402
from train_focus_resunet_loss_experiment import (  # noqa: E402
    FocusResUNet,
    HybridDFFLoss,
    augment_features,
    upgraded_channel_count,
)


DEFAULT_OUT_DIR = Path("tmp/ablation_results/runner_smoke")
PATCH_SIZE = 64
SEED = 20260619

VARIANT_SPECS = {
    "ABL-00": {
        "variant": "Full S2R-FocusNet",
        "feature_space": "upgraded_38_channel",
        "zero_channels": [],
        "forward_smoke": True,
        "decision": "final-method anchor",
    },
    "ABL-01": {
        "variant": "Direct image-to-depth",
        "feature_space": "raw_focus_stack_17_channel",
        "zero_channels": [],
        "forward_smoke": False,
        "decision": "requires derived lower-prior model; do not emulate with final FocusResUNet channel masking",
    },
    "ABL-02": {
        "variant": "w/o DFF/GADFF prior",
        "feature_space": "upgraded_38_channel",
        "zero_channels": [34, 35, 36, 37],
        "forward_smoke": True,
        "decision": "zero upgraded DFF/GADFF depth and confidence prior channels while keeping risk",
    },
    "ABL-03": {
        "variant": "w/o focal difference",
        "feature_space": "upgraded_38_channel",
        "zero_channels": list(range(17, 33)),
        "forward_smoke": True,
        "decision": "zero upgraded focal-difference input channels 17-32",
    },
    "ABL-04": {
        "variant": "w/o glare cue",
        "feature_space": "upgraded_38_channel",
        "zero_channels": [33],
        "forward_smoke": True,
        "decision": "zero explicit risk cue; stricter GADFF-derived glare removal remains a separate decision",
    },
}


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def build_p10_scenario():
    return surface_scenario(
        "test_V谷_P10_宽谷粗糙平底",
        960,
        540,
        611,
        1200,
        "v_valley",
        "perlin",
        tilt_x_um=80,
        tilt_y_um=-45,
        feature_amp_um=820,
        noise_amp_um=72,
        perlin_octaves=6,
        perlin_grid=160,
        perlin_persistence=0.60,
        valley_width=0.54,
        valley_floor=0.160,
        valley_sharpness=0.70,
        orientation_deg=-15,
        stray_level=0.20,
        roughness_base=0.40,
        f0=0.76,
    )


def center_patch(features: np.ndarray, truth: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = features.shape
    if height < size or width < size:
        raise ValueError(f"Patch size {size} is larger than feature map {features.shape}")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return features[:, y0 : y0 + size, x0 : x0 + size], truth[y0 : y0 + size, x0 : x0 + size]


def apply_zero_channels(features: np.ndarray, zero_channels: list[int]) -> np.ndarray:
    masked = features.copy()
    if zero_channels:
        masked[zero_channels, :, :] = 0.0
    return masked


def artifact_files(run_id: str) -> list[str]:
    run_root = Path("tmp/ablation_results") / run_id
    if not run_root.exists():
        return []
    risky_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".png", ".jpg", ".jpeg"}
    return [str(path) for path in sorted(run_root.rglob("*")) if path.is_file() and path.suffix.lower() in risky_suffixes]


def smoke_forward(
    model: FocusResUNet,
    criterion: HybridDFFLoss,
    features: np.ndarray,
    truth: np.ndarray,
    device: str,
) -> dict[str, Any]:
    x = torch.from_numpy(features[None].astype(np.float32)).to(device)
    y = torch.from_numpy(truth[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x)
        loss, parts = criterion(out, y, x)
    return {
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "output_shape": list(out.shape),
        "output_min": float(out.min().detach().cpu()),
        "output_max": float(out.max().detach().cpu()),
        "loss_total_diagnostic": float(loss.detach().cpu()),
        "loss_parts_diagnostic": parts,
    }


def summarize_variant(
    run_id: str,
    spec: dict[str, Any],
    upgraded_patch: np.ndarray,
    truth_patch: np.ndarray,
    model: FocusResUNet,
    criterion: HybridDFFLoss,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    zero_channels = list(spec["zero_channels"])
    row: dict[str, Any] = {
        "run_id": run_id,
        "variant": spec["variant"],
        "feature_space": spec["feature_space"],
        "zero_channels": zero_channels,
        "forward_smoke": bool(spec["forward_smoke"]),
        "decision": spec["decision"],
        "status": "planned_lower_prior_runner" if not spec["forward_smoke"] else "shape_smoke_pending",
    }
    artifacts = artifact_files(run_id)
    checks.append(check(f"{run_id} no accidental model/result artifacts", not artifacts, f"artifact_files={artifacts}", "warning"))

    if not spec["forward_smoke"]:
        raw_stack = upgraded_patch[:DEFAULT_STACK_LAYERS]
        row["raw_stack_shape"] = list(raw_stack.shape)
        row["status"] = "design_recorded_no_forward"
        checks.append(check(f"{run_id} lower-prior runner decision recorded", True, spec["decision"]))
        checks.append(check(f"{run_id} raw stack has expected channels", raw_stack.shape[0] == DEFAULT_STACK_LAYERS, str(raw_stack.shape)))
        return row, checks

    masked = apply_zero_channels(upgraded_patch, zero_channels)
    target = masked[zero_channels, :, :] if zero_channels else np.empty((0,), dtype=np.float32)
    original_target = upgraded_patch[zero_channels, :, :] if zero_channels else np.empty((0,), dtype=np.float32)
    keep_channels = [idx for idx in range(upgraded_patch.shape[0]) if idx not in set(zero_channels)]
    row.update(
        {
            "pre_mask_target_mean_abs": float(np.mean(np.abs(original_target))) if original_target.size else 0.0,
            "post_mask_target_mean_abs": float(np.mean(np.abs(target))) if target.size else 0.0,
        }
    )
    checks.append(check(f"{run_id} upgraded feature channels", masked.shape[0] == upgraded_channel_count(), str(masked.shape)))
    checks.append(check(f"{run_id} target channels zeroed", bool(target.size == 0 or np.max(np.abs(target)) == 0.0), f"zero_channels={zero_channels}"))
    checks.append(
        check(
            f"{run_id} non-target channels unchanged",
            bool(not keep_channels or np.allclose(masked[keep_channels], upgraded_patch[keep_channels])),
            f"kept_channels={len(keep_channels)}",
        )
    )
    if zero_channels:
        checks.append(check(f"{run_id} target had signal before mask", bool(np.max(np.abs(original_target)) > 0.0), f"pre_max={float(np.max(np.abs(original_target)))}"))

    smoke = smoke_forward(model, criterion, masked, truth_patch, device)
    row["forward_summary"] = smoke
    row["status"] = "shape_smoke_passed"
    checks.append(check(f"{run_id} forward output shape", smoke["output_shape"] == [1, 1, PATCH_SIZE, PATCH_SIZE], str(smoke["output_shape"])))
    checks.append(check(f"{run_id} diagnostic loss finite", np.isfinite(smoke["loss_total_diagnostic"]), str(smoke["loss_total_diagnostic"])))
    return row, checks


def build_report(device: str) -> dict[str, Any]:
    torch.manual_seed(SEED)
    scenario = build_p10_scenario()
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
    base = np.asarray(arrays["features"], dtype=np.float32)
    truth = np.asarray(arrays["truth"], dtype=np.float32)
    upgraded = augment_features(base)
    upgraded_patch, truth_patch = center_patch(upgraded, truth, PATCH_SIZE)

    model = FocusResUNet().to(device)
    model.eval()
    criterion = HybridDFFLoss()

    checks: list[dict[str, Any]] = [
        check("base feature shape", list(base.shape) == [22, 540, 960], str(base.shape)),
        check("upgraded feature count", upgraded.shape[0] == upgraded_channel_count(), f"{upgraded.shape[0]} vs {upgraded_channel_count()}"),
        check("patch shape", list(upgraded_patch.shape) == [upgraded_channel_count(), PATCH_SIZE, PATCH_SIZE], str(upgraded_patch.shape)),
        check("model is eval mode", not model.training, f"training={model.training}"),
    ]
    variants = []
    for run_id, spec in VARIANT_SPECS.items():
        row, variant_checks = summarize_variant(run_id, spec, upgraded_patch, truth_patch, model, criterion, device)
        variants.append(row)
        checks.extend(variant_checks)
    grad_files = [name for name, param in model.named_parameters() if param.grad is not None]
    checks.append(check("no gradients accumulated", not grad_files, f"params_with_grad={grad_files}"))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "sample_id": scenario.name,
        "device": device,
        "patch_size": PATCH_SIZE,
        "seed": SEED,
        "interpretation": "Shape/config smoke only. No training, optimizer step, checkpoint, saved prediction, metric table, or module-effectiveness claim was produced.",
        "base_feature_shape": list(base.shape),
        "upgraded_feature_shape": list(upgraded.shape),
        "variant_summaries": variants,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Minimal Ablation Runner Smoke Test",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Sample: {report['sample_id']}",
        f"- Device: {report['device']}",
        f"- Patch size: {report['patch_size']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Variant Summary",
        "",
        "| Run | Variant | Feature Space | Zero Channels | Status |",
        "|---|---|---|---|---|",
    ]
    for row in report["variant_summaries"]:
        lines.append(
            f"| {row['run_id']} | {row['variant']} | {row['feature_space']} | "
            f"{row['zero_channels']} | {row['status']} |"
        )
    lines.extend(["", "## Forward Smoke Diagnostics", "", "| Run | Input Shape | Output Shape | Diagnostic Loss |", "|---|---|---|---:|"])
    for row in report["variant_summaries"]:
        smoke = row.get("forward_summary")
        if not smoke:
            lines.append(f"| {row['run_id']} | skipped | skipped | n/a |")
            continue
        lines.append(
            f"| {row['run_id']} | {smoke['input_shape']} | {smoke['output_shape']} | "
            f"{smoke['loss_total_diagnostic']:.8f} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    report = build_report(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "minimal_ablation_runner_smoke.json"
    md_path = args.out_dir / "minimal_ablation_runner_smoke.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Minimal ablation runner smoke: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
