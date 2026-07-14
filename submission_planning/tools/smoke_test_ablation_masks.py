"""Smoke-test ablation input masks without training or inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_dataset_training import surface_scenario  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402
from train_focus_resunet_loss_experiment import augment_features  # noqa: E402


DEFAULT_OUT_DIR = Path("tmp/ablation_results/mask_smoke")

MASK_SPECS = {
    "ABL-01": {"feature_space": "base", "zero_channels": list(range(17, 22)), "description": "Direct image-to-depth"},
    "ABL-02": {"feature_space": "base", "zero_channels": list(range(18, 22)), "description": "w/o DFF/GADFF prior"},
    "ABL-03": {"feature_space": "upgraded", "zero_channels": list(range(17, 33)), "description": "w/o focal-difference input signal"},
    "ABL-04": {"feature_space": "base", "zero_channels": [17], "description": "w/o glare cue"},
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


def apply_mask(features: np.ndarray, zero_channels: list[int]) -> np.ndarray:
    masked = features.copy()
    masked[zero_channels, :, :] = 0.0
    return masked


def unchanged_channels_ok(original: np.ndarray, masked: np.ndarray, zero_channels: list[int]) -> bool:
    keep = [idx for idx in range(original.shape[0]) if idx not in set(zero_channels)]
    if not keep:
        return True
    return bool(np.allclose(original[keep], masked[keep]))


def summarize_mask(run_id: str, features: np.ndarray, zero_channels: list[int]) -> dict[str, Any]:
    masked = apply_mask(features, zero_channels)
    target = masked[zero_channels, :, :] if zero_channels else np.empty((0,), dtype=np.float32)
    original_target = features[zero_channels, :, :] if zero_channels else np.empty((0,), dtype=np.float32)
    checks = [
        check(f"{run_id} target channels zeroed", bool(target.size == 0 or np.max(np.abs(target)) == 0.0), f"max_abs={float(np.max(np.abs(target))) if target.size else 0.0}"),
        check(f"{run_id} non-target channels unchanged", unchanged_channels_ok(features, masked, zero_channels), f"zero_channels={zero_channels}"),
        check(f"{run_id} target had nonzero signal before mask", bool(original_target.size == 0 or np.max(np.abs(original_target)) > 0.0), f"pre_mask_max_abs={float(np.max(np.abs(original_target))) if original_target.size else 0.0}"),
    ]
    return {
        "run_id": run_id,
        "feature_shape": list(features.shape),
        "zero_channels": zero_channels,
        "pre_mask_target_mean_abs": float(np.mean(np.abs(original_target))) if original_target.size else 0.0,
        "post_mask_target_mean_abs": float(np.mean(np.abs(target))) if target.size else 0.0,
        "checks": checks,
    }


def build_report() -> dict[str, Any]:
    scenario = build_p10_scenario()
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
    base = np.asarray(arrays["features"], dtype=np.float32)
    upgraded = augment_features(base)
    summaries = []
    checks = [
        check("base feature shape", base.shape[0] == 22, f"base={base.shape}"),
        check("upgraded feature shape", upgraded.shape[0] == 38, f"upgraded={upgraded.shape}"),
    ]
    for run_id, spec in MASK_SPECS.items():
        features = base if spec["feature_space"] == "base" else upgraded
        summary = summarize_mask(run_id, features, spec["zero_channels"])
        summary["feature_space"] = spec["feature_space"]
        summary["description"] = spec["description"]
        summaries.append(summary)
        checks.extend(summary["checks"])
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    return {
        "status": "pass" if not errors else "fail",
        "sample_id": scenario.name,
        "base_feature_shape": list(base.shape),
        "upgraded_feature_shape": list(upgraded.shape),
        "mask_summaries": summaries,
        "check_count": len(checks),
        "error_count": len(errors),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Mask Smoke Test",
        "",
        f"- Status: {report['status']}",
        f"- Sample: {report['sample_id']}",
        f"- Base feature shape: {report['base_feature_shape']}",
        f"- Upgraded feature shape: {report['upgraded_feature_shape']}",
        f"- Errors: {report['error_count']}",
        "",
        "## Mask Summary",
        "",
        "| Run | Space | Zero Channels | Pre-mask Mean Abs | Post-mask Mean Abs |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["mask_summaries"]:
        lines.append(
            f"| {row['run_id']} | {row['feature_space']} | {row['zero_channels']} | "
            f"{row['pre_mask_target_mean_abs']:.8f} | {row['post_mask_target_mean_abs']:.8f} |"
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
    args = parser.parse_args()
    report = build_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ablation_mask_smoke_test.json"
    md_path = args.out_dir / "ablation_mask_smoke_test.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation mask smoke test: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

