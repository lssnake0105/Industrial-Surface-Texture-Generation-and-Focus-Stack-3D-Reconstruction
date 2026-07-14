"""Audit the actual focal-difference implementation for ABL-03."""

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
from train_focus_resunet_loss_experiment import augment_features, upgraded_channel_count  # noqa: E402


DEFAULT_OUT_DIR = Path("tmp/ablation_results/ABL-03/logs")


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


def build_report() -> dict[str, Any]:
    scenario = build_p10_scenario()
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
    base = np.asarray(arrays["features"], dtype=np.float32)
    stack = np.asarray(arrays["stack"], dtype=np.float32)
    upgraded = augment_features(base)
    expected_diff = np.diff(stack, axis=0)
    actual_diff = upgraded[DEFAULT_STACK_LAYERS : DEFAULT_STACK_LAYERS + DEFAULT_STACK_LAYERS - 1]
    prior = upgraded[DEFAULT_STACK_LAYERS + DEFAULT_STACK_LAYERS - 1 :]
    checks = [
        check("base feature shape", base.shape[0] == 22, f"base={base.shape}"),
        check("upgraded channel count", upgraded.shape[0] == upgraded_channel_count(), f"upgraded={upgraded.shape[0]}, expected={upgraded_channel_count()}"),
        check("upgraded spatial shape", upgraded.shape[1:] == base.shape[1:], f"upgraded={upgraded.shape}, base={base.shape}"),
        check("stack channels preserved", np.allclose(upgraded[:DEFAULT_STACK_LAYERS], stack), "channels 0-16 vs stack"),
        check("focal-difference channels match np.diff(stack)", np.allclose(actual_diff, expected_diff), "channels 17-32 vs np.diff(stack)"),
        check("prior channel count", prior.shape[0] == 5, f"prior channels={prior.shape[0]}"),
    ]
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    return {
        "status": "pass" if not errors else "fail",
        "sample_id": scenario.name,
        "base_feature_shape": list(base.shape),
        "upgraded_feature_shape": list(upgraded.shape),
        "stack_channel_range": [0, 16],
        "focal_difference_channel_range": [17, 32],
        "prior_channel_range": [33, 37],
        "abl03_recommended_action": "zero channels 17-32 on upgraded features while keeping the 38-channel Focus-ResUNet architecture",
        "check_count": len(checks),
        "error_count": len(errors),
        "checks": checks,
        "diff_stats": {
            "min": float(np.min(actual_diff)),
            "max": float(np.max(actual_diff)),
            "mean": float(np.mean(actual_diff)),
            "std": float(np.std(actual_diff)),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ABL-03 Focal-Difference Implementation Audit",
        "",
        f"- Status: {report['status']}",
        f"- Sample: {report['sample_id']}",
        f"- Base feature shape: {report['base_feature_shape']}",
        f"- Upgraded feature shape: {report['upgraded_feature_shape']}",
        f"- Focal-difference channels: {report['focal_difference_channel_range']}",
        f"- Recommended ABL-03 action: {report['abl03_recommended_action']}",
        "",
        "## Difference Channel Stats",
        "",
        f"- min: {report['diff_stats']['min']:.6f}",
        f"- max: {report['diff_stats']['max']:.6f}",
        f"- mean: {report['diff_stats']['mean']:.6f}",
        f"- std: {report['diff_stats']['std']:.6f}",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---|---|---|",
    ]
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
    json_path = args.out_dir / "abl03_focal_difference_implementation_audit.json"
    md_path = args.out_dir / "abl03_focal_difference_implementation_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"ABL-03 focal-difference audit: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

