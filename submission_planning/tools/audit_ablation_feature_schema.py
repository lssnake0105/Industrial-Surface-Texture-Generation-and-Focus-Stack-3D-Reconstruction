"""Audit feature-channel schema for planned ablation variants."""

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
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, feature_channel_count, generate_sample_arrays  # noqa: E402


DEFAULT_OUT_DIR = Path("tmp/ablation_results/schema_audit")


CHANNEL_SCHEMA = [
    {"name": "focus_stack", "start": 0, "end": 16, "channels": list(range(0, 17))},
    {"name": "risk_map", "start": 17, "end": 17, "channels": [17]},
    {"name": "dff_depth", "start": 18, "end": 18, "channels": [18]},
    {"name": "focus_confidence", "start": 19, "end": 19, "channels": [19]},
    {"name": "gadff_depth", "start": 20, "end": 20, "channels": [20]},
    {"name": "gadff_confidence", "start": 21, "end": 21, "channels": [21]},
]

ABLATION_MASKS = {
    "ABL-00": {"variant": "Full S2R-FocusNet", "zero_channels": [], "implementation": "keep all channels"},
    "ABL-01": {"variant": "Direct image-to-depth", "zero_channels": [17, 18, 19, 20, 21], "implementation": "keep focus stack only"},
    "ABL-02": {"variant": "w/o DFF/GADFF prior", "zero_channels": [18, 19, 20, 21], "implementation": "keep focus stack and risk map"},
    "ABL-03": {"variant": "w/o focal difference", "zero_channels": [], "implementation": "use upgraded 38-channel Focus-ResUNet features; zero upgraded channels 17-32"},
    "ABL-04": {"variant": "w/o glare cue", "zero_channels": [17], "implementation": "zero risk map; GADFF-derived channels need a separate decision"},
    "ABL-05": {"variant": "w/o domain randomization", "zero_channels": [], "implementation": "dataset-generation control, not channel mask"},
    "ABL-06": {"variant": "unbounded prediction", "zero_channels": [], "implementation": "model/loss/output-head control, not channel mask"},
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


def channel_stats(features: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for group in CHANNEL_SCHEMA:
        arr = features[group["channels"], :, :]
        rows.append(
            {
                "name": group["name"],
                "channels": group["channels"],
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    scenario = build_p10_scenario()
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
    features = np.asarray(arrays["features"], dtype=np.float32)
    stack = np.asarray(arrays["stack"], dtype=np.float32)
    risk = np.asarray(arrays["risk"], dtype=np.float32)
    dff = np.asarray(arrays["dff"], dtype=np.float32)
    gadff = np.asarray(arrays["gadff"], dtype=np.float32)
    checks = [
        check("feature channel count", features.shape[0] == feature_channel_count(DEFAULT_STACK_LAYERS), f"features={features.shape[0]}, expected={feature_channel_count(DEFAULT_STACK_LAYERS)}"),
        check("feature height/width", features.shape[1:] == stack.shape[1:], f"features={features.shape}, stack={stack.shape}"),
        check("risk channel matches generated risk", np.allclose(features[17], risk), "channel 17 vs risk"),
        check("DFF channel matches generated dff", np.allclose(features[18], dff), "channel 18 vs dff"),
        check("GADFF channel matches generated gadff", np.allclose(features[20], gadff), "channel 20 vs gadff"),
    ]
    warnings = [
        check("ABL-03 upgraded feature path audited", True, "see ABL-03 focal-difference implementation audit"),
        check("ABL-05 needs dataset-generation control", False, "domain randomization is not a channel mask", "warning"),
        check("ABL-06 needs model-output control", False, "residual/bounded behavior is not a channel mask", "warning"),
    ]
    all_checks = checks + warnings
    errors = [row for row in all_checks if not row["passed"] and row["severity"] == "error"]
    warning_rows = [row for row in all_checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "sample_id": scenario.name,
        "feature_shape": list(features.shape),
        "stack_shape": list(stack.shape),
        "expected_channel_count": feature_channel_count(DEFAULT_STACK_LAYERS),
        "channel_schema": CHANNEL_SCHEMA,
        "channel_stats": channel_stats(features),
        "ablation_masks": ABLATION_MASKS,
        "check_count": len(all_checks),
        "error_count": len(errors),
        "warning_count": len(warning_rows),
        "checks": all_checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Feature Schema Audit",
        "",
        f"- Status: {report['status']}",
        f"- Sample: {report['sample_id']}",
        f"- Feature shape: {report['feature_shape']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Channel Schema",
        "",
        "| Name | Channels | Mean | Std | Min | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["channel_stats"]:
        lines.append(
            f"| {row['name']} | {row['channels']} | {row['mean']:.6f} | {row['std']:.6f} | {row['min']:.6f} | {row['max']:.6f} |"
        )
    lines.extend(["", "## Ablation Masks", "", "| Run | Variant | Zero Channels | Implementation Note |", "|---|---|---|---|"])
    for run_id, row in report["ablation_masks"].items():
        lines.append(f"| {run_id} | {row['variant']} | {row['zero_channels']} | {row['implementation']} |")
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
    json_path = args.out_dir / "ablation_feature_schema_audit.json"
    md_path = args.out_dir / "ablation_feature_schema_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation feature schema audit: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
