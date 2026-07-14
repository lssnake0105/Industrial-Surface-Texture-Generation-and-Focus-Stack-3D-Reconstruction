"""Export recoverable sample imaging panels for the matched ablation runs.

The script writes only under tmp/ablation_results. It regenerates the fixed
synthetic test samples from the project generator, loads existing checkpoints,
and exports focus-stack, prior, prediction, and error-map panels for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEFAULT_CHECKPOINT_TAG = "2026-06-19_matched_training_longer_repeat"
DEFAULT_EXPORT_TAG = "2026-06-19_longer_repeat_sample_visuals"
CHECKPOINT_CONFIG_KEYS = [
    "matched_longer_repeat_training",
    "matched_full_candidate_training",
    "matched_smoke_training",
    "pilot_training",
    "debug_training",
]


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


def run_config(run_id: str) -> dict[str, Any] | None:
    return read_json(ABL_ROOT / run_id / "run_config.json")


def checkpoint_for_run(run_id: str, checkpoint_tag: str) -> Path:
    cfg = run_config(run_id) or {}
    for key in CHECKPOINT_CONFIG_KEYS:
        entry = cfg.get(key, {})
        if isinstance(entry, dict) and entry.get("tag") == checkpoint_tag and entry.get("checkpoint"):
            return Path(str(entry["checkpoint"]))
    return ABL_ROOT / run_id / "checkpoints" / f"{checkpoint_tag}.pt"


def load_model(device: str, checkpoint: Path) -> FocusResUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def show_image(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = "viridis", vmin: float | None = 0.0, vmax: float | None = 1.0) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def save_focus_panel(path: Path, stack: np.ndarray) -> None:
    indices = [0, 4, 8, 12, 16]
    fig, axes = plt.subplots(1, len(indices), figsize=(12, 3), constrained_layout=True)
    for ax, index in zip(axes, indices):
        show_image(ax, stack[index], f"Focus layer {index:02d}", cmap="gray")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prior_panel(path: Path, truth: np.ndarray, dff: np.ndarray, gadff: np.ndarray, risk: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    show_image(axes[0], truth, "Ground truth depth")
    show_image(axes[1], dff, "DFF depth prior")
    show_image(axes[2], gadff, "GADFF depth prior")
    show_image(axes[3], risk, "High-risk mask", cmap="magma")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prediction_panel(path: Path, predictions: dict[str, np.ndarray], truth: np.ndarray) -> None:
    run_ids = list(predictions)
    fig, axes = plt.subplots(2, len(run_ids), figsize=(3.1 * len(run_ids), 6), constrained_layout=True)
    error_max = max(float(np.percentile(np.abs(pred - truth), 99)) for pred in predictions.values())
    error_max = max(error_max, 1e-3)
    for col, run_id in enumerate(run_ids):
        variant = str(RUN_SPECS[run_id]["variant"])
        pred = predictions[run_id]
        show_image(axes[0, col], pred, f"{run_id}\n{variant}")
        show_image(axes[1, col], np.abs(pred - truth), "Absolute error", cmap="inferno", vmin=0.0, vmax=error_max)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Sample Imaging Recovery Export",
        "",
        f"- Status: {report['status']}",
        f"- Export tag: {report['export_tag']}",
        f"- Checkpoint tag: {report['checkpoint_tag']}",
        f"- Device: {report['device']}",
        f"- Samples: {report['sample_count']}",
        f"- Runs: {', '.join(report['run_ids'])}",
        f"- Output directory: `{report['output_dir']}`",
        "",
        "## Recovery Scope",
        "",
        "- The focus-stack and prior panels are regenerated from the fixed synthetic test scenarios.",
        "- Prediction and error panels are re-inferred from the saved matched-longer-repeat checkpoints.",
        "- These panels are reporting artifacts for supervisor discussion; metric eligibility remains governed by the existing audits.",
        "",
        "## Sample Mapping",
        "",
        "| Sample ID | Category | Scenario | Resolution | Depth range um | Focus stack | Priors | Predictions |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in report["samples"]:
        lines.append(
            f"| {row['sample_id']} | {row['category']} | {row['scenario']} | {row['resolution']} | "
            f"{float(row['depth_range_um']):.1f} | `{row['focus_panel']}` | `{row['prior_panel']}` | `{row['prediction_panel']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_visuals(args: argparse.Namespace) -> dict[str, Any]:
    export_tag = safe_tag(args.tag)
    checkpoint_tag = safe_tag(args.checkpoint_tag)
    run_ids = args.run_id or DEFAULT_RUN_IDS
    out_dir = ABL_ROOT / "sample_visual_recovery" / export_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoints: dict[str, Path] = {}
    models: dict[str, FocusResUNet] = {}
    for run_id in run_ids:
        checkpoint = checkpoint_for_run(run_id, checkpoint_tag)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint missing for {run_id}: {checkpoint}")
        checkpoints[run_id] = checkpoint
        models[run_id] = load_model(device, checkpoint)

    sample_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    test_items = build_dataset()["test"]
    if args.max_samples:
        test_items = test_items[: args.max_samples]

    for index, (category, scenario) in enumerate(test_items, start=1):
        sample_id = f"S{index:02d}"
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        stack = np.asarray(arrays["stack"], dtype=np.float32)
        base = np.asarray(arrays["features"], dtype=np.float32)
        truth = np.asarray(arrays["truth"], dtype=np.float32)
        risk = np.asarray(arrays["risk"], dtype=np.float32)
        dff = np.asarray(arrays["dff"], dtype=np.float32)
        gadff = np.asarray(arrays["gadff"], dtype=np.float32)
        upgraded = augment_features(base)

        focus_panel = Path("focus_stack") / f"{sample_id}_focus_stack_montage.png"
        prior_panel = Path("priors") / f"{sample_id}_reference_priors.png"
        prediction_panel = Path("predictions") / f"{sample_id}_longer_repeat_predictions_errors.png"
        save_focus_panel(out_dir / focus_panel, stack)
        save_prior_panel(out_dir / prior_panel, truth, dff, gadff, risk)

        predictions: dict[str, np.ndarray] = {}
        for run_id, model in models.items():
            spec = RUN_SPECS[run_id]
            masked = apply_zero_channels(upgraded, list(spec["zero_channels"]))
            pred = predict_tiled_upgraded(model, masked, device, tile=args.tile, overlap=args.overlap)
            predictions[run_id] = pred
            row_metrics = metrics(pred, truth, risk, scenario.depth_range_um)
            metric_rows.append(
                {
                    "sample_id": sample_id,
                    "run_id": run_id,
                    "variant": spec["variant"],
                    "category": category,
                    "scenario": scenario.name,
                    "mae_um": row_metrics["mae_um"],
                    "edge_mae_um": row_metrics["edge_mae_um"],
                    "high_risk_mae_um": row_metrics["high_risk_mae_um"],
                    "p90_norm": row_metrics["p90_norm"],
                    "checkpoint": str(checkpoints[run_id]),
                    "checkpoint_tag": checkpoint_tag,
                }
            )
        save_prediction_panel(out_dir / prediction_panel, predictions, truth)

        sample_rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "scenario": scenario.name,
                "resolution": f"{scenario.width}x{scenario.height}",
                "depth_range_um": scenario.depth_range_um,
                "focus_panel": str(focus_panel).replace("\\", "/"),
                "prior_panel": str(prior_panel).replace("\\", "/"),
                "prediction_panel": str(prediction_panel).replace("\\", "/"),
            }
        )

    write_csv(out_dir / "sample_visual_manifest.csv", sample_rows)
    write_csv(out_dir / "sample_prediction_metrics.csv", metric_rows)
    report = {
        "status": "pass",
        "export_tag": export_tag,
        "checkpoint_tag": checkpoint_tag,
        "device": device,
        "run_ids": run_ids,
        "sample_count": len(sample_rows),
        "output_dir": str(out_dir),
        "samples": sample_rows,
        "checkpoints": {run_id: str(path) for run_id, path in checkpoints.items()},
        "claim_eligible": False,
        "main_table_eligible": False,
    }
    (out_dir / "sample_visual_recovery_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_summary(out_dir / "sample_visual_recovery_summary.md", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=sorted(RUN_SPECS), help="Run id to export. May be repeated.")
    parser.add_argument("--tag", default=DEFAULT_EXPORT_TAG)
    parser.add_argument("--checkpoint-tag", default=DEFAULT_CHECKPOINT_TAG)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample limit. Use 0 for all fixed test samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export_visuals(args)
    print(json.dumps({"status": report["status"], "output_dir": report["output_dir"], "sample_count": report["sample_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
