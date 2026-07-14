"""Calibrate focus-confidence reliability for ABL-07 / CGP-FocusNet."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from run_confidence_weighted_loss_training import ABL_ROOT, DATE, RUN_ID, safe_tag, write_json  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402
from train_focus_resunet_loss_experiment import FocusResUNet, augment_features, predict_tiled_upgraded  # noqa: E402


DEFAULT_OUT = ROOT / "submission_planning" / "optical_mechanism_analysis" / "focus_confidence_reliability_calibration"
DEFAULT_CHECKPOINTS = [
    "2026-06-22_confidence_gated_prior_full_candidate",
    "2026-06-22_confidence_gated_prior_seed_repeat",
]
FIXED_EDGES = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.000001]
VARIANT = "CGP-FocusNet / ABL-07 confidence-gated DFF/GADFF prior"


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
    return ABL_ROOT / RUN_ID / "checkpoints" / f"{checkpoint_tag}.pt"


def load_model(checkpoint: Path, device: str) -> FocusResUNet:
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def focus_maps(base_features: np.ndarray) -> dict[str, np.ndarray]:
    prior_offset = DEFAULT_STACK_LAYERS
    risk = np.clip(base_features[prior_offset + 0], 0, 1).astype(np.float32)
    conf = np.clip(base_features[prior_offset + 2], 0, 1).astype(np.float32)
    ga_conf = np.clip(base_features[prior_offset + 4], 0, 1).astype(np.float32)
    focus_conf = np.clip(0.65 * conf + 0.35 * ga_conf, 0, 1).astype(np.float32)
    prior_weight = np.clip(np.power(focus_conf, 1.5) * (1.0 - 0.45 * risk), 0.02, 1.0).astype(np.float32)
    return {"risk": risk, "dff_conf": conf, "gadff_conf": ga_conf, "focus_conf": focus_conf, "prior_weight": prior_weight}


def ordinal_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def corr(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float:
    xf = np.asarray(x, dtype=np.float64).ravel()
    yf = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(xf) & np.isfinite(yf)
    xf = xf[mask]
    yf = yf[mask]
    if xf.size < 3:
        return float("nan")
    if spearman:
        xf = ordinal_rank(xf)
        yf = ordinal_rank(yf)
    xs = xf - float(np.mean(xf))
    ys = yf - float(np.mean(yf))
    denom = float(np.sqrt(np.sum(xs * xs) * np.sum(ys * ys)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(xs * ys) / denom)


def fixed_bucket_masks(focus_conf: np.ndarray) -> list[tuple[str, float, float, np.ndarray]]:
    masks = []
    for idx, (lo, hi) in enumerate(zip(FIXED_EDGES[:-1], FIXED_EDGES[1:]), start=1):
        if idx == len(FIXED_EDGES) - 1:
            mask = (focus_conf >= lo) & (focus_conf <= hi)
        else:
            mask = (focus_conf >= lo) & (focus_conf < hi)
        masks.append((f"F{idx}_{lo:.2f}_{min(hi, 1.0):.2f}", lo, min(hi, 1.0), mask))
    return masks


def quantile_bucket_masks(focus_conf: np.ndarray, bucket_count: int) -> list[tuple[str, float, float, np.ndarray]]:
    quantiles = np.quantile(focus_conf.ravel(), np.linspace(0.0, 1.0, bucket_count + 1))
    masks = []
    for idx in range(bucket_count):
        lo = float(quantiles[idx])
        hi = float(quantiles[idx + 1])
        if idx == bucket_count - 1:
            mask = (focus_conf >= lo) & (focus_conf <= hi)
        else:
            mask = (focus_conf >= lo) & (focus_conf < hi)
        masks.append((f"Q{idx + 1}_{lo:.4f}_{hi:.4f}", lo, hi, mask))
    return masks


def bucket_stats(
    *,
    checkpoint_tag: str,
    strategy: str,
    bucket_id: str,
    bucket_lo: float,
    bucket_hi: float,
    sample: str,
    category: str,
    depth_range_um: float,
    mask: np.ndarray,
    maps: dict[str, np.ndarray],
    truth: np.ndarray,
    dff: np.ndarray,
    gadff: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    pixel_count = int(np.sum(mask))
    if pixel_count == 0:
        return {
            "checkpoint_tag": checkpoint_tag,
            "bucket_strategy": strategy,
            "bucket_id": bucket_id,
            "bucket_lo": bucket_lo,
            "bucket_hi": bucket_hi,
            "category": category,
            "sample": sample,
            "depth_range_um": depth_range_um,
            "pixel_count": 0,
            "mean_focus_conf": float("nan"),
            "mean_dff_conf": float("nan"),
            "mean_gadff_conf": float("nan"),
            "mean_prior_weight": float("nan"),
            "mean_risk": float("nan"),
            "dff_mae_um": float("nan"),
            "gadff_mae_um": float("nan"),
            "model_mae_um": float("nan"),
            "model_vs_dff_gain_percent": float("nan"),
            "model_vs_gadff_gain_percent": float("nan"),
        }
    dff_err = np.abs(dff[mask] - truth[mask]) * depth_range_um
    gadff_err = np.abs(gadff[mask] - truth[mask]) * depth_range_um
    model_err = np.abs(pred[mask] - truth[mask]) * depth_range_um
    dff_mae = float(np.mean(dff_err))
    gadff_mae = float(np.mean(gadff_err))
    model_mae = float(np.mean(model_err))
    return {
        "checkpoint_tag": checkpoint_tag,
        "bucket_strategy": strategy,
        "bucket_id": bucket_id,
        "bucket_lo": bucket_lo,
        "bucket_hi": bucket_hi,
        "category": category,
        "sample": sample,
        "depth_range_um": depth_range_um,
        "pixel_count": pixel_count,
        "mean_focus_conf": float(np.mean(maps["focus_conf"][mask])),
        "mean_dff_conf": float(np.mean(maps["dff_conf"][mask])),
        "mean_gadff_conf": float(np.mean(maps["gadff_conf"][mask])),
        "mean_prior_weight": float(np.mean(maps["prior_weight"][mask])),
        "mean_risk": float(np.mean(maps["risk"][mask])),
        "dff_mae_um": dff_mae,
        "gadff_mae_um": gadff_mae,
        "model_mae_um": model_mae,
        "model_vs_dff_gain_percent": (dff_mae - model_mae) / max(dff_mae, 1e-8) * 100.0,
        "model_vs_gadff_gain_percent": (gadff_mae - model_mae) / max(gadff_mae, 1e-8) * 100.0,
    }


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    weights = np.asarray([float(row["pixel_count"]) for row in rows if np.isfinite(float(row.get(key, float("nan"))))], dtype=np.float64)
    vals = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row.get(key, float("nan"))))], dtype=np.float64)
    if vals.size == 0 or float(np.sum(weights)) <= 0:
        return float("nan")
    return float(np.sum(vals * weights) / np.sum(weights))


def aggregate_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["checkpoint_tag"], row["bucket_strategy"], row["bucket_id"]) for row in rows})
    for checkpoint_tag, strategy, bucket_id in keys:
        part = [row for row in rows if row["checkpoint_tag"] == checkpoint_tag and row["bucket_strategy"] == strategy and row["bucket_id"] == bucket_id and row["pixel_count"] > 0]
        if not part:
            continue
        dff = weighted_mean(part, "dff_mae_um")
        gadff = weighted_mean(part, "gadff_mae_um")
        model = weighted_mean(part, "model_mae_um")
        out.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "bucket_strategy": strategy,
                "bucket_id": bucket_id,
                "bucket_lo": float(part[0]["bucket_lo"]),
                "bucket_hi": float(part[0]["bucket_hi"]),
                "sample_count": len({row["sample"] for row in part}),
                "pixel_count": int(sum(int(row["pixel_count"]) for row in part)),
                "mean_focus_conf": weighted_mean(part, "mean_focus_conf"),
                "mean_dff_conf": weighted_mean(part, "mean_dff_conf"),
                "mean_gadff_conf": weighted_mean(part, "mean_gadff_conf"),
                "mean_prior_weight": weighted_mean(part, "mean_prior_weight"),
                "mean_risk": weighted_mean(part, "mean_risk"),
                "dff_mae_um": dff,
                "gadff_mae_um": gadff,
                "model_mae_um": model,
                "model_vs_dff_gain_percent": (dff - model) / max(dff, 1e-8) * 100.0,
                "model_vs_gadff_gain_percent": (gadff - model) / max(gadff, 1e-8) * 100.0,
            }
        )
    return sorted(out, key=lambda row: (row["checkpoint_tag"], row["bucket_strategy"], row["bucket_lo"]))


def plot_strategy(aggregate_rows: list[dict[str, Any]], checkpoint_tag: str, strategy: str, out: Path) -> None:
    rows = [row for row in aggregate_rows if row["checkpoint_tag"] == checkpoint_tag and row["bucket_strategy"] == strategy]
    if not rows:
        return
    x = np.asarray([row["mean_focus_conf"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    axes[0].plot(x, [row["dff_mae_um"] for row in rows], marker="o", label="DFF")
    axes[0].plot(x, [row["gadff_mae_um"] for row in rows], marker="o", label="GADFF")
    axes[0].plot(x, [row["model_mae_um"] for row in rows], marker="o", label="CGP-FocusNet")
    axes[0].set_xlabel("Mean focus confidence")
    axes[0].set_ylabel("MAE (um)")
    axes[0].set_title(f"{strategy}: confidence-error")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, [row["model_vs_dff_gain_percent"] for row in rows], marker="o", color="#2a7f62")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Mean focus confidence")
    axes[1].set_ylabel("Gain vs DFF (%)")
    axes[1].set_title(f"{strategy}: gain")
    axes[1].grid(alpha=0.25)
    fig.suptitle(checkpoint_tag)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def correlation_rows(pixel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in pixel_rows:
        focus = row["focus_conf"].ravel()
        for target_name, error_map in [
            ("dff_error", row["dff_error_um"]),
            ("gadff_error", row["gadff_error_um"]),
            ("model_error", row["model_error_um"]),
        ]:
            out.append(
                {
                    "checkpoint_tag": row["checkpoint_tag"],
                    "category": row["category"],
                    "sample": row["sample"],
                    "target": target_name,
                    "pearson_focus_conf_vs_abs_error": corr(focus, error_map.ravel(), spearman=False),
                    "spearman_focus_conf_vs_abs_error": corr(focus, error_map.ravel(), spearman=True),
                    "mean_focus_conf": float(np.mean(focus)),
                    "mean_abs_error_um": float(np.mean(error_map)),
                }
            )
    return out


def aggregate_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for checkpoint_tag in sorted({row["checkpoint_tag"] for row in rows}):
        for target in sorted({row["target"] for row in rows}):
            part = [row for row in rows if row["checkpoint_tag"] == checkpoint_tag and row["target"] == target]
            out.append(
                {
                    "checkpoint_tag": checkpoint_tag,
                    "target": target,
                    "sample_count": len(part),
                    "mean_pearson_focus_conf_vs_abs_error": float(np.mean([row["pearson_focus_conf_vs_abs_error"] for row in part])),
                    "mean_spearman_focus_conf_vs_abs_error": float(np.mean([row["spearman_focus_conf_vs_abs_error"] for row in part])),
                    "mean_abs_error_um": float(np.mean([row["mean_abs_error_um"] for row in part])),
                }
            )
    return out


def bucket_monotonic_score(rows: list[dict[str, Any]], key: str) -> float:
    if len(rows) < 3:
        return float("nan")
    x = np.asarray([row["mean_focus_conf"] for row in rows], dtype=np.float64)
    y = np.asarray([row[key] for row in rows], dtype=np.float64)
    return corr(x, y, spearman=True)


def write_report(
    path: Path,
    summary: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    corr_summary_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    lines = [
        "# Focus-Confidence Reliability Calibration Report",
        "",
        f"- 日期：{DATE}",
        f"- 方法：{VARIANT}",
        f"- 状态：{summary['status']}",
        f"- 样本：fixed synthetic test split，共 {summary['sample_count']} 个样本",
        "- 结论边界：synthetic GT reliability calibration only；real-height calibrated accuracy claim remains unsupported。",
        "",
        "## 1. 研究问题",
        "",
        "本实验检验 $C_{\\mathrm{focus}}$ 是否可以作为 DFF/GADFF prior 的可靠性信号。若 $C_{\\mathrm{focus}}$ 越低时 DFF 误差越高，且 CGP-FocusNet 的收益更集中在低置信桶中，则 confidence-gated prior consistency 具有更强的统计解释。",
        "",
        "当前 gate 定义为：",
        "",
        "$$C_{\\mathrm{focus}}=\\mathrm{clip}(0.65C_{\\mathrm{DFF}}+0.35C_{\\mathrm{GADFF}},0,1),$$",
        "",
        "$$W_{\\mathrm{prior}}=\\mathrm{clip}(C_{\\mathrm{focus}}^{1.5}(1-0.45R),0.02,1.0).$$",
        "",
        "## 2. Correlation Summary",
        "",
        "| Checkpoint | Target error | Samples | Pearson | Spearman | Mean error um |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in corr_summary_rows:
        lines.append(
            f"| {row['checkpoint_tag']} | {row['target']} | {row['sample_count']} | "
            f"{row['mean_pearson_focus_conf_vs_abs_error']:.4f} | {row['mean_spearman_focus_conf_vs_abs_error']:.4f} | "
            f"{row['mean_abs_error_um']:.2f} |"
        )
    lines.extend(["", "## 3. Bucket Summary: Quantile Buckets", ""])
    for checkpoint_tag in summary["checkpoint_tags"]:
        rows = [row for row in aggregate_rows if row["checkpoint_tag"] == checkpoint_tag and row["bucket_strategy"] == "quantile"]
        lines.extend(
            [
                f"### {checkpoint_tag}",
                "",
                "| Bucket | Focus conf | Prior weight | Risk | DFF MAE | GADFF MAE | Model MAE | Gain vs DFF |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['bucket_id']} | {row['mean_focus_conf']:.4f} | {row['mean_prior_weight']:.4f} | "
                f"{row['mean_risk']:.4f} | {row['dff_mae_um']:.2f} | {row['gadff_mae_um']:.2f} | "
                f"{row['model_mae_um']:.2f} | {row['model_vs_dff_gain_percent']:.2f}% |"
            )
        score = bucket_monotonic_score(rows, "dff_mae_um")
        gain_score = bucket_monotonic_score(rows, "model_vs_dff_gain_percent")
        lines.extend(
            [
                "",
                f"- $C_{{\\mathrm{{focus}}}}$ 与 DFF bucket MAE 的 Spearman 趋势：`{score:.4f}`。",
                f"- $C_{{\\mathrm{{focus}}}}$ 与 CGP gain bucket 的 Spearman 趋势：`{gain_score:.4f}`。",
                "",
            ]
        )
    lines.extend(["", "## 4. 原理解释", ""])
    lines.append(
        "如果 DFF 误差随 $C_{\\mathrm{focus}}$ 降低而升高，则说明 focus confidence 具有 prior reliability 解释价值。CGP-FocusNet 的训练目标把这个观测量放进 $W_{\\mathrm{prior}}$，使网络在低可靠 prior 区域减弱一致性约束，在可靠区域保留 DFF/GADFF 的轴向结构信息。"
    )
    lines.extend(["", "## 5. Claim Boundary", ""])
    lines.extend(
        [
            "- claim-ineligible for calibrated real-height accuracy.",
            "- real-stack evidence remains diagnostic alignment only.",
            "- audit should be rerun after any manuscript-level merge.",
            "- external baseline superiority remains unsupported until compatible baseline runs are completed.",
            "",
            "## 6. Artifacts",
            "",
            "| Artifact | Path |",
            "|---|---|",
            f"| bucket metrics CSV | `{summary['bucket_metrics_csv']}` |",
            f"| aggregate metrics CSV | `{summary['aggregate_metrics_csv']}` |",
            f"| correlation metrics CSV | `{summary['correlation_metrics_csv']}` |",
            f"| correlation summary CSV | `{summary['correlation_summary_csv']}` |",
            f"| summary JSON | `{summary['summary_json']}` |",
        ]
    )
    for plot_path in plot_paths:
        lines.append(f"| plot | `{plot_path}` |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tags = args.checkpoint_tag or DEFAULT_CHECKPOINTS
    checks: list[dict[str, Any]] = []
    dataset = build_dataset()["test"]
    if args.max_samples:
        dataset = dataset[: args.max_samples]
    sample_count = len(dataset)
    bucket_rows: list[dict[str, Any]] = []
    pixel_corr_payloads: list[dict[str, Any]] = []

    for checkpoint_tag in checkpoint_tags:
        tag = safe_tag(checkpoint_tag)
        checkpoint = checkpoint_for_tag(tag)
        checks.append(check(f"{tag} checkpoint exists", checkpoint.exists(), str(checkpoint)))
        if not checkpoint.exists():
            continue
        model = load_model(checkpoint, device)
        for category, scenario in dataset:
            arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
            base = np.asarray(arrays["features"], dtype=np.float32)
            truth = np.asarray(arrays["truth"], dtype=np.float32)
            dff = np.asarray(arrays["dff"], dtype=np.float32)
            gadff = np.asarray(arrays["gadff"], dtype=np.float32)
            maps = focus_maps(base)
            pred = predict_tiled_upgraded(model, augment_features(base), device, tile=args.tile, overlap=args.overlap)

            for strategy, masks in [
                ("fixed", fixed_bucket_masks(maps["focus_conf"])),
                ("quantile", quantile_bucket_masks(maps["focus_conf"], args.quantile_buckets)),
            ]:
                for bucket_id, lo, hi, mask in masks:
                    bucket_rows.append(
                        bucket_stats(
                            checkpoint_tag=tag,
                            strategy=strategy,
                            bucket_id=bucket_id,
                            bucket_lo=lo,
                            bucket_hi=hi,
                            sample=scenario.name,
                            category=category,
                            depth_range_um=float(scenario.depth_range_um),
                            mask=mask,
                            maps=maps,
                            truth=truth,
                            dff=dff,
                            gadff=gadff,
                            pred=pred,
                        )
                    )

            depth_range_um = float(scenario.depth_range_um)
            pixel_corr_payloads.append(
                {
                    "checkpoint_tag": tag,
                    "category": category,
                    "sample": scenario.name,
                    "focus_conf": maps["focus_conf"],
                    "dff_error_um": np.abs(dff - truth) * depth_range_um,
                    "gadff_error_um": np.abs(gadff - truth) * depth_range_um,
                    "model_error_um": np.abs(pred - truth) * depth_range_um,
                }
            )

    aggregate_rows = aggregate_bucket_rows(bucket_rows)
    corr_rows = correlation_rows(pixel_corr_payloads)
    corr_summary = aggregate_correlations(corr_rows)
    bucket_csv = out_dir / "focus_confidence_reliability_bucket_metrics.csv"
    aggregate_csv = out_dir / "focus_confidence_reliability_aggregate_metrics.csv"
    corr_csv = out_dir / "focus_confidence_reliability_correlation_metrics.csv"
    corr_summary_csv = out_dir / "focus_confidence_reliability_correlation_summary.csv"
    summary_json = out_dir / "focus_confidence_reliability_summary.json"
    report_md = out_dir / "focus_confidence_reliability_calibration_report.md"
    write_csv(bucket_csv, bucket_rows)
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(corr_csv, corr_rows)
    write_csv(corr_summary_csv, corr_summary)
    plot_paths: list[Path] = []
    for checkpoint_tag in checkpoint_tags:
        tag = safe_tag(checkpoint_tag)
        for strategy in ("quantile", "fixed"):
            plot_path = out_dir / f"{tag}_{strategy}_confidence_reliability_curve.png"
            plot_strategy(aggregate_rows, tag, strategy, plot_path)
            if plot_path.exists():
                plot_paths.append(plot_path)
    checks.extend(
        [
            check("sample count positive", sample_count > 0, str(sample_count)),
            check("bucket rows written", bucket_csv.exists() and len(bucket_rows) > 0, str(bucket_csv)),
            check("aggregate rows written", aggregate_csv.exists() and len(aggregate_rows) > 0, str(aggregate_csv)),
            check("correlation rows written", corr_csv.exists() and len(corr_rows) > 0, str(corr_csv)),
            check("plots written", len(plot_paths) == len(checkpoint_tags) * 2, f"{len(plot_paths)} plots"),
        ]
    )
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    summary = {
        "status": "pass" if not errors else "fail",
        "date": DATE,
        "run_id": RUN_ID,
        "variant": VARIANT,
        "device": device,
        "sample_count": sample_count,
        "checkpoint_tags": checkpoint_tags,
        "bucket_metrics_csv": str(bucket_csv),
        "aggregate_metrics_csv": str(aggregate_csv),
        "correlation_metrics_csv": str(corr_csv),
        "correlation_summary_csv": str(corr_summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "plot_paths": [str(path) for path in plot_paths],
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len([row for row in checks if not row["passed"] and row["severity"] == "warning"]),
        "checks": checks,
        "claim_eligible": False,
        "main_table_eligible": False,
        "claim_boundary": "Synthetic GT reliability calibration only. No calibrated real-height accuracy claim.",
    }
    write_json(summary_json, summary)
    write_report(report_md, summary, aggregate_rows, corr_summary, plot_paths)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-tag", action="append", help="ABL-07 checkpoint tag. May be repeated.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--quantile-buckets", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "sample_count": summary["sample_count"], "report": summary["report_md"]}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
