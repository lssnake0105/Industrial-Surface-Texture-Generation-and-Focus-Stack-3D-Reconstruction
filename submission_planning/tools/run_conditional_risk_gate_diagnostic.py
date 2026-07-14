"""Diagnose conditional risk gates for confidence-gated prior consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402


DATE = date.today().isoformat()
OUT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis" / "conditional_risk_gate_diagnostic"


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    label: str
    exponent: float
    risk_coeff: float
    condition: str
    threshold: float
    min_weight: float = 0.02


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ordinal_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def corr(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float:
    xf = np.asarray(x, dtype=np.float64).ravel()
    yf = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(xf) & np.isfinite(yf)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    xf = xf[mask]
    yf = yf[mask]
    if spearman:
        xf = ordinal_rank(xf)
        yf = ordinal_rank(yf)
    x_std = float(np.std(xf))
    y_std = float(np.std(yf))
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.mean((xf - np.mean(xf)) * (yf - np.mean(yf))) / (x_std * y_std))


def focus_maps(features: np.ndarray) -> dict[str, np.ndarray]:
    offset = DEFAULT_STACK_LAYERS
    risk = np.clip(features[offset + 0], 0, 1).astype(np.float32)
    dff_conf = np.clip(features[offset + 2], 0, 1).astype(np.float32)
    gadff_conf = np.clip(features[offset + 4], 0, 1).astype(np.float32)
    focus_conf = np.clip(0.65 * dff_conf + 0.35 * gadff_conf, 0, 1).astype(np.float32)
    return {"risk": risk, "focus_conf": focus_conf}


def finite_sample_indices(size: int, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    if max_pixels <= 0 or size <= max_pixels:
        return np.arange(size, dtype=np.int64)
    return np.sort(rng.choice(size, size=max_pixels, replace=False).astype(np.int64))


def collect_pixels(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[str]]:
    rng = np.random.default_rng(args.seed)
    dataset = build_dataset()["test"]
    if args.max_samples:
        dataset = dataset[: args.max_samples]
    chunks: dict[str, list[np.ndarray]] = {
        "focus_conf": [],
        "risk": [],
        "prior_error": [],
        "dff_error": [],
        "gadff_error": [],
        "sample_index": [],
    }
    sample_names: list[str] = []
    for sample_idx, (category, scenario) in enumerate(dataset):
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        features = np.asarray(arrays["features"], dtype=np.float32)
        truth = np.asarray(arrays["truth"], dtype=np.float32)
        dff = np.asarray(arrays["dff"], dtype=np.float32)
        gadff = np.asarray(arrays["gadff"], dtype=np.float32)
        maps = focus_maps(features)
        prior_target = 0.45 * dff + 0.55 * gadff
        scale = float(scenario.depth_range_um)
        prior_error = np.abs(prior_target - truth).astype(np.float32).ravel() * scale
        dff_error = np.abs(dff - truth).astype(np.float32).ravel() * scale
        gadff_error = np.abs(gadff - truth).astype(np.float32).ravel() * scale
        focus_conf = maps["focus_conf"].ravel()
        risk = maps["risk"].ravel()
        finite = np.isfinite(prior_error) & np.isfinite(dff_error) & np.isfinite(gadff_error) & np.isfinite(focus_conf) & np.isfinite(risk)
        idx = np.flatnonzero(finite)
        idx = idx[finite_sample_indices(idx.size, args.max_pixels_per_sample, rng)]
        chunks["focus_conf"].append(focus_conf[idx])
        chunks["risk"].append(risk[idx])
        chunks["prior_error"].append(prior_error[idx])
        chunks["dff_error"].append(dff_error[idx])
        chunks["gadff_error"].append(gadff_error[idx])
        chunks["sample_index"].append(np.full(idx.size, sample_idx, dtype=np.int32))
        sample_names.append(f"{category}:{scenario.name}")
    return {key: np.concatenate(value) for key, value in chunks.items()}, sample_names


def candidate_specs() -> list[GateSpec]:
    specs = [
        GateSpec("no_risk_cfocus_p15", "No-risk gate: C_focus^1.5", 1.5, 0.0, "none", 0.0),
        GateSpec("current_risk045_all", "Current gate: risk0.45 applied everywhere", 1.5, 0.45, "all", 1.0),
    ]
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        specs.append(
            GateSpec(
                f"conditional_risk045_cbelow{threshold:g}",
                f"Risk0.45 only when C_focus <= {threshold:g}",
                1.5,
                0.45,
                "focus_below_abs",
                threshold,
            )
        )
    for quantile in [0.20, 0.25, 0.30, 0.35]:
        specs.append(
            GateSpec(
                f"conditional_risk045_cq{quantile:g}",
                f"Risk0.45 only below focus-confidence q={quantile:g}",
                1.5,
                0.45,
                "focus_below_quantile",
                quantile,
            )
        )
    return specs


def gate_weight(spec: GateSpec, focus_conf: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.power(focus_conf, spec.exponent)
    if spec.condition == "none":
        active = np.zeros_like(focus_conf, dtype=bool)
    elif spec.condition == "all":
        active = np.ones_like(focus_conf, dtype=bool)
    elif spec.condition == "focus_below_abs":
        active = focus_conf <= spec.threshold
    elif spec.condition == "focus_below_quantile":
        active = focus_conf <= float(np.quantile(focus_conf, spec.threshold))
    else:
        raise ValueError(f"Unknown condition: {spec.condition}")
    modifier = np.ones_like(base, dtype=np.float32)
    modifier[active] = 1.0 - spec.risk_coeff * risk[active]
    weight = np.clip(base * modifier, spec.min_weight, 1.0).astype(np.float32)
    return weight, active


def summarize_gate(
    spec: GateSpec,
    focus_conf: np.ndarray,
    risk: np.ndarray,
    prior_error: np.ndarray,
    sample_index: np.ndarray,
    sample_count: int,
) -> dict[str, Any]:
    weight, active = gate_weight(spec, focus_conf, risk)
    order = np.argsort(weight, kind="mergesort")
    n = weight.size
    k20 = max(1, int(round(n * 0.20)))
    low20 = order[:k20]
    high20 = order[-k20:]
    overall = float(np.mean(prior_error))
    low20_error = float(np.mean(prior_error[low20]))
    high20_error = float(np.mean(prior_error[high20]))
    per_sample_spearman = []
    for idx in range(sample_count):
        mask = sample_index == idx
        per_sample_spearman.append(corr(weight[mask], prior_error[mask], spearman=True))
    return {
        "gate_id": spec.gate_id,
        "label": spec.label,
        "exponent": spec.exponent,
        "risk_coeff": spec.risk_coeff,
        "condition": spec.condition,
        "threshold": spec.threshold,
        "active_fraction": float(np.mean(active)),
        "active_focus_conf_mean": float(np.mean(focus_conf[active])) if np.any(active) else 0.0,
        "active_risk_mean": float(np.mean(risk[active])) if np.any(active) else 0.0,
        "weight_mean": float(np.mean(weight)),
        "weight_std": float(np.std(weight)),
        "spearman_weight_vs_prior_error": corr(weight, prior_error, spearman=True),
        "pearson_weight_vs_prior_error": corr(weight, prior_error, spearman=False),
        "mean_sample_spearman": float(np.mean(per_sample_spearman)),
        "min_sample_spearman": float(np.min(per_sample_spearman)),
        "max_sample_spearman": float(np.max(per_sample_spearman)),
        "prior_error_mean_um": overall,
        "low20_prior_mae_um": low20_error,
        "high20_prior_mae_um": high20_error,
        "low20_to_high20_error_ratio": low20_error / max(high20_error, 1e-8),
        "low20_lift_vs_overall": low20_error / max(overall, 1e-8),
        "high20_ratio_vs_overall": high20_error / max(overall, 1e-8),
    }


def quantile_curve(
    spec: GateSpec,
    focus_conf: np.ndarray,
    risk: np.ndarray,
    prior_error: np.ndarray,
    bucket_count: int,
) -> list[dict[str, Any]]:
    weight, active = gate_weight(spec, focus_conf, risk)
    edges = np.quantile(weight, np.linspace(0, 1, bucket_count + 1))
    edges = np.maximum.accumulate(edges)
    rows: list[dict[str, Any]] = []
    for idx in range(bucket_count):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        mask = (weight >= lo) & (weight <= hi) if idx == bucket_count - 1 else (weight >= lo) & (weight < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "gate_id": spec.gate_id,
                "bucket": f"Q{idx + 1}",
                "pixel_count": int(np.count_nonzero(mask)),
                "mean_weight": float(np.mean(weight[mask])),
                "mean_focus_conf": float(np.mean(focus_conf[mask])),
                "mean_risk": float(np.mean(risk[mask])),
                "active_fraction": float(np.mean(active[mask])),
                "prior_target_mae_um": float(np.mean(prior_error[mask])),
            }
        )
    return rows


def plot_summary(rows: list[dict[str, Any]], out: Path) -> None:
    top = rows[:8]
    labels = [row["gate_id"] for row in top]
    x = np.arange(len(top))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x, [row["low20_to_high20_error_ratio"] for row in top], color="#4464ad")
    axes[0].set_ylabel("Low20 / high20 error")
    axes[1].bar(x, [row["spearman_weight_vs_prior_error"] for row in top], color="#2a7f62")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Spearman")
    axes[1].set_xticks(x, labels, rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    no_risk = next(row for row in rows if row["gate_id"] == "no_risk_cfocus_p15")
    current = next(row for row in rows if row["gate_id"] == "current_risk045_all")
    rank1 = rows[0]
    conditional_rows = [row for row in rows if row["condition"].startswith("focus_below")]
    conditional_rank1 = conditional_rows[0] if conditional_rows else None
    lines: list[str] = []
    lines.append("# Conditional Risk Gate Diagnostic")
    lines.append("")
    lines.append(f"- 日期：{DATE}")
    lines.append(f"- 样本：fixed synthetic test split，共 {summary['sample_count']} 个样本")
    lines.append(f"- 像素：{summary['pixel_count']}")
    lines.append("- 结论边界：claim-ineligible diagnostic only；real-height calibrated accuracy claim remains unsupported。")
    lines.append("- real-stack evidence remains diagnostic alignment only；audit should be rerun after manuscript-level merge。")
    lines.append("- 本诊断只服务 confidence-gated prior consistency 的机制分析。")
    lines.append("")
    lines.append("## 1. 研究问题")
    lines.append("")
    lines.append("前序审计显示 risk 项在全局相乘时会误降权高 risk + 高 confidence 的可用 prior。本诊断比较 no-risk gate、当前全局 risk gate，以及只在低 $C_{\\mathrm{focus}}$ 区域启用 risk 的 conditional gate。")
    lines.append("")
    lines.append("## 2. 排名前八的候选")
    lines.append("")
    lines.append("| Rank | Gate | Active frac | Spearman | Sample Spearman | Low20/High20 | Low20 lift | High20 ratio |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows[:8]:
        lines.append(
            f"| {row['rank']} | {row['gate_id']} | {row['active_fraction']:.3f} | "
            f"{row['spearman_weight_vs_prior_error']:.4f} | {row['mean_sample_spearman']:.4f} | "
            f"{row['low20_to_high20_error_ratio']:.2f} | {row['low20_lift_vs_overall']:.2f} | "
            f"{row['high20_ratio_vs_overall']:.2f} |"
        )
    lines.append("")
    lines.append("## 3. 关键对照")
    lines.append("")
    lines.append("| Gate | Rank | Active frac | Spearman | Low20/High20 | Low20 MAE um | High20 MAE um |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    key_rows = [no_risk, current]
    if conditional_rank1 and conditional_rank1["gate_id"] not in {row["gate_id"] for row in key_rows}:
        key_rows.append(conditional_rank1)
    if rank1["gate_id"] not in {row["gate_id"] for row in key_rows}:
        key_rows.insert(0, rank1)
    for row in key_rows:
        lines.append(
            f"| {row['gate_id']} | {row['rank']} | {row['active_fraction']:.3f} | "
            f"{row['spearman_weight_vs_prior_error']:.4f} | {row['low20_to_high20_error_ratio']:.2f} | "
            f"{row['low20_prior_mae_um']:.2f} | {row['high20_prior_mae_um']:.2f} |"
        )
    lines.append("")
    lines.append("## 4. 原理判断")
    lines.append("")
    if conditional_rank1 and conditional_rank1["rank"] < current["rank"] and conditional_rank1["rank"] <= no_risk["rank"] + 3:
        lines.append("conditional risk gate 明显优于全局 risk gate，并接近 no-risk gate；它更适合作为保留 risk 诊断价值的折中候选，但当前 prior-error ranking 仍由 no-risk gate 占优。")
    elif no_risk["rank"] <= (conditional_rank1["rank"] if conditional_rank1 else 999):
        lines.append("no-risk gate 仍然是最稳的 prior-error ranking 选择，conditional risk gate 没有提供额外排序收益。")
    else:
        lines.append("conditional risk gate 在当前诊断中优于 no-risk gate，值得进入 matched retraining smoke。")
    lines.append("")
    lines.append("可支持的主张：")
    lines.append("")
    lines.append("- risk 项不适合全局乘到 $C_{\\mathrm{focus}}^{1.5}$ 上。")
    lines.append("- 如果保留 risk 项，应优先考虑 low-confidence 条件触发或更弱调制。")
    lines.append("- 当前证据仍以 prior-error ranking 为主，需要训练层面复核。")
    lines.append("")
    lines.append("暂不使用的主张：")
    lines.append("")
    lines.append("- 暂不否定 risk map 的诊断价值。")
    lines.append("- 不声明真实样本三维高度精度。")
    lines.append("- 不声明模型性能优势。")
    lines.append("")
    lines.append("## 5. Q1-Q6 分桶曲线")
    lines.append("")
    for gate_id in [row["gate_id"] for row in key_rows]:
        lines.append(f"### {gate_id}")
        lines.append("")
        lines.append("| Bucket | Weight | Focus | Risk | Active frac | Prior MAE um |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in [r for r in bucket_rows if r["gate_id"] == gate_id]:
            lines.append(
                f"| {row['bucket']} | {row['mean_weight']:.4f} | {row['mean_focus_conf']:.4f} | "
                f"{row['mean_risk']:.4f} | {row['active_fraction']:.3f} | {row['prior_target_mae_um']:.2f} |"
            )
        lines.append("")
    lines.append("## 6. 下一步")
    lines.append("")
    lines.append("1. 若 conditional gate 接近 no-risk gate，可把它作为 full-budget matched repeat 的候选。")
    lines.append("2. 若 no-risk gate 仍最稳，论文方法中把 risk 移到 diagnostic/failure-analysis 分支。")
    lines.append("3. 在 real-stack diagnostic alignment 中检查 low-confidence 条件是否比 high-risk 条件更贴近 spike/saturation。")
    lines.append("")
    lines.append("## 7. 文件索引")
    lines.append("")
    for key, value in artifacts.items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    payload, sample_names = collect_pixels(args)
    focus_conf = payload["focus_conf"]
    risk = payload["risk"]
    prior_error = payload["prior_error"]
    sample_index = payload["sample_index"]
    specs = candidate_specs()
    rows = [
        summarize_gate(spec, focus_conf, risk, prior_error, sample_index, len(sample_names))
        for spec in specs
    ]
    rows.sort(
        key=lambda row: (
            -float(row["low20_to_high20_error_ratio"]),
            float(row["spearman_weight_vs_prior_error"]),
            float(row["high20_ratio_vs_overall"]),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    key_gate_ids = {row["gate_id"] for row in rows[:6]}
    key_gate_ids.update({"no_risk_cfocus_p15", "current_risk045_all"})
    bucket_rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.gate_id in key_gate_ids:
            bucket_rows.extend(quantile_curve(spec, focus_conf, risk, prior_error, args.quantile_buckets))
    rows_csv = out / "conditional_risk_gate_metrics.csv"
    bucket_csv = out / "conditional_risk_gate_buckets.csv"
    summary_json = out / "conditional_risk_gate_summary.json"
    report_md = out / "conditional_risk_gate_report.md"
    plot_path = out / "conditional_risk_gate_top_candidates.png"
    artifacts = {
        "metrics_csv": str(rows_csv),
        "bucket_csv": str(bucket_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "plot": str(plot_path),
    }
    summary = {
        "status": "pass",
        "date": DATE,
        "sample_count": len(sample_names),
        "pixel_count": int(focus_conf.size),
        "candidate_count": len(specs),
        "rank1_gate_id": rows[0]["gate_id"],
        "no_risk_rank": next(row["rank"] for row in rows if row["gate_id"] == "no_risk_cfocus_p15"),
        "current_gate_rank": next(row["rank"] for row in rows if row["gate_id"] == "current_risk045_all"),
        "artifacts": artifacts,
        "claim_eligible": False,
        "main_table_eligible": False,
        "claim_boundary": "Claim-ineligible conditional risk gate diagnostic only. No model accuracy or calibrated real-height claim.",
    }
    write_csv(rows_csv, rows)
    write_csv(bucket_csv, bucket_rows)
    write_json(summary_json, {"summary": summary, "rows": rows, "bucket_rows": bucket_rows})
    plot_summary(rows, plot_path)
    write_report(report_md, summary, rows, bucket_rows, artifacts)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-pixels-per-sample", type=int, default=160000)
    parser.add_argument("--quantile-buckets", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps({"status": summary["status"], "rank1_gate_id": summary["rank1_gate_id"], "no_risk_rank": summary["no_risk_rank"], "current_gate_rank": summary["current_gate_rank"], "report": summary["artifacts"]["report_md"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
