"""Diagnose gate-shape choices for confidence-gated prior reliability.

This script does not train a model. It tests whether candidate reliability
weights can rank synthetic DFF/GADFF prior errors on the fixed test split.
"""

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
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402


OUT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis" / "gate_shape_reliability_diagnostic"
CURRENT_EXPONENT = 1.5
CURRENT_RISK_COEFF = 0.45
CURRENT_MIN_WEIGHT = 0.02
DATE = date.today().isoformat()


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    family: str
    exponent: float
    risk_coeff: float
    min_weight: float
    description: str


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


def focus_maps(base_features: np.ndarray) -> dict[str, np.ndarray]:
    prior_offset = DEFAULT_STACK_LAYERS
    risk = np.clip(base_features[prior_offset + 0], 0, 1).astype(np.float32)
    conf = np.clip(base_features[prior_offset + 2], 0, 1).astype(np.float32)
    ga_conf = np.clip(base_features[prior_offset + 4], 0, 1).astype(np.float32)
    focus_conf = np.clip(0.65 * conf + 0.35 * ga_conf, 0, 1).astype(np.float32)
    return {"risk": risk, "dff_conf": conf, "gadff_conf": ga_conf, "focus_conf": focus_conf}


def gate_weight(spec: GateSpec, focus_conf: np.ndarray, risk: np.ndarray) -> np.ndarray:
    if spec.family == "current_family":
        raw = np.power(focus_conf, spec.exponent) * (1.0 - spec.risk_coeff * risk)
    elif spec.family == "baseline_family":
        raw = np.power(focus_conf, spec.exponent) * np.power(1.0 - risk, spec.risk_coeff)
    elif spec.family == "focus_only":
        raw = np.power(focus_conf, spec.exponent)
    elif spec.family == "risk_only":
        raw = 1.0 - spec.risk_coeff * risk
    else:
        raise ValueError(f"Unknown gate family: {spec.family}")
    return np.clip(raw, spec.min_weight, 1.0).astype(np.float32)


def candidate_specs() -> list[GateSpec]:
    specs: list[GateSpec] = []
    for exponent in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        for risk_coeff in [0.0, 0.25, 0.45, 0.65, 0.85]:
            gate_id = f"cfocus_p{exponent:g}_risk{risk_coeff:g}_min002"
            specs.append(
                GateSpec(
                    gate_id=gate_id,
                    family="current_family",
                    exponent=exponent,
                    risk_coeff=risk_coeff,
                    min_weight=0.02,
                    description="clip(C_focus^p * (1 - lambda R), 0.02, 1)",
                )
            )
    for exponent in [1.0, 1.5, 2.0]:
        specs.append(
            GateSpec(
                gate_id=f"focus_only_p{exponent:g}",
                family="focus_only",
                exponent=exponent,
                risk_coeff=0.0,
                min_weight=0.02,
                description="clip(C_focus^p, 0.02, 1)",
            )
        )
    for risk_power in [0.5, 1.0, 1.5, 2.0]:
        specs.append(
            GateSpec(
                gate_id=f"baseline_cfocus1_1minusrisk_p{risk_power:g}",
                family="baseline_family",
                exponent=1.0,
                risk_coeff=risk_power,
                min_weight=0.0,
                description="clip(C_focus * (1 - R)^q, 0, 1)",
            )
        )
    specs.append(
        GateSpec(
            gate_id="risk_only_coeff1",
            family="risk_only",
            exponent=0.0,
            risk_coeff=1.0,
            min_weight=0.0,
            description="clip(1 - R, 0, 1)",
        )
    )
    return specs


def finite_sample_indices(size: int, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    if max_pixels <= 0 or size <= max_pixels:
        return np.arange(size, dtype=np.int64)
    return np.sort(rng.choice(size, size=max_pixels, replace=False).astype(np.int64))


def quantile_bucket_rows(
    sample: str,
    category: str,
    spec: GateSpec,
    weights: np.ndarray,
    prior_error: np.ndarray,
    dff_error: np.ndarray,
    gadff_error: np.ndarray,
    focus_conf: np.ndarray,
    risk: np.ndarray,
    bucket_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.quantile(weights, np.linspace(0.0, 1.0, bucket_count + 1))
    edges = np.maximum.accumulate(edges)
    for idx in range(bucket_count):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == bucket_count - 1:
            mask = (weights >= lo) & (weights <= hi)
        else:
            mask = (weights >= lo) & (weights < hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        rows.append(
            {
                "sample": sample,
                "category": category,
                "gate_id": spec.gate_id,
                "bucket": f"Q{idx + 1}",
                "bucket_lo": lo,
                "bucket_hi": hi,
                "pixel_count": count,
                "mean_weight": float(np.mean(weights[mask])),
                "mean_focus_conf": float(np.mean(focus_conf[mask])),
                "mean_risk": float(np.mean(risk[mask])),
                "prior_target_mae_um": float(np.mean(prior_error[mask])),
                "dff_mae_um": float(np.mean(dff_error[mask])),
                "gadff_mae_um": float(np.mean(gadff_error[mask])),
            }
        )
    return rows


def summarize_candidate_sample(
    sample: str,
    category: str,
    spec: GateSpec,
    weights: np.ndarray,
    prior_error: np.ndarray,
    dff_error: np.ndarray,
    gadff_error: np.ndarray,
    focus_conf: np.ndarray,
    risk: np.ndarray,
    bucket_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    order = np.argsort(weights, kind="mergesort")
    n = weights.size
    k10 = max(1, int(round(n * 0.10)))
    k20 = max(1, int(round(n * 0.20)))
    low10 = order[:k10]
    high10 = order[-k10:]
    low20 = order[:k20]
    high20 = order[-k20:]
    overall = float(np.mean(prior_error))
    low20_error = float(np.mean(prior_error[low20]))
    high20_error = float(np.mean(prior_error[high20]))
    bucket_rows = quantile_bucket_rows(
        sample=sample,
        category=category,
        spec=spec,
        weights=weights,
        prior_error=prior_error,
        dff_error=dff_error,
        gadff_error=gadff_error,
        focus_conf=focus_conf,
        risk=risk,
        bucket_count=bucket_count,
    )
    bucket_spearman = corr(
        np.asarray([row["mean_weight"] for row in bucket_rows], dtype=np.float64),
        np.asarray([row["prior_target_mae_um"] for row in bucket_rows], dtype=np.float64),
        spearman=True,
    )
    row = {
        "sample": sample,
        "category": category,
        "gate_id": spec.gate_id,
        "family": spec.family,
        "exponent": spec.exponent,
        "risk_coeff": spec.risk_coeff,
        "min_weight": spec.min_weight,
        "description": spec.description,
        "pixel_count": int(n),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "pearson_weight_vs_prior_error": corr(weights, prior_error, spearman=False),
        "spearman_weight_vs_prior_error": corr(weights, prior_error, spearman=True),
        "bucket_spearman_weight_vs_prior_error": bucket_spearman,
        "prior_target_mae_um": overall,
        "dff_mae_um": float(np.mean(dff_error)),
        "gadff_mae_um": float(np.mean(gadff_error)),
        "low10_prior_mae_um": float(np.mean(prior_error[low10])),
        "high10_prior_mae_um": float(np.mean(prior_error[high10])),
        "low20_prior_mae_um": low20_error,
        "high20_prior_mae_um": high20_error,
        "low20_to_high20_error_ratio": low20_error / max(high20_error, 1e-8),
        "low20_error_lift_vs_overall": low20_error / max(overall, 1e-8),
        "high20_error_ratio_vs_overall": high20_error / max(overall, 1e-8),
    }
    return row, bucket_rows


def weighted_mean(rows: list[dict[str, Any]], key: str, weight_key: str = "pixel_count") -> float:
    weight_sum = sum(float(row[weight_key]) for row in rows)
    if weight_sum <= 0:
        return float("nan")
    return sum(float(row[key]) * float(row[weight_key]) for row in rows) / weight_sum


def aggregate_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["gate_id"]), []).append(row)
    out: list[dict[str, Any]] = []
    for gate_id, part in grouped.items():
        first = part[0]
        out.append(
            {
                "gate_id": gate_id,
                "family": first["family"],
                "exponent": first["exponent"],
                "risk_coeff": first["risk_coeff"],
                "min_weight": first["min_weight"],
                "description": first["description"],
                "sample_count": len(part),
                "pixel_count": int(sum(int(row["pixel_count"]) for row in part)),
                "weight_mean": weighted_mean(part, "weight_mean"),
                "weight_std": weighted_mean(part, "weight_std"),
                "pearson_weight_vs_prior_error": float(np.mean([row["pearson_weight_vs_prior_error"] for row in part])),
                "spearman_weight_vs_prior_error": float(np.mean([row["spearman_weight_vs_prior_error"] for row in part])),
                "bucket_spearman_weight_vs_prior_error": float(np.mean([row["bucket_spearman_weight_vs_prior_error"] for row in part])),
                "prior_target_mae_um": weighted_mean(part, "prior_target_mae_um"),
                "dff_mae_um": weighted_mean(part, "dff_mae_um"),
                "gadff_mae_um": weighted_mean(part, "gadff_mae_um"),
                "low10_prior_mae_um": weighted_mean(part, "low10_prior_mae_um"),
                "high10_prior_mae_um": weighted_mean(part, "high10_prior_mae_um"),
                "low20_prior_mae_um": weighted_mean(part, "low20_prior_mae_um"),
                "high20_prior_mae_um": weighted_mean(part, "high20_prior_mae_um"),
                "low20_to_high20_error_ratio": weighted_mean(part, "low20_to_high20_error_ratio"),
                "low20_error_lift_vs_overall": weighted_mean(part, "low20_error_lift_vs_overall"),
                "high20_error_ratio_vs_overall": weighted_mean(part, "high20_error_ratio_vs_overall"),
            }
        )
    out.sort(
        key=lambda row: (
            -float(row["low20_to_high20_error_ratio"]),
            float(row["spearman_weight_vs_prior_error"]),
            float(row["high20_error_ratio_vs_overall"]),
        )
    )
    for idx, row in enumerate(out, start=1):
        row["rank"] = idx
        row["is_current_gate"] = (
            row["family"] == "current_family"
            and math.isclose(float(row["exponent"]), CURRENT_EXPONENT)
            and math.isclose(float(row["risk_coeff"]), CURRENT_RISK_COEFF)
            and math.isclose(float(row["min_weight"]), CURRENT_MIN_WEIGHT)
        )
    return out


def aggregate_bucket_rows(rows: list[dict[str, Any]], top_gate_ids: set[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        gate_id = str(row["gate_id"])
        if gate_id not in top_gate_ids:
            continue
        grouped.setdefault((gate_id, str(row["bucket"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (gate_id, bucket), part in grouped.items():
        out.append(
            {
                "gate_id": gate_id,
                "bucket": bucket,
                "pixel_count": int(sum(int(row["pixel_count"]) for row in part)),
                "mean_weight": weighted_mean(part, "mean_weight"),
                "mean_focus_conf": weighted_mean(part, "mean_focus_conf"),
                "mean_risk": weighted_mean(part, "mean_risk"),
                "prior_target_mae_um": weighted_mean(part, "prior_target_mae_um"),
                "dff_mae_um": weighted_mean(part, "dff_mae_um"),
                "gadff_mae_um": weighted_mean(part, "gadff_mae_um"),
            }
        )
    out.sort(key=lambda row: (str(row["gate_id"]), int(str(row["bucket"])[1:])))
    return out


def plot_top_gates(rows: list[dict[str, Any]], bucket_rows: list[dict[str, Any]], out: Path) -> None:
    selected = rows[:5]
    current = next((row for row in rows if row["is_current_gate"]), None)
    if current and current["gate_id"] not in {row["gate_id"] for row in selected}:
        selected.append(current)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    names = [str(row["gate_id"]) for row in selected]
    ratios = [float(row["low20_to_high20_error_ratio"]) for row in selected]
    spearmans = [float(row["spearman_weight_vs_prior_error"]) for row in selected]
    axes[0].barh(range(len(names)), ratios, color="#4464ad")
    axes[0].set_yticks(range(len(names)), names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Low-20% / high-20% prior-error ratio")
    axes[0].set_title("High-error concentration in low-weight region")
    axes[1].barh(range(len(names)), spearmans, color="#2a7f62")
    axes[1].set_yticks(range(len(names)), names, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Spearman(weight, prior error)")
    axes[1].set_title("Weight-error monotonicity")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    for row in selected[:2]:
        gate_id = str(row["gate_id"])
        part = [b for b in bucket_rows if b["gate_id"] == gate_id]
        if not part:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [b["mean_weight"] for b in part]
        y = [b["prior_target_mae_um"] for b in part]
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Mean reliability weight")
        ax.set_ylabel("Prior-target MAE (um)")
        ax.set_title(gate_id)
        fig.tight_layout()
        fig.savefig(out.with_name(f"{gate_id}_bucket_curve.png"), dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    summary: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    current = next(row for row in aggregate_rows if row["is_current_gate"])
    rank1 = aggregate_rows[0]
    lines: list[str] = []
    lines.append("# Gate-Shape Reliability Diagnostic")
    lines.append("")
    lines.append(f"- 日期：{DATE}")
    lines.append(f"- 样本：fixed synthetic test split，共 {summary['sample_count']} 个样本")
    lines.append(f"- 像素采样：每个样本最多 {summary['max_pixels_per_sample']} 个像素；用于原理诊断和门控排序，不作为模型重训结果。")
    lines.append("- 结论边界：claim-ineligible for manuscript main-table accuracy；synthetic GT prior-reliability diagnostic only；real-height calibrated accuracy claim remains unsupported。")
    lines.append("- real-stack evidence remains diagnostic alignment only；audit should be rerun after any manuscript-level merge。")
    lines.append("- 本诊断只服务 confidence-gated prior consistency 的机制分析。")
    lines.append("")
    lines.append("## 1. 研究问题")
    lines.append("")
    lines.append("当前 ABL-07 使用的门控形式为：")
    lines.append("")
    lines.append("$$W=\\mathrm{clip}(C_{\\mathrm{focus}}^{1.5}(1-0.45R),0.02,1).$$")
    lines.append("")
    lines.append("本诊断检验不同 $C_{\\mathrm{focus}}^p(1-\\lambda R)$ 形状能否更好地排序 DFF/GADFF prior target 的误差。理想门控应让低权重区域富集高 prior error，并让权重与 prior error 呈负相关。")
    lines.append("")
    lines.append("## 2. 排名前十的门控形状")
    lines.append("")
    lines.append("| Rank | Gate | Spearman | Low20/High20 error ratio | Low20 lift | High20 ratio | Mean weight |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in aggregate_rows[:10]:
        lines.append(
            f"| {row['rank']} | {row['gate_id']} | "
            f"{row['spearman_weight_vs_prior_error']:.4f} | "
            f"{row['low20_to_high20_error_ratio']:.2f} | "
            f"{row['low20_error_lift_vs_overall']:.2f} | "
            f"{row['high20_error_ratio_vs_overall']:.2f} | "
            f"{row['weight_mean']:.3f} |"
        )
    lines.append("")
    lines.append("## 3. 当前门控与 Rank-1 诊断候选")
    lines.append("")
    lines.append("| Gate | Rank | Spearman | Low20/High20 error ratio | Low20 MAE um | High20 MAE um |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in [rank1, current]:
        lines.append(
            f"| {row['gate_id']} | {row['rank']} | "
            f"{row['spearman_weight_vs_prior_error']:.4f} | "
            f"{row['low20_to_high20_error_ratio']:.2f} | "
            f"{row['low20_prior_mae_um']:.2f} | "
            f"{row['high20_prior_mae_um']:.2f} |"
        )
    lines.append("")
    lines.append("解释：如果 rank-1 诊断候选只比当前门控小幅提升，则当前 $p=1.5,\\lambda=0.45$ 更适合作为保守默认值；如果差距较大，下一步应做 matched retraining ablation。")
    lines.append("")
    lines.append("## 4. Q1-Q6 分桶曲线")
    lines.append("")
    for gate_id in [str(rank1["gate_id"]), str(current["gate_id"])]:
        part = [row for row in bucket_rows if row["gate_id"] == gate_id]
        lines.append(f"### {gate_id}")
        lines.append("")
        lines.append("| Bucket | Mean weight | Focus conf | Risk | Prior target MAE um | DFF MAE um | GADFF MAE um |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in part:
            lines.append(
                f"| {row['bucket']} | {row['mean_weight']:.4f} | {row['mean_focus_conf']:.4f} | "
                f"{row['mean_risk']:.4f} | {row['prior_target_mae_um']:.2f} | "
                f"{row['dff_mae_um']:.2f} | {row['gadff_mae_um']:.2f} |"
            )
        lines.append("")
    lines.append("## 5. 原理结论")
    lines.append("")
    if float(rank1["exponent"]) >= 1.5:
        lines.append("诊断排名更偏向较大的 confidence exponent，说明可靠性门控需要压低中低置信区域的 prior consistency 权重，避免把 DFF/GADFF 的错误结构强行写入网络目标。")
    else:
        lines.append("诊断排名更偏向较小的 confidence exponent，说明当前门控可能过度压低中等置信区域，后续需要检查是否损失了可用的轴向结构信息。")
    if float(rank1["risk_coeff"]) <= 0.25:
        lines.append("rank-1 诊断候选中的 risk 系数偏低，提示 risk map 更适合作为辅助调制项，不能单独承担 prior reliability 判断。")
    else:
        lines.append("rank-1 诊断候选中仍保留 risk 项，提示局部反光风险对 prior reliability 排序具有补充价值。")
    lines.append("")
    lines.append("可支持的主张：")
    lines.append("")
    lines.append("- gate-shape 可以用 prior-error ranking 进行机制诊断。")
    lines.append("- low-confidence 区域应被赋予更低 prior consistency 权重。")
    lines.append("- 当前门控是否进入论文主方法，仍需 matched retraining ablation 支撑。")
    lines.append("")
    lines.append("暂不使用的主张：")
    lines.append("")
    lines.append("- 该诊断不声明模型精度提升。")
    lines.append("- 该诊断不声明真实样本三维高度精度。")
    lines.append("- 该诊断不声明外部基线优势。")
    lines.append("")
    lines.append("## 6. 下一步")
    lines.append("")
    lines.append("1. 对 rank-1 诊断候选、当前门控、focus-only 三个配置做 matched retraining smoke。")
    lines.append("2. 如果趋势稳定，再做 full split seed repeat。")
    lines.append("3. 对高置信负收益区域做 per-sample failure audit，检查是否来自形貌参数或 risk/confidence 特征失配。")
    lines.append("")
    lines.append("## 7. 文件索引")
    lines.append("")
    for key, value in summary.items():
        if key.endswith("_path") or key.endswith("_csv") or key.endswith("_json"):
            lines.append(f"- {key}: `{value}`")
    for plot in plot_paths:
        lines.append(f"- plot: `{plot}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()["test"]
    if args.max_samples:
        dataset = dataset[: args.max_samples]
    specs = candidate_specs()
    sample_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for category, scenario in dataset:
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        base = np.asarray(arrays["features"], dtype=np.float32)
        truth = np.asarray(arrays["truth"], dtype=np.float32)
        dff = np.asarray(arrays["dff"], dtype=np.float32)
        gadff = np.asarray(arrays["gadff"], dtype=np.float32)
        maps = focus_maps(base)
        depth_range_um = float(scenario.depth_range_um)
        prior_target = 0.45 * dff + 0.55 * gadff
        prior_error = np.abs(prior_target - truth).astype(np.float32).ravel() * depth_range_um
        dff_error = np.abs(dff - truth).astype(np.float32).ravel() * depth_range_um
        gadff_error = np.abs(gadff - truth).astype(np.float32).ravel() * depth_range_um
        focus_conf = maps["focus_conf"].ravel()
        risk = maps["risk"].ravel()
        finite = np.isfinite(prior_error) & np.isfinite(dff_error) & np.isfinite(gadff_error) & np.isfinite(focus_conf) & np.isfinite(risk)
        idx = np.flatnonzero(finite)
        idx = idx[finite_sample_indices(idx.size, args.max_pixels_per_sample, rng)]
        prior_error_s = prior_error[idx]
        dff_error_s = dff_error[idx]
        gadff_error_s = gadff_error[idx]
        focus_conf_s = focus_conf[idx]
        risk_s = risk[idx]
        for spec in specs:
            weights = gate_weight(spec, focus_conf_s, risk_s)
            row, bucket_rows = summarize_candidate_sample(
                sample=scenario.name,
                category=category,
                spec=spec,
                weights=weights,
                prior_error=prior_error_s,
                dff_error=dff_error_s,
                gadff_error=gadff_error_s,
                focus_conf=focus_conf_s,
                risk=risk_s,
                bucket_count=args.quantile_buckets,
            )
            sample_rows.append(row)
            all_bucket_rows.extend(bucket_rows)
    aggregate_rows = aggregate_sample_rows(sample_rows)
    current_rows = [row for row in aggregate_rows if row["is_current_gate"]]
    checks.append({"check": "sample count positive", "passed": len(dataset) > 0, "detail": str(len(dataset)), "severity": "error"})
    checks.append({"check": "candidate count positive", "passed": len(specs) > 0, "detail": str(len(specs)), "severity": "error"})
    checks.append({"check": "current gate found", "passed": len(current_rows) == 1, "detail": str(len(current_rows)), "severity": "error"})
    top_gate_ids = {str(row["gate_id"]) for row in aggregate_rows[:5]}
    if current_rows:
        top_gate_ids.add(str(current_rows[0]["gate_id"]))
    aggregate_bucket = aggregate_bucket_rows(all_bucket_rows, top_gate_ids)
    sample_csv = out_dir / "gate_shape_sample_metrics.csv"
    aggregate_csv = out_dir / "gate_shape_aggregate_metrics.csv"
    bucket_csv = out_dir / "gate_shape_bucket_metrics.csv"
    summary_json = out_dir / "gate_shape_reliability_summary.json"
    report_md = out_dir / "gate_shape_reliability_diagnostic_report.md"
    plot_path = out_dir / "gate_shape_top_candidates.png"
    write_csv(sample_csv, sample_rows)
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(bucket_csv, aggregate_bucket)
    plot_top_gates(aggregate_rows, aggregate_bucket, plot_path)
    plot_paths = [plot_path]
    for path in out_dir.glob("*_bucket_curve.png"):
        plot_paths.append(path)
    checks.extend(
        [
            {"check": "sample CSV written", "passed": sample_csv.exists(), "detail": str(sample_csv), "severity": "error"},
            {"check": "aggregate CSV written", "passed": aggregate_csv.exists(), "detail": str(aggregate_csv), "severity": "error"},
            {"check": "bucket CSV written", "passed": bucket_csv.exists(), "detail": str(bucket_csv), "severity": "error"},
            {"check": "plot written", "passed": plot_path.exists(), "detail": str(plot_path), "severity": "error"},
        ]
    )
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    summary = {
        "status": "pass" if not errors else "fail",
        "date": DATE,
        "sample_count": len(dataset),
        "candidate_count": len(specs),
        "max_pixels_per_sample": args.max_pixels_per_sample,
        "quantile_buckets": args.quantile_buckets,
        "rank1_gate_id": aggregate_rows[0]["gate_id"] if aggregate_rows else "",
        "current_gate_rank": current_rows[0]["rank"] if current_rows else None,
        "current_gate_id": current_rows[0]["gate_id"] if current_rows else "",
        "sample_metrics_csv": str(sample_csv),
        "aggregate_metrics_csv": str(aggregate_csv),
        "bucket_metrics_csv": str(bucket_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "plot_paths": [str(path) for path in plot_paths],
        "check_count": len(checks),
        "error_count": len(errors),
        "checks": checks,
        "claim_eligible": False,
        "main_table_eligible": False,
        "claim_boundary": "Synthetic GT prior-reliability diagnostic only. No model accuracy or calibrated real-height claim.",
    }
    write_json(summary_json, summary)
    write_report(report_md, summary, aggregate_rows, aggregate_bucket, plot_paths)
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
    print(json.dumps({"status": summary["status"], "rank1_gate_id": summary["rank1_gate_id"], "current_gate_rank": summary["current_gate_rank"], "report": summary["report_md"]}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
