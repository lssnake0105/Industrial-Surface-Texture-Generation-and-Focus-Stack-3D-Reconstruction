"""Per-sample audit for risk-gate failure in confidence-gated prior design."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
OUT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis" / "per_sample_risk_gate_failure_audit"


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


def sample_arrays(category: str, scenario: Any, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
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
    return {
        "category": category,
        "sample": scenario.name,
        "depth_range_um": scale,
        "prior_error": prior_error[idx],
        "dff_error": dff_error[idx],
        "gadff_error": gadff_error[idx],
        "focus_conf": focus_conf[idx],
        "risk": risk[idx],
    }


def audit_sample(item: dict[str, Any]) -> dict[str, Any]:
    focus_conf = item["focus_conf"]
    risk = item["risk"]
    prior_error = item["prior_error"]
    dff_error = item["dff_error"]
    gadff_error = item["gadff_error"]
    w_no_risk = np.clip(np.power(focus_conf, 1.5), 0.02, 1.0)
    w_risk = np.clip(np.power(focus_conf, 1.5) * (1.0 - 0.45 * risk), 0.02, 1.0)
    delta_weight = w_no_risk - w_risk
    n = prior_error.size
    k20 = max(1, int(round(n * 0.20)))
    order_delta = np.argsort(delta_weight, kind="mergesort")
    order_error = np.argsort(prior_error, kind="mergesort")
    high_delta = order_delta[-k20:]
    low_delta = order_delta[:k20]
    high_error = order_error[-k20:]
    overlap = len(set(map(int, high_delta)).intersection(set(map(int, high_error)))) / k20
    high_delta_error = float(np.mean(prior_error[high_delta]))
    low_delta_error = float(np.mean(prior_error[low_delta]))
    high_error_delta = float(np.mean(delta_weight[high_error]))
    all_delta = float(np.mean(delta_weight))
    return {
        "category": item["category"],
        "sample": item["sample"],
        "pixel_count": int(n),
        "depth_range_um": item["depth_range_um"],
        "prior_error_mean_um": float(np.mean(prior_error)),
        "dff_error_mean_um": float(np.mean(dff_error)),
        "gadff_error_mean_um": float(np.mean(gadff_error)),
        "focus_conf_mean": float(np.mean(focus_conf)),
        "risk_mean": float(np.mean(risk)),
        "risk_vs_prior_error_spearman": corr(risk, prior_error, spearman=True),
        "focus_conf_vs_prior_error_spearman": corr(focus_conf, prior_error, spearman=True),
        "delta_weight_vs_prior_error_spearman": corr(delta_weight, prior_error, spearman=True),
        "high_delta_prior_error_um": high_delta_error,
        "low_delta_prior_error_um": low_delta_error,
        "high_delta_to_low_delta_error_ratio": high_delta_error / max(low_delta_error, 1e-8),
        "high_delta_focus_conf": float(np.mean(focus_conf[high_delta])),
        "high_delta_risk": float(np.mean(risk[high_delta])),
        "high_delta_overlap_top_error20": float(overlap),
        "top_error_delta_weight": high_error_delta,
        "overall_delta_weight": all_delta,
        "top_error_delta_weight_ratio": high_error_delta / max(all_delta, 1e-8),
        "risk_gate_failure_flag": bool((corr(delta_weight, prior_error, spearman=True) < 0) and (overlap < 0.10) and (high_delta_error < np.mean(prior_error))),
    }


def plot_sample_summary(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [row["sample"][:22] for row in rows]
    x = np.arange(len(rows))
    spearman = [row["delta_weight_vs_prior_error_spearman"] for row in rows]
    overlap = [row["high_delta_overlap_top_error20"] for row in rows]
    ratio = [row["high_delta_to_low_delta_error_ratio"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, spearman, color="#4464ad")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Spearman")
    axes[0].set_title("Risk-induced downweight vs prior error by sample")
    axes[1].bar(x, overlap, color="#2a7f62")
    axes[1].axhline(0.20, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Top20 overlap")
    axes[2].bar(x, ratio, color="#aa6f39")
    axes[2].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("High-delta / low-delta error")
    axes[2].set_xticks(x, labels, rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], artifacts: dict[str, str]) -> None:
    fail_rows = [row for row in rows if row["risk_gate_failure_flag"]]
    sorted_rows = sorted(rows, key=lambda row: row["delta_weight_vs_prior_error_spearman"])
    lines: list[str] = []
    lines.append("# Per-Sample Risk-Gate Failure Audit")
    lines.append("")
    lines.append(f"- 日期：{DATE}")
    lines.append(f"- 样本：fixed synthetic test split，共 {summary['sample_count']} 个样本")
    lines.append(f"- 结论边界：claim-ineligible diagnostic only；real-height calibrated accuracy claim remains unsupported。")
    lines.append("- real-stack evidence remains diagnostic alignment only；audit should be rerun after manuscript-level merge。")
    lines.append("- 本审计只服务 confidence-gated prior consistency 的机制分析。")
    lines.append("")
    lines.append("## 1. 审计问题")
    lines.append("")
    lines.append("上一轮全局诊断显示 risk-induced downweight 与 prior error 呈负相关。本轮检查这个现象是否由少数样本主导，或是否在 test split 中普遍存在。")
    lines.append("")
    lines.append("## 2. 汇总结论")
    lines.append("")
    lines.append(f"- `risk_gate_failure_flag` 样本数：`{summary['failure_count']}/{summary['sample_count']}`。")
    lines.append(f"- 样本级 delta-weight vs prior-error Spearman 均值：`{summary['mean_delta_spearman']:.4f}`。")
    lines.append(f"- 样本级 top20 overlap 均值：`{summary['mean_overlap']:.4f}`。")
    lines.append(f"- 样本级 high-delta / low-delta prior-error ratio 均值：`{summary['mean_high_low_ratio']:.4f}`。")
    lines.append("")
    if summary["failure_count"] >= max(1, int(round(summary["sample_count"] * 0.70))):
        lines.append("结论：risk 项的误降权现象具有跨样本一致性，不能解释为单一样本异常。")
    else:
        lines.append("结论：risk 项的误降权现象存在样本差异，后续需要把反例样本单独拆解。")
    lines.append("")
    lines.append("## 3. 样本级结果")
    lines.append("")
    lines.append("| Sample | Delta Spearman | Top20 overlap | High-delta error | Low-delta error | Ratio | Flag |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in sorted_rows:
        lines.append(
            f"| {row['sample']} | {row['delta_weight_vs_prior_error_spearman']:.4f} | "
            f"{row['high_delta_overlap_top_error20']:.3f} | {row['high_delta_prior_error_um']:.2f} | "
            f"{row['low_delta_prior_error_um']:.2f} | {row['high_delta_to_low_delta_error_ratio']:.3f} | "
            f"{'yes' if row['risk_gate_failure_flag'] else 'no'} |"
        )
    lines.append("")
    lines.append("## 4. 最强 failure 样本")
    lines.append("")
    for row in sorted_rows[:3]:
        lines.append(
            f"- `{row['sample']}`：delta Spearman `{row['delta_weight_vs_prior_error_spearman']:.4f}`，"
            f"top20 overlap `{row['high_delta_overlap_top_error20']:.3f}`，"
            f"high-delta error `{row['high_delta_prior_error_um']:.2f} um`。"
        )
    lines.append("")
    non_fail = [row for row in sorted_rows if not row["risk_gate_failure_flag"]]
    if non_fail:
        lines.append("## 5. 弱 failure / 边界样本")
        lines.append("")
        lines.append("未触发 flag 的样本并没有形成 risk 项的正证据；它们仍表现为负 Spearman，只是 top20 overlap 或 high-delta error ratio 没有达到本审计的强 failure 阈值。")
        lines.append("")
        for row in non_fail:
            lines.append(
                f"- `{row['sample']}`：delta Spearman `{row['delta_weight_vs_prior_error_spearman']:.4f}`，"
                f"top20 overlap `{row['high_delta_overlap_top_error20']:.3f}`，"
                f"ratio `{row['high_delta_to_low_delta_error_ratio']:.3f}`。"
            )
        lines.append("")
    lines.append("## 6. 原理解释")
    lines.append("")
    lines.append("risk 项当前描述的是反光几何倾向，而 prior reliability 更直接地由焦向响应一致性决定。样本级结果如果普遍显示 high-delta 区域误差较低，说明 risk gate 会把高 risk 但高 confidence 的可用 prior 也压低，从而削弱 DFF/GADFF prior 的结构保留作用。")
    lines.append("")
    lines.append("可支持的主张：")
    lines.append("")
    lines.append("- low-confidence 是 prior gate 的主要证据。")
    lines.append("- risk map 更适合作为 failure analysis 和 real-stack diagnostic alignment 的分区变量。")
    lines.append("- risk 项若进入训练，应作为弱调制或条件项，而非与 $C_{\\mathrm{focus}}$ 同等强度相乘。")
    lines.append("")
    lines.append("暂不使用的主张：")
    lines.append("")
    lines.append("- 暂不否定 risk map 的诊断价值。")
    lines.append("- 不声明真实样本三维高度精度。")
    lines.append("- 不声明模型性能优势。")
    lines.append("")
    lines.append("## 7. 下一步")
    lines.append("")
    lines.append("1. 设计 conditional risk gate：只在 low-confidence 或 saturation persistence 较高时启用 risk 调制。")
    lines.append("2. 对 failure 样本画 ROI 级 focus curve，确认 high risk + high confidence 区域为何 prior error 较低。")
    lines.append("3. 在 real-stack diagnostic alignment 中检查 high risk + high confidence 区域是否对应稳定表面结构。")
    lines.append("")
    lines.append("## 8. 文件索引")
    lines.append("")
    for key, value in artifacts.items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()["test"]
    if args.max_samples:
        dataset = dataset[: args.max_samples]
    rows: list[dict[str, Any]] = []
    for category, scenario in dataset:
        item = sample_arrays(category, scenario, args, rng)
        rows.append(audit_sample(item))
    summary = {
        "status": "pass",
        "date": DATE,
        "sample_count": len(rows),
        "max_pixels_per_sample": args.max_pixels_per_sample,
        "failure_count": int(sum(bool(row["risk_gate_failure_flag"]) for row in rows)),
        "mean_delta_spearman": float(np.mean([row["delta_weight_vs_prior_error_spearman"] for row in rows])),
        "mean_overlap": float(np.mean([row["high_delta_overlap_top_error20"] for row in rows])),
        "mean_high_low_ratio": float(np.mean([row["high_delta_to_low_delta_error_ratio"] for row in rows])),
        "claim_eligible": False,
        "main_table_eligible": False,
        "claim_boundary": "Claim-ineligible per-sample risk-gate failure audit only. No model accuracy or calibrated real-height claim.",
    }
    csv_path = out / "per_sample_risk_gate_failure_audit.csv"
    json_path = out / "per_sample_risk_gate_failure_audit_summary.json"
    report_md = out / "per_sample_risk_gate_failure_audit_report.md"
    plot_path = out / "per_sample_risk_gate_failure_audit.png"
    artifacts = {
        "sample_csv": str(csv_path),
        "summary_json": str(json_path),
        "report_md": str(report_md),
        "plot": str(plot_path),
    }
    summary["artifacts"] = artifacts
    write_csv(csv_path, rows)
    write_json(json_path, {"summary": summary, "rows": rows})
    plot_sample_summary(rows, plot_path)
    write_report(report_md, rows, summary, artifacts)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-pixels-per-sample", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps({"status": summary["status"], "failure_count": summary["failure_count"], "sample_count": summary["sample_count"], "report": summary["artifacts"]["report_md"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
