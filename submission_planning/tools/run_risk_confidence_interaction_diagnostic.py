"""Diagnose how risk interacts with focus confidence in prior gating."""

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
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402


DATE = date.today().isoformat()
OUT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis" / "risk_confidence_interaction_diagnostic"


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
    return {"risk": risk, "dff_conf": dff_conf, "gadff_conf": gadff_conf, "focus_conf": focus_conf}


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges[1:-1], values, side="right")
    return np.clip(idx, 0, len(edges) - 2)


def finite_sample_indices(size: int, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    if max_pixels <= 0 or size <= max_pixels:
        return np.arange(size, dtype=np.int64)
    return np.sort(rng.choice(size, size=max_pixels, replace=False).astype(np.int64))


def collect_pixels(args: argparse.Namespace) -> dict[str, np.ndarray]:
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
    payload = {key: np.concatenate(value) for key, value in chunks.items()}
    payload["sample_count"] = np.asarray([len(dataset)], dtype=np.int32)
    payload["sample_names"] = np.asarray(sample_names, dtype=object)
    return payload


def summarize_bins(payload: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    focus_conf = payload["focus_conf"]
    risk = payload["risk"]
    prior_error = payload["prior_error"]
    dff_error = payload["dff_error"]
    gadff_error = payload["gadff_error"]
    w_no_risk = np.clip(np.power(focus_conf, 1.5), 0.02, 1.0)
    w_risk = np.clip(np.power(focus_conf, 1.5) * (1.0 - 0.45 * risk), 0.02, 1.0)
    delta_weight = w_no_risk - w_risk
    focus_edges = np.quantile(focus_conf, np.linspace(0, 1, args.focus_bins + 1))
    risk_edges = np.quantile(risk, np.linspace(0, 1, args.risk_bins + 1))
    focus_edges = np.maximum.accumulate(focus_edges)
    risk_edges = np.maximum.accumulate(risk_edges)
    fidx = bin_index(focus_conf, focus_edges)
    ridx = bin_index(risk, risk_edges)
    rows: list[dict[str, Any]] = []
    for fi in range(args.focus_bins):
        for ri in range(args.risk_bins):
            mask = (fidx == fi) & (ridx == ri)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            rows.append(
                {
                    "focus_bin": f"F{fi + 1}",
                    "risk_bin": f"R{ri + 1}",
                    "pixel_count": count,
                    "focus_lo": float(focus_edges[fi]),
                    "focus_hi": float(focus_edges[fi + 1]),
                    "risk_lo": float(risk_edges[ri]),
                    "risk_hi": float(risk_edges[ri + 1]),
                    "mean_focus_conf": float(np.mean(focus_conf[mask])),
                    "mean_risk": float(np.mean(risk[mask])),
                    "prior_target_mae_um": float(np.mean(prior_error[mask])),
                    "dff_mae_um": float(np.mean(dff_error[mask])),
                    "gadff_mae_um": float(np.mean(gadff_error[mask])),
                    "no_risk_weight": float(np.mean(w_no_risk[mask])),
                    "risk_weight": float(np.mean(w_risk[mask])),
                    "delta_weight": float(np.mean(delta_weight[mask])),
                    "risk_penalty_ratio": float(np.mean(delta_weight[mask]) / max(float(np.mean(w_no_risk[mask])), 1e-8)),
                    "error_per_delta_weight": float(np.mean(prior_error[mask]) / max(float(np.mean(delta_weight[mask])), 1e-8)),
                }
            )
    # Extra downweight test: among pixels that risk gate demotes most, is error higher?
    order_delta = np.argsort(delta_weight, kind="mergesort")
    order_error = np.argsort(prior_error, kind="mergesort")
    k20 = max(1, int(round(delta_weight.size * 0.20)))
    high_delta = order_delta[-k20:]
    low_delta = order_delta[:k20]
    high_error = order_error[-k20:]
    overlap = len(set(map(int, high_delta)).intersection(set(map(int, high_error)))) / k20
    summary = {
        "pixel_count": int(delta_weight.size),
        "sample_count": int(payload["sample_count"][0]),
        "focus_conf_mean": float(np.mean(focus_conf)),
        "risk_mean": float(np.mean(risk)),
        "prior_error_mean_um": float(np.mean(prior_error)),
        "risk_vs_prior_error_pearson": corr(risk, prior_error, spearman=False),
        "risk_vs_prior_error_spearman": corr(risk, prior_error, spearman=True),
        "focus_conf_vs_prior_error_pearson": corr(focus_conf, prior_error, spearman=False),
        "focus_conf_vs_prior_error_spearman": corr(focus_conf, prior_error, spearman=True),
        "delta_weight_vs_prior_error_pearson": corr(delta_weight, prior_error, spearman=False),
        "delta_weight_vs_prior_error_spearman": corr(delta_weight, prior_error, spearman=True),
        "high_delta_prior_error_um": float(np.mean(prior_error[high_delta])),
        "low_delta_prior_error_um": float(np.mean(prior_error[low_delta])),
        "high_delta_focus_conf": float(np.mean(focus_conf[high_delta])),
        "high_delta_risk": float(np.mean(risk[high_delta])),
        "high_delta_overlap_top_error20": float(overlap),
    }
    return rows, summary


def plot_heatmap(rows: list[dict[str, Any]], out: Path, key: str, title: str) -> None:
    focus_bins = sorted({row["focus_bin"] for row in rows}, key=lambda x: int(x[1:]))
    risk_bins = sorted({row["risk_bin"] for row in rows}, key=lambda x: int(x[1:]))
    data = np.full((len(risk_bins), len(focus_bins)), np.nan, dtype=np.float64)
    for row in rows:
        fi = focus_bins.index(row["focus_bin"])
        ri = risk_bins.index(row["risk_bin"])
        data[ri, fi] = float(row[key])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(focus_bins)), focus_bins)
    ax.set_yticks(range(len(risk_bins)), risk_bins)
    ax.set_xlabel("Focus-confidence quantile bin")
    ax.set_ylabel("Risk quantile bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], artifacts: dict[str, str]) -> None:
    top_delta = sorted(rows, key=lambda row: float(row["delta_weight"]), reverse=True)[:8]
    top_error = sorted(rows, key=lambda row: float(row["prior_target_mae_um"]), reverse=True)[:8]
    lines: list[str] = []
    lines.append("# Risk-Confidence Interaction Diagnostic")
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
    lines.append("上一轮 gate-shape 诊断显示 $W=\\mathrm{clip}(C_{\\mathrm{focus}}^{1.5},0.02,1)$ 的 prior-error 排序优于 $W=\\mathrm{clip}(C_{\\mathrm{focus}}^{1.5}(1-0.45R),0.02,1)$。本诊断进一步检查 risk 项是否真的指向高 prior error，或是否在某些 focus-confidence 区域产生误降权。")
    lines.append("")
    lines.append("## 2. 全局相关性")
    lines.append("")
    lines.append("| Pair | Pearson | Spearman |")
    lines.append("|---|---:|---:|")
    lines.append(f"| risk vs prior error | {summary['risk_vs_prior_error_pearson']:.4f} | {summary['risk_vs_prior_error_spearman']:.4f} |")
    lines.append(f"| focus confidence vs prior error | {summary['focus_conf_vs_prior_error_pearson']:.4f} | {summary['focus_conf_vs_prior_error_spearman']:.4f} |")
    lines.append(f"| risk-induced downweight vs prior error | {summary['delta_weight_vs_prior_error_pearson']:.4f} | {summary['delta_weight_vs_prior_error_spearman']:.4f} |")
    lines.append("")
    lines.append("## 3. Risk 项额外降权是否命中高误差")
    lines.append("")
    lines.append(f"- 被 risk 项额外降权最多的 top 20% 像素，prior error 均值为 `{summary['high_delta_prior_error_um']:.2f} um`。")
    lines.append(f"- 被 risk 项额外降权最少的 bottom 20% 像素，prior error 均值为 `{summary['low_delta_prior_error_um']:.2f} um`。")
    lines.append(f"- top 20% 额外降权像素与 top 20% prior error 像素的重叠率为 `{summary['high_delta_overlap_top_error20']:.3f}`。")
    lines.append(f"- top 20% 额外降权区域的平均 $C_{{\\mathrm{{focus}}}}$ 为 `{summary['high_delta_focus_conf']:.3f}`，平均 risk 为 `{summary['high_delta_risk']:.3f}`。")
    lines.append("")
    lines.append("## 4. 额外降权最大的二维 bin")
    lines.append("")
    lines.append("| Focus bin | Risk bin | Focus | Risk | Prior MAE um | No-risk W | Risk W | Delta W |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in top_delta:
        lines.append(
            f"| {row['focus_bin']} | {row['risk_bin']} | {row['mean_focus_conf']:.3f} | {row['mean_risk']:.3f} | "
            f"{row['prior_target_mae_um']:.2f} | {row['no_risk_weight']:.4f} | {row['risk_weight']:.4f} | {row['delta_weight']:.4f} |"
        )
    lines.append("")
    lines.append("## 5. prior error 最高的二维 bin")
    lines.append("")
    lines.append("| Focus bin | Risk bin | Focus | Risk | Prior MAE um | Delta W | Penalty ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in top_error:
        lines.append(
            f"| {row['focus_bin']} | {row['risk_bin']} | {row['mean_focus_conf']:.3f} | {row['mean_risk']:.3f} | "
            f"{row['prior_target_mae_um']:.2f} | {row['delta_weight']:.4f} | {row['risk_penalty_ratio']:.3f} |"
        )
    lines.append("")
    lines.append("## 6. 原理判断")
    lines.append("")
    if summary["delta_weight_vs_prior_error_spearman"] < 0:
        lines.append("risk-induced downweight 与 prior error 呈负相关，说明当前 risk 项额外压低的区域并不稳定对应更高 prior error；这会削弱门控对 DFF/GADFF 可靠性的排序能力。")
    else:
        lines.append("risk-induced downweight 与 prior error 呈正相关，说明 risk 项对高误差区域有一定命中能力；仍需检查它是否同时误伤高置信低误差区域。")
    lines.append("当前更稳妥的解释是：$C_{\\mathrm{focus}}$ 直接描述焦向响应一致性，离 prior reliability 更近；risk map 描述反光几何倾向，适合作为辅助诊断或分区分析，不宜直接作为同等强度的 prior gate 因子。")
    lines.append("")
    lines.append("可支持的主张：")
    lines.append("")
    lines.append("- low-confidence 是 prior consistency 门控的核心证据。")
    lines.append("- risk map 更适合进入 failure analysis、real-stack diagnostic alignment 或弱调制项。")
    lines.append("- 是否保留 risk 项需要 full-budget matched repeat 进一步验证。")
    lines.append("")
    lines.append("暂不使用的主张：")
    lines.append("")
    lines.append("- 暂不否定 risk map 的诊断价值。")
    lines.append("- 不声明真实样本三维高度精度。")
    lines.append("- 不声明模型性能优势。")
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
    payload = collect_pixels(args)
    rows, summary = summarize_bins(payload, args)
    bin_csv = out / "risk_confidence_interaction_bins.csv"
    summary_json = out / "risk_confidence_interaction_summary.json"
    report_md = out / "risk_confidence_interaction_report.md"
    heat_error = out / "risk_confidence_prior_error_heatmap.png"
    heat_delta = out / "risk_confidence_delta_weight_heatmap.png"
    write_csv(bin_csv, rows)
    artifacts = {
        "bin_csv": str(bin_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "prior_error_heatmap": str(heat_error),
        "delta_weight_heatmap": str(heat_delta),
    }
    report = {
        "status": "pass",
        "date": DATE,
        "focus_bins": args.focus_bins,
        "risk_bins": args.risk_bins,
        "max_pixels_per_sample": args.max_pixels_per_sample,
        **summary,
        "artifacts": artifacts,
        "claim_eligible": False,
        "main_table_eligible": False,
        "claim_boundary": "Claim-ineligible risk-confidence diagnostic only. No model accuracy or calibrated real-height claim.",
    }
    write_json(summary_json, report)
    plot_heatmap(rows, heat_error, "prior_target_mae_um", "Prior target MAE by focus-risk bin")
    plot_heatmap(rows, heat_delta, "delta_weight", "Risk-induced downweight by focus-risk bin")
    write_report(report_md, report, rows, artifacts)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--focus-bins", type=int, default=5)
    parser.add_argument("--risk-bins", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-pixels-per-sample", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "report": report["artifacts"]["report_md"], "delta_spearman": report["delta_weight_vs_prior_error_spearman"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
