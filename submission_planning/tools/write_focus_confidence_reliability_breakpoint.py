from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis" / "focus_confidence_reliability_calibration"
AGG_CSV = OUT_DIR / "focus_confidence_reliability_aggregate_metrics.csv"
CORR_CSV = OUT_DIR / "focus_confidence_reliability_correlation_summary.csv"
SUMMARY_JSON = OUT_DIR / "focus_confidence_reliability_summary.json"
OUT_MD = OUT_DIR / "focus_confidence_reliability_breakpoint_2026-06-22.md"


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _q_index(bucket_id: str) -> str:
    m = re.match(r"^(Q\d+)_", bucket_id)
    if not m:
        raise ValueError(f"Unexpected bucket_id: {bucket_id}")
    return m.group(1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _load_corr() -> dict[tuple[str, str], dict[str, str]]:
    with CORR_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {(r["checkpoint_tag"], r["target"]): r for r in csv.DictReader(f)}


def _load_q_summary() -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with AGG_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["bucket_strategy"] != "quantile":
                continue
            tag = row["checkpoint_tag"]
            q = _q_index(row["bucket_id"])
            for key in [
                "mean_focus_conf",
                "mean_prior_weight",
                "mean_risk",
                "dff_mae_um",
                "gadff_mae_um",
                "model_mae_um",
                "model_vs_dff_gain_percent",
                "model_vs_gadff_gain_percent",
            ]:
                grouped[(tag, q)][key].append(_float(row, key))
            grouped[(tag, q)]["bucket_count"].append(1.0)

    by_tag: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for (tag, q), values in grouped.items():
        item: dict[str, float | str] = {"bucket": q}
        for key, vals in values.items():
            item[key] = _mean(vals)
        by_tag[tag].append(item)

    for tag in by_tag:
        by_tag[tag].sort(key=lambda x: int(str(x["bucket"])[1:]))
    return by_tag


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def _table(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "| Bucket | Focus conf | Prior weight | Risk | DFF MAE um | GADFF MAE um | Model MAE um | Gain vs DFF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {bucket} | {focus} | {prior} | {risk} | {dff} | {gadff} | {model} | {gain} |".format(
                bucket=r["bucket"],
                focus=_fmt(float(r["mean_focus_conf"]), 3),
                prior=_fmt(float(r["mean_prior_weight"]), 3),
                risk=_fmt(float(r["mean_risk"]), 3),
                dff=_fmt(float(r["dff_mae_um"])),
                gadff=_fmt(float(r["gadff_mae_um"])),
                model=_fmt(float(r["model_mae_um"])),
                gain=_fmt_pct(float(r["model_vs_dff_gain_percent"])),
            )
        )
    return "\n".join(lines)


def main() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    corr = _load_corr()
    q_summary = _load_q_summary()

    lines: list[str] = []
    lines.append("# Focus-Confidence Reliability Calibration 研究断点")
    lines.append("")
    lines.append(f"- 日期：{date.today().isoformat()}")
    lines.append("- 主题：验证 focus confidence 是否能解释 DFF/GADFF prior 的可靠性")
    lines.append(f"- 样本：fixed synthetic test split，共 {summary['sample_count']} 个样本")
    lines.append("- 结论边界：synthetic GT reliability calibration only；real-height calibrated accuracy claim remains unsupported。")
    lines.append("")
    lines.append("## 1. 当前结论")
    lines.append("")
    lines.append(
        "Focus confidence 可以作为 DFF/GADFF prior reliability 的统计信号。"
        "在两个 ABL-07 checkpoint 中，$C_{\\mathrm{focus}}$ 与 DFF/GADFF 绝对误差呈稳定负相关，"
        "说明低置信区域确实更容易对应传统 focus prior 的失效。"
    )
    lines.append("")
    lines.append("更关键的是，CGP-FocusNet 的收益主要集中在低置信桶；当 $C_{\\mathrm{focus}}$ 升高后，DFF/GADFF 本身已经较可靠，网络继续改写 prior 的收益下降，部分高置信桶还会出现负收益。这支持当前训练策略中“低可靠区域弱化 prior 一致性、高可靠区域保留轴向结构信息”的门控设计。")
    lines.append("")
    lines.append("## 2. 相关性证据")
    lines.append("")
    lines.append("| Checkpoint | Target | Pearson | Spearman | Mean error um |")
    lines.append("|---|---|---:|---:|---:|")
    for tag in summary["checkpoint_tags"]:
        for target in ["dff_error", "gadff_error", "model_error"]:
            row = corr[(tag, target)]
            lines.append(
                f"| {tag} | {target} | "
                f"{float(row['mean_pearson_focus_conf_vs_abs_error']):.4f} | "
                f"{float(row['mean_spearman_focus_conf_vs_abs_error']):.4f} | "
                f"{float(row['mean_abs_error_um']):.2f} |"
            )
    lines.append("")
    lines.append("解释：DFF/GADFF 的 Spearman 约为 -0.52 / -0.50，强于模型误差自身的相关性。这表明 $C_{\\mathrm{focus}}$ 更适合作为 prior reliability indicator，直接预测模型误差的证据仍然不足。")
    lines.append("")
    lines.append("## 3. Q1-Q6 归并桶趋势")
    lines.append("")
    for tag, rows in q_summary.items():
        lines.append(f"### {tag}")
        lines.append("")
        lines.append(_table(rows))
        lines.append("")
        low = rows[0]
        high = rows[-1]
        lines.append(
            f"- 低置信 Q1：DFF MAE {_fmt(float(low['dff_mae_um']))} um，"
            f"Model MAE {_fmt(float(low['model_mae_um']))} um，"
            f"Gain vs DFF {_fmt_pct(float(low['model_vs_dff_gain_percent']))}。"
        )
        lines.append(
            f"- 高置信 Q6：DFF MAE {_fmt(float(high['dff_mae_um']))} um，"
            f"Model MAE {_fmt(float(high['model_mae_um']))} um，"
            f"Gain vs DFF {_fmt_pct(float(high['model_vs_dff_gain_percent']))}。"
        )
        lines.append("")
    lines.append("## 4. 对论文故事线的意义")
    lines.append("")
    lines.append("这组结果把 ABL-07 从单纯的结果提升，推进到更清晰的原理解释：反光表面焦栈中的 DFF/GADFF prior 具有区域性可靠性差异，可靠性可以由焦向响应的一致性和风险项共同估计。因此，模型贡献可以表述为 confidence-gated prior consistency，避免把它写成额外堆叠的黑箱网络模块。")
    lines.append("")
    lines.append("可支持的论文主张：")
    lines.append("")
    lines.append("- $C_{\\mathrm{focus}}$ 与传统 prior 误差之间存在稳定负相关，可作为 prior reliability 的统计代理。")
    lines.append("- CGP-FocusNet 的主要收益来自 low-confidence、高风险、DFF/GADFF 更容易失败的区域。")
    lines.append("- 高置信区域中，传统 focus prior 已包含较可靠轴向结构，模型应减少不必要改写。")
    lines.append("")
    lines.append("暂不支持的主张：")
    lines.append("")
    lines.append("- real-height calibrated accuracy claim remains unsupported。")
    lines.append("- 外部基线总体优势仍需完成兼容评估后再判断。")
    lines.append("- 不支持把 real-stack alignment 当作带真值的定量评估。")
    lines.append("")
    lines.append("## 5. 下一步最有价值问题")
    lines.append("")
    lines.append("1. 做 gate-shape ablation：比较 $C^1.0$、$C^1.5$、$C^2.0$ 和 risk 权重系数，确认当前门控形状是否只是经验设定。")
    lines.append("2. 做 per-sample failure audit：定位高置信负收益桶来自哪些形貌、风险分布或仿真参数。")
    lines.append("3. 把 real-stack diagnostic alignment 与 synthetic reliability calibration 对齐：检查低 $C_{\\mathrm{focus}}$ 区域是否也对应真实焦栈中的 spike、saturation 或局部不连续。")
    lines.append("4. 增加文稿级图表：把 Q1-Q6 归并趋势图做成一张简洁 figure，用于支撑 confidence-gated prior consistency。")
    lines.append("")
    lines.append("## 6. 文件索引")
    lines.append("")
    lines.append(f"- 完整报告：`{OUT_DIR / 'focus_confidence_reliability_calibration_report.md'}`")
    lines.append(f"- 归并数据来源：`{AGG_CSV}`")
    lines.append(f"- 相关性数据：`{CORR_CSV}`")
    lines.append(f"- 运行摘要：`{SUMMARY_JSON}`")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
