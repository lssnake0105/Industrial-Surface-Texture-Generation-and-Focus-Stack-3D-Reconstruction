# Conditional Risk Gate Diagnostic

- 日期：2026-06-22
- 样本：fixed synthetic test split，共 7 个样本
- 像素：1120000
- 结论边界：claim-ineligible diagnostic only；real-height calibrated accuracy claim remains unsupported。
- real-stack evidence remains diagnostic alignment only；audit should be rerun after manuscript-level merge。
- 本诊断只服务 confidence-gated prior consistency 的机制分析。

## 1. 研究问题

前序审计显示 risk 项在全局相乘时会误降权高 risk + 高 confidence 的可用 prior。本诊断比较 no-risk gate、当前全局 risk gate，以及只在低 $C_{\mathrm{focus}}$ 区域启用 risk 的 conditional gate。

## 2. 排名前八的候选

| Rank | Gate | Active frac | Spearman | Sample Spearman | Low20/High20 | Low20 lift | High20 ratio |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | no_risk_cfocus_p15 | 0.000 | -0.4943 | -0.4655 | 4.61 | 1.68 | 0.36 |
| 2 | conditional_risk045_cq0.2 | 0.200 | -0.4913 | -0.4623 | 4.61 | 1.68 | 0.36 |
| 3 | conditional_risk045_cq0.25 | 0.250 | -0.4898 | -0.4607 | 4.34 | 1.58 | 0.36 |
| 4 | conditional_risk045_cbelow0.15 | 0.257 | -0.4896 | -0.4604 | 4.32 | 1.58 | 0.36 |
| 5 | conditional_risk045_cq0.3 | 0.300 | -0.4879 | -0.4587 | 4.32 | 1.57 | 0.36 |
| 6 | conditional_risk045_cbelow0.2 | 0.340 | -0.4862 | -0.4570 | 4.32 | 1.57 | 0.36 |
| 7 | conditional_risk045_cq0.35 | 0.350 | -0.4858 | -0.4565 | 4.32 | 1.57 | 0.36 |
| 8 | conditional_risk045_cbelow0.25 | 0.418 | -0.4821 | -0.4527 | 4.32 | 1.57 | 0.36 |

## 3. 关键对照

| Gate | Rank | Active frac | Spearman | Low20/High20 | Low20 MAE um | High20 MAE um |
|---|---:|---:|---:|---:|---:|---:|
| no_risk_cfocus_p15 | 1 | 0.000 | -0.4943 | 4.61 | 170.24 | 36.91 |
| current_risk045_all | 12 | 1.000 | -0.4291 | 3.21 | 159.37 | 49.58 |
| conditional_risk045_cq0.2 | 2 | 0.200 | -0.4913 | 4.61 | 170.24 | 36.91 |

## 4. 原理判断

conditional risk gate 明显优于全局 risk gate，并接近 no-risk gate；它更适合作为保留 risk 诊断价值的折中候选，但当前 prior-error ranking 仍由 no-risk gate 占优。

可支持的主张：

- risk 项不适合全局乘到 $C_{\mathrm{focus}}^{1.5}$ 上。
- 如果保留 risk 项，应优先考虑 low-confidence 条件触发或更弱调制。
- 当前证据仍以 prior-error ranking 为主，需要训练层面复核。

暂不使用的主张：

- 暂不否定 risk map 的诊断价值。
- 不声明真实样本三维高度精度。
- 不声明模型性能优势。

## 5. Q1-Q6 分桶曲线

### no_risk_cfocus_p15

| Bucket | Weight | Focus | Risk | Active frac | Prior MAE um |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.0214 | 0.0507 | 0.2988 | 0.000 | 174.05 |
| Q2 | 0.0569 | 0.1466 | 0.3602 | 0.000 | 140.93 |
| Q3 | 0.1254 | 0.2495 | 0.4004 | 0.000 | 113.79 |
| Q4 | 0.2253 | 0.3693 | 0.4386 | 0.000 | 85.62 |
| Q5 | 0.3762 | 0.5199 | 0.4615 | 0.000 | 58.27 |
| Q6 | 0.6883 | 0.7753 | 0.4315 | 0.000 | 34.61 |

### current_risk045_all

| Bucket | Weight | Focus | Risk | Active frac | Prior MAE um |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.0204 | 0.0523 | 0.3764 | 1.000 | 162.46 |
| Q2 | 0.0454 | 0.1507 | 0.4230 | 1.000 | 133.24 |
| Q3 | 0.0981 | 0.2559 | 0.4526 | 1.000 | 109.96 |
| Q4 | 0.1735 | 0.3770 | 0.4695 | 1.000 | 86.76 |
| Q5 | 0.2902 | 0.5246 | 0.4386 | 1.000 | 67.29 |
| Q6 | 0.5798 | 0.7508 | 0.2308 | 1.000 | 47.55 |

### conditional_risk045_cq0.2

| Bucket | Weight | Focus | Risk | Active frac | Prior MAE um |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.0204 | 0.0516 | 0.3564 | 1.000 | 165.49 |
| Q2 | 0.0563 | 0.1457 | 0.3026 | 0.200 | 149.49 |
| Q3 | 0.1254 | 0.2495 | 0.4004 | 0.000 | 113.79 |
| Q4 | 0.2253 | 0.3693 | 0.4386 | 0.000 | 85.62 |
| Q5 | 0.3762 | 0.5199 | 0.4615 | 0.000 | 58.27 |
| Q6 | 0.6883 | 0.7753 | 0.4315 | 0.000 | 34.61 |

## 6. 下一步

1. 若 conditional gate 接近 no-risk gate，可把它作为 full-budget matched repeat 的候选。
2. 若 no-risk gate 仍最稳，论文方法中把 risk 移到 diagnostic/failure-analysis 分支。
3. 在 real-stack diagnostic alignment 中检查 low-confidence 条件是否比 high-risk 条件更贴近 spike/saturation。

## 7. 文件索引

- metrics_csv: `submission_planning\optical_mechanism_analysis\conditional_risk_gate_diagnostic\conditional_risk_gate_metrics.csv`
- bucket_csv: `submission_planning\optical_mechanism_analysis\conditional_risk_gate_diagnostic\conditional_risk_gate_buckets.csv`
- summary_json: `submission_planning\optical_mechanism_analysis\conditional_risk_gate_diagnostic\conditional_risk_gate_summary.json`
- report_md: `submission_planning\optical_mechanism_analysis\conditional_risk_gate_diagnostic\conditional_risk_gate_report.md`
- plot: `submission_planning\optical_mechanism_analysis\conditional_risk_gate_diagnostic\conditional_risk_gate_top_candidates.png`
