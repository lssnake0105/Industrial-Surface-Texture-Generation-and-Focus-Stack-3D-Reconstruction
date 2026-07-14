# Risk-Confidence Interaction Diagnostic

- 日期：2026-06-22
- 样本：fixed synthetic test split，共 7 个样本
- 像素：1120000
- 结论边界：claim-ineligible diagnostic only；real-height calibrated accuracy claim remains unsupported。
- real-stack evidence remains diagnostic alignment only；audit should be rerun after manuscript-level merge。
- 本诊断只服务 confidence-gated prior consistency 的机制分析。

## 1. 研究问题

上一轮 gate-shape 诊断显示 $W=\mathrm{clip}(C_{\mathrm{focus}}^{1.5},0.02,1)$ 的 prior-error 排序优于 $W=\mathrm{clip}(C_{\mathrm{focus}}^{1.5}(1-0.45R),0.02,1)$。本诊断进一步检查 risk 项是否真的指向高 prior error，或是否在某些 focus-confidence 区域产生误降权。

## 2. 全局相关性

| Pair | Pearson | Spearman |
|---|---:|---:|
| risk vs prior error | -0.2804 | -0.3428 |
| focus confidence vs prior error | -0.2695 | -0.4944 |
| risk-induced downweight vs prior error | -0.2390 | -0.4660 |

## 3. Risk 项额外降权是否命中高误差

- 被 risk 项额外降权最多的 top 20% 像素，prior error 均值为 `26.32 um`。
- 被 risk 项额外降权最少的 bottom 20% 像素，prior error 均值为 `191.16 um`。
- top 20% 额外降权像素与 top 20% prior error 像素的重叠率为 `0.032`。
- top 20% 额外降权区域的平均 $C_{\mathrm{focus}}$ 为 `0.588`，平均 risk 为 `0.907`。

## 4. 额外降权最大的二维 bin

| Focus bin | Risk bin | Focus | Risk | Prior MAE um | No-risk W | Risk W | Delta W |
|---|---|---:|---:|---:|---:|---:|---:|
| F5 | R5 | 0.737 | 0.999 | 16.11 | 0.6388 | 0.3516 | 0.2872 |
| F5 | R4 | 0.728 | 0.824 | 23.37 | 0.6279 | 0.3944 | 0.2335 |
| F4 | R5 | 0.473 | 0.999 | 27.09 | 0.3267 | 0.1798 | 0.1469 |
| F4 | R4 | 0.470 | 0.807 | 37.82 | 0.3236 | 0.2060 | 0.1176 |
| F3 | R5 | 0.309 | 0.999 | 40.82 | 0.1730 | 0.0952 | 0.0778 |
| F3 | R4 | 0.309 | 0.800 | 55.72 | 0.1726 | 0.1105 | 0.0621 |
| F5 | R3 | 0.745 | 0.166 | 40.42 | 0.6492 | 0.6019 | 0.0474 |
| F2 | R5 | 0.177 | 0.999 | 55.60 | 0.0758 | 0.0417 | 0.0341 |

## 5. prior error 最高的二维 bin

| Focus bin | Risk bin | Focus | Risk | Prior MAE um | Delta W | Penalty ratio |
|---|---|---:|---:|---:|---:|---:|
| F1 | R1 | 0.056 | 0.000 | 231.82 | 0.0000 | 0.000 |
| F2 | R1 | 0.175 | 0.000 | 206.20 | 0.0000 | 0.000 |
| F1 | R2 | 0.057 | 0.004 | 205.69 | 0.0000 | 0.001 |
| F2 | R2 | 0.175 | 0.004 | 179.02 | 0.0001 | 0.002 |
| F1 | R3 | 0.068 | 0.156 | 174.51 | 0.0010 | 0.041 |
| F3 | R1 | 0.305 | 0.000 | 173.80 | 0.0000 | 0.000 |
| F3 | R2 | 0.307 | 0.004 | 142.47 | 0.0003 | 0.002 |
| F2 | R3 | 0.177 | 0.185 | 133.47 | 0.0063 | 0.084 |

## 6. 原理判断

risk-induced downweight 与 prior error 呈负相关，说明当前 risk 项额外压低的区域并不稳定对应更高 prior error；这会削弱门控对 DFF/GADFF 可靠性的排序能力。
当前更稳妥的解释是：$C_{\mathrm{focus}}$ 直接描述焦向响应一致性，离 prior reliability 更近；risk map 描述反光几何倾向，适合作为辅助诊断或分区分析，不宜直接作为同等强度的 prior gate 因子。

可支持的主张：

- low-confidence 是 prior consistency 门控的核心证据。
- risk map 更适合进入 failure analysis、real-stack diagnostic alignment 或弱调制项。
- 是否保留 risk 项需要 full-budget matched repeat 进一步验证。

暂不使用的主张：

- 暂不否定 risk map 的诊断价值。
- 不声明真实样本三维高度精度。
- 不声明模型性能优势。

## 7. 文件索引

- bin_csv: `submission_planning\optical_mechanism_analysis\risk_confidence_interaction_diagnostic\risk_confidence_interaction_bins.csv`
- summary_json: `submission_planning\optical_mechanism_analysis\risk_confidence_interaction_diagnostic\risk_confidence_interaction_summary.json`
- report_md: `submission_planning\optical_mechanism_analysis\risk_confidence_interaction_diagnostic\risk_confidence_interaction_report.md`
- prior_error_heatmap: `submission_planning\optical_mechanism_analysis\risk_confidence_interaction_diagnostic\risk_confidence_prior_error_heatmap.png`
- delta_weight_heatmap: `submission_planning\optical_mechanism_analysis\risk_confidence_interaction_diagnostic\risk_confidence_delta_weight_heatmap.png`
