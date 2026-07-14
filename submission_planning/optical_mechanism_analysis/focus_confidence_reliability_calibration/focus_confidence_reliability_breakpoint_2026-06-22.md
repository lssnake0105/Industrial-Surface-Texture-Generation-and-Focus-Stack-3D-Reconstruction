# Focus-Confidence Reliability Calibration 研究断点

- 日期：2026-06-22
- 主题：验证 focus confidence 是否能解释 DFF/GADFF prior 的可靠性
- 样本：fixed synthetic test split，共 7 个样本
- 结论边界：synthetic GT reliability calibration only；real-height calibrated accuracy claim remains unsupported。

## 1. 当前结论

Focus confidence 可以作为 DFF/GADFF prior reliability 的统计信号。在两个 ABL-07 checkpoint 中，$C_{\mathrm{focus}}$ 与 DFF/GADFF 绝对误差呈稳定负相关，说明低置信区域确实更容易对应传统 focus prior 的失效。

更关键的是，CGP-FocusNet 的收益主要集中在低置信桶；当 $C_{\mathrm{focus}}$ 升高后，DFF/GADFF 本身已经较可靠，网络继续改写 prior 的收益下降，部分高置信桶还会出现负收益。这支持当前训练策略中“低可靠区域弱化 prior 一致性、高可靠区域保留轴向结构信息”的门控设计。

## 2. 相关性证据

| Checkpoint | Target | Pearson | Spearman | Mean error um |
|---|---|---:|---:|---:|
| 2026-06-22_confidence_gated_prior_full_candidate | dff_error | -0.2450 | -0.5212 | 100.55 |
| 2026-06-22_confidence_gated_prior_full_candidate | gadff_error | -0.2505 | -0.4974 | 105.83 |
| 2026-06-22_confidence_gated_prior_full_candidate | model_error | -0.0819 | -0.0752 | 55.93 |
| 2026-06-22_confidence_gated_prior_seed_repeat | dff_error | -0.2450 | -0.5212 | 100.55 |
| 2026-06-22_confidence_gated_prior_seed_repeat | gadff_error | -0.2505 | -0.4974 | 105.83 |
| 2026-06-22_confidence_gated_prior_seed_repeat | model_error | -0.0868 | -0.0746 | 66.10 |

解释：DFF/GADFF 的 Spearman 约为 -0.52 / -0.50，强于模型误差自身的相关性。这表明 $C_{\mathrm{focus}}$ 更适合作为 prior reliability indicator，直接预测模型误差的证据仍然不足。

## 3. Q1-Q6 归并桶趋势

### 2026-06-22_confidence_gated_prior_full_candidate

| Bucket | Focus conf | Prior weight | Risk | DFF MAE um | GADFF MAE um | Model MAE um | Gain vs DFF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.054 | 0.022 | 0.309 | 159.49 | 163.49 | 69.60 | 46.37% |
| Q2 | 0.153 | 0.051 | 0.368 | 128.67 | 135.15 | 60.33 | 36.99% |
| Q3 | 0.256 | 0.107 | 0.407 | 106.95 | 114.21 | 55.25 | 23.78% |
| Q4 | 0.371 | 0.182 | 0.439 | 86.79 | 93.70 | 50.74 | 2.34% |
| Q5 | 0.513 | 0.294 | 0.451 | 70.79 | 75.81 | 49.45 | -41.10% |
| Q6 | 0.765 | 0.551 | 0.417 | 50.63 | 52.62 | 50.21 | -176.99% |

- 低置信 Q1：DFF MAE 159.49 um，Model MAE 69.60 um，Gain vs DFF 46.37%。
- 高置信 Q6：DFF MAE 50.63 um，Model MAE 50.21 um，Gain vs DFF -176.99%。

### 2026-06-22_confidence_gated_prior_seed_repeat

| Bucket | Focus conf | Prior weight | Risk | DFF MAE um | GADFF MAE um | Model MAE um | Gain vs DFF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.054 | 0.022 | 0.309 | 159.49 | 163.49 | 78.18 | 36.57% |
| Q2 | 0.153 | 0.051 | 0.368 | 128.67 | 135.15 | 70.18 | 21.89% |
| Q3 | 0.256 | 0.107 | 0.407 | 106.95 | 114.21 | 65.94 | 2.30% |
| Q4 | 0.371 | 0.182 | 0.439 | 86.79 | 93.70 | 62.50 | -28.94% |
| Q5 | 0.513 | 0.294 | 0.451 | 70.79 | 75.81 | 60.73 | -83.53% |
| Q6 | 0.765 | 0.551 | 0.417 | 50.63 | 52.62 | 59.03 | -233.64% |

- 低置信 Q1：DFF MAE 159.49 um，Model MAE 78.18 um，Gain vs DFF 36.57%。
- 高置信 Q6：DFF MAE 50.63 um，Model MAE 59.03 um，Gain vs DFF -233.64%。

## 4. 对论文故事线的意义

这组结果把 ABL-07 从单纯的结果提升，推进到更清晰的原理解释：反光表面焦栈中的 DFF/GADFF prior 具有区域性可靠性差异，可靠性可以由焦向响应的一致性和风险项共同估计。因此，模型贡献可以表述为 confidence-gated prior consistency，避免把它写成额外堆叠的黑箱网络模块。

可支持的论文主张：

- $C_{\mathrm{focus}}$ 与传统 prior 误差之间存在稳定负相关，可作为 prior reliability 的统计代理。
- CGP-FocusNet 的主要收益来自 low-confidence、高风险、DFF/GADFF 更容易失败的区域。
- 高置信区域中，传统 focus prior 已包含较可靠轴向结构，模型应减少不必要改写。

暂不支持的主张：

- real-height calibrated accuracy claim remains unsupported。
- 外部基线总体优势仍需完成兼容评估后再判断。
- 不支持把 real-stack alignment 当作带真值的定量评估。

## 5. 下一步最有价值问题

1. 做 gate-shape ablation：比较 $C^1.0$、$C^1.5$、$C^2.0$ 和 risk 权重系数，确认当前门控形状是否只是经验设定。
2. 做 per-sample failure audit：定位高置信负收益桶来自哪些形貌、风险分布或仿真参数。
3. 把 real-stack diagnostic alignment 与 synthetic reliability calibration 对齐：检查低 $C_{\mathrm{focus}}$ 区域是否也对应真实焦栈中的 spike、saturation 或局部不连续。
4. 增加文稿级图表：把 Q1-Q6 归并趋势图做成一张简洁 figure，用于支撑 confidence-gated prior consistency。

## 6. 文件索引

- 完整报告：`submission_planning\optical_mechanism_analysis\focus_confidence_reliability_calibration\focus_confidence_reliability_calibration_report.md`
- 归并数据来源：`submission_planning\optical_mechanism_analysis\focus_confidence_reliability_calibration\focus_confidence_reliability_aggregate_metrics.csv`
- 相关性数据：`submission_planning\optical_mechanism_analysis\focus_confidence_reliability_calibration\focus_confidence_reliability_correlation_summary.csv`
- 运行摘要：`submission_planning\optical_mechanism_analysis\focus_confidence_reliability_calibration\focus_confidence_reliability_summary.json`
