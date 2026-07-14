# Gate-Shape Matched Retraining Smoke

- 日期：2026-06-22
- 状态：pass
- 样本：train 27 / validation 10 / test 7
- 训练预算：2 epochs, 72 train patches, 18 validation patches
- 结论边界：claim-ineligible smoke only；real-height calibrated accuracy claim remains unsupported。
- real-stack evidence remains diagnostic alignment only；audit should be rerun after any manuscript-level merge。
- 本实验只检验 confidence-gated prior consistency 的训练可行性和方向性。

## 1. 实验设置

三组实验保持网络、数据 split、patch 采样预算、loss 其他项一致，只替换 prior consistency 权重：

| Tag | Gate |
|---|---|
| gate_rank1_cfocus_p15_risk0_smoke | Rank-1 diagnostic gate: C_focus^1.5 |
| gate_focus_only_p15_smoke | Focus-only gate: C_focus^1.5 |
| gate_current_cfocus_p15_risk045_smoke | Current ABL-07 gate: C_focus^1.5(1-0.45R) |

## 2. Full Test Split Smoke Summary

| Gate | Mean MAE um | High-risk MAE um | Low-confidence MAE um | Gain vs DFF | Win rate | Last val MAE norm | Prior weight mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_rank1_cfocus_p15_risk0_smoke | 270.10 | 302.67 | 264.25 | -168.61% | 0.14 | 0.23423 | 0.2062 |
| gate_focus_only_p15_smoke | 270.10 | 302.67 | 264.25 | -168.61% | 0.14 | 0.23423 | 0.2062 |
| gate_current_cfocus_p15_risk045_smoke | 272.52 | 308.66 | 266.53 | -171.02% | 0.14 | 0.23643 | 0.1749 |

## 3. 原理判断

在当前 smoke 预算下，rank-1 诊断候选的 mean MAE 比当前 risk0.45 门控低 2.42 um，方向上支持先把 risk 项降级为辅助调制。
rank-1 诊断候选与 focus-only 组数值完全一致，因为二者在当前候选集合中都等价于 $W=\mathrm{clip}(C_{\mathrm{focus}}^{1.5},0.02,1)$。

可支持的主张：

- low-confidence prior consistency 是当前门控设计的核心变量。
- gate-shape 诊断需要 matched retraining smoke 复核，不能只依赖 prior-error ranking。
- 本轮结果可以作为下一步 full split seed repeat 的候选筛选依据。

暂不使用的主张：

- 不声明模型精度提升已经成立。
- 不声明真实样本三维高度精度。
- 不声明外部基线优势。

## 4. 下一步

1. 如果 rank-1 候选继续占优，运行 full-budget matched repeat。
2. 如果 current gate 占优，分析 risk 项是否通过优化动态而非 prior-error 排序发挥作用。
3. 做高置信负收益区域的 per-sample failure audit。

## 5. 文件索引

- summary_csv: `tmp\ablation_results\gate_shape_retraining_smoke\gate_shape_retraining_smoke_summary.csv`
- per_sample_csv: `tmp\ablation_results\gate_shape_retraining_smoke\gate_shape_retraining_smoke_per_sample.csv`
- stratum_csv: `tmp\ablation_results\gate_shape_retraining_smoke\gate_shape_retraining_smoke_strata.csv`
- report_json: `tmp\ablation_results\gate_shape_retraining_smoke\gate_shape_retraining_smoke_report.json`
