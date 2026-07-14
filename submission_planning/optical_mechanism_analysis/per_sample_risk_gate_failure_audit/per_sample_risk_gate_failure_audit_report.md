# Per-Sample Risk-Gate Failure Audit

- 日期：2026-06-22
- 样本：fixed synthetic test split，共 7 个样本
- 结论边界：claim-ineligible diagnostic only；real-height calibrated accuracy claim remains unsupported。
- real-stack evidence remains diagnostic alignment only；audit should be rerun after manuscript-level merge。
- 本审计只服务 confidence-gated prior consistency 的机制分析。

## 1. 审计问题

上一轮全局诊断显示 risk-induced downweight 与 prior error 呈负相关。本轮检查这个现象是否由少数样本主导，或是否在 test split 中普遍存在。

## 2. 汇总结论

- `risk_gate_failure_flag` 样本数：`5/7`。
- 样本级 delta-weight vs prior-error Spearman 均值：`-0.4193`。
- 样本级 top20 overlap 均值：`0.0468`。
- 样本级 high-delta / low-delta prior-error ratio 均值：`0.2311`。

结论：risk 项的误降权现象具有跨样本一致性，不能解释为单一样本异常。

## 3. 样本级结果

| Sample | Delta Spearman | Top20 overlap | High-delta error | Low-delta error | Ratio | Flag |
|---|---:|---:|---:|---:|---:|---|
| test_A型突起刃脊_柏林粗糙 | -0.6481 | 0.001 | 10.22 | 243.72 | 0.042 | yes |
| test_V谷_P10_宽谷粗糙平底 | -0.5299 | 0.014 | 25.62 | 237.67 | 0.108 | yes |
| test_山脊_柏林粗糙 | -0.5013 | 0.035 | 13.89 | 187.63 | 0.074 | yes |
| test_阶跃_柏林粗糙 | -0.4547 | 0.017 | 8.44 | 66.48 | 0.127 | yes |
| test_山峰_分形粗糙 | -0.4458 | 0.025 | 17.66 | 117.12 | 0.151 | yes |
| test_周期_条纹粗糙 | -0.2431 | 0.125 | 159.09 | 259.62 | 0.613 | no |
| test_复合腐蚀凹坑 | -0.1121 | 0.110 | 33.48 | 66.51 | 0.503 | no |

## 4. 最强 failure 样本

- `test_A型突起刃脊_柏林粗糙`：delta Spearman `-0.6481`，top20 overlap `0.001`，high-delta error `10.22 um`。
- `test_V谷_P10_宽谷粗糙平底`：delta Spearman `-0.5299`，top20 overlap `0.014`，high-delta error `25.62 um`。
- `test_山脊_柏林粗糙`：delta Spearman `-0.5013`，top20 overlap `0.035`，high-delta error `13.89 um`。

## 5. 弱 failure / 边界样本

未触发 flag 的样本并没有形成 risk 项的正证据；它们仍表现为负 Spearman，只是 top20 overlap 或 high-delta error ratio 没有达到本审计的强 failure 阈值。

- `test_周期_条纹粗糙`：delta Spearman `-0.2431`，top20 overlap `0.125`，ratio `0.613`。
- `test_复合腐蚀凹坑`：delta Spearman `-0.1121`，top20 overlap `0.110`，ratio `0.503`。

## 6. 原理解释

risk 项当前描述的是反光几何倾向，而 prior reliability 更直接地由焦向响应一致性决定。样本级结果如果普遍显示 high-delta 区域误差较低，说明 risk gate 会把高 risk 但高 confidence 的可用 prior 也压低，从而削弱 DFF/GADFF prior 的结构保留作用。

可支持的主张：

- low-confidence 是 prior gate 的主要证据。
- risk map 更适合作为 failure analysis 和 real-stack diagnostic alignment 的分区变量。
- risk 项若进入训练，应作为弱调制或条件项，而非与 $C_{\mathrm{focus}}$ 同等强度相乘。

暂不使用的主张：

- 暂不否定 risk map 的诊断价值。
- 不声明真实样本三维高度精度。
- 不声明模型性能优势。

## 7. 下一步

1. 设计 conditional risk gate：只在 low-confidence 或 saturation persistence 较高时启用 risk 调制。
2. 对 failure 样本画 ROI 级 focus curve，确认 high risk + high confidence 区域为何 prior error 较低。
3. 在 real-stack diagnostic alignment 中检查 high risk + high confidence 区域是否对应稳定表面结构。

## 8. 文件索引

- sample_csv: `submission_planning\optical_mechanism_analysis\per_sample_risk_gate_failure_audit\per_sample_risk_gate_failure_audit.csv`
- summary_json: `submission_planning\optical_mechanism_analysis\per_sample_risk_gate_failure_audit\per_sample_risk_gate_failure_audit_summary.json`
- report_md: `submission_planning\optical_mechanism_analysis\per_sample_risk_gate_failure_audit\per_sample_risk_gate_failure_audit_report.md`
- plot: `submission_planning\optical_mechanism_analysis\per_sample_risk_gate_failure_audit\per_sample_risk_gate_failure_audit.png`
