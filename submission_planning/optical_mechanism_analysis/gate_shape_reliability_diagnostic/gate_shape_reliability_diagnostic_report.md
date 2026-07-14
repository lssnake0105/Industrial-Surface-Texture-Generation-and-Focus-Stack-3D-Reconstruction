# Gate-Shape Reliability Diagnostic

- 日期：2026-06-22
- 样本：fixed synthetic test split，共 7 个样本
- 像素采样：每个样本最多 160000 个像素；用于原理诊断和门控排序，不作为模型重训结果。
- 结论边界：claim-ineligible for manuscript main-table accuracy；synthetic GT prior-reliability diagnostic only；real-height calibrated accuracy claim remains unsupported。
- real-stack evidence remains diagnostic alignment only；audit should be rerun after any manuscript-level merge。
- 本诊断只服务 confidence-gated prior consistency 的机制分析。

## 1. 研究问题

当前 ABL-07 使用的门控形式为：

$$W=\mathrm{clip}(C_{\mathrm{focus}}^{1.5}(1-0.45R),0.02,1).$$

本诊断检验不同 $C_{\mathrm{focus}}^p(1-\lambda R)$ 形状能否更好地排序 DFF/GADFF prior target 的误差。理想门控应让低权重区域富集高 prior error，并让权重与 prior error 呈负相关。

## 2. 排名前十的门控形状

| Rank | Gate | Spearman | Low20/High20 error ratio | Low20 lift | High20 ratio | Mean weight |
|---:|---|---:|---:|---:|---:|---:|
| 1 | cfocus_p1.5_risk0_min002 | -0.4655 | 6.38 | 1.74 | 0.40 | 0.249 |
| 2 | focus_only_p1.5 | -0.4655 | 6.38 | 1.74 | 0.40 | 0.249 |
| 3 | cfocus_p1.25_risk0_min002 | -0.4661 | 6.38 | 1.74 | 0.40 | 0.293 |
| 4 | cfocus_p0.75_risk0_min002 | -0.4664 | 6.38 | 1.74 | 0.40 | 0.434 |
| 5 | cfocus_p1_risk0_min002 | -0.4663 | 6.38 | 1.74 | 0.40 | 0.352 |
| 6 | focus_only_p1 | -0.4663 | 6.38 | 1.74 | 0.40 | 0.352 |
| 7 | cfocus_p1.75_risk0_min002 | -0.4641 | 6.36 | 1.73 | 0.40 | 0.215 |
| 8 | cfocus_p2_risk0_min002 | -0.4617 | 6.29 | 1.73 | 0.40 | 0.189 |
| 9 | focus_only_p2 | -0.4617 | 6.29 | 1.73 | 0.40 | 0.189 |
| 10 | cfocus_p2.5_risk0_min002 | -0.4536 | 5.79 | 1.61 | 0.40 | 0.151 |

## 3. 当前门控与 Rank-1 诊断候选

| Gate | Rank | Spearman | Low20/High20 error ratio | Low20 MAE um | High20 MAE um |
|---|---:|---:|---:|---:|---:|
| cfocus_p1.5_risk0_min002 | 1 | -0.4655 | 6.38 | 154.08 | 54.11 |
| cfocus_p1.5_risk0.45_min002 | 21 | -0.4021 | 4.06 | 144.43 | 65.90 |

解释：如果 rank-1 诊断候选只比当前门控小幅提升，则当前 $p=1.5,\lambda=0.45$ 更适合作为保守默认值；如果差距较大，下一步应做 matched retraining ablation。

## 4. Q1-Q6 分桶曲线

### cfocus_p1.5_risk0_min002

| Bucket | Mean weight | Focus conf | Risk | Prior target MAE um | DFF MAE um | GADFF MAE um |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.0242 | 0.0609 | 0.3233 | 121.68 | 123.14 | 126.49 |
| Q2 | 0.0528 | 0.1275 | 0.3469 | 154.64 | 155.53 | 161.70 |
| Q3 | 0.1327 | 0.2564 | 0.4081 | 108.50 | 107.26 | 114.73 |
| Q4 | 0.2291 | 0.3706 | 0.4383 | 88.98 | 86.91 | 93.91 |
| Q5 | 0.3708 | 0.5125 | 0.4513 | 72.38 | 70.37 | 75.44 |
| Q6 | 0.6757 | 0.7644 | 0.4163 | 51.66 | 50.74 | 52.70 |

### cfocus_p1.5_risk0.45_min002

| Bucket | Mean weight | Focus conf | Risk | Prior target MAE um | DFF MAE um | GADFF MAE um |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.0219 | 0.0628 | 0.4024 | 112.61 | 114.07 | 117.51 |
| Q2 | 0.0425 | 0.1311 | 0.4120 | 146.59 | 147.33 | 153.80 |
| Q3 | 0.1024 | 0.2628 | 0.4603 | 104.59 | 103.31 | 110.90 |
| Q4 | 0.1756 | 0.3773 | 0.4616 | 90.59 | 88.60 | 95.38 |
| Q5 | 0.2881 | 0.5149 | 0.4152 | 78.93 | 76.96 | 81.89 |
| Q6 | 0.5710 | 0.7429 | 0.2368 | 64.25 | 63.44 | 65.19 |

## 5. 原理结论

诊断排名更偏向较大的 confidence exponent，说明可靠性门控需要压低中低置信区域的 prior consistency 权重，避免把 DFF/GADFF 的错误结构强行写入网络目标。
rank-1 诊断候选中的 risk 系数偏低，提示 risk map 更适合作为辅助调制项，不能单独承担 prior reliability 判断。

可支持的主张：

- gate-shape 可以用 prior-error ranking 进行机制诊断。
- low-confidence 区域应被赋予更低 prior consistency 权重。
- 当前门控是否进入论文主方法，仍需 matched retraining ablation 支撑。

暂不使用的主张：

- 该诊断不声明模型精度提升。
- 该诊断不声明真实样本三维高度精度。
- 该诊断不声明外部基线优势。

## 6. 下一步

1. 对 rank-1 诊断候选、当前门控、focus-only 三个配置做 matched retraining smoke。
2. 如果趋势稳定，再做 full split seed repeat。
3. 对高置信负收益区域做 per-sample failure audit，检查是否来自形貌参数或 risk/confidence 特征失配。

## 7. 文件索引

- sample_metrics_csv: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\gate_shape_sample_metrics.csv`
- aggregate_metrics_csv: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\gate_shape_aggregate_metrics.csv`
- bucket_metrics_csv: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\gate_shape_bucket_metrics.csv`
- summary_json: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\gate_shape_reliability_summary.json`
- plot: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\gate_shape_top_candidates.png`
- plot: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\cfocus_p1.5_risk0_min002_bucket_curve.png`
- plot: `submission_planning\optical_mechanism_analysis\gate_shape_reliability_diagnostic\focus_only_p1.5_bucket_curve.png`
