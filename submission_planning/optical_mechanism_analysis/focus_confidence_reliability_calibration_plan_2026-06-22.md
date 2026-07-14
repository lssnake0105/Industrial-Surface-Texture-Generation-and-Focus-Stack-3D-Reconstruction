# Focus-Confidence Reliability Calibration Plan

- 日期：2026-06-22
- 目的：验证 CGP-FocusNet 中的 focus confidence gate 是否具有统计可解释性
- 推荐产物形式：Markdown + CSV + PNG，不要求 LaTeX 报告

## 1. 研究问题

当前 ABL-07 的机制假设是：DFF/GADFF prior 的可靠性与 focus confidence 相关。因此，训练中使用

$$C_{\mathrm{focus}}=\mathrm{clip}(0.65C_{\mathrm{DFF}}+0.35C_{\mathrm{GADFF}},0,1)$$

以及

$$W_{\mathrm{prior}}=\mathrm{clip}(C_{\mathrm{focus}}^{1.5}(1-0.45R),0.02,1.0)$$

来门控 prior consistency。下一步需要验证：$C_{\mathrm{focus}}$ 低的区域，DFF/GADFF 的真实误差是否确实更高；CGP-FocusNet 的收益是否集中在这些低可靠区域。

## 2. 最小可行实验

### 数据

使用已有 fixed synthetic test split：

- test samples: 7
- height GT: available
- DFF/GADFF priors: available
- CGP-FocusNet predictions: use ABL-07 full candidate and seed repeat

### 分桶方式

按 $C_{\mathrm{focus}}$ 分桶：

| Bucket | 区间 |
|---|---|
| B1 | $[0.00,0.10)$ |
| B2 | $[0.10,0.20)$ |
| B3 | $[0.20,0.35)$ |
| B4 | $[0.35,0.50)$ |
| B5 | $[0.50,0.70)$ |
| B6 | $[0.70,1.00]$ |

也可以使用分位数分桶，避免像素数量过度不均衡。建议先做 quantile buckets，再补 fixed-range buckets。

### 指标

每个 bucket 统计：

| Metric | 含义 |
|---|---|
| pixel_count | 当前桶内像素数 |
| mean_focus_conf | 平均 focus confidence |
| dff_mae_um | DFF 相对 GT 的 MAE |
| gadff_mae_um | GADFF 相对 GT 的 MAE |
| cgp_mae_um | CGP-FocusNet 相对 GT 的 MAE |
| cgp_gain_vs_dff | $(DFF-CGP)/DFF$ |
| prior_weight_mean | 平均 prior weight |
| risk_mean | 平均 risk cue |

核心判断：

$$\mathrm{corr}(C_{\mathrm{focus}}, |P_{\mathrm{DFF}}-H^\ast|)<0$$

若该相关性成立，说明 focus confidence 可以作为 DFF reliability signal。若低 $C_{\mathrm{focus}}$ 桶中 CGP gain 更高，则支持当前 gate 机制。

## 3. 判定标准

强支持：

- focus confidence 与 DFF absolute error 呈稳定负相关；
- 低 confidence bucket 的 DFF MAE 明显高于高 confidence bucket；
- CGP-FocusNet 在低 confidence bucket 的 gain 明显高于高 confidence bucket；
- 两个 ABL-07 checkpoint 趋势一致。

中等支持：

- DFF error 与 confidence 的趋势存在，但部分样本不稳定；
- CGP gain 在低 confidence 区域更高，但 seed repeat 幅度下降；
- 可以写成“preliminary reliability evidence”。

不支持：

- DFF error 与 confidence 无明显关系；
- CGP gain 与 confidence bucket 无关；
- 需要重新设计 confidence map 或 gate 形状。

## 4. 推荐脚本

建议新增脚本：

`submission_planning/tools/calibrate_focus_confidence_reliability.py`

输入：

- ABL-07 checkpoint tag；
- synthetic test split；
- bucket strategy: `quantile` 或 `fixed`；
- output directory。

输出：

- `focus_confidence_reliability_bucket_metrics.csv`
- `focus_confidence_reliability_bucket_metrics.json`
- `focus_confidence_reliability_report.md`
- `focus_confidence_reliability_curve.png`

## 5. 图表设计

### 图 1：confidence-error curve

横轴：bucket mean focus confidence  
纵轴：DFF MAE / GADFF MAE / CGP MAE

目标：显示 focus confidence 越低，DFF/GADFF 越不可靠。

### 图 2：gain-confidence curve

横轴：bucket mean focus confidence  
纵轴：CGP gain vs DFF

目标：显示 CGP-FocusNet 的收益是否集中在低 confidence 区域。

### 表 1：bucket metrics

按 checkpoint 分别列出 bucket 结果，优先展示：

- low-confidence bucket；
- middle-confidence bucket；
- high-confidence bucket。

## 6. 写作价值

如果实验成立，论文中的机制链可以从：

> low-confidence 区域收益更明显

推进为：

> focus confidence 可以预测 DFF prior 的可靠性，因此 confidence-gated prior consistency 有统计支撑。

这会显著增强原理解释，因为它把 gate 从经验设计提升为可验证的 reliability modeling。

## 7. 风险与边界

- 该实验只验证 synthetic GT 下的 reliability，不直接证明真实高度精度。
- 若真实焦栈没有 GT，只能观察 confidence 与 DFF instability 的关系。
- 若不同样本的 confidence 分布差异很大，应使用 per-sample normalization 或 quantile bucket。
- 若 $C_{\mathrm{DFF}}$ 与 $C_{\mathrm{GADFF}}$ 表现差异明显，需要分别分析，而不只看融合后的 $C_{\mathrm{focus}}$。

## 8. 下一步执行顺序

1. 实现 `calibrate_focus_confidence_reliability.py`。
2. 先跑 ABL-07 full candidate。
3. 再跑 seed repeat。
4. 对比两个 checkpoint 的 bucket 曲线。
5. 写 `focus_confidence_reliability_calibration_report.md`。
6. 若结果支持机制，再把该实验加入 CGP-FocusNet 三层证据链。

