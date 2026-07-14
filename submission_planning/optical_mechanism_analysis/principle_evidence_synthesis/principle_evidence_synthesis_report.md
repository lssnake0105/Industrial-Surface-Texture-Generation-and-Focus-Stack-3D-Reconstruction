# SRTP 原理方向证据合成报告

日期：2026-06-22  
用途：为投稿故事线、实验设计和论文写作提供证据边界  

## 1. 当前论文主线建议

本项目当前最稳妥的投稿主线应收束为：

> 面向反光工业微结构表面的焦栈三维重建中，传统 DFF 的失效主要来自焦度响应不确定、局部饱和/高亮和弱信号区域，而不仅是简单的采集漂移或单一眩光几何风险。本文通过 Simulation-to-Real 数据构造、DFF/GADFF 先验、focus-confidence 诊断和置信度加权训练策略，构建一个先验引导的稳健重建框架。

这条主线的关键优点是把“Simulation-to-Real”定位为解决真实标注不足和退化覆盖不足的手段，同时把论文的科学问题放在“反光表面焦栈中 DFF 为什么失效、如何被识别和缓解”上。

## 2. 证据等级总表

| 编号 | 可支撑的机制判断 | 主要证据 | 证据等级 | 论文位置 | 主要限制 |
|---|---|---|---|---|---|
| E1 | DFF 失败更稳定地对应 top-2 focus margin 低，而非几何 glare risk 单独决定 | 135 个合成条件；`low_margin` mean AUC 0.6412，std 0.0133，AUC>0.55 rate 1.00；`risk_mean/risk_max` mean AUC 约 0.51 | 强机制证据 | Method motivation / quality prior design | 基于合成数据，需避免写成真实 MAE 结论 |
| E2 | 真实焦栈中的不可靠区域具有多种焦度曲线形态，不宜合并成单一失败类别 | 7 组真实焦栈；清晰单峰 49.5%，平坦歧义 25.5%，暗弱信号 15.2%，多峰 4.2%，局部跳变 3.8%，高亮饱和 1.8% | 中强真实诊断证据 | Real-sample diagnostic / Discussion | 无真实高度 GT，属于 no-reference morphology |
| E3 | 100um 真实焦栈的局部 DFF 跳变难以由小幅全局层间平移单独解释 | 最大层间平移 0.1118 px，中位数 0；配准后 peak layer 变化 4.68%；spike proxy Pearson 0.9478；quality proxy Pearson 0.9825 | 中强排除性证据 | Real-stack sanity check / Discussion | 只排除全局平移，不能排除非刚性、倍率、照明角变化 |
| E4 | 置信度加权伪标签训练在多退化混合场景中更有价值 | 96 个受控合成条件；总体 MAE 0.0883→0.0857，胜率 71.9%；mixed 模式改善 7.2%，胜率 100.0% | 中等训练策略代理证据 | Training strategy / Ablation motivation | 轻量学生模型，不等价于完整 FocusResUNet |
| E5 | DFF/GADFF 先验仍是当前模型体系中最稳定的几何基础 | 已有 ablation：移除 DFF/GADFF prior 后 MAE 133.4808，高于 Full S2R-FocusNet 109.2209；但 w/o focal difference 和 w/o glare cue 更优 | 中等模型消融证据 | Ablation table with caution | 完整模型未占优，需要写成“DFF prior 有价值，融合设计需重构” |

## 3. 可直接写进论文的主张

### 3.1 Focus-margin confidence 是当前最可靠的 DFF failure 指示器

可写主张：

> In the controlled synthetic sweep, the top-2 focus margin provides the most stable no-reference indicator of DFF failure regions. The low-margin score achieves a mean AUC of 0.6412 with a standard deviation of 0.0133 over 135 conditions and remains above 0.55 in all tested cases.

中文解释：

在当前仿真族中，DFF 的错误更直接对应焦度响应峰值竞争，而不是几何眩光风险本身。`low_margin` 可以作为训练权重和无参考质量诊断的主信号，`glare risk` 更适合作为物理解释和区域分层先验。

禁写边界：

- 不应写成“glare risk 无用”。它仍可解释反光形成条件。
- 不应写成“low_margin 已在真实高度 GT 上证明有效”。真实样本目前没有标定高度。

### 3.2 真实焦栈不可靠性是多形态问题

可写主张：

> Real focus stacks exhibit multiple focus-response morphologies rather than a single failure mode. Across seven real stacks, only 49.5% of pixels are classified as confident single-peak responses, while the rest are distributed among flat ambiguous responses, dark low-signal regions, multi-peak competition, local peak-layer spikes, and saturated highlights.

中文解释：

真实域中的 DFF 先验不能被视为均匀可靠的监督标签。更合理的处理方式是把 DFF 作为带置信度的观测，并在训练中显式区分可靠单峰、平坦歧义、多峰竞争、高亮饱和和暗弱信号区域。

禁写边界：

- 不应把这些类别写成真实错误类别。
- 不应声称比例等同于真实深度失败比例。

### 3.3 全局层间平移不是当前 100um 焦栈不稳定性的主要解释

可写主张：

> For the 100um key-texture real stack, the estimated global inter-layer shift is below 0.12 pixels. After translation compensation, only 4.68% of pixels change their selected DFF peak layer, and the low-margin AUC for identifying spike-proxy pixels remains nearly unchanged.

中文解释：

这条证据可以作为防御性实验：它不能证明所有几何误差都不存在，但能说明当前观察到的 focus-response instability 和局部 DFF layer-selection jumps 难以用小幅全局平移单独解释。

禁写边界：

- 不能写成“已排除所有采集几何误差”。
- 不能排除非刚性形变、倍率变化、入瞳/照明变化或离焦导致的结构外观变化。

### 3.4 Confidence-weighted pseudo-labeling 是有条件收益的训练策略

可写主张：

> In a controlled pseudo-labeling study, confidence-weighted supervision reduces the average MAE of a lightweight student from 0.0883 to 0.0857 over 96 conditions. The benefit is concentrated under mixed degradation, where the relative improvement reaches 7.2% with a 100.0% win rate.

中文解释：

这条证据支持在论文中引入 confidence-aware loss 或 sample weighting 的动机，但要写成“按退化类型调节伪标签可信度”的策略，而不是写成所有场景都稳定提升。

禁写边界：

- 不应声称已经完成完整 FocusResUNet 训练验证。
- 不应声称 weak-texture 场景也有明显收益。

## 4. 建议的论文贡献重写

### 贡献 1：反光表面 DFF 失效的焦度响应机制分析

建议表述：

> We analyze DFF failure in reflective focus-stack reconstruction from the perspective of focus-response ambiguity, showing that low top-2 focus margin is a more stable failure indicator than geometric glare risk alone.

支撑证据：E1、E2、E3。

### 贡献 2：面向真实退化的 Simulation-to-Real 数据与诊断先验

建议表述：

> We design a simulation-to-real data construction strategy that covers not only surface height variation, but also focus-curve morphologies observed in real stacks, including flat responses, multi-peak competition, saturated highlights, and low-signal regions.

支撑证据：E2、E4。

### 贡献 3：置信度感知的先验引导重建训练策略

建议表述：

> We formulate DFF/GADFF priors as confidence-aware observations and use focus-confidence maps to guide pseudo-label weighting and prior fusion, instead of treating all DFF-derived targets as equally reliable.

支撑证据：E1、E4、E5。

## 5. 当前最安全的方法公式

建议将 quality prior 写成分层形式：

```text
Q_margin(x) = 1 - M_focus(x)
R_glare(x)  = geometry-based glare risk
P_sat(x)    = saturation persistence
```

其中：

- `Q_margin` 是训练权重和 DFF 失败诊断的主信号；
- `R_glare` 是物理解释、区域分层和仿真条件控制信号；
- `P_sat` 用于区分饱和/高亮区域和普通低置信度区域。

对应的训练表述可写为：

```text
L = mean( w(x) * |H_pred(x) - H_target(x)| )
w(x) = f(Q_margin(x), P_sat(x), R_glare(x))
```

论文中应强调 `w(x)` 是 confidence-aware weighting，而不是把 `R_glare` 直接等价为 error mask。

## 6. 目前需要继续补强的缺口

| 缺口 | 为什么重要 | 可执行下一步 |
|---|---|---|
| 完整模型中的 confidence-weighted loss 尚未训练验证 | 代理实验不能替代神经网络主实验 | 在现有 ablation runner 中增加 margin-weighted loss 变体，做 matched split smoke + full candidate |
| w/o focal difference 和 w/o glare cue 反而优于 full model | 说明现有融合方式可能稀释稳定先验 | 将 glare/focal cue 从直接拼接改为 gating 或 confidence head |
| 真实样本缺少高度 GT | 限制真实结果只能写 no-reference diagnosis | 选择少量 ROI 做人工标注、重复采集或外部轮廓仪/共聚焦验证 |
| 真实焦栈只做了全局平移敏感性 | 仍可能存在非刚性、倍率、照明路径变化 | 增加局部块配准或 feature-flow 诊断，但只作为 sanity check |
| SOTA 对比仍需更新 | 投稿时必须避免过老 baseline | 保留 Depth Anything/Depth Anything V2 等 modern prior 作为相关工作，不一定作为直接 baseline |

## 7. 下一步实验建议

优先级最高的下一步是：

1. 在 `tmp/ablation_results` 内新增一个 `confidence_weighted_loss` 训练变体。
2. 保持现有 matched split 和 protected runner，不改动 `src/` 主项目交付代码。
3. 先做 smoke 验证，再做 4-epoch matched full candidate。
4. 指标重点看 global MAE、high-risk MAE、spike-region MAE 和 p90 error。
5. 如果结果优于 full model，再进入 seed repeat；如果结果不优于 full model，则把当前证据写成 training motivation，而不是主方法收益。

## 8. 可直接进入论文的英文段落

> Our diagnostic analysis suggests that the degradation of DFF in reflective focus stacks is governed more directly by focus-response ambiguity than by glare-prone geometry alone. In a 135-condition synthetic sweep, the low top-2 focus margin achieves a mean AUC of 0.6412 for identifying top DFF failure regions and remains above 0.55 in all conditions, whereas geometry-based glare-risk scores are close to random ranking. Real focus stacks further show that unreliable DFF observations are heterogeneous: across seven stacks, only 49.5% of pixels exhibit confident single-peak responses, while the remaining regions are distributed among flat ambiguous responses, dark low-signal areas, multi-peak competition, local peak-layer spikes, and saturated highlights. A registration sensitivity test on the 100um real stack also shows that the observed local DFF layer-selection jumps cannot be explained by small global inter-layer translation alone. These findings motivate treating DFF/GADFF priors as confidence-aware observations and using focus-confidence maps for loss weighting, pseudo-label selection, and region-wise analysis.

## 9. 可直接进入论文的中文段落

> 诊断实验表明，反光焦栈中 DFF 的退化更直接地受到焦度响应歧义影响，而不应仅由几何眩光风险解释。在 135 组合成条件中，top-2 focus margin 对 DFF 失败区域具有最稳定的识别能力，`low_margin` 的平均 AUC 为 0.6412，且在所有条件下均高于 0.55；相比之下，基于几何的 glare-risk 分数接近随机排序。真实焦栈进一步显示，DFF 不可靠区域具有多种形态：7 组样本中仅 49.5% 像素呈现清晰单峰响应，其余区域分布在平坦歧义、暗弱信号、多峰竞争、局部层跳变和高亮饱和等类别中。对 100um 真实焦栈的配准敏感性实验也表明，局部 DFF 层选择跳变难以由小幅全局层间平移单独解释。这些结果共同支持将 DFF/GADFF 先验作为带置信度的观测，并使用 focus-confidence map 进行损失加权、伪标签筛选和分区分析。

