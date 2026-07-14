# CGP-FocusNet 原理机制笔记

- 日期：2026-06-22
- 当前对象：ABL-07 / CGP-FocusNet
- 写作形式：中文 Markdown，保留内嵌 LaTeX 公式
- 证据状态：合成 GT 定量证据已通过内部审查；真实焦栈仅支持无参考诊断一致性

## 1. 当前核心判断

CGP-FocusNet 的本质贡献应收束为一个机制：把 DFF/GADFF 从“固定可信的深度答案”改写为“带置信度的物理先验观测”。模型仍然利用 DFF 的轴向聚焦信息，但训练时通过 focus confidence 和 risk cue 控制 prior consistency 的强度，使网络在焦点响应可靠处保留结构，在焦点响应不稳定处减少错误先验牵引。

这条机制比单纯强调网络结构更有解释力。当前证据显示，ABL-07 的整体 MAE 改善存在 seed sensitivity，但低置信焦点区域的收益在两次 full run 中都稳定出现，因此论文故事应把“focus-confidence-aware prior consistency”作为主线。

## 2. 问题本质：DFF 的失效来自焦点响应可靠性差异

传统 DFF 的基本假设是：像素在真实高度对应的焦平面附近具有最强清晰度响应。可抽象为：

$$\hat{z}(p)=\arg\max_{i} F(I_i,p),$$

其中 $F(I_i,p)$ 是第 $i$ 个焦平面在像素 $p$ 处的 focus measure。该假设在弱纹理、饱和高光、周期纹理、边界突变和多峰响应处容易失效。真实问题的关键不只在于 DFF 误差大，更在于不同像素的 DFF 可信度差异很大。

因此，DFF 更适合被建模为带噪声的观测：

$$P_{\mathrm{DFF}}(p)=H(p)+\epsilon(p), \quad \epsilon(p)\sim q(C_{\mathrm{focus}}(p),R(p)),$$

其中 $C_{\mathrm{focus}}(p)$ 表示焦点响应置信度，$R(p)$ 表示高光、饱和或局部风险。该视角给 CGP-FocusNet 的训练策略提供了合理性：网络不应均匀追随 DFF/GADFF，而应按局部可靠性调节先验约束。

## 3. 输入表示：让网络同时看到图像、轴向变化和物理先验

当前实现使用 38 个输入通道：

$$X=\mathrm{concat}(I,\Delta I,R,P_{\mathrm{DFF}},C_{\mathrm{DFF}},P_{\mathrm{GADFF}},C_{\mathrm{GADFF}}).$$

通道含义如下：

| 通道组 | 数量 | 含义 | 原理作用 |
|---|---:|---|---|
| $I$ | 17 | 原始焦栈强度层 | 提供真实图像证据 |
| $\Delta I$ | 16 | 相邻焦平面差分 | 捕捉轴向 focus response 变化 |
| $R$ | 1 | glare/risk prior | 标记高光、饱和或不稳定区域 |
| $P_{\mathrm{DFF}}$ | 1 | DFF 深度先验 | 提供可解释的轴向估计 |
| $C_{\mathrm{DFF}}$ | 1 | DFF focus confidence | 衡量 DFF 可靠性 |
| $P_{\mathrm{GADFF}}$ | 1 | GADFF 深度先验 | 提供 glare-aware 轴向估计 |
| $C_{\mathrm{GADFF}}$ | 1 | GADFF focus confidence | 衡量 GADFF 可靠性 |

这套输入设计的价值在于把“深度恢复”拆成三个层次：图像外观、焦平面变化、传统聚焦先验。网络学习的重点从直接拟合完整物理聚焦规律，转向基于已有物理线索进行局部纠偏。

## 4. 训练目标：置信门控的 DFF/GADFF prior consistency

ABL-07 的训练目标可写为：

$$\mathcal{L}=\mathcal{L}_{\mathrm{data}}+0.22\mathcal{L}_{\mathrm{grad}}+0.055\mathcal{L}_{\mathrm{curv}}+0.035\mathcal{L}_{\mathrm{normal}}+0.045\mathcal{L}_{\mathrm{prior}}.$$

其中，合成 GT 数据项 $\mathcal{L}_{\mathrm{data}}$ 保持均匀监督；门控只作用于 DFF/GADFF prior consistency。prior target 为：

$$P_{\mathrm{prior}}=0.45P_{\mathrm{DFF}}+0.55P_{\mathrm{GADFF}}.$$

focus confidence 融合方式为：

$$C_{\mathrm{focus}}=\mathrm{clip}(0.65C_{\mathrm{DFF}}+0.35C_{\mathrm{GADFF}},0,1).$$

prior 权重为：

$$W_{\mathrm{prior}}=\mathrm{clip}\left(C_{\mathrm{focus}}^{1.5}(1-0.45R),0.02,1.0\right).$$

最终 prior consistency 项为：

$$\mathcal{L}_{\mathrm{prior}}=\frac{\sum_p W_{\mathrm{prior}}(p)\rho(\hat{H}(p)-P_{\mathrm{prior}}(p))}{\max(\sum_p W_{\mathrm{prior}}(p),1)}.$$

这里 $\rho(\cdot)$ 是 Charbonnier penalty。这个设计有两个关键点：第一，GT 监督项不因 glare/risk 被直接放大；第二，DFF/GADFF 先验在低置信或高风险区域被软化，减少错误先验对网络的牵引。

## 5. 为什么低置信区域是最清晰的收益来源

合成测试中，低置信区域的 DFF 误差明显高于常规区域。若把 DFF/GADFF 作为均匀约束，网络会在这些区域被错误先验拖偏。CGP-FocusNet 通过 $W_{\mathrm{prior}}$ 降低这类像素的 prior consistency 权重，使模型更多依赖合成 GT、邻域结构和焦栈上下文。

当前两次 full run 的低置信区域结果如下：

| Run | Low-confidence MAE | Gain vs DFF | Win rate |
|---|---:|---:|---:|
| ABL-07 full candidate | 67.02 um | 55.64% | 6/7 |
| ABL-07 seed repeat | 75.94 um | 49.74% | 6/7 |

这说明 CGP-FocusNet 最稳定的改进来自“识别并软化不可靠 prior”，而非单纯增加模型容量。高风险区域虽然也有 ratio-of-means 改善，但 win rate 只有 3/7，因此 glare/risk 更适合写成辅助软化信号，当前不宜写成主要收益来源。

## 6. 真实焦栈证据：支持诊断一致性，真实高度精度仍需标定样本

真实焦栈没有 calibrated height ground truth，因此真实结果不能写成绝对高度误差。当前可支持的结论是：在 focus-margin、spike proxy、saturation persistence 和 morphology 诊断标记的不可靠区域，ABL-07 能减少 DFF 局部波动。

真实焦栈聚合结果：

| Checkpoint | Real stacks | Low-conf dev reduction | Spike-top10 dev reduction | Saturated dev reduction | Confident corr. |
|---|---:|---:|---:|---:|---:|
| full candidate | 7 | 94.85% | 96.25% | 83.16% | 0.5073 |
| seed repeat | 7 | 95.32% | 96.93% | 81.40% | 0.4213 |

这里的 local-deviation reduction 可以表示为：

$$G_{\mathrm{dev}}(M)=\frac{\mathbb{E}_{p\in M}[D_{\mathrm{dev}}^{\mathrm{DFF}}(p)]-\mathbb{E}_{p\in M}[D_{\mathrm{dev}}^{\mathrm{model}}(p)]}{\mathbb{E}_{p\in M}[D_{\mathrm{dev}}^{\mathrm{DFF}}(p)]}.$$

该指标衡量模型是否在诊断不可靠区域抑制 DFF 局部尖峰。它不等价于真实几何误差，只能作为 simulation-to-real alignment 的无参考证据。

## 7. 当前研究主线的更本质表达

可以把本项目写成一条由数据缺口驱动的原理链：

1. 真实工业反光表面缺少可同步的 dense height GT，因此直接监督学习受到限制。
2. 合成焦栈提供可控高度真值，但 synthetic-to-real gap 需要被显式处理。
3. DFF/GADFF 提供跨域较稳定的物理聚焦先验，但先验本身存在局部可靠性差异。
4. CGP-FocusNet 通过 focus confidence 将传统先验转化为可调约束，在低置信区域降低错误先验牵引。
5. 合成 GT 评价证明定量误差改善；低置信分层解释收益来源；真实焦栈无参考诊断验证同一机制在真实样本上具有一致行为。

这条主线比“提出一个网络并取得更低 MAE”更有投稿价值，因为它回答了一个更本质的问题：当真实 GT 不足时，如何利用可控仿真和物理先验构建可验证的深度恢复机制。

## 8. 下一轮最值得做的原理实验

### 8.1 Focus-confidence 可靠性校准

目标：证明 $C_{\mathrm{focus}}$ 与 DFF 误差存在可解释关系。

可做实验：

- 在 synthetic test split 上按 $C_{\mathrm{focus}}$ 分桶；
- 统计每个桶的 DFF MAE、GADFF MAE、CGP-FocusNet MAE；
- 画出 reliability curve：$C_{\mathrm{focus}} \downarrow$ 时 DFF error 是否上升；
- 计算 Spearman correlation 或 AUC，把 confidence 从“启发式指标”提升为“可校准可靠性信号”。

建议输出：

| Bucket | Mean focus confidence | DFF MAE | CGP MAE | Gain |
|---|---:|---:|---:|---:|

若该实验成立，论文中的机制表述会更强：CGP-FocusNet 的 gate 可被解释为基于可观测 reliability signal 的先验调节。

### 8.2 Gate 形状消融

目标：验证当前 $C_{\mathrm{focus}}^{1.5}$ 与 $(1-0.45R)$ 的设计是否合理。

可测变量：

- exponent：$C_{\mathrm{focus}}^\gamma$，取 $\gamma\in\{1.0,1.5,2.0\}$；
- risk coefficient：$1-\lambda R$，取 $\lambda\in\{0,0.25,0.45,0.65\}$；
- prior blend：$P_{\mathrm{prior}}=\alpha P_{\mathrm{DFF}}+(1-\alpha)P_{\mathrm{GADFF}}$，取 $\alpha\in\{0.25,0.45,0.65,0.85\}$。

最小可行方案：先只做 smoke/full-split 小矩阵，不追求一次跑完所有组合。优先看 low-confidence gain 和 high-risk win rate。

### 8.3 真实焦栈可视化诊断面板精选

目标：为论文或答辩提供更直观的机制证据。

建议选 1-2 个真实样本，比如 `钥匙纹路100um` 和 `钥匙尖头50um`，展示：

- 中间焦平面；
- DFF depth；
- CGP-FocusNet depth；
- low-margin map；
- spike proxy；
- morphology class map；
- DFF local deviation 与 model local deviation。

写作重点：展示“诊断不可靠区域的 DFF spike 被抑制”，不写真实高度正确性。

### 8.4 小规模真实标定样本

目标：把真实结果从 diagnostic alignment 推进到 real-height validation。

最低成本方案：

- 选择一个有台阶高度或可测深度的样本；
- 用显微台、轮廓仪、共聚焦或其他可用方式获取 sparse/region-level height reference；
- 只做少量 ROI 的相对高度或 profile curve 对齐；
- 把指标限定为 profile consistency 或 region-level height error。

该实验是当前 evidence chain 的关键缺口。只要有一个小规模标定子集，论文 claim 会明显更稳。

## 9. 当前 claim 边界

可安全表述：

> CGP-FocusNet 在合成 GT 测试中降低高度预测误差，并在真实无 GT 焦栈上与无参考诊断指标保持一致，表现为低置信和 spike-prone 区域的 DFF 局部波动被抑制。

需要谨慎表述：

- ABL-07 是当前内部主候选，但仍需要最终 manuscript-level claim review；
- 两次 seed 的整体 MAE 有差异，说明模型仍有 seed sensitivity；
- high-risk/glare 区域的 win rate 不够稳定，glare cue 当前更适合作为辅助信号。

当前不应表述：

- 把真实无 GT 诊断结果写成定量高度结论；
- 宣称优于尚未复现的外部方法；
- 真实无参考指标等价于几何正确性；
- glare/risk cue 是主要收益来源。

## 10. 下一步断点

下一步最有价值的动作是做 `focus-confidence reliability calibration`。它直接服务于原理主线，可以把当前“低置信区域收益明显”的现象进一步解释为“focus confidence 是有效可靠性信号”。建议产物保持 Markdown 格式：

- `focus_confidence_reliability_calibration_report.md`
- 一张 reliability table；
- 一张 confidence bucket 曲线；
- 一个简短结论：confidence gate 是否有统计支撑。

若时间只够做一件事，优先做这个实验；普通 seed repeat 的边际价值低于 reliability calibration。

## 11. Machine-Readable Claim Boundary

- claim-ineligible for calibrated real-height accuracy.
- real-stack evidence is diagnostic alignment only.
- audit status should be checked again after any manuscript-level merge.
- external baseline superiority remains unsupported until compatible baseline runs are completed.
