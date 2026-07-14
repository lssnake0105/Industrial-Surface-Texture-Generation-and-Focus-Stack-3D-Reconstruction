# 论文主张与证据矩阵

更新日期：2026-06-18  
用途：约束投稿论文中的 claim，确保每个主张都能被当前项目证据或后续实验支撑。

## 1. Claim 强度分级

| 等级 | 含义 | 写作方式 |
|---|---|---|
| A | 当前证据可直接支持 | 可写入 Abstract / Introduction / Conclusion |
| B | 当前证据部分支持，需要限定条件 | 可写入 Results / Discussion，并说明边界 |
| C | 当前证据不足，需要新增实验 | 暂写入 Future Work 或计划，不作为主贡献 |
| D | 当前证据不支持 | 不写入论文主张 |

## 2. 可强写主张

| Claim | 等级 | 当前证据 | 可写位置 | 建议表述 |
|---|---|---|---|---|
| 项目构建了一个可控 synthetic focus-stack 数据集，并具有 height ground truth | A | `dataset_split.csv` 中有 split、category、resolution、depth_range_um、stack_layers、z_step_um | Method / Experiments | We construct a controllable synthetic focus-stack dataset with known height maps for quantitative evaluation. |
| 真实样本缺少 calibrated height ground truth | A | 真实结果文件只包含 no-reference metrics；没有真实 height GT 文件 | Experiments / Discussion | Real samples are evaluated with no-reference morphology metrics due to the lack of calibrated height ground truth. |
| Focus-ResUNet 在当前 7 个 synthetic test samples 上取得最低 mean MAE | A | `paper_algorithm_comparison_metrics.csv` 聚合结果：Focus-ResUNet mean MAE = 53.22 um | Results | On the current synthetic test set, Focus-ResUNet achieves the lowest mean MAE among evaluated internal baselines. |
| Focus-ResUNet 在真实 no-reference 指标中显著降低 low-conf spike count | A | `real_midterm_method_summary.csv`：Focus-ResUNet spike count = 2.0，Original DFF = 9179.7 | Results | On real focus stacks, the proposed learning-based reconstruction substantially suppresses low-confidence spikes under no-reference evaluation. |
| Depth Anything V2 适合作为训练策略和单帧先验参考 | A | Depth Anything V2 论文与项目页；输入为单张图，与 focus-stack 主任务不同 | Related Work / Discussion | Monocular foundation depth models provide useful training-strategy references but are not direct focus-stack baselines. |
| DFV 是当前最关键外部 deep DFF 对比 | A | DFV 的 differential focus volume 与本项目 focal-difference 设计高度相关 | Related Work / Experiments | DFV is selected as a primary external baseline because it explicitly models differential focus information. |

## 3. 需谨慎限定的主张

| Claim | 等级 | 当前证据 | 必要限定 | 建议写法 |
|---|---|---|---|---|
| S2R-FocusNet 提升反光工业表面缺陷形貌恢复稳定性 | B | 合成 mean MAE / edge MAE 较优；真实 spike count 较低 | 限定为当前样本与 no-reference 指标 | The method improves morphology stability on the evaluated synthetic and real focus-stack samples. |
| prior-guided correction 优于直接 image-to-depth | B/C | 当前有 Focus-ResUNet 与 TinyDepthNet 对比，但缺少严格 ablation | 需要补 w/o prior 和 direct image-to-depth ablation | Preliminary results suggest the value of prior-guided correction; ablation is needed for a stronger claim. |
| focal-difference volume 是模型性能提升的关键 | B/C | 方法逻辑和 DFV 文献支持，但当前缺少去除该模块的实验 | 需要补 w/o focal-difference ablation | The focal-difference representation is designed to capture axial focus-response variation and will be evaluated through ablation. |
| glare-aware cue 改善 high-risk 区域 | B/C | high-risk 指标存在，但当前 Focus-ResUNet high-risk MAE 不是最优 | 需要谨慎，只能说仍需改进 | High-risk regions remain challenging, motivating more explicit glare-aware modeling. |
| simulation-to-real 训练策略有效 | B | 合成训练 + 真实无参考稳定性可间接支持 | 不写真实绝对精度 | The simulation-trained model shows promising no-reference stability on real focus stacks. |
| 当前方法可用于工业缺陷相对形貌分析 | B | 真实样本可视化和 no-reference 指标支持 | 限定为 relative morphology | The method is suitable for relative morphology visualization under the evaluated conditions. |

## 4. 需要新增实验后才能强写的主张

| Claim | 等级 | 缺失证据 | 最小补强 |
|---|---|---|---|
| proposed method 超过最新 deep DFF SOTA | C | 缺少 DFV/DDFFNet/HybridDepth 实测结果 | 至少复现 DFV 或 DDFFNet |
| 模型模块各自有效 | C | 缺少系统消融 | 补 w/o prior、w/o focal difference、w/o glare cue |
| 方法在真实样本上具有绝对高度精度 | C/D | 缺少真实 calibrated height GT | 采集 step-height / profilometer / confocal 标定样本 |
| domain randomization 提升真实泛化 | C | 缺少 w/o domain randomization 对照 | 补训练策略消融 |
| teacher-student pseudo-label 可以提升真实域效果 | C | 当前只来自 Depth Anything V2 方法论启发 | 后续做真实无标签 self-training prototype |

## 5. 不应写入的主张

| Claim | 等级 | 原因 | 替代表述 |
|---|---|---|---|
| 真实样本 absolute height error 更低 | D | 没有真实 height GT | 真实样本上 low-conf spike count 更低，形貌输出更稳定 |
| Depth Anything V2 是本文主 SOTA 对比 | D | 它是 single-image monocular depth，不使用 focus-stack axial response | Depth Anything V2 作为训练策略和单帧先验参考 |
| 本方法适用于全部工业表面 | D | 当前样本类型有限 | 本方法在评估的反光/弱纹理/缺陷样本上表现出潜力 |
| GADFF 单独优于全部传统方法 | D | 当前平均 MAE 和 edge MAE 不支持 | GADFF 提供 glare-aware prior，但单独使用仍不稳定 |
| 真实 no-reference metrics 等价于真实几何精度 | D | 指标没有真实高度标定 | no-reference metrics 只反映形貌稳定性和视觉一致性 |

## 6. Abstract 可用 claim 组合

安全组合：

1. 真实工业表面缺陷难以获得 calibrated height ground truth。
2. 本文构建可控 synthetic focus-stack 数据，用于 supervised training 和 quantitative evaluation。
3. 本文提出 prior-guided focus-stack reconstruction 框架，结合 DFF/GADFF priors、focal-difference representation 和 learning-based correction。
4. 在 synthetic test set 上，当前模型相对内部传统和学习基线降低 mean MAE。
5. 在真实 focus stacks 上，本文仅进行 no-reference morphology validation，结果显示 spike suppression 和形貌稳定性改善。

需等新增实验后加入：

1. 与 DFV/DDFFNet 等外部 deep DFF baseline 的数值比较。
2. 模块消融结论。
3. domain randomization 或 pseudo-label training 的增益。

## 7. Discussion 必须承认的边界

1. 真实样本缺少 calibrated height ground truth。
2. 当前 high-risk 区域误差仍需进一步优化。
3. 外部 deep DFF 对比尚需复现才能支撑 SOTA claim。
4. synthetic-to-real gap 仍需要通过更多真实样本和标定子集验证。
5. 单目 foundation depth 模型可提供先验，但不能替代 focus-stack 物理线索。

## 8. 投稿前 claim audit

| 检查项 | 通过标准 |
|---|---|
| Abstract 中是否出现 real absolute accuracy | 不出现 |
| Introduction 中是否夸大适用场景 | 限定 reflective / microscopic / evaluated samples |
| Results 中是否区分 synthetic 和 real | 分开写 synthetic quantitative 与 real no-reference |
| Related Work 中是否把 Depth Anything V2 放错位置 | 放在 foundation depth / training strategy |
| Conclusion 中是否声称 SOTA | 只有补外部基线后才能写 strong SOTA claim |
