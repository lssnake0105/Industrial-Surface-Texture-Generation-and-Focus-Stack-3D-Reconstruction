# 面向 Supervisor 的阶段性实验汇报：Simulation-to-Real Focus-Stack 形貌重建

日期：2026-06-19  
汇报定位：从原型项目出发，整理当前轮次自主研究的实验流程、验证结果、主要判断和下一步 prospect。  
证据边界：本文档汇总 `submission_planning/` 与 `tmp/ablation_results/` 中已完成的受保护实验；不声称真实样本 absolute height accuracy，也不把尚未运行的 DFV/DDFFNet 写成已完成 SOTA 对比。

## 摘要

本轮研究将原型中的 focus-stack height reconstruction 进一步整理为一个投稿导向的 **simulation-to-real (S2R) industrial surface morphology reconstruction** 框架。核心策略是用 synthetic height ground truth 解决真实工业反光表面难以标定的问题，并通过 DFF/GADFF prior、focal-difference signal 和 glare-aware cue 引导深度模型学习。

当前已完成 ABL-00/02/03/04 的 matched ablation training、7-sample full-split evaluation、eligibility audit，以及一个 longer-budget repeat。实验最稳定支持的结论是：**DFF/GADFF prior 是当前最关键的物理锚点**；去除该 prior 会显著增加整体误差和 high-risk 区域误差。与此同时，longer repeat 显示 full model 仍未优于 w/o focal difference 和 w/o glare cue，说明下一步重点应从单纯增加训练预算转向 **auxiliary cue fusion / cue quality audit**。

## 1. 从原型到研究问题的更新

原型阶段主要证明：模型可以从焦堆图像输出高度图，并在部分 synthetic / real cases 上表现出较好的形貌稳定性。当前阶段将其重新组织为一个更适合投稿和 supervisor 讨论的研究问题：

> 在缺少真实 calibrated height ground truth 的工业反光表面场景中，能否通过可控仿真数据构造、传统 focus-derived priors 和深度模型修正，实现可解释的 focus-stack 3D morphology reconstruction？

这一重构带来三个关键变化：

1. **Data strategy**：synthetic samples 提供可计算 absolute error 的 ground truth；real samples 只支持 no-reference morphology stability。
2. **Model strategy**：不再把多个模型产物平铺展示，而是把 Focus-ResUNet / S2R-FocusNet 作为主线，其他结构进入 baseline 或 ablation。
3. **Validation strategy**：所有新增实验都有 fixed split、protected runner、full-split evaluator 和 eligibility audit，避免 debug 结果误入论文主表。

## 2. 当前实验流程

本轮实验按以下顺序推进：

1. **Matched smoke**：验证 ABL-00/02/03/04 能在相同 train/validation/test split 边界下完成最小训练链路。
2. **Full candidate configuration**：固定第一轮正式候选训练预算：4 epochs、128 train patches/epoch、32 validation patches/epoch。
3. **Matched evaluator smoke**：新增支持 `checkpoint-tag` 的 evaluator，并用 smoke checkpoint 验证读取、tiled inference 和 metric output。
4. **Full candidate training and evaluation**：运行 4-epoch matched full-candidate training，并在 7 个 test samples 上评估。
5. **Longer-budget repeat**：将预算提高到 8 epochs、192 train patches/epoch、48 validation patches/epoch，判断 full model 未占优是否来自训练不足。
6. **Result interpretation and supervisor update**：基于结果判断哪些模块贡献稳定，哪些应进入下一轮模型设计。

## 3. 实验设置

### 3.1 Ablation Variants

| ID | Variant | 技术含义 |
|---|---|---|
| ABL-00 | Full S2R-FocusNet | 使用 upgraded 38-channel input，包括 focus stack、risk cue、DFF/GADFF prior、focal-difference signal 和 glare cue |
| ABL-02 | w/o DFF/GADFF prior | 去除 DFF、DFF confidence、GADFF、GADFF confidence，测试 focus-derived prior 的作用 |
| ABL-03 | w/o focal difference | 去除 17-32 通道的 adjacent focal-plane difference，测试焦向差分信息的作用 |
| ABL-04 | w/o glare cue | 去除 glare/risk cue，测试反光区域提示信息的作用 |

### 3.2 Training Budget

![Experiment budget timeline](D:/Documents/Desktop/projects/SRTP/submission_planning/autonomous_research/report_figures_2026-06-19/fig2_experiment_budget_timeline.png)

| Stage | Epochs | Train patches/epoch | Validation patches/epoch | Purpose |
|---|---:|---:|---:|---|
| Matched smoke | 1 | 2 | 1 | 验证训练链路可运行 |
| Full candidate | 4 | 128 | 32 | 形成第一轮阶段性消融证据 |
| Longer repeat | 8 | 192 | 48 | 判断 full model 未占优是否来自训练预算不足 |

### 3.3 Test Sample Conditions

本轮 full-split evaluation 覆盖 7 个 synthetic test samples。每个样本都有 synthetic height ground truth，因此可计算 MAE、edge MAE、high-risk MAE 和 P90。

![Test sample conditions](D:/Documents/Desktop/projects/SRTP/submission_planning/autonomous_research/report_figures_2026-06-19/fig6_test_sample_conditions.png)

| Sample ID | Category | Original sample name | Resolution | Depth range um |
|---|---|---|---|---:|
| S1 | P10 V谷-宽谷粗糙平底 | test_V谷_P10_宽谷粗糙平底 | 960x540 | 1200 |
| S2 | A型刃脊-柏林粗糙 | test_A型突起刃脊_柏林粗糙 | 960x540 | 1280 |
| S3 | 山峰-分形粗糙 | test_山峰_分形粗糙 | 640x360 | 1100 |
| S4 | 山脊-柏林粗糙 | test_山脊_柏林粗糙 | 640x360 | 1020 |
| S5 | 阶跃-柏林粗糙 | test_阶跃_柏林粗糙 | 640x360 | 1040 |
| S6 | 周期-条纹粗糙 | test_周期_条纹粗糙 | 640x360 | 920 |
| S7 | 腐蚀凹坑-复合 | test_复合腐蚀凹坑 | 960x540 | 1420 |

## 4. 实验结果

### 4.1 Full Candidate vs Longer Repeat

![Ablation metrics candidate vs repeat](D:/Documents/Desktop/projects/SRTP/submission_planning/autonomous_research/report_figures_2026-06-19/fig3_ablation_metrics_candidate_vs_repeat.png)

| Variant | Candidate MAE um | Longer MAE um | Candidate high-risk MAE um | Longer high-risk MAE um |
|---|---:|---:|---:|---:|
| Full S2R-FocusNet | 130.9028 | 109.2209 | 117.9743 | 86.6455 |
| w/o DFF/GADFF prior | 245.3440 | 133.4808 | 261.3550 | 121.3387 |
| w/o focal difference | 113.1038 | 90.4542 | 103.4093 | 57.2526 |
| w/o glare cue | 111.8795 | 75.4572 | 110.4368 | 60.7381 |

### 4.2 Longer-Budget Improvement

![Longer repeat improvement](D:/Documents/Desktop/projects/SRTP/submission_planning/autonomous_research/report_figures_2026-06-19/fig4_longer_repeat_improvement.png)

Longer repeat 降低了所有变体的 Mean MAE，说明训练预算确实影响模型收敛。但排序没有发生预期反转：Full S2R-FocusNet 仍没有超过 w/o focal difference 和 w/o glare cue。这个现象把下一步问题从 “training budget” 推向 “auxiliary signal fusion”。

### 4.3 Per-Sample Evaluation

![Per-sample longer repeat heatmap](D:/Documents/Desktop/projects/SRTP/submission_planning/autonomous_research/report_figures_2026-06-19/fig5_per_sample_longer_repeat_heatmap.png)

Per-sample heatmap 显示，w/o glare cue 在多数样本上具有更低 MAE。这说明 glare cue 当前可能存在两类问题：

1. glare/risk cue 本身与真实误差区域不完全一致；
2. full model 采用直接 channel concatenation 时，网络可能把 noisy cue 当成强监督信号，从而影响泛化。

## 5. 主要研究判断

### 5.1 DFF/GADFF prior 是当前最稳定的贡献

在 full candidate 中，移除 DFF/GADFF prior 后 Mean MAE 从 130.9028 um 上升到 245.3440 um；在 longer repeat 中，移除 prior 后 Mean MAE 仍从 109.2209 um 上升到 133.4808 um。high-risk MAE 也保持同方向恶化。

这说明 focus-derived prior 不只是辅助输入，而是当前小数据 S2R setting 下的关键物理锚点。它降低了网络从原始焦堆直接学习高度映射的负担，也让论文中 “prior-guided correction” 的主线更有支撑。

### 5.2 Full model 当前没有稳定整合所有 auxiliary cues

Longer repeat 已经排除了“训练预算过短”这一单一解释。full model 有明显改善，但 w/o focal difference 和 w/o glare cue 仍更强。当前更合理的解释是：focal-difference signal 和 glare cue 的信息质量或融合方式尚未被模型稳定利用。

因此，focal-difference 和 glare cue 不应在当前汇报中被表述为已验证的正贡献。更合适的表达是：这些信号具有物理动机，但需要 **learnable gating, confidence weighting, cue denoising, or high-risk loss redesign**。

### 5.3 目前的投稿风险和机会

当前结果已经足以支持一次高质量 supervisor discussion：项目从原型走向了有证据边界的研究框架，并完成了内部消融闭环。主要风险是 full model 作为最终方法仍需改进；主要机会是 DFF/GADFF prior 的贡献非常清楚，可以作为论文主线的稳定支点。

## 6. 面向 Supervisor 的建议表达

可以这样汇报：

> I reorganized the prototype into a simulation-to-real focus-stack reconstruction study with a fixed synthetic split, protected ablation runner, full-split evaluator, and eligibility audit. The most stable result is that DFF/GADFF priors are important: removing them consistently increases both overall MAE and high-risk error. I also tested a longer-budget repeat. It improves all variants, but the full model is still not the best, which suggests the current issue is more about auxiliary cue fusion and cue quality than training budget alone. My next step would be to redesign the full model with gated fusion or confidence weighting, while continuing the DFV/DDFFNet external SOTA comparison.

## 7. 下一步 Prospect

| Direction | Why | Concrete next step |
|---|---|---|
| Gated auxiliary fusion | full model 未能稳定利用 focal-difference / glare cue | 将 fixed concatenation 改为 learnable channel gate 或 attention fusion |
| Glare cue quality audit | w/o glare cue 当前最优，提示 glare cue 可能有噪声 | 分析 glare cue 与 error map / high-risk mask 的相关性 |
| Seed repeat | 判断 ABL-03/04 排序是否稳定 | 对 ABL-00、ABL-03、ABL-04 做至少 2 个 seeds |
| External SOTA | 投稿需要外部 deep DFF 对比 | 优先推进 DFV，再做 DDFFNet |
| Real validation | 强化 S2R 可信度 | 保持 no-reference 指标，同时争取 calibrated real subset |

## 8. 当前产物索引

| Artifact | Path |
|---|---|
| longer repeat result note | `submission_planning/autonomous_research/ablation_matched_longer_repeat_results.md` |
| supervisor update note | `submission_planning/autonomous_research/supervisor_update_2026-06-19.md` |
| full candidate result note | `submission_planning/autonomous_research/ablation_matched_full_candidate_results.md` |
| training summary | `tmp/ablation_results/training_runner_matched_longer_repeat/2026-06-19_matched_training_longer_repeat_summary.md` |
| evaluation summary | `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_longer_repeat_eval/2026-06-19_matched_longer_repeat_eval_summary.md` |
| report figures | `submission_planning/autonomous_research/report_figures_2026-06-19/` |

## 9. 证据边界

1. 当前 synthetic test split 可支持 absolute error comparison。
2. 当前结果不支持真实样本 absolute height accuracy。
3. 当前结果支持 DFF/GADFF prior 的阶段性贡献。
4. 当前结果不支持 focal-difference / glare cue 已经稳定提升 full model。
5. 当前结果尚未包含 DFV/DDFFNet 外部 deep DFF 数值对比。
