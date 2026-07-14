# Supervisor Update: From Prototype to Submission-Oriented S2R Focus-Stack Reconstruction

日期：2026-06-19  
用途：面向 supervisor 的阶段性 update，聚焦项目从原型到可投稿研究的改进点、当前实验验证成果和下一步 prospect。

## 1. One-Sentence Update

我已将原型项目推进为一个以 simulation-to-real 为主线的 focus-stack 3D morphology reconstruction 研究框架，并完成了受保护的 matched ablation training pipeline 与 longer-budget repeat；当前实验确认 DFF/GADFF prior 是最关键的稳定信号，同时更明确地显示 full model 对 focal-difference 与 glare cue 的融合方式需要重新设计。

## 2. From Prototype to Research Story

原型阶段主要证明模型可以从 focus stack 输出相对稳定的高度图。现在的研究故事被收束为：真实工业反光表面的 calibrated height GT 难以获取，因此先用仿真构造可控 ground truth，再用传统 DFF/GADFF prior、focal-difference volume 和 glare-aware cue 引导神经网络做 morphology correction，最后用真实样本 no-reference morphology 指标验证 sim-to-real 的可迁移性。

这个转变的关键是把项目从“模型效果展示”推进到“数据构造、物理先验、训练策略和证据边界”的研究框架。当前所有新增实验产物都限制在 `submission_planning/` 和 `tmp/`，没有改写原型源码或交付包。

## 3. Main Improvements Since the Prototype

| Improvement | What Changed | Why It Matters |
|---|---|---|
| Research framing | 将核心叙事改为 simulation-to-real focus-stack reconstruction | 解释真实 GT 不足的研究动机，并让仿真标签成为方法论核心 |
| Data/evidence boundary | 明确 synthetic GT 支持 absolute error，real samples 只支持 no-reference morphology stability | 降低投稿中真实高度精度 claim 的审稿风险 |
| Model story | 收束为一个 final prior-guided Focus-ResUNet/S2R-FocusNet 框架 | 避免多个模型产物分散主线，其他模型转为 baseline 或 ablation |
| Protected experiment pipeline | 新增 ablation runner、matched split、evaluator、eligibility audit | 实验可复现、可恢复，并能防止 debug 结果误入论文主表 |
| SOTA strategy | 建立 DFV/DDFFNet adapter、evaluation 和 eligibility gate | 为后续补充外部 deep DFF 对比预留可执行路径 |

## 4. Current Experimental Validation

### 4.1 Matched Ablation Training

已完成 ABL-00/02/03/04 的 matched full-candidate training：

| Item | Value |
|---|---|
| train / validation / test split | 27 / 10 / 7 |
| training budget | 4 epochs, 128 train patches/epoch, 32 validation patches/epoch |
| device | cuda |
| training status | pass |
| training checks | 52 checks, 0 errors, 0 warnings |
| full test evaluation | 7 test samples, 28 per-sample rows, 4 summary rows |
| eligibility audit | pass, 104 checks, 0 errors, 0 warnings |

随后已完成 longer-budget repeat：8 epochs、192 train patches/epoch、48 validation patches/epoch，仍使用相同 27/10/7 split。该 repeat 用于验证 full model 未占优是否主要来自短训练预算。

### 4.2 Ablation Results

| Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um | Interpretation |
|---|---:|---:|---:|---|
| Full S2R-FocusNet | 130.9028 | 183.7727 | 117.9743 | 当前 full candidate baseline |
| w/o DFF/GADFF prior | 245.3440 | 233.6410 | 261.3550 | prior 移除后显著退化 |
| w/o focal difference | 113.1038 | 161.9272 | 103.4093 | 当前短预算下优于 full，提示融合或训练策略需复核 |
| w/o glare cue | 111.8795 | 155.0823 | 110.4368 | 当前短预算下优于 full，提示 glare cue 可能引入噪声或需 gating |

### 4.3 Longer-Budget Repeat

| Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um | Interpretation |
|---|---:|---:|---:|---|
| Full S2R-FocusNet | 109.2209 | 153.0310 | 86.6455 | 更长训练后明显改善 |
| w/o DFF/GADFF prior | 133.4808 | 181.9107 | 121.3387 | 仍明显弱于 full，prior 贡献稳定 |
| w/o focal difference | 90.4542 | 158.5932 | 57.2526 | 仍优于 full，说明 focal-difference 融合需重审 |
| w/o glare cue | 75.4572 | 126.8816 | 60.7381 | 当前最优，说明 glare cue 可能引入噪声或需要 gating |

### 4.4 What the Result Supports

最可靠的结论是：DFF/GADFF prior 对任务非常重要。去掉 DFF/GADFF 后，Mean MAE 几乎翻倍，high-risk MAE 也显著变差。这直接支持论文主线中的 prior-guided correction 设计。

更需要讨论的结论是：longer-budget repeat 已经降低了所有变体误差，但没有让 full model 成为最优。因此问题更像辅助信号融合和 cue quality，而非单纯训练不足。focal-difference 和 glare cue 需要更好的融合方式、confidence gate、loss 权重或 denoising 机制。

## 5. Suggested Message to Supervisor

可以用下面这段作为口头 update：

> I have reorganized the project from a prototype demonstration into a submission-oriented simulation-to-real study. The current pipeline now has a fixed synthetic split, protected ablation training, full test-split evaluation, and eligibility auditing. The ablation results consistently show that DFF/GADFF priors are important: removing them increases both overall MAE and high-risk error. I also ran a longer-budget repeat, which improved all variants but did not make the full model the best. This suggests the issue is not only training budget; the focal-difference and glare cues likely need better fusion, gating, or confidence weighting before we report them as settled contributions. My next step is to refine the full model and continue the DFV/DDFFNet external SOTA comparison.

## 6. Prospect

| Prospect | Goal | Next Action |
|---|---|---|
| Seed repeat | 判断当前排序是否稳定 | 至少 2 seeds，优先 ABL-00/03/04 |
| Better auxiliary-signal fusion | 让 focal-difference / glare cue 以更稳的方式进入模型 | 尝试 channel gate、attention fusion 或 confidence weighting |
| High-risk loss refinement | 改善 edge/glare/low-texture 区域 | 增加 high-risk mask weighting 或 profile consistency loss |
| External SOTA comparison | 提升投稿说服力 | 优先跑 DFV，随后 DDFFNet，使用已建立的 batch evaluator |
| Real-domain validation | 强化 sim-to-real | 保持 no-reference metrics，争取一个 calibrated real subset |

## 7. Discussion Questions for Supervisor

1. 是否同意把 DFF/GADFF prior 作为当前最强、最稳的模块贡献？
2. focal-difference 和 glare cue 应先作为 pending refinement 继续实验，还是在论文中降级为 exploratory component？
3. 下一阶段是否优先做 gated fusion / seed repeat，还是先补 DFV/DDFFNet 外部 SOTA？
4. 是否有机会补一个 calibrated real sample，用于增强真实域说服力？
