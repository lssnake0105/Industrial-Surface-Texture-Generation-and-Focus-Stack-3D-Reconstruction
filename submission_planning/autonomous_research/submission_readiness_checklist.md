# 投稿准备检查清单

更新日期：2026-06-18  
目标：判断当前课题从项目报告推进到投稿论文还缺哪些证据、实验和写作材料。

## 1. 论文主线检查

| 项目 | 状态 | 证据 / 缺口 |
|---|---|---|
| 明确任务定义 | 基本完成 | 建议统一为 reflective surface defect morphology reconstruction from focus stacks |
| 单一最终方法 | 待收束 | 建议统一命名 S2R-FocusNet；TinyDepthNet、Residual Focus-ResUNet 放入 ablation 或 development history |
| Simulation-to-real 叙事 | 基本完成 | 需要把仿真作为数据引擎写进 Introduction 和 Method |
| 真实样本边界 | 完成但需强调 | 真实样本没有 calibrated height GT，只能报告 no-reference metrics |
| 外部 SOTA 对比 | 未完成 | 至少补 DFV 和 DDFFNet；若时间允许补 HybridDepth 或 DfF in the Wild |

## 2. 实验完整性检查

### 2.1 已具备

| 证据 | 当前文件 | 可支持内容 |
|---|---|---|
| synthetic train/validation/test split | `论文与PPT制作项目包/03_Data/synthetic_training/dataset_split.csv` | 数据构造、实验划分 |
| synthetic quantitative metrics | `论文与PPT制作项目包/03_Data/algorithm_comparison/paper_algorithm_comparison_metrics.csv` | 传统方法与项目内模型对比 |
| real no-reference summary | `论文与PPT制作项目包/03_Data/real_sample_comparison/real_midterm_method_summary.csv` | 真实样本稳定性对比 |
| method and pipeline figures | `results/figures/`, `assets/figures/`, `output/imagegen/` | 论文图示与 README 展示 |
| SOTA planning | `submission_planning/sota_comparison_chinese.md` | Related Work 和实验计划 |

### 2.2 仍缺

| 缺口 | 最小可接受补法 | 更强补法 |
|---|---|---|
| 最新 deep DFF 外部对比 | DFV + DDFFNet 至少一个跑通 | DFV + DDFFNet + HybridDepth / DfF in the Wild |
| 消融实验 | full, w/o prior, w/o focal difference, w/o glare cue | 加 w/o domain randomization, unbounded prediction |
| 真实样本定量 GT | 在文中明确无 GT 边界 | 采集 step-height / profilometer / confocal 标定子集 |
| 仿真真实性说明 | 描述 geometry, roughness, texture, reflectance, glare, focus response | 加 domain randomization 表和 synthetic-real gap 可视化 |
| 失败分析 | 说明 high-risk 区域仍存在误差 | 按 surface type、risk mask、edge mask 分组分析 |

## 3. SOTA 引用检查

| 类别 | 必须覆盖 | 当前建议 |
|---|---|---|
| Classical SFF/DFF | Nayar, Pertuz, Lee2013, Li2019 | 作为传统基线背景 |
| Deep DFF | DDFFNet, DFV, AiFDepthNet | DFV 是最关键外部对比 |
| Real-world / sim-to-real DFF | DfF in the Wild, DDFS, Focus on Defocus, DEReD | 支撑真实域和无 GT 训练策略 |
| 2025-2026 trend | HybridDepth, FAD, DualFocus, DDL-Recurrent SFF, Minimal Focal Stack | 写入 Related Work / Discussion |
| Foundation depth | Depth Anything V2, Marigold, ZoeDepth, Metric3D | 只作辅助背景和训练策略论据 |

## 4. 审稿风险与应对

| 风险 | 严重度 | 应对 |
|---|---|---|
| 缺少最新 deep DFF 对比 | 高 | 优先补 DFV；DDFFNet 作为基础 deep baseline |
| 真实样本没有 GT | 高 | 明确 no-reference validation；把 calibrated GT 放入 future work 或新增小样本 |
| 多个模型版本造成贡献分散 | 中-高 | 收束到 S2R-FocusNet，其余进入 ablation |
| high-risk 区域不是最优 | 中 | 作为 limitation，并用 glare-aware loss / confidence calibration 作为后续方向 |
| Depth Anything V2 被误用为主基线 | 中 | 只用于 auxiliary qualitative 和训练策略讨论 |
| 仿真域与真实域差距 | 高 | 增加 domain randomization 和 synthetic-real gap 分析 |

## 5. 投稿前必须完成项

| 编号 | 必须完成项 | 判定标准 |
|---|---|---|
| M1 | 统一最终方法名称 | 全文只把 S2R-FocusNet 或 Prior-Guided Focus-ResUNet 作为 final method |
| M2 | 明确真实样本无 GT | Abstract、Experiments、Discussion 均不宣称真实 absolute error |
| M3 | 补外部 SOTA 对比 | 至少 DFV 或 DDFFNet 一个外部 deep DFF 方法进入主表 |
| M4 | 补消融表 | 至少包含 w/o prior、w/o focal difference、w/o glare cue |
| M5 | 更新 Related Work | 覆盖 2025-2026 最新方法和 Depth Anything V2 |
| M6 | 补公平比较说明 | 说明输入帧数、训练设置、尺度对齐、输出单位 |
| M7 | 补 failure analysis | 至少分析 P10 V-valley、periodic stripe、real glare/texture 样本 |

## 6. 可推迟项

| 项目 | 可推迟原因 |
|---|---|
| 完整真实 calibrated GT | 需要额外设备和采集流程，可作为后续扩展 |
| 所有最新 SOTA 复现 | FAD、DualFocus、DDL-Recurrent SFF 可能暂无易用代码 |
| 大规模真实无标签自训练 | 工作量较大，适合下一版论文 |
| 端到端系统部署评估 | 当前投稿核心在 reconstruction method 与 sim-to-real 论证 |

## 7. 论文表述红线

1. 不写“真实样本绝对高度误差更低”。
2. 不把 Depth Anything V2 写成 focus-stack SOTA 主基线。
3. 不把多个项目内模型都写成同等级贡献。
4. 不把 synthetic test set 的结论泛化到全部工业表面。
5. 不把 no-reference 指标解释为真实几何精度。

## 8. 当前 readiness 结论

当前项目已经具备投稿雏形：问题、数据构造、内部方法、合成定量结果和真实无参考验证都存在。最主要短板是外部 deep DFF SOTA 对比与消融实验。若能补齐 DFV/DDFFNet 中至少一个强基线，并完成三个核心消融，稿件可以进入初投稿准备阶段。若目标是更稳的期刊版本，建议继续补真实标定子集和更完整的 domain gap 分析。
