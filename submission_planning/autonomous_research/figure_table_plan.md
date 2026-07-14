# Figure and Table Plan

更新日期：2026-06-18  
用途：规划投稿论文中的图表、caption、数据来源和缺失项，确保图表服务论文主线。

## 1. 图表总原则

1. 图表顺序必须跟论文叙事一致：problem -> simulation -> method -> synthetic results -> real validation -> ablation/failure。
2. Synthetic 图表可以报告 absolute error；real 图表只报告 no-reference morphology metrics。
3. 每张图都应回答一个明确问题，避免重复展示多个模型版本。
4. 所有图表中的 final method 统一写作 S2R-FocusNet 或最终确定名称。
5. 外部 baseline 结果未完成前，DFV/DDFFNet 只出现在计划表或 Related Work，不放入结果图。

## 2. 推荐 Figures

### Fig. 1: Simulation-to-Real Focus-Stack Pipeline

| 项目 | 内容 |
|---|---|
| 目的 | 说明本文核心在于 synthetic-to-real 数据与评估路线，并由网络结构承担形貌校正 |
| 内容 | synthetic height map -> rendered focus stack -> DFF/GADFF priors -> focal-difference volume -> S2R-FocusNet training -> real focus-stack inference |
| 数据源 | `output/imagegen/focus_stack_dff_pipeline_v2.png` 可作为基础图 |
| 章节 | Introduction / Method |
| 状态 | 已有图，可进一步按论文术语统一 |

Caption draft:

> Overview of the proposed simulation-to-real focus-stack reconstruction pipeline. Synthetic surfaces with known height maps are rendered into focus stacks for supervised training, while DFF/GADFF priors and focal-difference representations provide interpretable axial cues. Real focus stacks are evaluated with no-reference morphology metrics because calibrated real height ground truth is unavailable.

### Fig. 2: S2R-FocusNet Architecture

| 项目 | 内容 |
|---|---|
| 目的 | 收束多个模型产物，突出一个 final method |
| 内容 | focus-stack frames, DFF/GADFF prior channels, focal-difference channels, glare/high-risk cue, encoder-decoder, residual correction output |
| 数据源 | `output/imagegen/focus-resunet-industrial-network-diagram-final-2048x1152.png` 可作为基础图 |
| 章节 | Method |
| 状态 | 已有图，但需确认是否与最终方法命名一致 |

Caption draft:

> Architecture of the prior-guided focus-stack correction network. The model integrates raw focus-stack information, DFF/GADFF priors, focal-difference features, and glare-aware cues to predict a corrected height map while preserving focus-based physical cues.

### Fig. 3: Synthetic Dataset Examples

| 项目 | 内容 |
|---|---|
| 目的 | 展示 synthetic data engine 的覆盖范围 |
| 内容 | mountain, ridge, step, periodic stripe, V-valley, pit/groove 的 height map 和代表焦堆帧 |
| 数据源 | `论文与PPT制作项目包/06_Samples/simulated_generated_samples/` 与相关 preview 图 |
| 章节 | Method / Experiments |
| 状态 | 待整理 |

Caption draft:

> Representative synthetic surface categories used for supervised focus-stack reconstruction. The synthetic set covers multiple defect geometries, roughness patterns, depth ranges, and glare levels, enabling controlled quantitative evaluation with known height maps.

### Fig. 4: Synthetic Quantitative Comparison

| 项目 | 内容 |
|---|---|
| 目的 | 展示当前内部方法与传统基线的整体对比 |
| 内容 | mean MAE, edge MAE, high-risk MAE bar chart |
| 数据源 | `paper_algorithm_comparison_metrics.csv`, `results/figures/paper_comparison_mean_mae.png` |
| 章节 | Results |
| 状态 | 已有部分图；high-risk / edge 可能需补 |

Caption draft:

> Quantitative comparison on the synthetic test set. The proposed Focus-ResUNet achieves the lowest mean MAE and edge MAE among the evaluated internal baselines, while high-risk regions remain challenging and require further glare-aware modeling.

### Fig. 5: Difficult Synthetic Sample Visual Comparison

| 项目 | 内容 |
|---|---|
| 目的 | 用最难样本解释方法优势和失败边界 |
| 内容 | GT, Original DFF, DFF + post, Lee2013, Li2019, GADFF, Focus-ResUNet, residual variant |
| 推荐样本 | `test_V谷_P10_宽谷粗糙平底`, `test_周期_条纹粗糙` |
| 数据源 | `results/figures/simulation_multisample_algorithm_panel.png` 或项目包对应 comparison panel |
| 章节 | Results / Failure Analysis |
| 状态 | 已有综合图，可挑选困难样本做单图 |

Caption draft:

> Visual comparison on challenging synthetic samples. Traditional DFF-based methods are sensitive to focus ambiguity near defect boundaries and textured regions, whereas the learning-based correction produces smoother relative morphology. Remaining errors in high-risk regions are analyzed separately.

### Fig. 6: Real No-Reference Morphology Comparison

| 项目 | 内容 |
|---|---|
| 目的 | 展示真实样本的形貌稳定性和 spike suppression |
| 内容 | real input frame, DFF result, traditional baselines, Focus-ResUNet result, 3D surface |
| 数据源 | `results/figures/real_midterm_multisample_panel.png`, `assets/figures/readme_showcase/real_*_3d.png` |
| 章节 | Results |
| 状态 | 已有图 |

Caption draft:

> Real focus-stack reconstruction results under no-reference evaluation. Since calibrated real height ground truth is unavailable, the comparison focuses on morphology stability, relative dynamic range, and low-confidence spike suppression rather than absolute height accuracy.

### Fig. 7: Ablation and Failure Analysis

| 项目 | 内容 |
|---|---|
| 目的 | 证明模块贡献并解释失败模式 |
| 内容 | ablation MAE bar chart, profile curves, high-risk mask overlay, spike map |
| 数据源 | 待消融实验生成 |
| 章节 | Ablation / Discussion |
| 状态 | 待实验 |

Caption draft:

> Ablation and failure analysis of the proposed framework. The comparison isolates the effects of DFF/GADFF priors, focal-difference representation, and glare-aware cues, and highlights remaining errors in high-risk reflective regions.

## 3. 推荐 Tables

### Table 1: Synthetic Dataset Split

| 项目 | 内容 |
|---|---|
| 数据源 | `dataset_split.csv` |
| 列 | split, sample count, resolution, stack layers, depth range, z-step, surface types |
| 目的 | 证明数据构造可控且划分清晰 |
| 状态 | 可直接整理 |

### Table 2: Synthetic Quantitative Comparison

| 项目 | 内容 |
|---|---|
| 数据源 | `paper_algorithm_comparison_metrics.csv` |
| 列 | method, type, mean MAE, edge MAE, high-risk MAE, P90 error, notes |
| 目的 | 支撑主要合成定量结果 |
| 状态 | 已有内部方法结果；外部 DFV/DDFFNet 待补 |

### Table 3: Real No-Reference Evaluation

| 项目 | 内容 |
|---|---|
| 数据源 | `real_midterm_method_summary.csv` |
| 列 | method, roughness, edge retention, relative dynamic range, low-conf spike count |
| 目的 | 支撑真实样本稳定性验证 |
| 状态 | 可直接整理 |

### Table 4: Ablation Study

| 项目 | 内容 |
|---|---|
| 数据源 | 待消融实验 |
| 列 | variant, DFF prior, focal difference, glare cue, domain randomization, mean MAE, edge MAE, high-risk MAE |
| 目的 | 证明模块贡献 |
| 状态 | 待实验 |

### Table 5: External Baseline Fairness Checklist

| 项目 | 内容 |
|---|---|
| 数据源 | `baseline_reproduction_decision_table.md`, future baseline logs |
| 列 | method, input frames, training setting, scale alignment, output type, table eligibility |
| 目的 | 处理审稿人对公平比较的质疑 |
| 状态 | 可作为 supplementary 或 experiment notes |

## 4. 图表优先级

| 优先级 | 图表 | 原因 |
|---|---|---|
| P0 | Fig. 1 pipeline | 决定论文故事是否清楚 |
| P0 | Fig. 2 architecture | 收束 final method |
| P0 | Table 2 synthetic comparison | 支撑核心定量结果 |
| P0 | Table 3 real no-reference | 支撑真实样本验证边界 |
| P1 | Fig. 5 difficult sample | 展示方法优势与失败模式 |
| P1 | Table 4 ablation | 支撑方法贡献 |
| P1 | Fig. 7 ablation/failure | 提升论文说服力 |
| P2 | Table 5 fairness checklist | 视投稿格式决定是否加入 |

## 5. 当前缺失图表

| 缺失项 | 影响 | 最小补法 |
|---|---|---|
| 外部 DFV/DDFFNet 对比图 | 影响 SOTA 可信度 | 先补数值表，图可后补 |
| ablation 图表 | 影响模块贡献 | 先用表格，后补曲线/可视化 |
| high-risk mask 可视化 | 影响 glare-aware claim | 选 P10 或周期纹理样本叠加 mask |
| profile curve | 影响形貌连续性说明 | 从 synthetic GT 和预测高度图抽一条横截线 |

## 6. Caption 写作约束

1. Real sample captions must include `no-reference` or explicitly state no calibrated height ground truth.
2. Synthetic captions can mention absolute error only when height GT is used.
3. Depth Anything V2 captions must state it is a single-image auxiliary reference.
4. Ablation captions cannot claim module effectiveness before results exist.
5. Figure labels should use one final method name consistently.
