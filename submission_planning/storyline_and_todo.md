# SRTP 投稿故事线与 TODO / SRTP Manuscript Storyline and TODO

更新日期 / Updated: 2026-06-18

## 推荐故事线 / Recommended Storyline

### 工作标题 / Working Title

**中文：** 面向反光表面缺陷形貌的 Simulation-to-Real 焦栈三维重建：基于先验引导的深度校正方法

**English:** Simulation-to-Real Focus-Stack Reconstruction for Reflective Surface Defect Morphology Using Prior-Guided Deep Correction

### 一句话主张 / One-Sentence Claim

**中文：** 本研究提出一个面向反光工业表面缺陷的 Simulation-to-Real 焦栈重建框架，结合物理启发的仿真数据、DFF/GADFF 先验、焦向差分表示和紧凑型深度校正模型，在眩光、弱纹理和聚焦歧义条件下提升相对三维形貌恢复的稳定性。

**English:** This work proposes a simulation-to-real focus-stack reconstruction framework for reflective industrial surface defects, combining physics-inspired synthetic data, DFF/GADFF priors, focal-difference representation, and a compact deep correction model to improve relative 3D morphology recovery under glare, weak texture, and focus ambiguity.

### 核心叙事 / Core Narrative

**中文：** 论文应围绕工业表面形貌重建中的数据瓶颈展开。真实样本很难获得校准高度真值，尤其是在反光材料、微尺度缺陷和非均匀纹理场景中。因此，仿真不应只是补充实验，而应作为使监督学习和可控评估成为可能的核心数据引擎。

**English:** The paper should be framed around the data bottleneck in industrial surface morphology reconstruction. Real samples are difficult to annotate with calibrated height ground truth, especially under reflective materials, micro-scale defects, and uneven texture. Therefore, simulation should be presented as the central data engine that makes supervised learning and controlled evaluation possible.

**中文技术主线：**

1. 传统 DFF 提供可解释的聚焦几何线索，但在复杂表面上较脆弱。
2. 真实校准高度标签稀缺，合成焦栈成为可控真值来源。
3. 仿真质量取决于能否正确拆解表面因素：缺陷几何、粗糙度、材料反射、眩光、纹理和焦向响应。
4. 先验引导网络应学习对 DFF/GADFF 输出进行校正，而非完全从图像直接学习深度。
5. 真实样本用于验证形貌一致性和工程可行性，合成样本用于支持绝对误差评估。

**English technical story:**

1. Traditional DFF provides interpretable but fragile focus-based geometry cues.
2. Real calibrated depth labels are scarce, so synthetic focus-stack generation becomes the controllable source of ground truth.
3. Simulation quality depends on whether surface factors are decomposed properly: defect geometry, roughness, material reflectance, glare, texture, and focal response.
4. A prior-guided network should learn correction over DFF/GADFF outputs, rather than learning depth from images alone.
5. Real samples should validate morphology consistency and engineering plausibility, while synthetic samples support absolute error metrics.

### 推荐定位 / Recommended Positioning

**中文：** 最强定位是：

**面向反光表面缺陷形貌的 Simulation-to-Real、先验引导焦栈重建方法。**

该定位与当前证据匹配，因为项目已经具备合成真值、真实焦栈样本、DFF/GADFF 与传统自适应基线，以及学习型校正模型。同时，真实样本缺少校准高度真值这一边界也比较清晰。

**English:** The strongest positioning is:

**A simulation-to-real, prior-guided focus-stack reconstruction method for reflective surface defect morphology.**

This positioning fits the current evidence because the project already has synthetic ground truth, real focus-stack samples, DFF/GADFF and traditional adaptive baselines, and learning-based correction models. It also has a clear experimental boundary: real samples lack calibrated absolute height labels.

### 需要避免的表述 / Claims to Avoid

**中文：**

- 不要声称真实样本上实现了绝对计量精度，除非加入校准高度真值。
- 真实样本结果应使用相对形貌、视觉一致性、结构连续性、尖峰抑制、工程可行性等表述。
- 不要把多个网络变体都包装成最终贡献；最终故事应收敛到一个主模型，其余模型作为基线或消融。

**English:**

- Avoid claiming absolute metrology accuracy on real samples unless calibrated height ground truth is added.
- For real samples, use terms such as relative morphology, visual consistency, structural continuity, spike suppression, and engineering plausibility.
- Avoid presenting multiple network variants as separate final contributions. The final story should converge to one main model, with other models treated as baselines or ablations.

## 贡献结构 / Contribution Structure

### 贡献 1：基于仿真的真值构造 / Contribution 1: Simulation-Based Ground Truth Construction

**中文：** 构建一个可控的合成焦栈数据引擎，为缺陷表面生成已知高度图，并强调仿真用于缓解真实校准高度标签稀缺的问题。

**English:** Build a controllable synthetic focus-stack data engine for defect surfaces with known height maps, and emphasize that simulation is used to overcome the scarcity of calibrated real height labels.

### 贡献 2：先验引导的焦栈重建 / Contribution 2: Prior-Guided Focus-Stack Reconstruction

**中文：** 将 DFF/GADFF 结果、相邻焦向差分和原始焦栈帧作为互补线索，使模型学习在可靠区域保留传统聚焦估计，在失效区域进行校正。

**English:** Use DFF/GADFF results, adjacent focal differences, and original focus-stack frames as complementary cues, allowing the model to learn when to preserve traditional focus estimates and when to correct failure regions.

### 贡献 3：Synthetic-to-Real 评估协议 / Contribution 3: Synthetic-to-Real Evaluation Protocol

**中文：** 将评估分为两层：合成数据上的绝对 MAE 评估，以及真实数据上的无真值形貌指标评估。这样可以明确实验边界，使论证更稳健。

**English:** Separate evaluation into two layers: synthetic data with absolute MAE and real data with no-reference morphology indicators. This makes the experimental boundary explicit and defensible.

### 贡献 4：工业表面缺陷案例研究 / Contribution 4: Industrial Surface Defect Case Study

**中文：** 展示反光、弱纹理和复杂缺陷表面对直接 DFF 的挑战，并说明先验引导校正如何提升代表性缺陷样本中的形貌稳定性。

**English:** Show that reflective and low-texture surface defects expose the limitations of direct DFF, and that prior-guided correction improves morphology stability in representative defect cases.

## 模型精简方案 / Model Simplification Plan

### 最终模型命名 / Final Model Name

**中文：** 建议统一使用一个最终模型名称：

**S2R-FocusNet**

如果希望名称更描述性，也可以使用：

**Prior-Guided Focus-ResUNet**

**English:** Use one final model name consistently:

**S2R-FocusNet**

Alternative if a more descriptive name is preferred:

**Prior-Guided Focus-ResUNet**

### 推荐模型层级 / Recommended Model Hierarchy

**中文最终方法：**

- S2R-FocusNet / Prior-Guided Focus-ResUNet

**English final method:**

- S2R-FocusNet / Prior-Guided Focus-ResUNet

**中文基线：**

- Original DFF
- Original DFF + post-processing
- GADFF
- Lee2013 adaptive window
- Li2019 adaptive iteration
- DDFFNet / DFV-style external baseline if reproducible

**English baselines:**

- Original DFF
- Original DFF + post-processing
- GADFF
- Lee2013 adaptive window
- Li2019 adaptive iteration
- DDFFNet / DFV-style external baseline if reproducible

**中文消融：**

- 去除 DFF/GADFF 先验
- 去除焦向差分通道
- 去除眩光感知先验
- 去除 domain randomization
- 去除后处理或残差保护机制

**English ablations:**

- without DFF/GADFF priors
- without focal-difference channels
- without glare-aware prior
- without domain randomization
- without post-processing or residual protection

**中文：** TinyDepthNet 和 residual variants 应作为内部消融或开发历史保留，除非它们能形成清晰的实验对照点。

**English:** TinyDepthNet and residual variants should be kept as internal ablations or development history, unless they provide a clean experimental comparison point.

## Related Work 矩阵 / Related Work Matrix

| 主题 / Theme                               | 代表文献 / Representative Works                                                     | 与本项目关系 / Relation to This Project                                                                       | 论文中用途 / Use in Manuscript                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Classical SFF/DFF                        | Nayar and Nakagawa 1994; Pertuz et al. 2013; Lee et al. 2013; Li et al. 2019    | 建立聚焦测度、自适应窗口和传统 DFF 局限 / Establishes focus measures, adaptive windowing, and traditional DFF limits     | 背景与传统基线 / Background and traditional baselines                    |
| Learning-Based Depth from Focus          | DDFFNet; AiFDepthNet; DFV; Learning Depth from Focus in the Wild; DDFS          | 说明深度模型可以利用焦栈和 focus-volume 线索 / Shows that deep models can exploit focal stacks and focus-volume cues   | 主要技术比较组 / Main technical comparison group                         |
| Defocus / Sim-to-Real Depth Learning     | Focus on Defocus; DEReD; Dr.Bokeh; Depth Anything; Depth Anything V2            | 支撑 synthetic-to-real、伪标签和数据规模化叙事 / Supports synthetic-to-real, pseudo-label, and data scaling arguments | Related work 与 discussion / Related work and discussion           |
| Industrial Surface Defect Reconstruction | Surface defect detection reviews; Att-PU-Net structured illumination microscopy | 将深度重建连接到工业缺陷形貌场景 / Connects depth reconstruction to industrial defect morphology                        | 应用动机与未来对比 / Application motivation and future comparison          |
| Large-Scale Depth Foundation Models      | Depth Anything; Depth Anything V2                                               | 说明数据规模和伪监督能提升深度泛化 / Shows that data scale and pseudo-supervision improve depth generalization           | 概念支撑，不作为直接 DFF 基线 / Conceptual support, not a direct DFF baseline |

## Depth Anything 的放置方式 / Depth Anything Placement

**中文：** Depth Anything 适合用来说明大规模未标注数据和 teacher-student 伪标签可以提升深度模型泛化能力。它不适合作为本项目的直接 SOTA 基线，因为它处理的是单张 RGB 图像的单目相对深度估计，而本项目处理的是焦栈输入下的表面形貌重建。

**English:** Depth Anything should be cited as evidence that large-scale unlabeled data and teacher-student pseudo-labeling can improve depth model generalization. It should not be used as a direct SOTA baseline for this project because it estimates monocular relative depth from a single RGB image, while this project reconstructs surface morphology from focus stacks.

**建议写法 / Suggested manuscript wording:**

> Recent monocular depth foundation models such as Depth Anything demonstrate that large-scale unlabeled images and teacher-student pseudo-labeling can substantially improve depth generalization. However, monocular relative-depth estimation does not directly address focus-stack-based surface morphology reconstruction, where axial focus response, material reflectance, glare, and calibrated height supervision are central. This motivates a task-specific simulation-to-real strategy for focus-stack reconstruction.

## 实验 TODO / Experiment TODO

### 优先级 1：明确最终方法 / Priority 1: Clarify Final Method

**中文：**

- 统一重命名最终模型，并同步到论文、图、表和 README。
- 在 S2R-FocusNet 和 Prior-Guided Focus-ResUNet 中选择一个。
- 将 TinyDepthNet 和 residual Focus-ResUNet 移入消融或 appendix。
- 更新方法图，使其展示一个最终模型管线。

**English:**

- Rename the final model consistently across paper, figures, tables, and README.
- Choose either S2R-FocusNet or Prior-Guided Focus-ResUNet.
- Move TinyDepthNet and residual Focus-ResUNet into ablation or appendix.
- Update the method diagram to show one final model pipeline.

### 优先级 2：强化 Simulation-to-Real 故事 / Priority 2: Strengthen Simulation-to-Real Story

**中文：**

- 增加一节说明真实高度真值为什么难以获得或成本高。
- 描述合成数据生成因素：缺陷几何、粗糙度、纹理、反射、眩光、焦向响应和噪声。
- 增加 synthetic-real domain gap 分析段落。
- 增加 synthetic-to-real 流程图：synthetic height map -> rendered focus stack -> DFF/GADFF priors -> model training -> real focus-stack inference。

**English:**

- Add a subsection explaining why real ground truth is unavailable or expensive.
- Describe the synthetic data generation factors: defect geometry, roughness, texture, reflectance, glare, focus response, and noise.
- Add a domain gap paragraph comparing synthetic and real samples.
- Add a figure showing the synthetic-to-real flow: synthetic height map -> rendered focus stack -> DFF/GADFF priors -> model training -> real focus-stack inference.

### 优先级 3：增加消融实验 / Priority 3: Add Ablation Experiments

**中文最小消融：**

- full model
- without focal-difference channels
- without DFF/GADFF prior channels
- without glare-aware prior
- without domain randomization

**English minimum ablations:**

- full model
- without focal-difference channels
- without DFF/GADFF prior channels
- without glare-aware prior
- without domain randomization

**中文至少报告：**

- average MAE
- edge MAE
- high-risk region MAE
- P10 V-valley 或其他困难样本的可视化对比

**English report at least:**

- average MAE
- edge MAE
- high-risk region MAE
- visual comparison on P10 V-valley or another difficult sample

### 优先级 4：更新 SOTA 对比 / Priority 4: Update SOTA Comparison

**中文应纳入文献：**

- DDFFNet: Deep Depth from Focus
- AiFDepthNet: all-in-focus supervision
- DFV: Differential Focus Volume
- Learning Depth from Focus in the Wild
- DDFS: camera-setting-invariant defocus model
- Focus on Defocus
- DEReD
- Depth Anything and Depth Anything V2 as general depth foundation references
- Att-PU-Net structured illumination microscopy for industrial 3D defect reconstruction

**English literature to include:**

- DDFFNet: Deep Depth from Focus
- AiFDepthNet: all-in-focus supervision
- DFV: Differential Focus Volume
- Learning Depth from Focus in the Wild
- DDFS: camera-setting-invariant defocus model
- Focus on Defocus
- DEReD
- Depth Anything and Depth Anything V2 as general depth foundation references
- Att-PU-Net structured illumination microscopy for industrial 3D defect reconstruction

**中文实现优先级：**

1. 如果可行，优先复现或比较 DFV。
2. 如果代码和数据适配可控，保留 DDFFNet 作为学习型焦栈基线。
3. Depth Anything 仅作为概念相关文献，不作为公平数值基线。

**English implementation priority:**

1. Reproduce or compare against DFV if feasible.
2. Keep DDFFNet as a learning-based focal-stack baseline if code/data adaptation is manageable.
3. Use Depth Anything only as a qualitative conceptual reference, not a fair numerical baseline.

### 优先级 5：改进真实样本评估 / Priority 5: Improve Real-Sample Evaluation

**中文：** 当前真实样本结果应表述为无真值评估。建议补充或细化：

- roughness stability
- relative dynamic range
- spike count
- edge continuity or edge correlation
- defect region profile curves

**English:** Current real-sample results should be framed as no-reference evaluation. Add or refine:

- roughness stability
- relative dynamic range
- spike count
- edge continuity or edge correlation
- visual profile curves across defect regions

**中文未来改进：**

- 采集一个校准 step-height 样本。
- 使用轮廓仪、共聚焦显微镜、白光干涉仪获取小规模真实真值子集。
- 只在该子集上报告真实绝对误差。

**English future improvement:**

- collect one calibrated step-height sample;
- use profilometer/confocal/white-light interferometry for a small real ground-truth subset;
- report real absolute error on that subset only.

## 论文修改 TODO / Paper Revision TODO

### 摘要 / Abstract

**中文：**

- 从 ground-truth scarcity 问题切入。
- 引入 simulation-to-real focus-stack reconstruction。
- 说明合成数据支持定量评估，真实样本支持形貌验证。
- 避免过度声称真实绝对精度。

**English:**

- Start from the ground-truth scarcity problem.
- Introduce simulation-to-real focus-stack reconstruction.
- State that synthetic data supports quantitative evaluation and real samples support morphology validation.
- Avoid overclaiming real absolute accuracy.

### 引言 / Introduction

**中文：**

- 第一层 gap：工业表面形貌需要超越 2D 缺陷检测的 3D 线索。
- 第二层 gap：校准真实高度标签稀缺。
- 第三层 gap：传统 DFF 在眩光、弱纹理和复杂聚焦响应下失效。
- 将仿真数据构造和先验引导校正作为解决路径。

**English:**

- Make the first gap about industrial surface morphology requiring 3D cues beyond 2D defect detection.
- Make the second gap about calibrated real height labels.
- Make the third gap about traditional DFF failure under glare, weak texture, and complex focus response.
- Present simulation-based data construction and prior-guided correction as the solution.

### Related Work

**中文建议顺序：**

1. Classical SFF/DFF and focus measures
2. Learning-based depth from focus
3. Simulation-to-real and pseudo-supervised depth learning
4. Industrial surface defect 3D reconstruction

**English suggested order:**

1. Classical SFF/DFF and focus measures
2. Learning-based depth from focus
3. Simulation-to-real and pseudo-supervised depth learning
4. Industrial surface defect 3D reconstruction

### 方法 / Method

**中文：**

- 分开描述 simulator、traditional priors 和 learning model。
- 明确定义模型输入通道。
- 解释焦向差分通道为什么有用。
- 解释 DFF/GADFF 先验为什么能降低学习负担。

**English:**

- Separate the simulator, traditional priors, and learning model.
- Define model input channels clearly.
- Explain why focal-difference channels are useful.
- Explain why DFF/GADFF priors reduce learning burden.

### 实验 / Experiments

**中文：**

- 区分 synthetic quantitative experiments 和 real qualitative/no-reference experiments。
- 增加 ablation table。
- 在真实样本结果后增加 domain-gap discussion。

**English:**

- Separate synthetic quantitative experiments from real qualitative/no-reference experiments.
- Add an ablation table.
- Add a domain-gap discussion after real-sample results.

### 讨论 / Discussion

**中文：**

- 明确主要限制：真实校准高度真值缺失。
- 解释当前结果的意义：合成真值支持受控误差分析，真实样本展示部署可行性。
- 将真实校准样本采集作为后续工作。

**English:**

- State the main limitation: real calibrated ground truth is missing.
- Explain why the current result is still meaningful: synthetic ground truth gives controlled error analysis; real samples show deployment plausibility.
- Propose calibrated real sample collection as next work.

## 建议时间线 / Suggested Timeline

### 第 1 周 / Week 1

**中文：**

- 确定故事线和模型名称。
- 更新 Related Work 和 citation matrix。
- 围绕 Simulation-to-Real 清理论文结构。

**English:**

- Finalize story line and method name.
- Update Related Work and citation matrix.
- Clean paper structure around Simulation-to-Real.

### 第 2 周 / Week 2

**中文：**

- 运行最小消融实验。
- 生成更新后的图和表。
- 更新方法图和实验流程图。

**English:**

- Run minimum ablations.
- Generate updated figures and tables.
- Update method diagram and experiment pipeline diagram.

### 第 3 周 / Week 3

**中文：**

- 重写 abstract、introduction、related work 和 discussion。
- 对齐所有术语和图注。
- 准备投稿风格 PDF。

**English:**

- Rewrite abstract, introduction, related work, and discussion.
- Align all terminology and figure captions.
- Prepare submission-style PDF.

### 第 4 周 / Week 4

**中文：**

- 增加可行的外部基线。
- 打磨 limitation 和 future work。
- 最终检查 claims、figures、citations 和 real-data boundary。

**English:**

- Add any feasible external baseline.
- Polish limitation and future work.
- Do final consistency check for claims, figures, citations, and real-data boundaries.

## 立即行动项 / Immediate Next Actions

**中文：**

1. 决定最终模型名：S2R-FocusNet 或 Prior-Guided Focus-ResUNet。
2. 基于 Zotero core 15-paper BibTeX 文件写 Related Work。
3. 增加输入通道贡献的消融实验脚本或表格。
4. 将多个模型产物的表述收敛为一个最终方法叙事。
5. 增加一段明确区分 synthetic ground-truth evaluation 和 real no-reference validation 的文字。

**English:**

1. Decide the final model name: S2R-FocusNet or Prior-Guided Focus-ResUNet.
2. Create a Related Work section from the Zotero core 15-paper BibTeX file.
3. Add an ablation experiment script or table for input-channel contributions.
4. Replace multiple model-product wording with one final-method narrative.
5. Add a paragraph explicitly distinguishing synthetic ground-truth evaluation from real no-reference validation.

