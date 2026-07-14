# Related Work 文献矩阵：反光表面焦栈三维重建

日期：2026-06-21

## 1. 推荐分组

| 组别 | 代表文献 | 核心问题 | 与本项目关系 | 写作位置 | 实验优先级 |
|---|---|---|---|---|---|
| Classical SFF/DFF | Nayar1994, Pertuz2013, Lee2013, Li2019 | 如何从焦点评价恢复深度 | 提供传统 DFF 逻辑、focus measure 和自适应窗口 baseline | Background / Baselines | 已有结果，必须保留 |
| Learning-based DFF | DDFFNet, AiFDepthNet, DFV, DfF in the Wild, DDFS | 如何用网络建模焦堆序列 | 是最公平的外部 SOTA 对比组 | Related Work / Experiments | DFV、DDFFNet 优先 |
| Defocus + Sim-to-Real | Focus on Defocus, DEReD, Dr.Bokeh, DDFS | 如何用光学/散焦模型弥合合成到真实 | 支撑仿真、无真实真值、自监督与可微渲染叙事 | Related Work / Discussion | 以引用为主，少量可复现 |
| Industrial 3D Defect | Att-PU-Net BDSIM, surface defect review | 工业缺陷如何从 2D 检测转向 3D 形貌 | 建立应用价值和真实验证标准 | Introduction / Related Work | 不作为直接 baseline |
| Depth Foundation Models | Depth Anything, Depth Anything V2 | 大规模数据、合成标签和伪标签如何提升泛化 | 方法论参照，不是直接数值对比 | Discussion | 可做定性辅助，不进主表 |
| Optical Reflection / NA | Numerical aperture, microfacet/BRDF models | 反射光是否进入成像孔径 | 支撑 flare-risk prior 的物理解释 | Method | 需要仿真验证 |

## 2. 入文角度

### Nayar and Nakagawa 1994

- 类型：Classical Shape from Focus。
- 作用：说明 DFF/SFF 的基本思想是从多焦图像中寻找最大清晰度响应。
- 本文使用：作为传统 DFF 背景和 Original DFF baseline 的理论起点。

### Pertuz et al. 2013

- 类型：Focus measure operator survey。
- 作用：系统整理不同 focus measure 的性质。
- 本文使用：解释为什么 Laplacian、Tenengrad、方差等评价会受纹理、噪声和亮斑边缘影响。

### Lee2013 / Li2019

- 类型：Adaptive window SFF/DFF。
- 作用：解决固定窗口在弱纹理和边缘区域的局限。
- 本文使用：作为传统增强 baseline；对比说明单纯改窗口不足以解决反光/眩光伪峰。

### DDFFNet

- 类型：早期端到端 DFF 网络。
- 作用：证明深度学习可以从 focal stack 学习深度。
- 本文使用：最基础 learning-based baseline。若适配成本可控，建议在 synthetic split 上重新训练或微调。

### AiFDepthNet

- 类型：All-in-focus supervision。
- 作用：在缺少深度标签时用 AIF supervision 桥接监督和无监督。
- 本文使用：讨论真实样本无高度真值时的训练替代路径。

### DFV

- 类型：CVPR 2022 learning-based DFF。
- 作用：通过 differential focus volume 捕捉焦向一阶变化，并给出 focus probability。
- 本文使用：最关键外部 SOTA。它与本项目 focal-difference prior 最接近，建议优先复现。

### Learning Depth from Focus in the Wild

- 类型：真实相机焦堆深度估计。
- 作用：强调真实焦堆中的图像对齐、sharp-region detection、弱纹理和相机仿真。
- 本文使用：支撑真实焦堆数据质量检查和配准检查。

### DDFS

- 类型：camera-setting-invariant DFF。
- 作用：将 defocus model、plane sweep volume 和相机参数纳入网络。
- 本文使用：支撑“光学参数不能被网络黑箱吞掉”的论点。若焦距、f-number、焦层间距记录不完整，作为 Related Work 强相关文献。

### Focus on Defocus

- 类型：synthetic-to-real depth from defocus。
- 作用：用 defocus cue 作为更域不变的监督信号。
- 本文使用：支持仿真到真实迁移，但需要强调本文的对象是显微反光缺陷焦堆。

### DEReD

- 类型：fully self-supervised DFD from sparse focal stack。
- 作用：不依赖 depth 或 AIF ground truth，通过 optical model 验证和 refine 预测。
- 本文使用：支撑真实无真值场景下的后续自监督扩展。

### Dr.Bokeh

- 类型：可微离焦渲染。
- 作用：将物理 bokeh/defocus 形成过程做成可微模块。
- 本文使用：支持把 defocus PSF、遮挡边界和可微仿真引入训练框架。

### Depth Anything / Depth Anything V2

- 类型：单目深度基础模型。
- 作用：展示大规模无标签真实数据、合成标签和 teacher-student pseudo-label bridge 的价值。
- 本文使用：方法论参照。不可放入主数值对比，因为输入不是焦堆且输出不是显微形貌高度。

### Att-PU-Net BDSIM 2026

- 类型：工业光学元件微纳缺陷 3D 重建。
- 作用：用 bright-field/dark-field structured illumination microscopy 和深度网络重建缺陷点云，并用 WLI 验证。
- 本文使用：说明工业缺陷 3D 形貌重建是正在发展的相关方向，也提示本文需要真实校准真值子集。

## 3. 主结果表建议

### 必须进入主表

| 方法 | 角色 | 当前状态 |
|---|---|---|
| Original DFF | 传统下限 | 已有结果 |
| DFF + post-processing | 工程传统基线 | 已有结果 |
| Lee2013 adaptive window | 传统 SFF baseline | 已有结果 |
| Li2019 adaptive iteration | 传统 SFF baseline | 已有结果 |
| DDFFNet | learning-based baseline | 待适配 |
| DFV | 最强相关 SOTA | 优先适配 |
| Proposed Flare-FocusNet | 本文方法 | 需要统一命名和输入 prior |

### 可进入补充表或讨论

| 方法 | 使用方式 |
|---|---|
| DfF in the Wild | 若代码适配可控，进入补充实验；否则重点讨论真实焦堆配准与相机仿真 |
| DDFS | 若相机参数完整，尝试复现；否则用于讨论 camera-setting invariance |
| AiFDepthNet | 用于真实无高度真值的训练策略讨论 |
| DEReD | 用于自监督/无 AIF 真值扩展 |
| Depth Anything V2 | 用于 synthetic labels + pseudo-labeled real images 方法论，不进主表 |

## 4. 论文 Related Work 段落骨架

1. **Classical focus-stack reconstruction.** 从 Nayar1994 到 Pertuz2013、Lee2013、Li2019，强调传统 DFF 的解释性和局限。
2. **Learning from focal stacks.** DDFFNet、AiFDepthNet、DFV、DfF in the Wild、DDFS，强调从单点 focus measure 到焦堆体建模、相机参数建模和真实焦堆适配。
3. **Defocus modeling and simulation-to-real depth learning.** Focus on Defocus、DEReD、Dr.Bokeh，强调物理成像模型、自监督和可微渲染。
4. **Optical surface defect 3D reconstruction.** Att-PU-Net BDSIM 2026 和工业缺陷综述，强调工业缺陷定量 3D 的必要性。
5. **Large-scale depth priors.** Depth Anything / V2 作为训练策略和数据策略参照，明确与 focus-stack 任务的边界。

## 5. 近期引用优先级

| 优先级 | 文献 | 理由 |
|---|---|---|
| P0 | DFV | 与 focal-difference 表征最接近 |
| P0 | DDFFNet | learning-based DFF 基础 baseline |
| P0 | DDFS | 最新且有明确光学模型、相机参数不变性和 S2R 讨论 |
| P1 | DfF in the Wild | 真实焦堆配准、弱纹理和仿真器相关 |
| P1 | DEReD | 无真实 depth/AIF GT 的自监督路径 |
| P1 | Att-PU-Net BDSIM | 工业光学 3D 缺陷重建强相关 |
| P2 | Depth Anything V2 | 训练策略与数据策略参照 |
| P2 | Dr.Bokeh | 可微 defocus/PSF 建模参照 |

## 6. 参考链接

- DFV: https://arxiv.org/abs/2112.01712
- DDFFNet: https://arxiv.org/abs/1704.01085
- AiFDepthNet: https://arxiv.org/abs/2108.10843
- DfF in the Wild: https://arxiv.org/abs/2207.09658
- DDFS: https://arxiv.org/abs/2202.13055
- Focus on Defocus: https://arxiv.org/abs/2005.09623
- DEReD: https://arxiv.org/abs/2303.10752
- Dr.Bokeh: https://arxiv.org/abs/2308.08843
- Depth Anything: https://arxiv.org/abs/2401.10891
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Nikon Numerical Aperture: https://www.microscopyu.com/microscopy-basics/numerical-aperture
- Att-PU-Net BDSIM: https://doi.org/10.1364/AO.587592
