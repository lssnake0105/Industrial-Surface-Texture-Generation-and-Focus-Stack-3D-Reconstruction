# Manuscript Blueprint

更新日期：2026-06-18  
用途：将当前研究材料收束成可直接写英文论文的结构蓝图。  
建议题目：**Simulation-to-Real Focus-Stack Reconstruction for Reflective Surface Defect Morphology Using Prior-Guided Deep Correction**

## 1. 论文核心定位

本文应定位为一个面向反光工业表面缺陷形貌的 simulation-to-real focus-stack reconstruction 研究。核心价值不只是提出一个网络结构，还包括：

1. 将真实 height ground truth 稀缺作为主要研究动机；
2. 用可控 synthetic focus-stack 数据构造监督信号；
3. 用 DFF/GADFF priors 和 focal-difference representation 保留焦堆物理线索；
4. 用真实样本的 no-reference morphology metrics 验证部署可行性；
5. 清楚区分 synthetic quantitative evaluation 与 real no-reference validation。

## 2. 推荐标题

### 稳妥标题

**Simulation-to-Real Focus-Stack Reconstruction for Reflective Surface Defect Morphology Using Prior-Guided Deep Correction**

优点：准确覆盖 simulation-to-real、focus-stack、reflective surface defect、prior-guided correction。

### 更方法导向标题

**Prior-Guided Focus-Stack Depth Reconstruction for Reflective Industrial Surface Defect Morphology**

优点：更简洁；适合目标期刊偏工程应用。

### 更数据策略导向标题

**Synthetic-to-Real Focus-Stack Learning for 3D Morphology Reconstruction of Reflective Surface Defects**

优点：突出数据构造和迁移学习；适合强调 simulation-based research insight。

## 3. Abstract 结构

### 句子功能

| 句子 | 功能 | 安全 claim |
|---|---|---|
| 1 | 任务背景 | Reflective industrial surface defects require 3D morphology reconstruction beyond 2D visual inspection. |
| 2 | 核心困难 | Calibrated real height ground truth is difficult to obtain for microscopic reflective surfaces. |
| 3 | 方法概述 | We propose a simulation-to-real focus-stack reconstruction framework using synthetic height supervision and prior-guided correction. |
| 4 | 技术组成 | The framework combines DFF/GADFF priors, focal-difference representation, glare-aware cues, and a compact learning-based correction model. |
| 5 | 合成结果 | On the current synthetic test set, the proposed model achieves the lowest mean MAE among evaluated internal baselines. |
| 6 | 真实结果 | On real focus stacks, no-reference morphology metrics show reduced spike artifacts and improved reconstruction stability. |
| 7 | 边界 | Real absolute height accuracy remains future work because calibrated real height ground truth is unavailable. |

### 暂不写入 Abstract 的内容

1. 最新 SOTA superiority，除非补齐 DFV/DDFFNet 实测。
2. 真实样本 absolute metrology accuracy。
3. domain randomization 的确定增益，除非完成对应消融。

## 4. Introduction 结构

### Paragraph 1: 工业缺陷从 2D 检测走向 3D 形貌

写作目标：说明 2D defect detection 无法完整表达缺陷深度、边界形貌和微结构起伏。  
应引入：industrial surface defect inspection, reflective materials, micro-scale morphology, 3D reconstruction。

### Paragraph 2: Focus-stack / DFF 的价值与局限

写作目标：说明 focus-stack reconstruction 设备门槛较低，能利用显微调焦序列恢复相对高度；同时传统 DFF 对弱纹理、眩光和边界敏感。  
应引入：DFF, SFF, focus measure, adaptive window, focus ambiguity。

### Paragraph 3: 真实 GT 稀缺与 simulation-to-real 必要性

写作目标：把仿真从“补充数据”提升为核心研究策略。  
安全表述：真实显微反光表面的 calibrated height GT 获取困难，因此 synthetic height maps and rendered focus stacks allow controlled supervision and quantitative evaluation。

### Paragraph 4: 本文方法

写作目标：提出 S2R-FocusNet / Prior-Guided Focus-ResUNet。  
应包含：DFF/GADFF prior, focal-difference volume, glare-aware cue, bounded/residual correction。

### Paragraph 5: Contributions

建议贡献：

1. A controllable synthetic focus-stack data engine for reflective defect morphology with known height maps.
2. A prior-guided focus-stack reconstruction framework combining traditional DFF/GADFF priors and focal-difference representation.
3. A two-level evaluation protocol with synthetic absolute metrics and real no-reference morphology metrics.
4. A focused case study on reflective industrial surface defects, including failure analysis for high-risk regions.

## 5. Related Work 结构

### 5.1 Classical SFF/DFF

核心文献：Nayar, Pertuz, Lee2013, Li2019。  
对本文作用：建立传统 focus measure 和 adaptive window baseline。

### 5.2 Learning-Based Depth from Focus

核心文献：DDFFNet, AiFDepthNet, DFV, DfF in the Wild, DDFS, HybridDepth。  
对本文作用：说明 deep focal-stack representation 的发展，并把 DFV 定位为最关键外部 baseline。

### 5.3 Recent Focus-Stack Trends

核心文献：FAD, DualFocus, DDL-Recurrent SFF, Minimal Focal Stack。  
对本文作用：证明文献更新到 2025-2026，且研究趋势正在转向频域、spatio-focal constraints、序列建模和少帧采集。

### 5.4 Simulation-to-Real and Foundation Depth

核心文献：Focus on Defocus, DEReD, Depth Anything V2, Marigold, ZoeDepth, Metric3D。  
对本文作用：支持 synthetic labels、pseudo-labeled real data 和 teacher-student training strategy，但需说明单目模型不是直接 DFF baseline。

### 5.5 Industrial Surface Morphology

核心文献：surface defect detection reviews, structured illumination / microscopy 3D reconstruction。  
对本文作用：从 2D detection 引出 3D morphology reconstruction 的应用必要性。

## 6. Method 结构

### 6.1 Problem Formulation

定义：

| 符号 | 含义 |
|---|---|
| `I = {I_1, ..., I_N}` | N-frame focus stack |
| `z_i` | 第 i 帧焦平面位置 |
| `H` | 待恢复 height map |
| `P_dff` | DFF prior depth |
| `P_gadff` | glare-aware DFF prior |
| `D_focal` | focal-difference representation |
| `M_risk` | high-risk / glare mask |

目标：学习 `f(I, P_dff, P_gadff, D_focal, M_risk) -> H`。

### 6.2 Synthetic Focus-Stack Data Generation

写清变量：

1. defect geometry: ridge, valley, step, periodic, pit/groove；
2. roughness/noise: Perlin, fractal, stripe, procedural；
3. depth range and z-step；
4. reflectance / stray light / glare simulation；
5. focus response and stack rendering；
6. known height ground truth。

### 6.3 Traditional Priors

解释 Original DFF、DFF + post-processing、GADFF、Lee2013、Li2019 的角色。  
重点：它们既是 baseline，也是 proposed method 的可解释物理锚点。

### 6.4 Focal-Difference Representation

说明相邻焦平面或多尺度焦平面差分如何捕捉 focus response 变化。  
写作联系：与 DFV 的 differential focus volume 形成相关但不完全相同的设计逻辑。

### 6.5 S2R-FocusNet

建议只写一个 final method：

1. input channels；
2. encoder-decoder / ResUNet backbone；
3. prior fusion；
4. residual or bounded correction；
5. output height map。

TinyDepthNet、Residual Focus-ResUNet、PA-FRU 等放入 ablation 或 implementation history。

### 6.6 Training Objective

候选 loss：

1. height reconstruction loss；
2. edge-aware loss；
3. high-risk weighted loss；
4. smoothness / spike penalty；
5. prior consistency or residual bound。

只有实际实现过的 loss 才写入主方法；未实现的放到 future work。

## 7. Experiments 结构

### 7.1 Dataset and Setup

必须列出：

1. train / validation / test sample counts；
2. 17-frame focus stacks；
3. image resolution；
4. depth range and z-step；
5. synthetic GT availability；
6. real sample no-GT boundary。

### 7.2 Baselines

当前可写：

1. Original DFF；
2. DFF + post-processing；
3. GADFF；
4. Lee2013；
5. Li2019；
6. TinyDepthNet；
7. Focus-ResUNet / final method；
8. Residual variants as ablation。

需要补实验后写：

1. DDFFNet；
2. DFV；
3. HybridDepth or DfF in the Wild。

### 7.3 Metrics

Synthetic：

1. Mean MAE；
2. Edge MAE；
3. High-risk MAE；
4. P90 error；
5. RMSE normalized。

Real：

1. roughness；
2. edge retention；
3. relative dynamic range；
4. low-conf spike count；
5. profile curve consistency。

### 7.4 Synthetic Quantitative Results

安全主张：

1. Focus-ResUNet mean MAE = 53.22 um on current 7 synthetic test samples；
2. Focus-ResUNet edge MAE = 86.68 um；
3. high-risk MAE 不是最优，应放入 limitation。

### 7.5 Real No-Reference Results

安全主张：

1. Focus-ResUNet spike count = 2.0，Original DFF = 9179.7；
2. Focus-ResUNet edge retention 接近正值；
3. 真实结果只代表 morphology stability。

### 7.6 Ablation Study

最小消融：

1. w/o DFF/GADFF prior；
2. w/o focal-difference volume；
3. w/o glare-aware cue；
4. direct image-to-depth。

### 7.7 Failure Analysis

重点分析：

1. P10 V-valley；
2. periodic stripe；
3. high-risk glare mask；
4. real key texture / pit / tip samples。

## 8. Discussion 结构

### 8.1 Why Simulation Matters

强调 synthetic GT 的作用：可控变量、可重复评估、能训练 deep model。

### 8.2 Why Prior-Guided Correction Matters

强调传统 DFF 的物理锚点与 deep model 的局部修正互补。

### 8.3 Limits of Real Evaluation

明确真实样本没有 calibrated height GT，因此只做 no-reference morphology validation。

### 8.4 Relation to Foundation Depth

Depth Anything V2 的启发：synthetic labels + pseudo-labeled real data + teacher-student bridge。  
边界：单帧 depth model 不替代 focus-stack axial response。

### 8.5 Future Work

1. calibrated real height subset；
2. DFV/DDFFNet full reproduction；
3. pseudo-label training on real focus stacks；
4. stronger high-risk region modeling；
5. fewer-frame focus-stack acquisition。

## 9. Figure / Table 蓝图

| 编号 | 类型 | 内容 | 对应章节 |
|---|---|---|---|
| Fig. 1 | pipeline | synthetic-to-real focus-stack pipeline | Introduction / Method |
| Fig. 2 | method | S2R-FocusNet architecture | Method |
| Fig. 3 | dataset | synthetic sample categories and focus stack examples | Method / Experiments |
| Fig. 4 | table/bar | synthetic MAE / edge MAE / high-risk MAE | Results |
| Fig. 5 | visual | P10 difficult sample comparison | Results |
| Fig. 6 | visual | real no-reference morphology comparison | Results |
| Fig. 7 | ablation | module ablation chart and profile curves | Ablation |
| Table 1 | dataset | train/validation/test split | Experiments |
| Table 2 | quantitative | synthetic comparison with baselines | Results |
| Table 3 | no-reference | real sample metrics | Results |
| Table 4 | ablation | component ablation | Ablation |

## 10. 当前稿件缺口

| 缺口 | 影响 | 最小补法 |
|---|---|---|
| 外部 deep DFF baseline 未完成 | 影响 SOTA claim | 先跑 DFV 或 DDFFNet |
| 模块消融未完成 | 影响方法可信度 | 补三项核心消融 |
| 真实 calibrated GT 缺失 | 影响真实绝对精度 claim | 明确 no-reference 边界 |
| high-risk 区域仍不稳定 | 影响 glare-aware claim | 写入 limitation，补 high-risk loss 或 mask 分析 |
| 模型命名不统一 | 影响论文叙事 | 全文统一 S2R-FocusNet |

## 11. 一句话收束

本文最稳的论文故事是：**在真实高度标签稀缺的反光工业表面缺陷场景中，用可控仿真构造监督信号，再用传统 DFF/GADFF 先验与焦向差分表征引导深度模型完成相对三维形貌重建，并通过合成绝对误差与真实无参考形貌指标分别验证方法边界。**
