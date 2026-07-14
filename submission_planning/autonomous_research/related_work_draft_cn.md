# Related Work 中文草稿

更新日期：2026-06-18  
用途：为后续英文论文 Related Work 提供中文逻辑稿。正式投稿时需转换为英文并补齐 BibTeX 引用格式。

## 1. Classical Shape / Depth from Focus

Shape from Focus (SFF) 和 Depth from Focus (DFF) 通过多焦图像序列中的清晰度变化恢复物体表面高度，是显微测量和三维形貌重建中的经典路线。早期方法通常先定义 focus measure operator，再沿焦平面维度寻找最大响应，从而得到每个像素对应的深度或焦平面位置。Nayar and Nakagawa 建立了 SFF 的基础问题形式，Pertuz et al. 系统分析了多类 focus measure operator 的性质。此类方法具有可解释性强、训练数据需求低的优势，但对窗口大小、噪声、弱纹理和局部反光高度敏感。

为缓解固定窗口带来的尺度问题，Lee et al. 提出了 adaptive window selection，根据局部强度离散度调整焦点评价窗口；Li et al. 进一步提出 adaptive window iteration，通过窗口自适应和迭代增强改善深度恢复。这些方法与本项目中的 Lee2013 和 Li2019 基线直接相关，也说明传统 DFF 的核心限制：当表面存在低纹理、周期纹理、眩光或深度突变时，单一 focus measure 很难同时兼顾边界锐度与区域稳定性。

## 2. Learning-Based Depth from Focus

深度学习方法将 DFF 从手工 focus measure 推向了数据驱动的焦堆特征建模。DDFFNet 是早期端到端 deep DFF 方法之一，直接从 focal stack 估计 depth/disparity，证明卷积网络可以学习焦堆中的深度线索。AiFDepthNet 进一步将 depth estimation 与 all-in-focus image reconstruction 结合，用 AiF supervision 连接监督和无监督 DFF 训练。此类方法表明，焦堆不仅可以提供最大清晰度索引，还可以通过神经网络学习局部上下文、纹理变化和跨焦平面的响应模式。

DFV 是本研究最重要的外部对比方法之一。它提出 differential focus volume，通过焦平面维度上一阶差分特征捕捉 focus response 的变化，并利用 probability regression 输出深度估计。这与本项目使用 focal-difference volume 的思想高度相关。相比直接从单帧或堆叠图像回归深度，基于焦向差分的表示更强调轴向响应变化，适合解释焦栈重建中的深度来源。因此，DFV 应作为本文投稿时优先复现的 deep DFF baseline。

近年来，学习型 SFF/DFF 继续向更复杂的序列建模、频域建模和物理约束方向发展。FAD 引入 frequency-aware representation，将空间局部特征与频域结构信息结合，对周期纹理和低纹理区域具有启发意义。DualFocus 强调 spatio-focal constraints，尝试同时约束空间结构和焦平面维度的一致性。2026 年 IJCV 的 multiscale Directional Dilated Laplacian recurrent network 则从多尺度 Laplacian 特征和递归序列建模角度改进 SFF。这些工作共同表明，最新 deep DFF/SFF 研究正在从单点清晰度选择转向焦堆序列结构、频域特征和物理一致性的联合建模。

## 3. Simulation-to-Real and Defocus-Based Learning

工业显微表面往往难以获得高质量真实高度真值，尤其在反光材料、微尺度缺陷和复杂纹理场景中。Simulation-to-real 因此成为焦栈重建中的关键策略：通过合成样本提供可控 height ground truth，再通过随机化、先验约束或无标签真实样本减小域差距。Focus on Defocus 通过散焦线索讨论 synthetic-to-real depth estimation，说明 defocus cue 可以作为跨域训练的重要监督信号。DDFS 将相机设置和 defocus model 显式引入网络，以提升不同相机参数下的泛化能力；DfF in the Wild 则面向真实 focal stack 场景，处理真实相机中的视场变化、焦平面变化和图像不稳定问题。

无真实 depth GT 的训练策略同样重要。DEReD 从 defocus clue 出发进行 fully self-supervised depth estimation，通过 optical model 重建焦堆，实现无需深度真值的训练。Depth Anything V2 虽然属于单目深度基础模型，但其训练路线对本项目有直接方法论价值：高质量 synthetic depth labels、真实无标签图像和 teacher-student pseudo labeling 可以共同提升深度模型泛化。对本文而言，Depth Anything V2 不适合作为 focus-stack 数值主基线，但可支持 synthetic labels、pseudo-labeled real samples 和 teacher-student bridge 的后续训练策略。

## 4. Industrial Surface Defect Morphology

工业表面缺陷研究长期以 2D detection、classification 和 segmentation 为主，常见任务包括金属、陶瓷、电子器件和复合材料表面缺陷识别。综述类工作表明，深度学习已经显著提升了表面缺陷检测性能，但 2D 图像难以完整表达缺陷深度、凹坑轮廓、边缘形貌和微观起伏。对于反光或弱纹理表面，单帧图像还容易受到照明、眩光和纹理伪影影响。

三维形貌重建可以为缺陷理解提供更丰富的信息，例如凹坑深度、划痕截面、边界连续性和表面粗糙变化。结构光、显微成像和焦栈成像都可以服务该目标。与结构光或干涉测量相比，focus-stack reconstruction 的设备门槛较低，适合在显微调焦过程中获取多焦图像序列。本文关注的反光工业表面缺陷形貌重建正位于这一交叉点：它需要传统 DFF 的物理可解释性，也需要深度模型处理眩光、弱纹理和局部焦点歧义。

## 5. 本文定位

现有 classical SFF/DFF 方法可解释性强，但在反光和弱纹理区域容易产生错误焦点响应；现有 deep DFF 方法能够学习焦堆特征，但通常依赖真实深度数据或自然场景焦堆，和工业显微反光表面存在域差异。本文的核心定位是：以可控 synthetic focus-stack 数据构造 height ground truth，用 DFF/GADFF prior、focal-difference volume 和 glare-aware cues 引导一个紧凑深度校正模型，并在 synthetic absolute metrics 与 real no-reference morphology metrics 两个层面评估。

该定位与最新文献的关系可以概括为三点：

1. 相比传统 DFF，本文强调 prior-guided learning 对失效区域的校正能力。
2. 相比一般 deep DFF，本文突出 simulation-to-real 数据构造和工业反光表面场景。
3. 相比单目 foundation depth，本文保留 focus-stack 的轴向焦点响应，避免将单帧视觉先验直接替代焦堆三维线索。

## 6. 建议英文论文段落结构

1. **Classical SFF/DFF and focus measures**  
   介绍 Nayar, Pertuz, Lee2013, Li2019，指出 focus measure 和窗口尺度限制。

2. **Deep depth from focus and focal-stack modeling**  
   介绍 DDFFNet, AiFDepthNet, DFV, FAD, DualFocus, DDL-Recurrent SFF，突出 deep focal-stack representation。

3. **Simulation-to-real and defocus supervision**  
   介绍 Focus on Defocus, DfF in the Wild, DDFS, DEReD, Depth Anything V2，突出真实 GT 缺失和 pseudo-label 训练策略。

4. **Industrial surface morphology reconstruction**  
   从表面缺陷检测扩展到三维形貌重建，说明本文任务的应用动机。
