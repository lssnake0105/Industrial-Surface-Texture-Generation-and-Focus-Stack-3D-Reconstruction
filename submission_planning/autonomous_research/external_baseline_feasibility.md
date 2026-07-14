# 外部 SOTA 基线复现可行性清单

更新日期：2026-06-18  
用途：为投稿前的 SOTA 对比选择可执行方案。  
原则：优先复现与 focus-stack / DFF 任务同构的方法；单目深度基础模型只作为辅助，不进入主数值表。

## 1. 优先级总表

| 优先级 | 方法 | 类型 | 投稿价值 | 复现成本 | 推荐处理 |
|---|---|---|---|---|---|
| P0 | DFV | deep DFF / differential focus volume | 最高 | 中 | 第一优先级复现 |
| P0 | DDFFNet | early deep DFF | 高 | 中 | 第二优先级复现 |
| P1 | HybridDepth | focal stack + single-image prior | 高 | 中-高 | 若代码环境可跑，作为 2025 强基线 |
| P1 | DfF in the Wild | real-world DFF | 高 | 中-高 | 强化 simulation-to-real 叙事 |
| P1 | DDFS | defocus model + camera setting | 高 | 高 | 相机参数充分时复现 |
| P1 | AiFDepthNet | AiF supervision | 中-高 | 中-高 | 更适合作为训练策略参考 |
| P1 | DEReD | self-supervised defocus | 中-高 | 高 | 作为无 GT 真实样本讨论 |
| P2 | FAD | frequency-aware DFF | 中-高 | 未定 | 先引用，等待代码 |
| P2 | DDL-Recurrent SFF | learning-based SFF | 中-高 | 未定 | 先引用，等待代码 |
| P3 | Depth Anything V2 | monocular foundation depth | 中 | 低 | 做 qualitative auxiliary，不做主数值基线 |

## 2. P0 方法

### 2.1 DFV / Deep Depth from Focus with Differential Focus Volume

**定位：** 当前最关键外部 SOTA。  
**原因：** DFV 与本项目的 focal-difference volume 思路高度接近，审稿人很可能把它视为必须比较的方法。  
**建议实验：**

| 项目 | 建议 |
|---|---|
| 输入 | 本项目 synthetic focus stack，必要时按 DFV 需求重排为焦平面序列 |
| 输出 | depth / disparity / focus probability |
| 评价 | MAE、edge MAE、high-risk MAE、P10 难样本可视化 |
| 训练策略 | 优先本项目 synthetic train split 重新训练；若时间不足，再尝试官方预训练 zero-shot |
| 风险 | 数据格式、焦距/深度尺度映射、GPU 环境 |

**论文写法：** DFV 是最接近本项目表征路线的强基线。若 proposed method 在整体 MAE 或真实样本稳定性上优于 DFV，投稿说服力会显著提升。

### 2.2 DDFFNet / Deep Depth from Focus

**定位：** learning-based DFF 的基础深度学习基线。  
**原因：** 它代表从传统 focus measure 到端到端网络的早期转折，适合作为 deep baseline lower bound。  
**建议实验：**

| 项目 | 建议 |
|---|---|
| 输入 | 合成焦堆，按官方接口转换为 HDF5 或等价张量 |
| 输出 | depth / disparity |
| 评价 | 与 DFV 使用同一 synthetic test split |
| 训练策略 | 优先重新训练，避免自然场景预训练和显微工业表面域差距过大 |
| 风险 | 老代码依赖、数据接口、尺度对齐 |

**论文写法：** DDFFNet 可说明单纯端到端焦堆学习在数据规模有限、反光/弱纹理场景下的表现边界。

## 3. P1 方法

### 3.1 HybridDepth

**定位：** 2025 年 focal stack 与 single-image prior 融合方向。  
**原因：** 它能连接 focus stack 与 foundation depth prior，是当前稿件可考虑的较新强对比。  
**建议实验：**

| 项目 | 建议 |
|---|---|
| 输入 | focal stack + 单帧或 all-in-focus 图像 |
| 输出 | metric / relative depth |
| 评价 | synthetic MAE + qualitative real comparison |
| 风险 | 原方法面向移动端/自然场景，与显微反光表面存在域差异 |

**处理建议：** 若两周内能跑通，加入增强表；若适配成本过高，写入 Related Work 和 Discussion。

### 3.2 Learning Depth from Focus in the Wild

**定位：** simulation-to-real 叙事强相关方法。  
**原因：** 真实 focal stack 常存在视场变化、错位、相机设置差异和纹理复杂性，该方法正好服务这类问题。  
**建议实验：**

| 项目 | 建议 |
|---|---|
| 输入 | 真实焦堆或合成焦堆 |
| 输出 | depth |
| 评价 | 真实 no-reference metrics、合成 MAE |
| 风险 | 官方流程可能依赖特定数据准备和相机模拟参数 |

**处理建议：** 作为 P1 复现候选；即使不复现，也应在 Related Work 中重点讨论。

### 3.3 DDFS

**定位：** 光学参数显式建模的深度 DFF。  
**原因：** 它把 defocus model、plane sweep volume 和 camera setting invariance 放入学习框架，和本项目“物理先验 + 学习校正”主线一致。  
**建议实验：**

| 项目 | 建议 |
|---|---|
| 输入 | 焦堆 + 相机参数/等效焦平面参数 |
| 输出 | depth |
| 评价 | synthetic MAE、domain gap discussion |
| 风险 | 本项目现有真实样本可能缺少完整相机参数 |

**处理建议：** 参数齐全时复现；参数不足时不做主数值比较，避免不公平对比。

### 3.4 AiFDepthNet 与 DEReD

**定位：** 无真实 depth GT 情况下的训练策略参考。  
**原因：** AiFDepthNet 使用 all-in-focus supervision，DEReD 使用 self-supervised defocus reconstruction，二者可支撑真实样本无 GT 的后续路线。  
**处理建议：**

| 方法 | 当前用途 | 后续可能 |
|---|---|---|
| AiFDepthNet | Related Work + 半监督训练策略讨论 | 若可生成 AiF 图，尝试补充实验 |
| DEReD | Related Work + self-supervised future work | 若真实焦堆充足，可探索重建一致性 loss |

## 4. P2/P3 方法

### 4.1 FAD 与 DDL-Recurrent SFF

**定位：** 最新 SFF/DFF 文献更新。  
**原因：** FAD 强调频域特征，适合讨论周期纹理；DDL-Recurrent SFF 强调多尺度 Laplacian 与焦堆序列递归建模，适合讨论显微 SFF 最新趋势。  
**处理建议：** 先写入 Related Work。若公开代码或实现细节充足，再评估复现。

### 4.2 Depth Anything V2

**定位：** 单目基础模型与训练策略论据。  
**原因：** 它表明高质量 synthetic labeled data 与真实无标签 pseudo labels 可以共同提升泛化，和本项目 simulation-to-real 训练策略相关。  
**建议实验：**

| 项目 | 建议 |
|---|---|
| 输入 | 真实焦堆的最佳聚焦帧或 all-in-focus 图 |
| 输出 | relative depth |
| 评价 | qualitative side-by-side，不进入 synthetic MAE 主表 |
| 重点观察 | 是否保留微小缺陷形貌、是否受反光区域干扰、是否输出自然图像式深度先验 |

**论文写法：** Depth Anything V2 适合作为大规模先验和 pseudo-label 训练路线的背景，也可作为单帧辅助可视化。主结论仍应围绕 focus-stack 重建。

## 5. 数据接口准备

为减少后续适配成本，建议先定义一个统一中间格式：

| 字段 | 含义 | 备注 |
|---|---|---|
| `stack` | shape = `[N, H, W]` 或 `[N, C, H, W]` 的焦堆 | N 为焦平面数量 |
| `focus_positions` | 每帧对应焦平面位置或归一化索引 | 若无绝对单位，至少保存相对顺序 |
| `height_gt` | 合成样本 height ground truth | 真实样本为空 |
| `valid_mask` | 有效区域 mask | 排除边界、黑边或缺失区域 |
| `edge_mask` | 边缘/深度突变区域 | 用于 edge MAE |
| `high_risk_mask` | glare / weak texture / high-risk 区域 | 用于 high-risk MAE |
| `dff_prior` | Original DFF depth | proposed method 与 ablation 使用 |
| `confidence` | focus confidence / glare confidence | 后续 self-training 可用 |

## 6. 最小可执行顺序

1. 固定 synthetic train/test split 与统一中间格式。
2. 先适配 DFV，记录输入尺寸、帧数、尺度对齐方式和运行时间。
3. 再适配 DDFFNet，形成一个新旧 deep DFF 对比。
4. 若还有时间，尝试 HybridDepth 或 DfF in the Wild。
5. Depth Anything V2 只做真实样本 qualitative auxiliary figure。
6. 汇总所有方法的公平比较设置，写入论文 Experiments。

## 7. 当前不建议做的事

1. 不把单目基础模型作为主表数值 SOTA。
2. 不在真实样本上报告 absolute height error。
3. 不把多个项目内网络都写成最终贡献。
4. 不在相机参数不完整时强行比较 DDFS。
5. 不为了堆方法数量加入与 focus stack 无关的检测/分割 SOTA。
