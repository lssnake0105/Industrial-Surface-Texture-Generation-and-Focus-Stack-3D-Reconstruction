# DFF / Focus-Stack 3D Reconstruction 中文 SOTA 对比清单

更新日期：2026-06-18  
项目定位：面向反光工业表面缺陷形貌的 simulation-to-real focus-stack 3D reconstruction。  
建议主线：用合成数据提供可控 ground truth，用真实样本验证无参考形貌稳定性，用一个 prior-guided deep correction model 收束最终方法。

## 1. 投稿对比结论

当前稿件最需要补强的是学习型 focus-stack / depth-from-focus 对比，而非继续扩展传统 DFF 变体。建议将对比方法分为四层：

| 层级 | 用途 | 方法 |
|---|---|---|
| P0 必须保留 / 优先复现 | 构成主结果表的最低可信对比 | Original DFF, DFF + post-processing, Lee2013, Li2019, DDFFNet, DFV |
| P1 强相关补充 | 用于增强 simulation-to-real 和模型训练策略叙事 | DfF in the Wild, DDFS, HybridDepth, AiFDepthNet, DEReD, Focus on Defocus |
| P2 最新趋势引用 | 用于 Related Work 和 Discussion，代码或适配成本不确定 | FAD, DualFocus, DDL-Recurrent SFF, SAS（待核验）, Minimal Focal Stack, Event Focal Stack |
| P3 辅助参考 | 只能作为单帧深度先验、训练策略论据或概念对照 | Depth Anything V2, Marigold, ZoeDepth, Metric3D / Metric3Dv2 |

最推荐的投稿实验组合：

1. 主表：Original DFF, DFF + post-processing, Lee2013, Li2019, DDFFNet, DFV, Proposed S2R-FocusNet。
2. 增强表：加入 DfF in the Wild 或 HybridDepth，二者至少选一个；若相机参数整理充分，再尝试 DDFS。
3. 无真实 ground truth 场景：真实样本只报告 no-reference morphology metrics，例如 roughness stability, spike count, dynamic range, edge continuity, profile consistency。
4. 相关工作：必须讨论 FAD、DualFocus、DDL-Recurrent SFF 和 Minimal Focal Stack，体现文献更新到 2025-2026 时间线；SAS 暂列为待核验候选。

## 2. 主对比清单

| 优先级 | 方法 | 年份 / 来源 | 核心思想 | 与本项目关系 | 复现建议 |
|---|---|---|---|---|---|
| P0 | Original DFF | 传统基线 | 逐像素搜索焦点响应最大帧并映射为高度 | 已有实现，可解释性强，是全部 learning 方法的下限参照 | 保留现有结果 |
| P0 | DFF + post-processing | 项目内工程基线 | 对原始 DFF 深度图做平滑、异常值抑制或结构修正 | 说明传统方法通过后处理可以改善，但受焦点评价错误限制 | 保留现有结果 |
| P0 | Lee2013 adaptive window | Optics & Laser Technology 2013 | 根据局部强度离散度自适应选择焦点评价窗口 | 对应固定窗口在弱纹理/边缘处失效的问题 | 已有实现，放入传统 SFF/DFF 组 |
| P0 | Li2019 adaptive window iteration | Chinese Optics Letters 2019 | 结合自适应窗口与迭代增强焦点评价值 | 与 Lee2013 共同代表传统 adaptive focus measure 路线 | 已有实现，保留 |
| P0 | DDFFNet / Deep Depth from Focus | ACCV 2018 / LNCS 2019 | 端到端 CNN 从 focal stack 直接估计 depth/disparity | 第一代深度学习 DFF，适合作为 learning-based lower bound | 优先适配其 HDF5 数据接口，用本项目 synthetic split 训练或微调 |
| P0 | DFV / Deep Depth from Focus with Differential Focus Volume | CVPR 2022 | 通过 differential focus volume 捕捉焦平面维度的一阶特征变化，并输出 focus probability / depth | 与本项目 focal-difference volume 思路最接近，是最关键外部 SOTA 对比 | 必须优先复现；可用同一 synthetic test set 比较 MAE、edge MAE、high-risk MAE |

## 3. 强相关补充方法

| 优先级 | 方法 | 年份 / 来源 | 核心思想 | 为什么重要 | 对本项目的使用方式 |
|---|---|---|---|---|---|
| P1 | Learning Depth from Focus in the Wild | ECCV 2022 | 面向真实相机 focal stack，加入图像对齐、sharp-region detection 和真实相机模拟 | 直接服务 simulation-to-real 叙事，尤其适合解释真实焦堆中的视场变化、微小错位和弱纹理 | 若代码可运行，作为真实适配能力对比；若时间不足，作为 Related Work 重点引用 |
| P1 | DDFS / Deep Depth from Focal Stack with Defocus Model | IJCV 2024 | 将相机设置、散焦模型和平面扫描体积纳入网络输入，实现 camera-setting invariance | 与本项目“光学先验 + 学习模型”的思想相近，并明确讨论 synthetic-to-real gap | 若能整理焦距、光圈、焦平面间距等参数，则尝试复现；参数不完整时作为强相关工作讨论 |
| P1 | HybridDepth | WACV 2025 | 融合 focal stack 与 single-image depth priors，提升 metric depth 和结构细节 | 代表 2025 年 focal stack + foundation prior 的实用路线，代码公开，投稿时很有说服力 | 可作为 P1 可执行外部强基线；需要注意其移动端/自然场景设定与显微工业表面存在域差异 |
| P1 | AiFDepthNet | ICCV 2021 | 同时估计 depth map 与 all-in-focus image，可用 AiF supervision 桥接有监督和无监督训练 | 对本项目真实样本缺少 height GT 的问题有方法论意义 | 可作为无真实深度标签训练策略的讨论对象；若能生成 AiF 图，可尝试补充实验 |
| P1 | DEReD | CVPR 2023 | 仅用 sparse focal stack 自监督估计 depth 和 AiF image，并用 optical model 重建焦堆 | 与“真实样本无 ground truth”高度相关 | 优先放入 Related Work / Future Work；直接复现实验成本较高 |
| P1 | Focus on Defocus / DefocusNet | CVPR 2020 | 用散焦线索作为 synthetic-to-real 的域不变监督信号 | 支撑“仿真训练可迁移到真实图像”的叙事 | 可作为 sim-to-real 深度估计相关工作；若复现，用作 defocus-based 辅助对比 |

## 4. 最新趋势与需要引用的方法

| 优先级 | 方法 | 年份 / 来源 | 核心思想 | 对稿件的价值 | 建议动作 |
|---|---|---|---|---|---|
| P2 | FAD / Frequency-Aware Deep Depth from Focus | IJCAI 2025 | 结合多尺度空间局部特征和频域全局结构特征，用 Fourier 模块增强 DFF | 对本项目周期纹理、弱纹理和表面细节重建很相关 | 作为 2025 SOTA 引用；若无公开代码，不强行复现 |
| P2 | DualFocus | arXiv / OpenReview 2025 | 同时约束空间维度和焦平面维度，利用 spatio-focal variational constraints 缓解纹理伪影和深度突变 | 与本项目 focal-difference、edge / defect boundary 失败区域分析相关 | 作为最新方法讨论；等待代码或正式出版信息 |
| P2 | Robust Shape from Focus via Multiscale Directional Dilated Laplacian Recurrent Network | IJCV 2026 | 结合多尺度 Directional Dilated Laplacian 特征和 recurrent 网络建模焦堆序列 | 对显微 SFF/DFF 场景比单目基础模型更接近，可作为 2026 年最新学习型 SFF 引用 | 加入 Related Work；若代码公开再评估复现 |
| P2 | SAS / Sequence Association for Shape from Focus | IEEE TPAMI 2025，待核验 | 将多焦图像序列作为完整 3D 数据处理，强调 sequence association、selective fusion 和 multiscale aggregation | 可用于说明 SFF 从单帧 focus measure 转向序列建模 | 当前公开信息需再次核验，不放入主表，暂作候选引用 |
| P2 | Towards Minimal Focal Stack in Shape from Focus | CVPRW 2026 | 用两张输入图像生成 AiF 和 Energy-of-Difference cues，减少焦堆采集数量 | 对本项目 17 帧真实焦堆的采集成本有后续意义 | 放入 Discussion / Future Work，说明未来可减少采集帧数 |
| P2 | Dense Depth from Event Focal Stack | WACV 2025 | 在焦平面扫描时用 event camera 构造 event focal stack 并估计 dense depth | 属于硬件路线，启发高时间分辨率 focus sweep | 只作为趋势引用，不进入主对比 |
| P2 | Learning Monocular Depth from Focus with Event Focal Stack | arXiv 2024 | 利用 event voxel 和 focal-distance-guided cross-modal attention 估计 event focal stack depth | 与显微平台硬件不同，但说明 focus sweep 可结合新型传感器 | 作为扩展方向引用 |

## 5. 单目深度基础模型的边界

| 方法 | 年份 / 来源 | 可用价值 | 边界 |
|---|---|---|---|
| Depth Anything V2 | arXiv:2406.09414v1 / NeurIPS 2024 | 证明精确 synthetic depth labels、large-scale unlabeled real images、teacher-student pseudo labeling 和 gradient-aware supervision 可提升细节、反光/透明场景鲁棒性与泛化；也可作为单帧深度 sanity check | 输入为单张 RGB，缺少 focal stack 的轴向焦点响应，不适合放入主数值对比；可作为“simulation-to-real 训练策略”“高质量 GT 标注价值”和“反光表面深度先验”的强相关背景 |
| Marigold | CVPR 2024 | diffusion prior 和 synthetic fine-tuning 对 zero-shot monocular depth 有参考价值 | 输出 affine-invariant depth，和显微表面相对高度/焦平面索引不直接一致 |
| ZoeDepth | 2023 | relative + metric depth 的单目融合路线，可作为深度基础模型背景 | 自然图像训练分布与工业显微反光表面差异大 |
| Metric3D / Metric3Dv2 | ICCV 2023 / TPAMI 2024 | 代表 single-image metric depth / normal foundation model | 对显微小尺度缺陷形貌缺少直接可比性 |

论文中建议写法：这些模型可用于说明大规模数据和先验模型对深度泛化的重要性，但主对比仍应围绕 focus-stack / DFF / SFF 方法展开。

### Depth Anything V2 / arXiv:2406.09414v1 单独判断

Depth Anything V2 应保留在 P3 辅助参考层，但需要在 Related Work 和 Discussion 中更突出。它是 2024 年单目深度基础模型的重要代表，论文核心结论与本项目的 simulation-to-real 叙事高度相关：高质量合成深度标签能提供更细粒度的监督，真实无标签图像经过 teacher pseudo-label 后可以缓解 synthetic-to-real domain gap。该文训练流程使用 595K synthetic labeled images 和 62M real unlabeled images，并通过大模型 teacher 生成 pseudo depth，再训练不同规模 student models。

该文对本项目的新增价值不在主结果表，而在训练策略与研究叙事：

| 维度 | 对 Depth Anything V2 的判断 | 对本项目的启发 |
|---|---|---|
| Ground truth 质量 | 文章指出真实深度标签可能因传感器、stereo matching 或 SfM 引入噪声，而 synthetic labels 可提供更完整细节 | 支撑本项目用可控仿真构造显微形貌 GT，尤其用于边界、细小缺陷和反光区域监督 |
| Synthetic-to-real gap | 纯 synthetic 训练受分布差异和场景覆盖限制影响 | 本项目应把仿真样本多样性、真实无标签焦堆和 domain randomization 写成核心训练策略 |
| Pseudo-label bridge | teacher 在真实无标签图像上生成伪深度，再训练 student models | 可作为后续方案：用 Focus-ResUNet、传统 DFF ensemble 或外部 priors 为真实焦堆生成 pseudo depth / confidence |
| Reflective / transparent robustness | 文章明确关注反光、透明表面和细节边界 | 与本项目工业反光表面相关，但其单帧 RGB 任务与 focus-stack 形貌恢复仍需区分 |
| Model scale and efficiency | 提供从 25M 到 1.3B parameters 的模型尺度 | 可作为真实样本辅助 inference 或轻量 qualitative comparison，不应作为直接 DFF/SFF 数值 SOTA |

Depth Anything V2 对本项目最有价值的定位是方法论参照和辅助先验，可支持三个论点：

1. ground truth 质量比数据来源“真实”本身更关键；对反光、透明、细结构和边界区域，真实传感器标签也可能有系统误差。
2. simulation-to-real 可以通过 pseudo-label / teacher-student bridge 增强，而非只依赖直接从合成域迁移到真实域。
3. 单帧基础模型可以作为真实样本的辅助 depth prior 或 qualitative sanity check，但不能替代 focus-stack 中的焦平面响应、DFF/GADFF prior 和 focal-difference volume。

在实验上，Depth Anything V2 可选做一个轻量补充：取真实焦堆中的最佳聚焦帧或 all-in-focus 融合图作为输入，输出单帧 relative depth，与 S2R-FocusNet 的真实样本形貌可视化并排展示。该结果只能作为 qualitative auxiliary comparison，不进入 synthetic MAE 主表。

## 6. 公平比较原则

| 问题 | 建议 |
|---|---|
| 输出尺度不一致 | 对 synthetic GT 使用统一高度单位；对只输出相对 depth/disparity 的方法，报告 scale-aligned MAE，同时标注是否经过 affine / min-max alignment |
| 输入帧数不同 | 优先统一为本项目的固定焦堆帧数；无法统一时报告采样策略，例如等间隔抽帧、中心帧补齐或模型默认帧数 |
| 预训练与重新训练混用 | 将 zero-shot pretrained、synthetic retraining、fine-tuning 分开列，不在同一数值栏中混淆 |
| 相机参数缺失 | 对 DDFS 等依赖相机设置的方法，明确记录焦距、焦平面间距、光圈或等效参数；无法提供时不做强行数值对比 |
| 真实样本无 GT | 只报告 no-reference metrics 和可视化，不宣称真实样本的 absolute height accuracy |
| 工业显微域差异 | 对自然场景模型增加 domain gap 说明，重点分析反光、弱纹理、周期纹理、眩光和微小缺陷边界 |

## 7. 推荐实验表结构

### Table 1: Synthetic Quantitative Comparison

列建议：

| Method | Type | Training Data | MAE | Edge MAE | High-Glare / High-Risk MAE | Runtime | Notes |
|---|---|---|---|---|---|---|---|

推荐方法顺序：

1. Original DFF
2. DFF + post-processing
3. Lee2013
4. Li2019
5. DDFFNet
6. DFV
7. DfF in the Wild 或 HybridDepth
8. Proposed S2R-FocusNet

### Table 2: Ablation Study

列建议：

| Variant | DFF / GADFF Prior | Focal Difference | Glare Prior | Domain Randomization | MAE | Edge MAE |
|---|---|---|---|---|---|---|

最低限度 ablation：

1. Full model
2. w/o DFF prior
3. w/o focal-difference volume
4. w/o glare-aware prior
5. w/o domain randomization
6. direct image-to-depth network

### Table 3: Real No-Reference Morphology Evaluation

列建议：

| Method | Roughness Stability | Spike Count | Dynamic Range | Edge Continuity | Visual Failure Mode |
|---|---|---|---|---|---|

真实样本建议只比较：

1. Original DFF
2. DFF + post-processing
3. Lee2013 / Li2019
4. DFV 或 DDFFNet
5. Proposed S2R-FocusNet

## 8. Related Work 推荐组织

1. Classical SFF / DFF and focus measures  
   Nayar and Nakagawa, Pertuz et al., Lee2013, Li2019。强调传统方法的可解释性、窗口尺度依赖、弱纹理失效和噪声敏感。

2. Learning-based depth from focus  
   DDFFNet, AiFDepthNet, DFV, DfF in the Wild, DDFS, HybridDepth, FAD, DualFocus, DDL-Recurrent SFF, SAS（待核验）。强调从单点 focus measure 转向焦堆序列建模、体数据建模、频域建模和物理约束。

3. Simulation-to-real depth / defocus learning  
   Focus on Defocus, DEReD, DDFS, DfF in the Wild。强调真实 depth GT 缺失时，仿真、光学模型、自监督和域随机化的价值。

4. Industrial surface defect morphology  
   表面缺陷检测综述、结构光/显微 3D 缺陷重建、Att-PU-Net 等。强调工业缺陷研究从 2D detection 走向 3D morphology 的必要性。

5. Monocular depth foundation models  
   Depth Anything V2, Marigold, ZoeDepth, Metric3D。强调大规模先验和 synthetic data 的趋势，同时说明其与 focus-stack 形貌恢复的任务差异。

建议在这一段中特别点出 Depth Anything V2：其“precise synthetic labels + pseudo-labeled real images”的训练路线为本项目的仿真数据构造、真实无标签样本利用和 teacher-student 后续扩展提供了直接方法论参照。

## 9. 对本项目最关键的写作判断

1. 最强外部对比是 DFV，因为它与本项目的 focal-difference 表征最接近。
2. 最能强化投稿故事的是 DfF in the Wild、DDFS 和 HybridDepth，因为它们直接关联真实相机、仿真到真实、相机参数不变性和单目先验融合。
3. 最适合解释真实样本无 GT 困境的是 AiFDepthNet 和 DEReD，它们提供了 AiF supervision 与 self-supervised defocus reconstruction 的方法论参照。
4. 最新文献必须覆盖 FAD、DualFocus、DDL-Recurrent SFF 和 Minimal Focal Stack；SAS 需要再次核验后再决定是否写入正式 Related Work。
5. Depth Anything V2 / Marigold 等基础模型可作为“大规模数据与视觉先验”的背景，主结果表应以 focus-stack 方法为中心。
6. Depth Anything V2 对本文最直接的启发是：合成标签质量、真实无标签图像覆盖度和 teacher-student pseudo-label bridge 可以共同构成 simulation-to-real 训练策略。

## 10. 可执行复现顺序

| 阶段 | 时间成本 | 目标 |
|---|---|---|
| Phase 1 | 1-2 周 | 复现 DDFFNet 和 DFV，在本项目 synthetic test set 上得到 MAE / edge MAE / high-risk MAE |
| Phase 2 | 2-3 周 | 尝试 DfF in the Wild 或 HybridDepth，至少得到一个 2022 以后强外部学习型对比 |
| Phase 3 | 3-4 周 | 若相机参数可整理，尝试 DDFS；若参数不足，保留为强相关讨论 |
| Phase 4 | 4-6 周 | 视时间尝试 AiFDepthNet / DEReD 的无监督或半监督变体，服务真实样本无 GT 叙事 |
| Phase 5 | 投稿前 | 补齐 FAD、DualFocus、DDL-Recurrent SFF、Minimal Focal Stack 的 Related Work 段落和 citation，并二次核验 SAS |

## 11. 参考来源

- DDFFNet / Deep Depth from Focus: https://github.com/soyers/ddff-pytorch ; https://github.com/hazirbas/ddff-toolbox
- DFV / Deep Depth from Focus with Differential Focus Volume: https://github.com/fuy34/DFV ; https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Deep_Depth_From_Focus_With_Differential_Focus_Volume_CVPR_2022_paper.pdf
- Learning Depth from Focus in the Wild: https://github.com/wcy199705/DfFintheWild ; https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610001.pdf
- DDFS / Deep Depth from Focal Stack with Defocus Model: https://github.com/yfujimura/DDFS ; https://link.springer.com/article/10.1007/s11263-023-01964-x
- AiFDepthNet: https://github.com/albert100121/AiFDepthNet ; https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Bridging_Unsupervised_and_Supervised_Depth_From_Focus_via_All-in-Focus_Supervision_ICCV_2021_paper.pdf
- DEReD: https://github.com/Ehzoahis/DEReD ; https://openaccess.thecvf.com/content/CVPR2023/papers/Si_Fully_Self-Supervised_Depth_Estimation_From_Defocus_Clue_CVPR_2023_paper.pdf
- Focus on Defocus / DefocusNet: https://github.com/dvl-tum/defocus-net ; https://openaccess.thecvf.com/content_CVPR_2020/papers/Maximov_Focus_on_Defocus_Bridging_the_Synthetic_to_Real_Domain_Gap_CVPR_2020_paper.pdf
- HybridDepth: https://github.com/cake-lab/HybridDepth ; https://openaccess.thecvf.com/content/WACV2025/html/Ganj_HybridDepth_Robust_Metric_Depth_Fusion_by_Leveraging_Depth_from_Focus_WACV_2025_paper.html
- FAD / Frequency-Aware Deep Depth from Focus: https://www.ijcai.org/proceedings/2025/241
- DualFocus: https://arxiv.org/abs/2509.21992 ; https://openreview.net/forum?id=OZUl49U6p6
- Robust Shape from Focus via Multiscale Directional Dilated Laplacian Recurrent Network: https://link.springer.com/article/10.1007/s11263-025-02396-1 ; https://arxiv.org/html/2512.10498v1
- SAS / Sequence Association for Shape from Focus（待核验）: https://www.computer.org/csdl/journal/tp/2025/10/11028609/27pwJbOE30Q
- Towards Minimal Focal Stack in Shape from Focus: https://openaccess.thecvf.com/content/CVPR2026W/3DMV/html/Ashfaq_Towards_Minimal_Focal_Stack_in_Shape_from_Focus_CVPRW_2026_paper.html
- Dense Depth from Event Focal Stack: https://arxiv.org/abs/2412.08120
- Depth Anything V2: https://depth-anything-v2.github.io/ ; https://github.com/DepthAnything/Depth-Anything-V2
- Depth Anything V2 arXiv HTML: https://arxiv.org/html/2406.09414v1
- Marigold: https://marigoldmonodepth.github.io/ ; https://github.com/prs-eth/marigold
- ZoeDepth: https://github.com/isl-org/ZoeDepth
- Metric3D / Metric3Dv2: https://github.com/YvanYin/Metric3D ; https://jugghm.github.io/Metric3Dv2/
