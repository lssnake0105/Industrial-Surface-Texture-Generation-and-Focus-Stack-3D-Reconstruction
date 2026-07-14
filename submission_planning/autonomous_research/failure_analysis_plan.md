# Failure Analysis Plan

更新日期：2026-06-18  
用途：规划论文中的失败分析，明确哪些失败是当前方法的限制，哪些失败能支撑后续研究方向。

## 1. 失败分析目标

失败分析不只是展示错误结果，而是回答三个审稿问题：

1. 当前方法在什么条件下仍不稳定？
2. 错误来自数据域差异、focus ambiguity、glare，还是模型结构限制？
3. 后续改进应从 high-risk modeling、external baseline、real GT calibration 还是 pseudo-label training 入手？

## 2. 推荐分析对象

| 对象 | 类型 | 选择原因 | 主要观察 |
|---|---|---|---|
| `test_V谷_P10_宽谷粗糙平底` | synthetic | 宽谷、粗糙平底、深度范围大 | 边界误差、谷底平滑、high-risk 区域 |
| `test_周期_条纹粗糙` | synthetic | 周期纹理容易干扰 focus measure | 周期伪影、频域/焦向歧义 |
| `test_阶跃_柏林粗糙` | synthetic | 阶跃边界明确 | edge MAE、profile continuity |
| `test_复合腐蚀凹坑` | synthetic | 复合工业缺陷形貌 | 局部凹坑深度与边界 |
| 真实磕碰孔 5um | real | 真实微小凹坑 | spike suppression、形貌连续性 |
| 真实钥匙纹路 100um | real | 纹理/反光复杂 | 纹理误判、动态范围 |
| 真实钥匙尖头 50um | real | 尖锐边缘 | 边界连续性、尖峰 |

## 3. 错误类型分类

| 错误类型 | 表现 | 可能原因 | 对应改进 |
|---|---|---|---|
| Boundary oversmoothing | 缺陷边界被抹平 | loss 偏整体 MAE，缺少 edge-aware 约束 | edge loss, profile loss, boundary mask |
| High-risk region error | glare/弱纹理区域误差大 | focus response 不可靠，mask 或 cue 不足 | glare-aware confidence, high-risk weighted loss |
| Periodic texture artifact | 周期条纹导致高度周期伪影 | focus measure 被纹理频率误导 | frequency-aware features, FAD-inspired module |
| Spike artifact | 局部高度尖峰 | DFF prior 不稳定或 unrestricted correction | bounded residual, confidence filtering |
| Flattened morphology | 起伏过度平滑 | 网络偏向平均化 | dynamic range loss, profile constraint |
| Domain gap artifact | 真实样本与合成表现不一致 | 仿真 reflectance/noise/focus response 不充分 | domain randomization, pseudo-label real training |

## 4. Synthetic Failure Analysis

### 4.1 P10 V-Valley

分析问题：

1. 宽谷边界是否被准确恢复？
2. 谷底是否出现过度平滑或局部尖峰？
3. high-risk mask 内误差是否显著高于全图？
4. Focus-ResUNet 的 edge MAE 改善是否来自边界修正？

建议图：

| 子图 | 内容 |
|---|---|
| a | GT height map |
| b | Original DFF |
| c | Lee2013 / Li2019 |
| d | Focus-ResUNet |
| e | error map |
| f | cross-section profile |

### 4.2 Periodic Stripe

分析问题：

1. 周期纹理是否造成重复高度伪影？
2. DFF/GADFF 是否把纹理频率误判为焦点变化？
3. focal-difference representation 是否减弱周期伪影？
4. 是否需要引入 frequency-aware module 或 FAD 相关设计？

建议图：

| 子图 | 内容 |
|---|---|
| a | representative focus frame |
| b | GT height |
| c | Original DFF prediction |
| d | Focus-ResUNet prediction |
| e | Fourier / frequency note if available |
| f | profile curve |

### 4.3 Step Boundary

分析问题：

1. 阶跃边缘位置是否准确？
2. 边缘附近是否出现 halo 或 transition blur？
3. edge MAE 能否解释视觉差异？
4. 是否需要边界加权 loss？

## 5. Real Failure Analysis

真实样本没有 calibrated height GT，因此分析语言必须限定为 morphology stability、visual plausibility、spike suppression 和 relative profile consistency。

### 5.1 Real Pit 5um

观察：

1. 凹坑是否形成连续局部形貌；
2. DFF 是否出现低置信尖峰；
3. Focus-ResUNet 是否减少 spike；
4. dynamic range 是否过度压缩。

### 5.2 Real Key Texture 100um

观察：

1. 纹理是否被误解释为高度起伏；
2. 反光区域是否造成异常 ridge；
3. 方法是否保持相对形貌连续性。

### 5.3 Real Key Tip 50um

观察：

1. 尖锐边缘是否被过度平滑；
2. 局部尖峰是否被抑制；
3. profile 是否有不合理跳变。

## 6. Metrics for Failure Analysis

| 指标 | Synthetic | Real | 用途 |
|---|---|---|---|
| error map | yes | no | 定位误差区域 |
| edge MAE | yes | no | 边界误差 |
| high-risk MAE | yes | no | glare/weak texture 区域误差 |
| spike count | yes | yes | 异常尖峰 |
| profile curve | yes | yes | 形貌连续性 |
| dynamic range | yes | yes | 是否过度平滑 |
| visual consistency | yes | yes | 解释图像结构 |

## 7. 写作模板

### Synthetic failure wording

> The P10 V-valley sample reveals that wide concave structures remain challenging for both traditional focus measures and learning-based correction. Although the proposed model reduces the overall MAE and improves boundary continuity, errors remain concentrated in high-risk regions where glare and weak texture reduce the reliability of focus responses.

### Real failure wording

> Since calibrated real height ground truth is unavailable, the real-sample failure analysis focuses on no-reference morphology behavior. The proposed model suppresses many low-confidence spikes observed in direct DFF outputs, but some reflective regions still show compressed dynamic range or over-smoothed local morphology.

### Limitation wording

> These observations indicate that high-risk reflective regions should be treated as a separate modeling target rather than only as ordinary pixels in a global reconstruction loss.

## 8. 后续实验建议

| 方向 | 目的 |
|---|---|
| high-risk weighted loss | 改善 glare/weak texture 区域 |
| edge-aware loss | 改善缺陷边界 |
| frequency-aware feature | 改善周期纹理 |
| confidence calibration | 区分可靠 DFF prior 和失效区域 |
| pseudo-label real training | 缓解 synthetic-to-real gap |
| calibrated real subset | 验证真实绝对高度误差 |

## 9. 失败分析红线

1. 真实样本不写 absolute error。
2. 真实 profile 只解释 relative morphology。
3. 不把 high-risk 区域问题写成已经解决。
4. 不用单个样本证明全部工业场景泛化。
5. 不把 Depth Anything V2 单帧结果作为 focus-stack 失败/成功的直接判据。
