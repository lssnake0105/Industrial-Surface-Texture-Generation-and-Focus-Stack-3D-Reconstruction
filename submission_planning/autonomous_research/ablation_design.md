# 消融实验设计

更新日期：2026-06-18  
目标：将当前模型叙事收束到一个 final method，并用消融实验证明 DFF/GADFF prior、focal-difference volume、glare-aware cue 和 domain randomization 的必要性。

## 1. 消融主问题

| 编号 | 研究问题 | 对应投稿论点 |
|---|---|---|
| Q1 | DFF/GADFF prior 是否降低了模型学习难度？ | prior-guided correction 比直接 image-to-depth 更适合小规模合成数据 |
| Q2 | focal-difference volume 是否提供了焦平面方向的关键变化信息？ | 模型需要利用 focus-stack 的轴向响应，不能只依赖单帧纹理 |
| Q3 | glare-aware cue 是否能改善反光/眩光区域稳定性？ | 反光工业表面需要针对 high-risk 区域的显式处理 |
| Q4 | domain randomization 是否提升真实样本形貌稳定性？ | simulation-to-real 的关键在于扰动覆盖与真实域鲁棒性 |
| Q5 | residual correction / bounded correction 是否优于完全自由预测？ | 保留传统 DFF 的物理锚点可减少异常尖峰 |

## 2. 推荐消融组合

| Variant | DFF/GADFF Prior | Focal Difference | Glare Cue | Domain Randomization | Residual / Bounded Correction | 目的 |
|---|---|---|---|---|---|---|
| Full S2R-FocusNet | yes | yes | yes | yes | yes | 最终方法 |
| Direct image-to-depth | no | no | no | yes | no | 检验先验引导的必要性 |
| w/o DFF/GADFF prior | no | yes | yes | yes | no/yes | 检验传统 focus prior 的贡献 |
| w/o focal difference | yes | no | yes | yes | yes | 检验焦向差分表征的贡献 |
| w/o glare cue | yes | yes | no | yes | yes | 检验 high-risk 区域处理 |
| w/o domain randomization | yes | yes | yes | no | yes | 检验真实域稳定性 |
| unbounded prediction | yes | yes | yes | yes | no | 检验残差/边界保护对尖峰的抑制 |

## 3. 指标设计

### 3.1 Synthetic metrics

| 指标 | 计算范围 | 解释 |
|---|---|---|
| Mean MAE | valid mask | 整体高度误差 |
| Edge MAE | edge mask | 深度突变、缺陷边界、形貌轮廓稳定性 |
| High-Risk MAE | high-risk mask | 反光、弱纹理、眩光区域误差 |
| P90 Error | valid mask | 尾部异常误差 |
| Spike Count | thresholded error / low confidence region | 异常尖峰控制能力 |

### 3.2 Real no-reference metrics

| 指标 | 解释 | 注意 |
|---|---|---|
| Roughness Stability | 输出表面是否过度噪声化 | 不等同于真实 roughness accuracy |
| Low-Conf Spike Count | 异常尖峰数量 | 用于证明稳定性 |
| Relative Dynamic Range | 输出是否保留形貌起伏 | 过低可能过度平滑，过高可能尖峰失控 |
| Edge Retention | 与输入边缘/纹理结构的相关性 | 只作为结构一致性参考 |
| Profile Consistency | 缺陷截面曲线是否连续 | 适合图示，不宜过度量化 |

## 4. 预期结果与解释

| Variant | 预期变化 | 若结果不符合预期，如何解释 |
|---|---|---|
| Direct image-to-depth | synthetic MAE 上升，真实样本 spike 增多 | 说明模型容量或数据增强已部分补偿 prior；需检查是否发生过拟合 |
| w/o DFF/GADFF prior | edge MAE 与 P90 error 上升 | 若变化小，说明 focal difference 已吸收大部分 focus prior |
| w/o focal difference | 边界、周期纹理和焦向歧义样本变差 | 若变化小，说明网络可能主要利用 DFF prior；需要增加焦向差分可视化 |
| w/o glare cue | high-risk MAE 或真实 glare 区域 spike 增多 | 若变化小，说明当前 glare cue 设计不足或 high-risk mask 不够敏感 |
| w/o domain randomization | synthetic 结果可能接近，真实样本稳定性下降 | 若真实结果无变化，说明当前真实样本覆盖不足或随机化范围偏离真实域 |
| unbounded prediction | 真实样本 spike count 增加 | 若更好，说明 residual bound 限制过强，需要重设 correction range |

## 5. 关键样本选择

| 样本 | 用途 |
|---|---|
| `test_V谷_P10_宽谷粗糙平底` | 最难样本之一，用于展示宽谷、边界和 high-risk 失败 |
| `test_周期_条纹粗糙` | 周期纹理，用于观察 frequency-aware / focal-difference 是否有效 |
| `test_阶跃_柏林粗糙` | 阶跃边界，用于 edge MAE 和 profile curve |
| `test_复合腐蚀凹坑` | 复合缺陷，用于工业形貌可视化 |
| 真实钥匙纹路/磕碰孔/钥匙尖头 | 用于 no-reference morphology stability |

## 6. 表格模板

### Synthetic ablation table

| Variant | Mean MAE | Edge MAE | High-Risk MAE | P90 Error | Main Failure |
|---|---:|---:|---:|---:|---|
| Full S2R-FocusNet | 待填 | 待填 | 待填 | 待填 | - |
| Direct image-to-depth | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o DFF/GADFF prior | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o focal difference | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o glare cue | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o domain randomization | 待填 | 待填 | 待填 | 待填 | 待分析 |

### Real no-reference ablation table

| Variant | Roughness | Spike Count | Dynamic Range | Edge Retention | Visual Comment |
|---|---:|---:|---:|---:|---|
| Full S2R-FocusNet | 待填 | 待填 | 待填 | 待填 | 待分析 |
| Direct image-to-depth | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o DFF/GADFF prior | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o focal difference | 待填 | 待填 | 待填 | 待填 | 待分析 |
| w/o glare cue | 待填 | 待填 | 待填 | 待填 | 待分析 |

## 7. 图示计划

| 图 | 内容 | 目的 |
|---|---|---|
| Ablation bar chart | Mean MAE / Edge MAE / High-Risk MAE | 快速展示各模块贡献 |
| Difficult sample panel | GT, Full, w/o prior, w/o focal difference, w/o glare cue | 展示边界与反光区域失败模式 |
| Profile curve | 选定截面上的 GT 与各 ablation 曲线 | 说明形貌连续性 |
| Real spike map | 真实样本不同 ablation 的异常尖峰位置 | 说明 no-reference 稳定性 |

## 8. 写作模板

> To isolate the contribution of each component, we evaluate several ablated variants of S2R-FocusNet under the same synthetic train/test split. Removing the DFF/GADFF prior tests whether traditional focus-based estimates provide useful physical anchors. Removing the focal-difference volume evaluates the importance of axial focus-response variation. Removing the glare-aware cue examines whether reflective high-risk regions require explicit modeling. The real samples are evaluated only with no-reference morphology metrics because calibrated height ground truth is unavailable.

中文说明：

> 为了拆分各模块贡献，我们在相同合成训练/测试划分下比较 S2R-FocusNet 的多个消融版本。去除 DFF/GADFF 先验用于检验传统聚焦估计是否提供有效物理锚点；去除焦向差分体用于检验轴向聚焦响应变化的重要性；去除眩光感知线索用于检验反光高风险区域是否需要显式建模。真实样本由于缺少校准高度真值，只使用无参考形貌指标进行评价。
