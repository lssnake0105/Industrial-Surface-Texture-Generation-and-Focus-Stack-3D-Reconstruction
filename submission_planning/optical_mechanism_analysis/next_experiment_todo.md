# 下一轮可执行实验 TODO

日期：2026-06-21  
目标：把“反光/眩光导致 DFF 失效”从故事线推进到可验证证据链。

## A. 焦堆证据补强

| 优先级 | 任务 | 产物 | 判断标准 |
|---|---|---|---|
| P0 | 对 `钥匙纹路100um` 选 3 个 ROI：高亮边缘、普通纹理、暗区 | ROI 逐层小图、亮度曲线、focus curve | 高亮 ROI 的 focus peak 是否与 p99/saturation peak 重合 |
| P0 | 对所有真实样本生成 saturation persistence map | `I>=0.90`、`I>=0.98` 两类图 | 是否能划分 recoverable / unrecoverable glare 区域 |
| P0 | 计算高亮区与非高亮区的 DFF spike count | 分区表格 | 高亮区是否贡献主要尖峰 |
| P1 | 做局部配准检查 | 每层相对位移曲线和配准前后对比 | 排除机械位移造成的伪机理 |
| P1 | 输出代表性失败案例图 | 原图焦层、DFF、Focus-ResUNet、残差图 | 可直接进入论文 Figure 1 或 Figure 4 |

## B. 光学仿真

| 优先级 | 任务 | 产物 | 判断标准 |
|---|---|---|---|
| P0 | 生成微米级高度图：V 谷、圆孔、刃脊、Perlin 粗糙 | height map + normal map | 是否覆盖当前真实样本形态 |
| P0 | 实现法线-接收孔径 glare-risk map | risk map | 高曲率/斜坡/边缘是否出现高风险 |
| P0 | 模拟不同 NA、照明锥角、粗糙度 | 参数扫描图 | 高亮风险是否随光学参数系统变化 |
| P1 | 加入 defocus PSF 和 clipping | 合成焦堆 | 能否产生离焦亮斑边缘和伪 focus peak |
| P1 | 与真实 `钥匙纹路100um` 的 high-intensity persistence 做形态对比 | side-by-side 图 | 是否能复现“边缘高亮持续存在”现象 |

## C. 模型与消融

| 优先级 | 任务 | 产物 | 判断标准 |
|---|---|---|---|
| P0 | 统一最终模型名为 Flare-FocusNet 或 Flare-Aware Focus-ResUNet | README、论文、图表统一命名 | 正文只出现一个最终方法 |
| P0 | 增加 saturation persistence / focal std prior | 输入通道说明和 ablation | high-risk MAE 是否下降 |
| P0 | 按 glare-risk mask 报告 synthetic MAE | 分区结果表 | 方法优势是否集中在反光风险区 |
| P1 | 适配 DFV | 外部 SOTA 表 | 至少 synthetic test set 可跑通 |
| P1 | 适配 DDFFNet | 外部学习型 baseline | 与 DFV 形成 learning-based baseline 组 |

## D. 真实验证

| 优先级 | 任务 | 产物 | 判断标准 |
|---|---|---|---|
| P0 | 固定曝光重拍代表样本 | 新焦堆 | 能否复现高亮层变化 |
| P1 | 包围曝光或不同曝光时间采集 | 多曝光焦堆 | 判断过曝是动态范围问题还是结构性反光问题 |
| P1 | 记录物镜 NA、焦层间距、曝光时间、照明设置 | metadata 表 | 支持 DDFS / optical simulation 参数 |
| P2 | 获取 WLI / 共聚焦 / 轮廓仪小规模 GT | 真实高度子集 | 真实 absolute MAE 可报告 |

## E. 投稿写作

| 优先级 | 任务 | 产物 |
|---|---|---|
| P0 | 重写 Introduction，突出 reflective microstructure / glare / DFF failure | Introduction 草稿 |
| P0 | 重写 Method，分成 simulator、flare prior、network 三段 | Method 草稿 |
| P0 | 更新 Related Work，加入 DFV、DfF in the Wild、DDFS、DEReD、Depth Anything V2、Att-PU-Net | Related Work 草稿 |
| P1 | 新增 Limitations，明确真实无绝对高度真值 | Discussion 段落 |
| P1 | 设计 Figure 1：问题机制图 | 图像生成或手工图 |

## 最小闭环

最小可投稿前闭环为：

1. `钥匙纹路100um` ROI 证据图；已生成 `roi_probe/钥匙纹路100um_roi_curves.png`、`roi_probe/钥匙纹路100um_roi_montage.png`。
2. 法线-孔径 glare-risk simulator；已生成 `glare_sim/` 下 4 类微表面风险图和 `glare_risk_simulation_note.md`。
3. saturation persistence prior 消融；
4. synthetic high-risk MAE 表；
5. DFV 或 DDFFNet 至少一个学习型外部 baseline；
6. 真实样本 no-reference 指标与限制说明。
