# 外部基线复现决策表

更新日期：2026-06-18  
用途：为后续真实执行外部 SOTA 复现时提供决策规则，减少盲目下载和无效适配。

## 1. 决策原则

| 原则 | 说明 |
|---|---|
| 任务同构优先 | 优先选择输入为 focus stack / focal stack 的方法 |
| 可公平评估优先 | 能输出 depth / focus index 并可与 synthetic GT 对齐的方法优先 |
| 低污染执行 | 外部仓库和中间数据放入 `tmp/` 或独立未跟踪目录 |
| 失败可降级 | 无法复现时保留 Related Work 讨论，不强行进入主结果表 |
| 记录可追溯 | 每个 baseline 记录 commit、环境、输入帧数、训练设置、尺度对齐方式 |

## 2. Baseline 执行顺序

| 顺序 | 方法 | 进入下一步条件 | 失败后处理 |
|---|---|---|---|
| 1 | DFV | 代码可运行，能读入 17-frame 或采样 focal stack | 若训练失败，尝试官方预训练 zero-shot；仍失败则写 Related Work + feasibility note |
| 2 | DDFFNet | 数据能转为官方格式，模型可输出 depth/disparity | 若依赖过旧，改做方法引用和复现失败说明 |
| 3 | HybridDepth | 能构造 focal stack + single-image / AiF 输入 | 若尺度不可比，只做 qualitative 或 Related Work |
| 4 | DfF in the Wild | 真实/合成焦堆可适配其输入流程 | 若相机模拟参数不足，作为 sim-to-real 相关工作 |
| 5 | DDFS | 相机参数或等效焦平面参数齐全 | 参数不足时不进入数值表 |
| 6 | Depth Anything V2 | 可对真实焦堆最佳聚焦帧或 AiF 图做 inference | 只做 auxiliary figure，不进入主表 |

## 3. DFV 决策细则

| 检查项 | 通过标准 | 不通过处理 |
|---|---|---|
| 代码获取 | 官方仓库可下载或已有源码 | 仅引用论文，不做复现 |
| 环境 | PyTorch/CUDA 版本可安装 | 尝试 CPU 小样例；仍失败则停止 |
| 数据接口 | 可读入 `[N,H,W]` 或 `[N,C,H,W]` 焦堆 | 写 adapter；若框架强绑定原数据集，改 zero-shot |
| 输出 | 可得到 depth/focus probability | 若只输出中间概率，按 focus position 转 height |
| 尺度对齐 | 可映射到 um 或做明确 scale alignment | 不能对齐时不进主表 |
| 训练成本 | 可在可用 GPU/CPU 预算内跑小规模训练 | 改用预训练或只做 qualitative |

**最小成功定义：** 在 7 个 synthetic test samples 上得到可解释输出，并能计算 MAE、edge MAE、high-risk MAE 中至少 mean MAE。

## 4. DDFFNet 决策细则

| 检查项 | 通过标准 | 不通过处理 |
|---|---|---|
| 数据格式 | 可生成 HDF5 或官方 dataloader 支持格式 | 写最小 adapter；若成本过高则暂停 |
| 输入帧数 | 支持 17 frames 或可等间隔采样 | 记录采样策略 |
| 输出类型 | depth/disparity 可转换 | 只能输出相对值时做 scale-aligned MAE |
| 依赖版本 | 老代码能在隔离环境运行 | 保留为 Related Work 和 early deep DFF baseline |

**最小成功定义：** 在 synthetic test set 上跑通一个统一模型输出，并给出至少 mean MAE。

## 5. HybridDepth 决策细则

| 检查项 | 通过标准 | 不通过处理 |
|---|---|---|
| 输入构造 | 可从焦堆生成 single-image / AiF 输入 | 若 AiF 不可靠，使用最佳聚焦帧 |
| 输出尺度 | metric depth 或可 scale-align | 不可对齐时做 qualitative |
| 域差异 | 自然场景偏置可在 Discussion 说明 | 不用于强结论 |

**最小成功定义：** 生成 synthetic 或 real 样本的可视化结果，能说明 focal stack + single-image prior 的优势或局限。

## 6. Depth Anything V2 决策细则

| 检查项 | 通过标准 | 不通过处理 |
|---|---|---|
| 输入 | 最佳聚焦帧、中心帧或 AiF 图可生成 | 不做实验，只引用训练策略 |
| 输出 | relative depth map 可视化 | 不计算主表 MAE |
| 对比 | 与 S2R-FocusNet 真实样本形貌图并排 | 只用于 Discussion |

**最小成功定义：** 得到一张真实样本 relative depth qualitative figure，并明确其作为单帧先验的边界。

## 7. 结果进入论文表格的门槛

| 表格 | 进入条件 |
|---|---|
| Synthetic main table | 使用同一 test split；输出可与 height GT 对齐；记录训练设置和 scale alignment |
| Synthetic auxiliary table | 可输出相对 depth，但只可做 scale-aligned error |
| Real no-reference table | 不需要 GT，但必须使用同一真实样本和同一 no-reference 指标 |
| Qualitative figure | 输出可视化清晰，且 caption 明确任务差异和限制 |
| Related Work only | 代码不可复现、输出不可比或任务差异过大 |

## 8. 失败记录模板

```text
Method:
Source:
Date:
Code version / commit:
Environment:
Input adapter attempted:
Failure point:
Can be used in paper as:
Next action:
```

## 9. 推荐优先执行计划

### Day 1: baseline inventory

1. 记录 DFV、DDFFNet、HybridDepth、Depth Anything V2 的官方链接、代码状态和依赖。
2. 不下载到主项目目录；如需下载，放入 `tmp/external_repos/`。
3. 建立 `tmp/external_baseline_data/` 中间数据目录。

### Day 2-3: DFV adapter

1. 准备 1 个 synthetic sample 的 17-frame stack。
2. 跑通官方最小 inference 或 dataloader。
3. 记录输出尺度与 height GT 的映射。

### Day 4-5: DFV evaluation

1. 扩展到 7 个 synthetic test samples。
2. 计算 mean MAE、edge MAE、high-risk MAE。
3. 生成 P10 difficult sample 可视化。

### Day 6-7: DDFFNet adapter

1. 转换 HDF5 或等价格式。
2. 尝试 inference / training。
3. 若依赖阻碍过大，写复现失败说明并转 HybridDepth。

## 10. 停止条件

| 情况 | 停止动作 |
|---|---|
| 方法无法读入 focus stack | 停止主表复现，转 Related Work |
| 输出无法和 GT 建立任何尺度关系 | 停止 synthetic 数值对比，转 qualitative |
| 依赖环境需要大规模破坏本项目环境 | 停止本地复现，改隔离环境或记录风险 |
| 复现时间超过计划且没有可视化输出 | 暂停，优先完成已有论文结构 |
| 方法任务与 focus-stack reconstruction 差异过大 | 不进入主表 |
