# 数据接口与外部基线适配说明

更新日期：2026-06-18  
用途：为 DFV、DDFFNet、HybridDepth、Depth Anything V2 等外部方法准备统一数据接口，避免后续复现时临时改动项目原始数据。

## 1. 当前数据资产

### 1.1 Synthetic split

来源：`论文与PPT制作项目包/03_Data/synthetic_training/dataset_split.csv`

| Split | 样本数 | 分辨率 | Stack Layers | Depth Range | z-step | 主要表面类型 |
|---|---:|---|---:|---|---|---|
| train | 12 | 640x360 | 17 | 860-1260 um | 53.75-78.75 um | mountain, ridge, a_ridge, step, periodic, v_valley, pit/groove |
| validation | 5 | 640x360 | 17 | 900-1260 um | 56.25-78.75 um | mountain, ridge, step, periodic, pit/groove |
| test | 7 | 640x360 或 960x540 | 17 | 920-1420 um | 57.5-88.75 um | v_valley, a_ridge, mountain, ridge, step, periodic, pit/groove |

关键字段：

| 字段 | 含义 | 对外部方法的作用 |
|---|---|---|
| `split` | train / validation / test | 保持所有模型公平划分 |
| `sample` | 样本 ID | 作为结果表主键 |
| `category` | 表面类别 | 用于按缺陷类型分析失败模式 |
| `resolution` | 图像尺寸 | 判断是否需要 resize / crop |
| `depth_range_um` | 高度范围 | 输出尺度还原与 MAE 单位 |
| `stack_layers` | 焦堆帧数 | 外部模型输入帧数适配 |
| `z_step_um` | 相邻焦平面间距 | depth/disparity/focus index 转换 |
| `stray_level` | 杂散光/眩光强度近似 | domain randomization 与 high-risk 分析 |
| `surface_baseline` | 基础几何类型 | 按几何形态分析 |
| `surface_noise` | 表面噪声类型 | 关联周期纹理/柏林噪声/分形噪声 |

### 1.2 Existing metric files

| 文件 | 用途 |
|---|---|
| `论文与PPT制作项目包/03_Data/algorithm_comparison/paper_algorithm_comparison_metrics.csv` | 现有方法在 7 个 synthetic test samples 上的统一对比 |
| `论文与PPT制作项目包/03_Data/synthetic_training/final_metrics.csv` | Focus-ResUNet / DFF / GADFF 在 train/validation/test 上的训练期指标 |
| `论文与PPT制作项目包/03_Data/real_sample_comparison/real_midterm_method_summary.csv` | 真实样本 no-reference 指标汇总 |
| `论文与PPT制作项目包/03_Data/real_sample_comparison/real_midterm_all_algorithm_metrics.csv` | 真实样本逐方法细粒度指标 |

## 2. 推荐统一中间格式

外部方法复现时，建议先导出到独立临时目录，例如 `tmp/external_baseline_data/`，不要覆盖项目已有数据。

### 2.1 样本目录结构

```text
sample_id/
  meta.json
  stack/
    000.png
    001.png
    ...
    016.png
  height_gt.npy
  masks/
    valid_mask.png
    edge_mask.png
    high_risk_mask.png
  priors/
    dff_depth.npy
    gadff_depth.npy
    focus_confidence.npy
```

### 2.2 `meta.json` 字段

```json
{
  "sample_id": "test_V谷_P10_宽谷粗糙平底",
  "split": "test",
  "category": "P10 V谷-宽谷粗糙平底",
  "resolution": [960, 540],
  "stack_layers": 17,
  "focus_positions_um": [0.0, 75.0, 150.0],
  "z_step_um": 75.0,
  "depth_range_um": 1200.0,
  "height_unit": "um",
  "surface_baseline": "v_valley",
  "surface_noise": "perlin",
  "stray_level": 0.2,
  "has_height_gt": true
}
```

说明：

| 字段 | 规则 |
|---|---|
| `resolution` | 使用 `[width, height]`，避免和数组 `[H, W]` 混淆 |
| `focus_positions_um` | 若没有绝对物理起点，可用以 0 开始的相对焦平面位置 |
| `height_gt.npy` | 使用 um 单位，和项目 MAE 指标一致 |
| `valid_mask.png` | 1 表示参与评价区域，0 表示边界或无效区域 |
| `edge_mask.png` | 用于 edge MAE |
| `high_risk_mask.png` | 用于 glare / weak texture / high-risk MAE |

## 3. 输出尺度对齐

外部方法常输出 depth、disparity、focus probability 或归一化 relative depth。投稿表格必须明确尺度处理。

| 输出类型 | 建议对齐方式 | 可否进入主表 |
|---|---|---|
| metric depth / height | 直接换算到 um | 可以 |
| focus index | `height_um = focus_index * z_step_um` 或按焦平面位置查表 | 可以 |
| disparity | 需要按官方定义转 depth，再与 GT 做 scale alignment | 可以，但必须标注 |
| relative depth | 仅做 affine alignment 后报告 scale-aligned error | 谨慎进入主表 |
| single-image relative depth | 只做 qualitative auxiliary | 不进入 synthetic MAE 主表 |

推荐同时报告：

1. `Raw MAE`：外部方法可直接输出物理尺度时使用。
2. `Scale-aligned MAE`：输出相对深度时使用，需在表注说明。
3. `Edge MAE`：只在 `edge_mask` 区域计算。
4. `High-risk MAE`：只在 `high_risk_mask` 区域计算。

## 4. 方法适配要点

### 4.1 DFV

| 适配点 | 建议 |
|---|---|
| 输入帧数 | 优先使用全部 17 帧；若模型要求固定帧数，采用等间隔采样并记录索引 |
| 输入格式 | 将 `[N, H, W]` 或 `[N, 3, H, W]` 转为官方 dataloader 所需格式 |
| 输出 | depth / focus probability |
| 训练 | 首选 synthetic train split 训练、validation split 调参、test split 评估 |
| 公平性 | 记录是否使用官方预训练，避免和本项目 synthetic-only training 混淆 |

### 4.2 DDFFNet

| 适配点 | 建议 |
|---|---|
| 输入格式 | 可能需要 HDF5 或类似 dataset 文件 |
| 分辨率 | 先统一 resize/crop 策略，确保 GT 同步变换 |
| 输出尺度 | 若输出 disparity，需要明确转换或 alignment |
| 训练 | 从头训练或 fine-tune 都要单独标注 |

### 4.3 HybridDepth

| 适配点 | 建议 |
|---|---|
| 输入 | focal stack + 单帧/all-in-focus 图 |
| 风险 | 原任务可能偏自然场景或移动端，显微反光域差异较大 |
| 使用方式 | 若能跑通可进入增强表；若输出尺度难以匹配，作为 qualitative 或 Related Work |

### 4.4 Depth Anything V2

| 适配点 | 建议 |
|---|---|
| 输入 | 最佳聚焦帧、中心帧或 all-in-focus 图 |
| 输出 | relative depth |
| 使用方式 | 真实样本 qualitative auxiliary figure |
| 禁止事项 | 不纳入 synthetic MAE 主表，不与 focus-stack 方法直接宣称数值优劣 |

## 5. 评估协议

### 5.1 Synthetic test set

固定使用 7 个 test samples：

| Sample | Category | Resolution | Depth Range (um) | z-step (um) |
|---|---|---|---:|---:|
| `test_V谷_P10_宽谷粗糙平底` | P10 V谷-宽谷粗糙平底 | 960x540 | 1200 | 75.0 |
| `test_A型突起刃脊_柏林粗糙` | A型刃脊-柏林粗糙 | 960x540 | 1280 | 80.0 |
| `test_山峰_分形粗糙` | 山峰-分形粗糙 | 640x360 | 1100 | 68.75 |
| `test_山脊_柏林粗糙` | 山脊-柏林粗糙 | 640x360 | 1020 | 63.75 |
| `test_阶跃_柏林粗糙` | 阶跃-柏林粗糙 | 640x360 | 1040 | 65.0 |
| `test_周期_条纹粗糙` | 周期-条纹粗糙 | 640x360 | 920 | 57.5 |
| `test_复合腐蚀凹坑` | 腐蚀凹坑-复合 | 960x540 | 1420 | 88.75 |

### 5.2 Real no-reference set

真实样本没有 calibrated height GT。外部方法在真实样本上的评价只使用：

1. roughness stability；
2. spike count；
3. relative dynamic range；
4. edge retention / edge continuity；
5. profile curve consistency；
6. visual failure mode。

## 6. 结果记录模板

```csv
method,training_setting,sample,mae_um,edge_mae_um,high_risk_mae_um,scale_alignment,runtime_ms,notes
DFV,synthetic_retrain,test_V谷_P10_宽谷粗糙平底,,,,,,
DDFFNet,synthetic_retrain,test_V谷_P10_宽谷粗糙平底,,,,,,
HybridDepth,zero_shot_or_finetune,test_V谷_P10_宽谷粗糙平底,,,,,,
```

## 7. 风险控制

1. 所有外部方法输出先进入临时目录，人工确认后再决定是否纳入论文材料。
2. 每个方法必须记录输入帧数、resize 策略、训练数据、预训练权重和尺度对齐方式。
3. 对真实样本只写 no-reference 评价。
4. 对无法公平适配的方法，放入 Related Work 或 Discussion，不强行进入结果主表。
5. 后续若下载外部仓库，优先使用 `tmp/external_repos/` 并确认不会被误提交。
