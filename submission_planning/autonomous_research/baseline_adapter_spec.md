# Baseline Adapter Specification

更新日期：2026-06-18  
用途：为后续 DFV、DDFFNet、HybridDepth 等外部基线复现设计数据适配规范。  
边界：本文档只定义接口和日志规范，不下载外部仓库，不生成大文件，不修改现有项目数据。

## 1. Adapter 目标

1. 将项目 synthetic focus-stack 数据转换为外部 baseline 可读取的标准中间格式。
2. 保持 train / validation / test split 不变。
3. 记录输入帧数、resize/crop、尺度对齐、训练设置和输出单位。
4. 所有中间数据写入 `tmp/external_baseline_data/`，避免污染项目主资源。
5. 所有外部仓库建议写入 `tmp/external_repos/`，避免误提交。

## 2. 推荐目录结构

```text
tmp/
  external_baseline_data/
    manifest.csv
    samples/
      sample_id/
        meta.json
        stack/
          000.png
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
  external_baseline_results/
    method_name/
      run_config.json
      predictions/
      metrics.csv
      logs/
  external_repos/
    DFV/
    ddff-pytorch/
```

说明：`tmp/` 当前已是非核心临时区。后续若需要长期保留外部结果，应先确认 `.gitignore` 和提交边界。

## 3. Manifest 设计

### 3.1 `manifest.csv`

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `sample_id` | string | `test_V谷_P10_宽谷粗糙平底` | 样本主键 |
| `split` | string | `test` | train / validation / test |
| `category` | string | `P10 V谷-宽谷粗糙平底` | 缺陷类别 |
| `width` | int | 960 | 图像宽度 |
| `height` | int | 540 | 图像高度 |
| `stack_layers` | int | 17 | 焦堆帧数 |
| `z_step_um` | float | 75.0 | 相邻焦平面间距 |
| `depth_range_um` | float | 1200.0 | 高度范围 |
| `surface_baseline` | string | `v_valley` | 基础几何 |
| `surface_noise` | string | `perlin` | 纹理/噪声 |
| `stray_level` | float | 0.2 | 杂散光/眩光等级 |
| `has_gt` | bool | true | 是否有 height GT |
| `stack_path` | string | `samples/.../stack` | 焦堆目录 |
| `gt_path` | string | `samples/.../height_gt.npy` | GT 路径 |

### 3.2 `run_config.json`

```json
{
  "method": "DFV",
  "source": "official_or_local",
  "code_version": "unknown",
  "input_frames": 17,
  "frame_sampling": "all",
  "resize": null,
  "training_setting": "synthetic_retrain",
  "pretrained_weights": null,
  "scale_alignment": "raw_um",
  "evaluation_split": "test",
  "notes": ""
}
```

## 4. Conversion Rules

| 项目 | 规则 |
|---|---|
| frame order | 按焦平面顺序保存为 `000.png` 到 `016.png` |
| image dtype | PNG 使用 8-bit 或 16-bit 时必须记录 |
| normalization | 外部模型需要归一化时写入 `run_config.json` |
| height unit | `height_gt.npy` 统一使用 um |
| focus position | 用 `z_step_um` 和 frame index 构造相对焦平面 |
| resize/crop | 必须同步处理 stack、height_gt、masks |
| invalid pixels | 用 `valid_mask` 排除 |
| edge/high-risk | synthetic 评价时分别使用 `edge_mask` 和 `high_risk_mask` |

## 5. Method-Specific Adapter

### 5.1 DFV

| 项目 | 规格 |
|---|---|
| 输入 | 17-frame focus stack，必要时转 `[N, C, H, W]` |
| 帧数策略 | 默认 all frames；如官方模型固定帧数，采用等间隔采样并记录 |
| 输出 | depth / focus probability |
| 尺度 | focus probability -> expected focus position -> height um |
| 评价 | MAE, edge MAE, high-risk MAE |
| 最小验收 | 单样本 inference 输出非空预测图 |

### 5.2 DDFFNet

| 项目 | 规格 |
|---|---|
| 输入 | HDF5 或官方 dataloader 支持的焦堆格式 |
| 帧数策略 | 17 frames 或等间隔采样 |
| 输出 | depth / disparity |
| 尺度 | disparity 输出需做 scale alignment 或按官方公式转换 |
| 评价 | MAE, edge MAE, high-risk MAE |
| 最小验收 | 7 个 test samples 可批量输出预测 |

### 5.3 HybridDepth

| 项目 | 规格 |
|---|---|
| 输入 | focal stack + selected frame / all-in-focus image |
| 输出 | metric or relative depth |
| 尺度 | metric 可进主表；relative 只做 auxiliary |
| 评价 | synthetic scale-aligned metrics 或 real qualitative |
| 最小验收 | 至少生成真实样本 qualitative result |

### 5.4 Depth Anything V2

| 项目 | 规格 |
|---|---|
| 输入 | real best-focus frame / AiF image |
| 输出 | relative depth |
| 评价 | qualitative only |
| 禁止 | 不进入 synthetic main MAE table |
| 最小验收 | 一张真实样本辅助可视化 |

## 6. Metrics Output

### 6.1 `metrics.csv`

```csv
method,run_id,training_setting,sample_id,split,mae_um,edge_mae_um,high_risk_mae_um,p90_um,scale_alignment,runtime_ms,notes
DFV,dfv_run_001,synthetic_retrain,test_V谷_P10_宽谷粗糙平底,test,,,,,,,
```

### 6.2 `real_metrics.csv`

```csv
method,run_id,sample_id,roughness,edge_retention,relative_dynamic_range,low_conf_spike_count,notes
DFV,dfv_run_001,real_key_texture,,,,,
```

## 7. Logging Requirements

每次 baseline run 必须记录：

1. method name；
2. code source / commit；
3. environment；
4. input frame count；
5. resize/crop；
6. train/validation/test split；
7. scale alignment；
8. output unit；
9. failed samples；
10. whether result is eligible for main table。

## 8. Main Table Eligibility

| 条件 | 必须满足 |
|---|---|
| 使用同一 synthetic test split | yes |
| 输出可与 height GT 对齐 | yes |
| 记录训练设置 | yes |
| 记录尺度对齐 | yes |
| 没有使用 test GT 进行训练或调参 | yes |
| 真实样本不报告 absolute error | yes |

## 9. Adapter Stop Conditions

| 条件 | 处理 |
|---|---|
| 外部模型强依赖无法获得的数据字段 | 停止主表适配，转 Related Work |
| 输出尺度完全不可解释 | 只做 qualitative 或停止 |
| 依赖安装会影响当前项目环境 | 转隔离环境或停止 |
| 中间数据体积过大 | 缩小到单样本 smoke test |
| 结果无法复现 | 不进入论文主表 |

## 10. Next Implementation Step

后续真正执行时，建议只先实现一个 smoke-test adapter：

1. 选择 `test_V谷_P10_宽谷粗糙平底`；
2. 导出 17-frame stack；
3. 保存 `meta.json`；
4. 生成空白 `run_config.json`；
5. 不训练模型，只验证外部 dataloader 能否读取样本。

这个 smoke test 成功后，再进入 DFV 或 DDFFNet 的完整适配。
