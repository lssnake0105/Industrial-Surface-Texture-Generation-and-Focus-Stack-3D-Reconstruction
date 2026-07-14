# DFV Smoke-Test Checklist

更新日期：2026-06-18  
用途：为后续 DFV 单样本 smoke test 提供执行清单。  
边界：本文档只定义检查项，不下载外部仓库，不运行训练，不生成中间数据。

## 1. Smoke Test Goal

The smoke test is successful if one synthetic focus-stack sample can be converted into the expected external-baseline format and loaded by the DFV pipeline without touching the original project resources.

Minimum target sample:

```text
test_V谷_P10_宽谷粗糙平底
```

Reason: this is a difficult synthetic sample with 17 focus-stack layers, 960x540 resolution, 1200 um depth range, and 75.0 um z-step.

## 2. Preconditions

| Item | Required Status |
|---|---|
| Original project data | read-only |
| Intermediate output directory | `tmp/external_baseline_data/` |
| External repository directory | `tmp/external_repos/` |
| Current git worktree | no core source/data/result modifications |
| Network download | only after explicit execution decision |
| Model training | not part of smoke test |

## 3. Data Preparation Checklist

| Step | Check | Status |
|---|---|---|
| 1 | Locate sample metadata in `dataset_split.csv` | pending |
| 2 | Confirm stack layer count = 17 | pending |
| 3 | Confirm resolution = 960x540 | pending |
| 4 | Confirm `z_step_um = 75.0` | pending |
| 5 | Confirm `depth_range_um = 1200` | pending |
| 6 | Identify source focus-stack image files | pending |
| 7 | Identify height GT file or generation route | pending |
| 8 | Identify or define valid / edge / high-risk masks | pending |
| 9 | Export frames to `tmp/external_baseline_data/samples/<sample_id>/stack/` | pending |
| 10 | Write `meta.json` | pending |

## 4. DFV Input Assumptions to Verify

| Item | Question |
|---|---|
| frame order | Does DFV expect near-to-far or far-to-near focal ordering? |
| input channels | Does DFV expect grayscale or RGB? |
| image range | Does DFV expect [0,1], [-1,1], ImageNet normalization, or raw intensity? |
| stack shape | Does DFV expect `[N,C,H,W]`, `[C,N,H,W]`, or dataset-specific tensor layout? |
| focus positions | Does DFV require physical focal distances or only ordered frames? |
| output type | depth, disparity, focus probability, or index distribution? |
| scale | Can output be mapped to micrometers? |

## 5. Minimal `meta.json`

```json
{
  "sample_id": "test_V谷_P10_宽谷粗糙平底",
  "split": "test",
  "category": "P10 V谷-宽谷粗糙平底",
  "resolution": [960, 540],
  "stack_layers": 17,
  "focus_positions_um": [0.0, 75.0, 150.0, 225.0, 300.0, 375.0, 450.0, 525.0, 600.0, 675.0, 750.0, 825.0, 900.0, 975.0, 1050.0, 1125.0, 1200.0],
  "z_step_um": 75.0,
  "depth_range_um": 1200.0,
  "height_unit": "um",
  "surface_baseline": "v_valley",
  "surface_noise": "perlin",
  "stray_level": 0.2,
  "has_height_gt": true
}
```

## 6. Smoke-Test Success Criteria

| Level | Criteria |
|---|---|
| L1 data export | 17 frames and `meta.json` exist in temporary directory |
| L2 adapter load | DFV-side dataloader or minimal tensor loader can read the stack |
| L3 inference shape | DFV model or placeholder loader returns a prediction-shaped tensor |
| L4 scale mapping | predicted focus/depth output can be mapped or aligned to height GT |
| L5 metric prototype | MAE can be computed for the single sample |

L1-L2 are enough for a pure smoke test. L3-L5 are required before full 7-sample evaluation.

## 7. Failure Conditions

| Failure | Decision |
|---|---|
| source focus-stack frames cannot be located | stop and inspect project generation scripts |
| DFV requires unavailable camera parameters | try ordered-frame mode; if impossible, move DFV to Related Work only |
| DFV hardcodes original dataset format | write a dataset shim or use one-sample export; if too costly, stop |
| output scale cannot be interpreted | keep qualitative only or scale-aligned auxiliary result |
| environment conflicts with current project | move to isolated environment |

## 8. Logging Template

```text
Smoke test date:
Sample:
Frames exported:
Input shape:
Normalization:
DFV code source:
DFV code version:
Dataloader status:
Inference status:
Output shape:
Scale mapping:
Metric prototype:
Blocking issue:
Next action:
```

## 9. No-Pollution Rules

1. Do not write generated data outside `tmp/`.
2. Do not modify `src/`, `data/`, `results/`, `assets/`, or `论文与PPT制作项目包/`.
3. Do not stage external repositories or generated model outputs.
4. Do not overwrite current manuscript files.
5. Record every generated path before considering further runs.
