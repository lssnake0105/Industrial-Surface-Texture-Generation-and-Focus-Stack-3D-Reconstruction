# DFV / DDFFNet Integration Protocol

Updated: 2026-06-18

Purpose: define the next executable protocol for integrating DFV and DDFFNet as external SOTA baselines without modifying project source data, figures, manuscripts, or model outputs.

## 1. Current Readiness

| Component | Status | Evidence |
|---|---|---|
| P10 synthetic export | ready | `tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底/` |
| Dataloader-only smoke test | pass | `tmp/external_baseline_data/smoke_reports/p10_dataloader_smoke.md` |
| Single-prediction evaluator | pass | `submission_planning/tools/evaluate_external_prediction.py` |
| Batch evaluator | pass | `submission_planning/tools/evaluate_external_prediction_batch.py` |
| Metric mask synchronization | pass | evaluator matches original `metrics()` high-risk and Sobel-edge rules |
| Full 7-sample external export | pending | requires explicit `--split test` export |
| DFV/DDFFNet external code | pending | should be downloaded only into `tmp/external_repos/` |

## 2. Non-Pollution Rules

1. External repositories must be placed under `tmp/external_repos/`.
2. Exported focus-stack samples must be placed under `tmp/external_baseline_data/`.
3. External predictions and logs must be placed under `tmp/external_baseline_results/`.
4. Do not modify `src/`, `results/`, `output/`, `README.md`, `Updated_English_Project_Paper.tex`, or `论文与PPT制作项目包/`.
5. Do not copy model weights or generated external predictions into `submission_planning/`.
6. Only small Markdown, CSV templates, and tool scripts belong in `submission_planning/`.

## 3. Phase A: External Code Inventory

Before downloading or running any code, record:

| Field | DFV | DDFFNet |
|---|---|---|
| Repository URL | `https://github.com/fuy34/DFV` | `https://github.com/soyers/ddff-pytorch` |
| Paper role | primary modern deep DFF baseline | early deep DFF learning baseline |
| Expected input | focal stack / differential focus volume | focal stack, possibly HDF5 dataset |
| Expected output | depth / focus probability | depth / disparity |
| Main risk | dataset-specific loader and output scale | old dependency and dataset format |
| First success target | read exported P10 package | read exported P10 package or generated HDF5 |

Create a run note before execution:

```text
tmp/external_baseline_results/<method>/logs/<date>_inventory.md
```

## 4. Phase B: Data Export

Single-sample smoke export:

```powershell
python -X utf8 submission_planning/tools/export_one_external_baseline_sample.py --overwrite
```

Full synthetic test export, only when ready to run an external baseline:

```powershell
python -X utf8 submission_planning/tools/export_one_external_baseline_sample.py --split test --overwrite
```

Expected output:

```text
tmp/external_baseline_data/
  manifest.csv
  samples/<sample_id>/
    meta.json
    stack/000.png ... 016.png
    height_gt.npy
    masks/high_risk_mask.npy
    masks/risk_layers.npy
    priors/
```

## 5. Phase C: Dataloader Adapter

The first adapter should not load model weights. It should only prove that exported data can be transformed into the expected tensor layout.

Minimum checks:

| Check | Required Evidence |
|---|---|
| frame count | 17 frames loaded |
| frame shape | `[17, 540, 960]` for P10 |
| grayscale tensor | `[1, 17, 1, 540, 960]` or method-specific equivalent |
| RGB tensor if needed | `[1, 17, 3, 540, 960]` |
| frame order | documented as exported order from focus position 1.0 to 0.0 |
| normalization | recorded before model inference |
| output path | no files outside `tmp/` |

Use current smoke test as the local reference:

```powershell
python -X utf8 submission_planning/tools/smoke_test_external_baseline_package.py
```

## 6. Phase D: Prediction Export Contract

Each external method should write one `.npy` prediction per sample:

```text
tmp/external_baseline_results/<method>/predictions/<sample_id>.npy
```

Prediction contract:

| Field | Requirement |
|---|---|
| shape | `[H, W]`, same as `height_gt.npy` |
| dtype | float32 preferred |
| scale | document as `raw_norm`, `minmax`, `affine`, or method-specific mapping |
| post-processing | record any resize, crop, interpolation, clipping, or scale alignment |
| failed samples | list explicitly in run log |

Prediction manifest template:

```text
submission_planning/autonomous_research/templates/external_prediction_manifest_template.csv
```

## 7. Phase E: Metric Evaluation

Single prediction:

```powershell
python -X utf8 submission_planning/tools/evaluate_external_prediction.py `
  --prediction tmp/external_baseline_results/DFV/predictions/test_V谷_P10_宽谷粗糙平底.npy `
  --method DFV `
  --scale-mode raw_norm `
  --out-dir tmp/external_baseline_results/DFV/evaluation/test_V谷_P10_宽谷粗糙平底
```

Batch evaluation:

```powershell
python -X utf8 submission_planning/tools/evaluate_external_prediction_batch.py `
  --prediction-manifest tmp/external_baseline_results/DFV/prediction_manifest.csv `
  --out-dir tmp/external_baseline_results/DFV/batch_evaluation
```

Metrics use the same synthetic mask rules as the original project:

- high-risk: `risk > max(percentile(risk, 84), 0.08)`;
- edge: Sobel gradient magnitude on GT height, thresholded by percentile 88.

## 8. Main-Table Eligibility Gate

An external method can enter the synthetic main table only if all gates pass:

| Gate | Pass Condition |
|---|---|
| same split | uses the fixed synthetic test split |
| sample coverage | reports every evaluated and skipped sample |
| output alignment | scale mode documented and reproducible |
| metric pipeline | uses current evaluator or exact project `metrics()` equivalent |
| training setting | zero-shot, synthetic retrain, or fine-tune clearly separated |
| no test leakage | no test GT used for training, tuning, or per-sample fitting except explicitly labeled affine evaluation |
| run log | code source, environment, command, failures, and output paths recorded |

If any gate fails, keep the method in Related Work or qualitative discussion.

## 9. Recommended Execution Order

1. Run DFV dataloader-only adapter on P10.
2. If loader succeeds, run DFV inference on P10.
3. Export DFV P10 prediction as `.npy`.
4. Evaluate DFV P10 prediction with `evaluate_external_prediction.py`.
5. If P10 output is interpretable, export all 7 test samples.
6. Run DFV batch prediction and `evaluate_external_prediction_batch.py`.
7. Repeat the same workflow for DDFFNet only after DFV reaches a clear pass/fail state.

## 10. Stop Conditions

| Stop Condition | Paper Handling |
|---|---|
| repository cannot run in isolated temp environment | Related Work only |
| dataloader cannot accept exported focal stack | feasibility note, no main-table result |
| output shape cannot be mapped to GT | qualitative only |
| scale alignment depends on test GT per sample | auxiliary table only unless clearly justified |
| runtime or dependency cost blocks progress | prioritize manuscript, ablation, and claim audit |

## 11. Manuscript Wording Rule

Before external results exist, write:

> DFV and DDFFNet are identified as priority external deep DFF baselines. Their adaptation requires a unified focal-stack export, documented scale alignment, and metric evaluation under the fixed synthetic test split.

After valid results exist, replace the statement with the actual method setting, split coverage, and measured metrics.
