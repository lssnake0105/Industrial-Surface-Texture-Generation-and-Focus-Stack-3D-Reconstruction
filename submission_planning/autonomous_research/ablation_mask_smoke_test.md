# Ablation Mask Smoke Test

Updated: 2026-06-18

Purpose: verify that planned ablation masks alter the intended input channels before any training run. This smoke test does not train models, run inference, or create predictions.

## 1. Tested Variants

| Run ID | Feature space | Mask action |
|---|---|---|
| ABL-01 | base 22-channel features | zero channels 17-21 |
| ABL-02 | base 22-channel features | zero channels 18-21 |
| ABL-03 | upgraded 38-channel Focus-ResUNet features | zero channels 17-32 |
| ABL-04 | base 22-channel features | zero channel 17 |

ABL-05 and ABL-06 are not included because they require dataset-generation or model-output changes rather than a simple input mask.

## 2. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/smoke_test_ablation_masks.py
```

Default output:

```text
tmp/ablation_results/mask_smoke/ablation_mask_smoke_test.md
tmp/ablation_results/mask_smoke/ablation_mask_smoke_test.json
```

## 3. Pass Condition

The smoke test passes when:

1. each targeted channel range becomes exactly zero after masking;
2. non-targeted channels remain unchanged;
3. ABL-03 is applied only to upgraded 38-channel features;
4. no model weights, predictions, or figures are created.

## 4. Interpretation

Passing this smoke test means the input-mask definitions are operationally consistent. It does not validate any ablation result or module contribution.

