# Ablation Small-Training Debug

Updated: 2026-06-19

Purpose: record the first explicitly guarded small-scale training debug run for ABL-00 and ABL-03.

## 1. Boundary

This run is a runner-debug step only. It does not provide manuscript-ready ablation evidence.

It used:

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-id ABL-00 --run-id ABL-03 --max-epochs 1 --train-patches 8 --val-patches 4 --batch-size 1
```

All outputs were written under:

```text
tmp/ablation_results/
```

## 2. Current Result

The 2026-06-19 small-training debug run reports `status=pass` with 16 checks, 0 errors, and 0 warnings.

| Run | Status | Last debug val MAE norm | Checkpoint |
|---|---|---:|---|
| ABL-00 | small training debug completed | 0.18825971 | `tmp/ablation_results/ABL-00/checkpoints/2026-06-19_small_training_debug.pt` |
| ABL-03 | small training debug completed | 0.27409419 | `tmp/ablation_results/ABL-03/checkpoints/2026-06-19_small_training_debug.pt` |

The metric values above are debug diagnostics from a single P10 patch-sampling setup with 1 epoch, 8 train patches, and 4 validation patches. They must not be interpreted as ablation performance.

## 3. Eligibility State

Both run configs remain excluded from manuscript claims:

| Run | `status` | `main_table_eligible` | `claim_eligible` |
|---|---|---|---|
| ABL-00 | `small_training_debug_run` | `false` | `false` |
| ABL-03 | `small_training_debug_run` | `false` | `false` |

## 4. Why This Matters

This debug run proves that the protected training runner can:

1. load the target ABL workspace;
2. build upgraded 38-channel Focus-ResUNet inputs;
3. apply the ABL-03 focal-difference mask;
4. run a minimal optimizer/backward/update loop;
5. write checkpoint and debug metrics only under `tmp/ablation_results/`;
6. keep claim eligibility disabled.

## 5. Next Step

The next safe step is to expand from debug training to a controlled ablation pilot:

```text
ABL-00, ABL-02, ABL-03, ABL-04
max_epochs: 2-3
train_patches: 64-128
val_patches: 16-32
output: tmp/ablation_results/<run_id>/
```

Before entering the manuscript, a later run must cover the intended synthetic split, write per-sample metrics, include code-state logs, and pass a claim eligibility audit.
