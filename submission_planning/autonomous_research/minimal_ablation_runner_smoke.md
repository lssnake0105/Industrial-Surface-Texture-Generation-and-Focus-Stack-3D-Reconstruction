# Minimal Ablation Runner Smoke

Updated: 2026-06-19

Purpose: validate a minimal, non-training runner interface for ABL-00 to ABL-04 after the training-entry preflight corrected the final-method ablation path.

## 1. Boundary

This smoke test does not:

1. train any model;
2. create an optimizer or run an optimizer step;
3. save checkpoint files;
4. save prediction arrays or figures;
5. update `claim_eligible`;
6. modify `src/` or existing result assets.

The generated report stays under:

```text
tmp/ablation_results/runner_smoke/
```

## 2. Why This Smoke Test Is Needed

The previous training-entry preflight established that final-method ablations should be anchored to:

```text
src/train_focus_resunet_loss_experiment.py
```

This script uses `augment_features()` to convert base 22-channel features into upgraded 38-channel Focus-ResUNet features:

| Channel range | Meaning |
|---|---|
| 0-16 | raw 17-layer focus stack |
| 17-32 | adjacent focal-difference channels |
| 33 | risk / glare cue |
| 34 | DFF depth prior |
| 35 | DFF confidence |
| 36 | GADFF depth prior |
| 37 | GADFF confidence |

The smoke test checks whether the corrected ABL variants can be represented in this upgraded feature space before any real training job is launched.

## 3. Current Variant Decisions

| Run | Variant | Smoke decision |
|---|---|---|
| ABL-00 | Full S2R-FocusNet | use full 38-channel feature input |
| ABL-01 | Direct image-to-depth | record as separate lower-prior runner; do not emulate by masking final Focus-ResUNet only |
| ABL-02 | w/o DFF/GADFF prior | zero upgraded channels 34-37 |
| ABL-03 | w/o focal difference | zero upgraded channels 17-32 |
| ABL-04 | w/o glare cue | zero upgraded channel 33 as explicit risk cue; stricter GADFF-derived glare removal remains a later design decision |

## 4. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/run_ablation_variant_smoke.py
```

Default outputs:

```text
tmp/ablation_results/runner_smoke/minimal_ablation_runner_smoke.json
tmp/ablation_results/runner_smoke/minimal_ablation_runner_smoke.md
```

## 5. Current Result

The 2026-06-19 CPU smoke test reports `status=pass` with 35 checks, 0 errors, and 0 warnings.

On a 64 x 64 P10 center patch:

| Run | Input | Output | Status |
|---|---|---|---|
| ABL-00 | `[1, 38, 64, 64]` | `[1, 1, 64, 64]` | shape smoke passed |
| ABL-01 | raw stack design only | skipped | lower-prior design recorded |
| ABL-02 | `[1, 38, 64, 64]` | `[1, 1, 64, 64]` | shape smoke passed |
| ABL-03 | `[1, 38, 64, 64]` | `[1, 1, 64, 64]` | shape smoke passed |
| ABL-04 | `[1, 38, 64, 64]` | `[1, 1, 64, 64]` | shape smoke passed |

The diagnostic losses in the smoke report only prove finite forward/loss computation on an untrained, randomly initialized model. They must not be interpreted as ablation performance.

## 6. Interpretation

Passing this smoke test means the corrected runner interface is coherent enough for the next implementation step. It supports the following planning statement:

> ABL-00/02/03/04 can share the upgraded Focus-ResUNet input path with controlled channel masking, while ABL-01 requires a separate lower-prior architecture or runner.

It does not support claims that any module improves MAE, edge MAE, high-risk MAE, or real-sample morphology stability.

## 7. Next Step

The next safe ablation step is to implement a training runner that writes run logs, seeds, code status, split definitions, synthetic metrics, and claim eligibility flags under `tmp/ablation_results/<run_id>/`. Training results should enter the manuscript only after metrics and audit evidence exist.
