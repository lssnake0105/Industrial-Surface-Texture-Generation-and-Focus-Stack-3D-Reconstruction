# Ablation Training Runner Preflight

Updated: 2026-06-19

Purpose: define and verify the boundary for a future ablation training runner after the minimal runner smoke test passed.

## 1. Boundary

This preflight does not:

1. create an optimizer;
2. enter a training loop;
3. save checkpoint files;
4. save prediction arrays, figures, or metric results;
5. update any `claim_eligible` field;
6. call the original experiment script's `main()`.

The generated report stays under:

```text
tmp/ablation_results/training_runner_preflight/
```

## 2. Key Safety Finding

The original final-method script defines:

```text
src/train_focus_resunet_loss_experiment.py
```

but its `main()`, `evaluate_split()`, `write_metric_plots()`, and `write_report()` write to the original project delivery package under:

```text
src/结题交付包/05_图表与结果/模型与损失函数升级实验/
```

Therefore, the ablation training runner must reuse components such as `FocusResUNet`, `HybridDFFLoss`, `augment_features()`, `generate_sample_arrays()`, and `metrics()`, while redirecting all logs, metrics, checkpoints, predictions, and figures under:

```text
tmp/ablation_results/<run_id>/
```

## 3. Current Variant Plan

| Run | Variant | Runner mode | Trainable after runner implementation | Input channels | Zero channels |
|---|---|---|---|---:|---|
| ABL-00 | Full S2R-FocusNet | `focus_resunet_upgraded` | yes | 38 | `[]` |
| ABL-01 | Direct image-to-depth | `lower_prior_focus_stack_only` | no, decision needed first | 17 | `[]` |
| ABL-02 | w/o DFF/GADFF prior | `focus_resunet_upgraded_masked` | yes | 38 | `[34, 35, 36, 37]` |
| ABL-03 | w/o focal difference | `focus_resunet_upgraded_masked` | yes | 38 | `[17, 18, ..., 32]` |
| ABL-04 | w/o glare cue | `focus_resunet_upgraded_masked` | yes | 38 | `[33]` |

ABL-04 currently removes the explicit risk channel only. A stricter glare-removal condition that also separates GADFF-derived information remains a later design decision.

## 4. Current Result

The 2026-06-19 training-runner preflight reports `status=pass` with 63 checks, 0 errors, and 0 warnings.

It confirms:

1. the final-method source entry exists;
2. the upgraded feature count remains 38;
3. the original output folders are outside `tmp/`;
4. the future runner should not call original output-writing functions directly;
5. ABL-00/02/03/04 are trainable after runner implementation;
6. ABL-01 is gated by a lower-prior architecture decision;
7. all planned outputs stay under `tmp/ablation_results/<run_id>/`.

## 5. Interpretation

Passing this preflight means the training-runner design boundary is safe enough for implementation. It does not mean ablation training has started, checkpoints exist, or module-effectiveness claims are supported.

The next implementation should first support ABL-00 and ABL-03 because they test the final full model and the focal-difference contribution most directly.
