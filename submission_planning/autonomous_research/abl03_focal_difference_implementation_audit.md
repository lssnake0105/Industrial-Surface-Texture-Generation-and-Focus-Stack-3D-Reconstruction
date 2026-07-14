# ABL-03 Focal-Difference Implementation Audit

Updated: 2026-06-18

Purpose: resolve the ambiguity found by the ablation feature schema audit. The base `features_for_model()` tensor does not include focal-difference channels, but the upgraded Focus-ResUNet path may add them through a separate feature-augmentation step.

## 1. Finding

The current Focus-ResUNet implementation uses:

```text
src/train_focus_resunet_loss_experiment.py::augment_features()
```

The function constructs upgraded features by concatenating:

```text
[17 stack frames, 16 adjacent-frame differences, 5 prior channels]
```

Therefore, ABL-03 can be implemented as an upgraded-feature ablation, not as a mask on the original 22-channel base tensor.

## 2. Required ABL-03 Definition

| Item | Definition |
|---|---|
| baseline input | 38-channel upgraded feature tensor |
| stack channels | 0-16 |
| focal-difference channels | 17-32 |
| prior channels | 33-37 |
| ABL-03 action | zero or remove channels 17-32 while keeping stack and prior channels |
| model implication | if channels are removed, model architecture must be adjusted; if zeroed, architecture can remain unchanged |

## 3. Recommended Implementation

The safer first implementation is a zero-mask ablation:

1. use the same 38-channel Focus-ResUNet architecture;
2. construct upgraded features with `augment_features()`;
3. zero channels 17-32 before training and inference;
4. keep all training split, seed, loss, epoch, and evaluation settings the same as ABL-00.

This isolates the information content of focal-difference channels while minimizing architecture changes.

## 4. Remaining Risk

Zeroing channels and removing channels are not identical. Zeroing preserves architecture and parameter count; removing channels changes the first-layer dimensions and may alter optimization behavior. If reviewers question this, report the ablation as "w/o focal-difference input signal" rather than "w/o focal-difference module".

## 5. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/audit_abl03_focal_difference_implementation.py
```

Default output:

```text
tmp/ablation_results/ABL-03/logs/abl03_focal_difference_implementation_audit.md
tmp/ablation_results/ABL-03/logs/abl03_focal_difference_implementation_audit.json
```

