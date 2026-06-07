# Model Notes

This project's model weights are not fully uploaded to GitHub.

## Model Description

The local workspace contains several PyTorch checkpoints for focus-stack depth reconstruction and residual correction experiments, including TinyDepthNet, Focus-ResUNet variants, residual Focus-ResUNet variants, and baseline-guided residual correction models.

## File Size

Model weights are stored as `.pt` files and may be large when combined with training outputs, logs, validation panels, and checkpoint variants.

## Access

Weights are kept in local archive material and should only be shared after confirming file size, authorship, project constraints, and whether the corresponding training data can be released.

## Reproduction

Training scripts are available in `src/`. Reproducing trained weights requires the excluded datasets, compatible PyTorch installation, and enough compute for the selected experiment. Public figures in `assets/` and `results/` provide a lightweight way to review the completed outcomes.
