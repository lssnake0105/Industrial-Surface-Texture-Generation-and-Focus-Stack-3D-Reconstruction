# English Experiments Outline

Updated: 2026-06-18  
Purpose: section-level Experiments outline for the future English manuscript.  
Scope: separates current evidence from required future experiments.

## 1. Experiments Section Goal

The Experiments section should demonstrate three points:

1. Synthetic focus-stack data enables absolute quantitative evaluation.
2. Prior-guided learning improves current internal reconstruction results under the evaluated synthetic split.
3. Real focus-stack samples can be evaluated only with no-reference morphology metrics until calibrated real height ground truth is available.

## 2. Suggested Section Structure

1. Dataset and Evaluation Protocol
2. Baselines
3. Metrics
4. Synthetic Quantitative Results
5. Real No-Reference Morphology Evaluation
6. Ablation Study
7. Failure Analysis
8. External Baseline Plan or Results

## 3. Dataset and Evaluation Protocol

### Synthetic Dataset

Draft content:

The synthetic dataset is divided into 12 training samples, 5 validation samples, and 7 test samples. Each sample contains a 17-layer focus stack and a known height map. The test set covers V-valley, A-ridge, mountain, ridge, step, periodic stripe, and pit/groove structures, with resolutions of 640x360 or 960x540, depth ranges from 920 to 1420 um, and z-step values from 57.5 to 88.75 um. This controlled setting enables absolute quantitative evaluation.

### Real Dataset

Draft content:

The real focus-stack samples are used for deployment-oriented morphology validation. Because calibrated real height ground truth is unavailable, real results are evaluated using no-reference morphology metrics rather than absolute height error. This design keeps the synthetic quantitative evaluation and real morphology validation explicitly separated.

### Table

Use Table 1: Synthetic dataset split and sample statistics.

## 4. Baselines

### Current Internal Baselines

| Method | Role |
|---|---|
| Original DFF | classical focus-based baseline |
| DFF + post-processing | engineering baseline for spike/noise reduction |
| GADFF | glare-aware DFF baseline / prior |
| Lee2013 adaptive window | traditional adaptive SFF/DFF baseline |
| Li2019 adaptive iteration | traditional adaptive SFF/DFF baseline |
| TinyDepthNet | lightweight learning baseline |
| Focus-ResUNet / S2R-FocusNet | current final method candidate |
| Residual Focus-ResUNet | residual correction ablation or internal variant |

### External Baselines to Add

| Method | Status | Role |
|---|---|---|
| DFV | pending | primary external deep DFF baseline |
| DDFFNet | pending | early deep DFF baseline |
| HybridDepth | optional | recent focal-stack + single-image prior baseline |
| Depth Anything V2 | auxiliary only | single-image qualitative reference / training-strategy discussion |

### Writing Rule

If DFV/DDFFNet results are not yet available, write them as future comparison targets or planned external baselines, not as completed results.

## 5. Metrics

### Synthetic Metrics

| Metric | Meaning |
|---|---|
| MAE | absolute height error in um |
| Edge MAE | error around depth discontinuities and defect boundaries |
| High-risk MAE | error in glare/weak-texture/high-risk regions |
| P90 error | tail error behavior |
| normalized RMSE | scale-normalized reconstruction error |

### Real No-Reference Metrics

| Metric | Meaning |
|---|---|
| roughness | surface stability / excessive noise tendency |
| edge retention | consistency with visible structure |
| relative dynamic range | preservation of morphology variation |
| low-confidence spike count | local unstable spike artifacts |
| profile consistency | qualitative continuity along selected cross-sections |

### Warning

Real no-reference metrics should not be interpreted as calibrated geometric accuracy.

## 6. Synthetic Quantitative Results

### Current Supported Result

Current synthetic test-set averages:

| Method | Mean MAE (um) | Mean Edge MAE (um) | Mean High-Risk MAE (um) |
|---|---:|---:|---:|
| Focus-ResUNet | 53.22 | 86.68 | 40.14 |
| TinyDepthNet | 57.31 | 89.10 | 42.63 |
| Lee2013 adaptive window | 62.35 | 146.83 | 30.52 |
| Residual Focus-ResUNet | 62.52 | 126.91 | 30.08 |
| Li2019 adaptive iteration | 62.95 | 145.33 | 31.24 |
| DFF + post-processing | 63.81 | 149.04 | 29.72 |
| Original DFF | 100.55 | 206.61 | 46.32 |
| GADFF | 105.83 | 210.60 | 46.38 |

### Safe Interpretation

Focus-ResUNet achieves the lowest mean MAE and edge MAE among the current evaluated internal baselines on the synthetic test set. However, high-risk MAE is not the best among all methods, indicating that glare/weak-texture regions remain a key limitation.

### Figure/Table

Use Table 2 and Fig. 4.

## 7. Real No-Reference Morphology Evaluation

### Current Supported Result

Current real no-reference summary:

| Method | Roughness | Edge Retention | Dynamic Range | Low-Conf Spike Count |
|---|---:|---:|---:|---:|
| Focus-ResUNet | 0.0078 | 0.0009 | 0.5317 | 2.0 |
| TinyDepthNet | 0.0067 | -0.0243 | 0.3623 | 49.0 |
| DFF + post-processing | 0.0130 | -0.0296 | 0.3944 | 4091.3 |
| GADFF | 0.0316 | -0.0295 | 0.5406 | 3351.4 |
| Lee2013 adaptive window | 0.0295 | -0.0209 | 0.6325 | 5018.7 |
| Li2019 adaptive iteration | 0.0318 | -0.0233 | 0.6515 | 5433.6 |
| Original DFF | 0.0977 | -0.0394 | 0.6964 | 9179.7 |
| Residual Focus-ResUNet | 0.0594 | -0.0343 | 0.5203 | 9262.1 |

### Safe Interpretation

On real focus stacks, Focus-ResUNet greatly reduces low-confidence spike artifacts compared with direct DFF-based methods. The result supports morphology stability under no-reference evaluation, not absolute height accuracy.

### Figure/Table

Use Table 3 and Fig. 6.

## 8. Ablation Study

### Required Variants

| Variant | Purpose |
|---|---|
| full S2R-FocusNet | final method |
| w/o DFF/GADFF prior | test prior contribution |
| w/o focal-difference representation | test axial focus-response contribution |
| w/o glare-aware cue | test high-risk region modeling |
| direct image-to-depth | test need for prior-guided correction |

### Current Status

These ablations are planned but not yet available. Do not write ablation conclusions until results exist.

## 9. Failure Analysis

### Recommended Cases

| Case | Reason |
|---|---|
| P10 V-valley | wide valley and high-risk regions |
| periodic stripe | texture-induced focus ambiguity |
| step boundary | edge accuracy and profile continuity |
| real pit/key texture/key tip | real morphology stability and spike suppression |

### Safe Interpretation

Failure analysis should emphasize remaining limitations, especially high-risk reflective regions and the absence of calibrated real height ground truth.

## 10. External Baseline Results or Plan

### If Results Are Not Available

Write:

> DFV and DDFFNet are identified as priority external deep DFF baselines because they directly process focal stacks and provide stronger comparison with learning-based depth-from-focus methods. Their adaptation requires a unified data interface and scale-alignment protocol, which is planned as part of future experimental extension.

### If Results Become Available

Add:

1. training setting;
2. input frame count;
3. scale alignment;
4. MAE / edge MAE / high-risk MAE;
5. qualitative comparison on P10 V-valley.

## 11. Experiments Section Red Lines

1. Do not put Depth Anything V2 in the synthetic main MAE table.
2. Do not report real absolute height error.
3. Do not compare methods with different scale alignment without table notes.
4. Do not claim module effectiveness before ablation.
5. Do not claim SOTA superiority before external baselines are evaluated.
