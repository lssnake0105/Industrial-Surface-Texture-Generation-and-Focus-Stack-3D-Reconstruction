# English Method Outline

Updated: 2026-06-18  
Purpose: section-level Method outline for the future English manuscript.  
Scope: only claims supported by current project evidence are written as main content; uncertain implementation details are marked as verification items.

## 1. Method Section Goal

The Method section should explain why the proposed framework is a prior-guided focus-stack correction method. It should avoid presenting multiple internal networks as separate final contributions. The final method name should be unified as **S2R-FocusNet** or **Prior-Guided Focus-ResUNet** before manuscript submission.

## 2. Suggested Section Structure

1. Problem Formulation
2. Synthetic Focus-Stack Data Construction
3. Traditional Focus-Based Priors
4. Focal-Difference Representation
5. Prior-Guided Correction Network
6. Training Objective and Implementation Notes
7. Inference and Real-Sample Evaluation Protocol

## 3. Problem Formulation

### Main Message

Reflective surface morphology reconstruction is formulated as a height-map estimation problem from a focus stack and auxiliary focus-based priors.

### Draft Content

Let \(I=\{I_1,\ldots,I_N\}\) denote an \(N\)-frame focus stack captured at different focal positions \(z=\{z_1,\ldots,z_N\}\). The goal is to estimate a relative surface height map \(H\) from the focus-stack observations. For synthetic samples, the ground-truth height map \(H^\*\) is available and used for supervised training and quantitative evaluation. For real samples, calibrated height ground truth is unavailable, so evaluation is restricted to no-reference morphology indicators.

The proposed framework estimates height using both image observations and focus-based priors:

\[
\hat{H}=f_{\theta}(I, P_{\mathrm{DFF}}, P_{\mathrm{GADFF}}, D_{\mathrm{focal}}, M_{\mathrm{risk}}),
\]

where \(P_{\mathrm{DFF}}\) is the traditional depth-from-focus prior, \(P_{\mathrm{GADFF}}\) is the glare-aware DFF prior, \(D_{\mathrm{focal}}\) denotes focal-difference features, and \(M_{\mathrm{risk}}\) denotes high-risk or glare-related cues when available.

### Verification Needed

| Item | Needed Before Final Paper |
|---|---|
| exact input channels | confirm from training code |
| whether `M_risk` is an explicit network input | confirm from implementation |
| final model name | choose S2R-FocusNet or Prior-Guided Focus-ResUNet |

## 4. Synthetic Focus-Stack Data Construction

### Main Message

Synthetic data is the controlled source of height ground truth and should be presented as a central data engine.

### Draft Content

The synthetic dataset is constructed to represent multiple surface defect geometries and texture conditions. The current split contains 12 training samples, 5 validation samples, and 7 test samples, each with 17 focus-stack layers. The test set includes V-valley, ridge, mountain, step, periodic, and pit/groove structures, with depth ranges from 920 to 1420 um and z-step values from 57.5 to 88.75 um. Each synthetic sample includes a known height map, enabling absolute quantitative metrics such as MAE, edge MAE, and high-risk MAE.

The simulation factors should be described along six dimensions:

1. surface geometry: mountain, ridge, a-ridge, step, periodic, V-valley, pit/groove;
2. surface texture and roughness: Perlin, fractal, stripe, procedural noise;
3. axial sampling: 17-layer focus stacks and z-step values;
4. depth range: sample-specific height scale in micrometers;
5. glare/stray light level: approximate reflective disturbance indicator;
6. rendered focus response: synthetic image sequence used for DFF and learning.

### Table Pointer

Use Table 1 for dataset split summary.

### Verification Needed

| Item | Needed Before Final Paper |
|---|---|
| exact rendering formula | verify from `surface_sample_generator.py` and simulation scripts |
| blur/focus response model | verify from code |
| reflectance/glare implementation | verify from `simulate_antiglare_*` and `glare_aware_dff.py` |

## 5. Traditional Focus-Based Priors

### Main Message

Traditional DFF outputs are not only baselines; they are interpretable priors that reduce the burden on learning-based correction.

### Draft Content

The framework uses traditional focus-based estimates as priors. Original DFF provides a direct estimate by selecting the focal position with the strongest focus response. DFF with post-processing reduces local noise and spike artifacts. Lee2013 and Li2019 represent adaptive-window SFF/DFF baselines, addressing the limitations of fixed local windows. GADFF introduces glare-aware processing to account for reflective high-risk regions.

In the proposed framework, these priors serve two roles. First, they define classical baselines for quantitative comparison. Second, DFF/GADFF-related outputs provide physically interpretable axial cues for the learning-based correction model.

### Caution

Do not claim GADFF alone is superior. Current synthetic metrics show that GADFF is unstable as a standalone method, but it remains useful as a glare-aware cue or baseline.

## 6. Focal-Difference Representation

### Main Message

Focal-difference features encode how focus responses change along the axial dimension, making the model sensitive to depth-related focus variation.

### Draft Content

Instead of treating the focus stack as an unordered set of images, the method uses focal-difference features to represent axial changes between neighboring focal positions. This representation is motivated by the observation that depth cues in focus-stack imaging are encoded not only in individual frame sharpness but also in how local image structures change across focal planes. This idea is closely related to the differential focus volume used in DFV, making DFV an important external baseline for future comparison.

### Verification Needed

| Item | Needed Before Final Paper |
|---|---|
| exact difference definition | adjacent-frame difference, feature difference, or focus-measure difference |
| whether difference is computed in image space or feature space | confirm from code |
| whether multi-scale difference exists | confirm from implementation |

## 7. Prior-Guided Correction Network

### Main Message

The final model should be described as a compact prior-guided correction network, not as several independent model products.

### Draft Content

The proposed correction network takes the focus-stack observations and auxiliary priors as input and predicts a corrected height map. A ResUNet-style encoder-decoder backbone is used to combine local image features with prior depth cues. The network is designed to correct unstable DFF estimates in ambiguous regions while preserving reliable focus-based structures. In reflective or weak-texture areas, glare-aware cues and focal-difference features provide additional information for reducing spike artifacts and improving morphology continuity.

### Naming Rule

Use one final model name throughout:

| Option | Use Case |
|---|---|
| S2R-FocusNet | stronger simulation-to-real story |
| Prior-Guided Focus-ResUNet | more architecture-specific |

### Internal Variants

| Variant | Manuscript Role |
|---|---|
| TinyDepthNet | lightweight internal baseline or ablation |
| Focus-ResUNet | current main model / candidate final model |
| Residual Focus-ResUNet | ablation for residual correction |
| GADFF | classical/glare-aware prior or baseline |

## 8. Training Objective and Implementation Notes

### Current Safe Content

For synthetic samples, the network is trained with known height maps. The primary supervision should be described as height reconstruction loss only if confirmed by code. Edge-aware, high-risk weighted, smoothness, or prior-consistency losses should be written as ablation/future candidates unless they are already implemented.

### Candidate Loss Terms

| Loss | Status |
|---|---|
| height reconstruction loss | likely implemented; verify |
| edge-aware loss | candidate or future unless verified |
| high-risk weighted loss | candidate or future unless verified |
| smoothness / spike penalty | candidate or future unless verified |
| residual bound / prior consistency | candidate or future unless verified |

### Writing Rule

Only implemented losses should appear in the final Method section. Unimplemented losses should move to Discussion or Future Work.

## 9. Inference and Real-Sample Evaluation Protocol

### Main Message

The same trained model can be applied to real focus stacks, but real evaluation must be no-reference.

### Draft Content

During inference, the trained model receives real focus stacks and the corresponding focus-based priors, and outputs a relative morphology map. Since calibrated real height maps are not available, real-sample evaluation uses no-reference morphology metrics, including roughness stability, edge retention, relative dynamic range, and low-confidence spike count. These metrics are used to assess output stability and visual plausibility rather than absolute geometric accuracy.

## 10. Method Section Red Lines

1. Do not claim real absolute height accuracy.
2. Do not describe unimplemented losses as part of the final method.
3. Do not present TinyDepthNet, Focus-ResUNet, and residual variants as separate final contributions.
4. Do not claim focal-difference effectiveness before ablation is completed.
5. Do not claim superiority over DFV before running the external baseline.
