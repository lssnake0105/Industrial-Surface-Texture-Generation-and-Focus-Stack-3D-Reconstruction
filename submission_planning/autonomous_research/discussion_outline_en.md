# English Discussion Outline

Updated: 2026-06-18  
Purpose: section-level Discussion outline for the future English manuscript.  
Scope: emphasizes interpretation, limitations, and future work without overclaiming.

## 1. Discussion Section Goal

The Discussion should explain what the results mean, what they do not prove, and how the project can be strengthened for submission. It should connect the technical result to the broader research logic: simulation-based ground truth, prior-guided correction, real no-reference validation, and remaining synthetic-to-real gaps.

## 2. Suggested Section Structure

1. Why simulation-based supervision matters
2. Why prior-guided correction is useful
3. Interpretation of real no-reference results
4. Remaining high-risk region limitations
5. Relation to foundation depth and pseudo-label training
6. Limitations
7. Future work

## 3. Why Simulation-Based Supervision Matters

### Main Point

The key research bottleneck is not only network design; it is the scarcity of calibrated height labels for real microscopic reflective surfaces.

### Draft Skeleton

The synthetic focus-stack dataset plays a central role in this study because it provides known height maps for supervised training and absolute quantitative evaluation. For reflective microscopic surfaces, collecting calibrated real height ground truth for every defect sample can be difficult, especially when surface reflectance, local glare, and micro-scale morphology interact with the imaging process. The simulation-based data engine allows surface geometry, roughness, depth range, focal sampling, and glare-related factors to be controlled independently. This makes it possible to analyze reconstruction behavior under known conditions and to separate synthetic absolute evaluation from real no-reference validation.

### Safe Claim

Simulation provides controlled ground truth and evaluation; it does not prove full real-world absolute accuracy.

## 4. Why Prior-Guided Correction Is Useful

### Main Point

DFF/GADFF priors reduce the learning burden and provide interpretable axial cues, while the neural model corrects unstable regions.

### Draft Skeleton

The current results suggest that learning-based correction can improve overall reconstruction accuracy and edge stability on the synthetic test set. A purely traditional DFF pipeline is interpretable but sensitive to weak texture, glare, and boundary ambiguity. A purely image-to-depth network may require more data to learn focus geometry reliably. Prior-guided correction provides a practical compromise: traditional focus-based estimates serve as physical anchors, and the network focuses on correcting local artifacts and improving morphology continuity.

### Boundary

The stronger statement that each prior module is necessary requires ablation results.

## 5. Interpretation of Real No-Reference Results

### Main Point

Real results support morphology stability and spike suppression, not calibrated geometric accuracy.

### Draft Skeleton

On real focus-stack samples, the proposed learning-based model reduces low-confidence spike artifacts compared with direct DFF-based methods. This suggests that the model can suppress some unstable focus responses when applied outside the synthetic domain. However, because calibrated real height maps are unavailable, the real-sample metrics should be interpreted as no-reference indicators of morphology stability, visual plausibility, and relative dynamic behavior. They should not be interpreted as absolute metrology accuracy.

### Safe Claim

The method shows promising real-sample stability under no-reference evaluation.

## 6. Remaining High-Risk Region Limitations

### Main Point

High-risk reflective regions remain an unresolved modeling target.

### Draft Skeleton

Although the current model achieves the lowest mean MAE and edge MAE among internal baselines on the synthetic test set, high-risk MAE is not consistently the best. This indicates that glare, weak texture, and ambiguous focus responses require more targeted modeling. Treating all pixels with a global reconstruction objective may underemphasize localized reflective failure regions. Future versions should consider high-risk weighted losses, confidence calibration, edge-aware constraints, and explicit glare-region modeling.

### Future Link

Connect this to FAD, DualFocus, and DDL-Recurrent SFF as recent works exploring richer focus-stack representations.

## 7. Relation to Foundation Depth and Pseudo-Label Training

### Main Point

Depth Anything V2 is useful as a training-strategy reference, not as a direct focus-stack baseline.

### Draft Skeleton

Recent monocular foundation depth models, such as Depth Anything V2, show that high-quality synthetic labels and pseudo-labeled real images can improve depth generalization. This strategy is relevant to the present project because the real focus-stack samples lack calibrated height ground truth. A possible future extension is to generate pseudo depth or pseudo confidence maps for real focus stacks using an ensemble of focus-based methods and the current model, then use consistency training or teacher-student learning to reduce the synthetic-to-real gap. However, monocular depth models process single images and do not directly use axial focus responses, so they should be treated as auxiliary references rather than direct DFF baselines.

## 8. Limitations

### Current Limitations

| Limitation | Explanation | Mitigation |
|---|---|---|
| No calibrated real height GT | Real results cannot prove absolute height accuracy | collect step-height / profilometer / confocal subset |
| External deep DFF baselines pending | SOTA claim remains incomplete | reproduce DFV and DDFFNet |
| Ablation pending | module contribution not fully isolated | run w/o prior, w/o focal difference, w/o glare cue |
| High-risk region errors | glare/weak texture remains difficult | high-risk weighted loss and confidence modeling |
| Limited real-sample diversity | current real samples do not cover all materials | expand real focus-stack set |

## 9. Future Work

### Near-Term

1. Reproduce DFV and DDFFNet under the unified synthetic test protocol.
2. Add core ablations for priors, focal-difference representation, and glare-aware cues.
3. Generate high-risk mask visualizations and profile curves for failure analysis.
4. Unify final model naming as S2R-FocusNet or Prior-Guided Focus-ResUNet.

### Mid-Term

1. Collect a small calibrated real height subset.
2. Add pseudo-label or teacher-student training on real focus stacks.
3. Improve high-risk region modeling with confidence calibration.
4. Explore frequency-aware representations for periodic textures.

### Long-Term

1. Reduce required focal-stack frame count.
2. Evaluate cross-device and cross-material generalization.
3. Develop deployment-oriented real-time reconstruction.

## 10. Discussion Red Lines

1. Do not claim real absolute metrology accuracy.
2. Do not claim external SOTA superiority before DFV/DDFFNet results.
3. Do not claim all modules are proven without ablation.
4. Do not treat Depth Anything V2 as a direct focus-stack baseline.
5. Do not imply current real samples cover all reflective industrial surfaces.

## 11. Possible Closing Paragraph

Overall, the current results support a cautious but coherent conclusion: simulation-based supervision can provide controlled height labels for focus-stack reconstruction, and prior-guided learning can improve relative morphology stability under the evaluated conditions. The main remaining challenge is to bridge synthetic quantitative performance with calibrated real-world validation. Addressing this challenge will require stronger external baselines, targeted ablation, calibrated real samples, and more explicit modeling of reflective high-risk regions.
