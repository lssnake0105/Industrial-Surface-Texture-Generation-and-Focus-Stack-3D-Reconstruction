# English Abstract Draft

Updated: 2026-06-18  
Purpose: safe abstract draft for the future manuscript.  
Working title: **Simulation-to-Real Focus-Stack Reconstruction for Reflective Surface Defect Morphology Using Prior-Guided Deep Correction**

## Version A: Current-Evidence Safe Abstract

Reflective industrial surface defects require three-dimensional morphology reconstruction to characterize local depth variations, boundary structures, and micro-scale surface irregularities that are difficult to describe from two-dimensional images alone. However, calibrated height ground truth is difficult to obtain for real microscopic reflective surfaces, which limits the direct training and quantitative evaluation of learning-based reconstruction methods. To address this data bottleneck, this work studies a simulation-to-real focus-stack reconstruction framework for reflective surface defect morphology. The framework uses controllable synthetic focus stacks with known height maps for supervised training and quantitative evaluation, and combines traditional depth-from-focus priors, glare-aware focus cues, focal-difference representation, and a compact learning-based correction model. On the current synthetic test set, the proposed Focus-ResUNet achieves the lowest mean absolute error among the evaluated internal classical and learning-based baselines, with a mean MAE of 53.22 um and an edge MAE of 86.68 um. On real focus-stack samples without calibrated height ground truth, no-reference morphology metrics show reduced low-confidence spike artifacts and improved reconstruction stability compared with direct DFF-based methods. The results suggest that prior-guided correction over simulation-derived supervision is a practical route for relative morphology reconstruction of reflective surface defects, while calibrated real-height validation and broader external deep DFF comparisons remain important future work.

## Version B: Stronger Abstract After External Baselines

Use this version only after DFV/DDFFNet or another external deep DFF baseline has been successfully evaluated.

Reflective industrial surface defects require three-dimensional morphology reconstruction to characterize local depth variations, boundary structures, and micro-scale surface irregularities beyond two-dimensional visual inspection. A key obstacle is the scarcity of calibrated real height ground truth for microscopic reflective surfaces, which makes supervised learning and real-world quantitative validation difficult. This work proposes a simulation-to-real focus-stack reconstruction framework for reflective surface defect morphology. The method constructs controllable synthetic focus stacks with known height maps and integrates DFF/GADFF priors, focal-difference representation, glare-aware cues, and a compact prior-guided correction network. Synthetic experiments are conducted with absolute height metrics, while real focus-stack samples are evaluated using no-reference morphology metrics because calibrated real height ground truth is unavailable. Compared with [external baseline names], the proposed method achieves [insert external comparison result] on the synthetic test set. On real focus stacks, it reduces low-confidence spike artifacts and improves morphology stability under no-reference evaluation. These results indicate that simulation-based supervision and prior-guided focus-stack correction can provide an effective route toward deployable relative morphology reconstruction for reflective industrial surface defects.

## Claims Included

| Claim | Evidence Status | Included In |
|---|---|---|
| Real calibrated height GT is difficult / unavailable | supported by current real-sample evaluation boundary | Version A, B |
| Synthetic focus stacks provide known height maps | supported by dataset split and metric files | Version A, B |
| Focus-ResUNet has lowest mean MAE among internal evaluated baselines | supported by current synthetic comparison | Version A |
| Real samples show lower spike artifacts under no-reference metrics | supported by real summary metrics | Version A, B |
| Proposed method beats external SOTA | not yet supported | only placeholder in Version B |

## Claims Excluded

1. Real absolute height accuracy.
2. Superiority over DFV/DDFFNet before external baseline experiments.
3. Confirmed gain from domain randomization before ablation.
4. Confirmed effectiveness of each model component before ablation.
5. Generalization to all industrial surfaces.

## Notes for Later Revision

1. Replace `Focus-ResUNet` with the final model name once S2R-FocusNet / Prior-Guided Focus-ResUNet is fixed.
2. Add external baseline comparison only after DFV/DDFFNet results are available.
3. Add ablation conclusions only after w/o prior, w/o focal difference, and w/o glare cue experiments are completed.
4. If a calibrated real sample is collected, revise the final sentence to include real absolute validation.
