# English Introduction Outline

Updated: 2026-06-18  
Purpose: paragraph-level outline for the future English manuscript.  
Working title: **Simulation-to-Real Focus-Stack Reconstruction for Reflective Surface Defect Morphology Using Prior-Guided Deep Correction**

## 1. Target Structure

The Introduction should be written as five paragraphs:

1. Industrial surface defects require 3D morphology.
2. Focus-stack reconstruction is useful but fragile on reflective surfaces.
3. Real calibrated height labels are scarce, motivating simulation-to-real learning.
4. Prior-guided deep correction is the proposed solution.
5. Contributions.

## 2. Paragraph 1: From 2D Defect Inspection to 3D Morphology

### Main Point

Two-dimensional visual inspection can identify visible surface anomalies, but it cannot fully characterize defect depth, boundary geometry, and micro-scale surface morphology.

### Draft Skeleton

Industrial surface inspection has increasingly relied on image-based methods for detecting scratches, pits, dents, and texture anomalies. While two-dimensional images are effective for locating visible defects, many industrial scenarios require a more detailed description of surface morphology, including local depth variation, boundary continuity, and micro-scale roughness. This requirement is particularly important for reflective surfaces, where illumination changes and specular highlights can obscure the structural meaning of image intensity. Therefore, recovering three-dimensional morphology from microscopic image sequences is a relevant step toward more informative defect characterization.

### Evidence / Citation Needs

| Need | Candidate Source |
|---|---|
| Surface defect detection review | industrial defect detection review papers |
| 3D morphology motivation | structured illumination / microscopy 3D reconstruction references |
| Reflective surface challenge | project discussion + optical inspection literature |

## 3. Paragraph 2: Focus-Stack Reconstruction and Its Limitations

### Main Point

Focus-stack imaging provides axial cues for surface reconstruction, but traditional DFF/SFF methods are sensitive to focus measure reliability, texture, window size, glare, and depth discontinuities.

### Draft Skeleton

Focus-stack imaging provides a practical way to infer surface height by capturing a sequence of images at different focal positions. Classical shape-from-focus and depth-from-focus methods estimate depth by evaluating local sharpness responses along the focal dimension. These methods are interpretable and do not require large training datasets, but their performance strongly depends on the reliability of the focus measure. Fixed or adaptive window strategies can reduce some local ambiguity, yet weak texture, periodic patterns, reflective glare, and sharp defect boundaries can still lead to unstable depth estimates and spike artifacts. These limitations motivate methods that can preserve the physical interpretability of DFF while correcting its failure cases.

### Evidence / Citation Needs

| Need | Candidate Source |
|---|---|
| Classical SFF/DFF | Nayar and Nakagawa |
| Focus measure review | Pertuz et al. |
| Adaptive window baselines | Lee2013, Li2019 |
| Project evidence | Original DFF and DFF + post-processing metrics |

## 4. Paragraph 3: Ground-Truth Scarcity and Simulation-to-Real Strategy

### Main Point

The central bottleneck is not only model design; it is the lack of calibrated real height ground truth for reflective microscopic surfaces.

### Draft Skeleton

A major obstacle for learning-based focus-stack reconstruction is the lack of calibrated height ground truth for real microscopic reflective samples. Profilometer, confocal, or interferometric measurements can provide accurate references, but such measurements may be costly, time-consuming, or unavailable for every defect sample. As a result, direct supervised learning from real focus stacks is difficult. A simulation-to-real strategy offers a practical alternative: synthetic surfaces can be generated with known height maps, rendered into focus stacks under controlled depth ranges, roughness patterns, reflectance changes, and glare conditions, and then used for supervised training and quantitative evaluation. Real focus stacks can then be used to assess morphology stability and deployment plausibility under a no-reference evaluation protocol.

### Evidence / Citation Needs

| Need | Candidate Source |
|---|---|
| Current synthetic split | `dataset_split.csv` |
| Current synthetic GT metrics | `paper_algorithm_comparison_metrics.csv` |
| Real no-GT boundary | `real_midterm_method_summary.csv` |
| Sim-to-real depth references | Focus on Defocus, DDFS, DfF in the Wild, DEReD |
| Foundation data strategy | Depth Anything V2 |

## 5. Paragraph 4: Proposed Prior-Guided Correction Framework

### Main Point

The method should be framed as prior-guided correction over focus-stack evidence, not as an unconstrained image-to-depth model.

### Draft Skeleton

In this work, we study a prior-guided focus-stack reconstruction framework for reflective surface defect morphology. Instead of relying only on raw focus-stack frames, the framework combines traditional DFF/GADFF priors, focal-difference representation, glare-aware cues, and a compact learning-based correction network. The DFF-related priors provide interpretable axial estimates, while the focal-difference representation captures changes in focus response across adjacent focal positions. The learning-based correction model is then used to reduce unstable artifacts and improve relative morphology continuity, particularly around defect boundaries and reflective high-risk regions. This design aims to reduce the learning burden under limited data while retaining the physical cues embedded in the focus stack.

### Evidence / Citation Needs

| Need | Candidate Source |
|---|---|
| DFF/GADFF priors | project methods and metrics |
| Focal-difference relevance | DFV paper |
| Internal result | Focus-ResUNet synthetic MAE and real spike count |
| Future ablation | `ablation_design.md` |

## 6. Paragraph 5: Contributions

### Safe Contribution List

Current-evidence version:

1. We construct a controllable synthetic focus-stack dataset for reflective defect morphology, where each sample has a known height map for quantitative evaluation.
2. We develop a prior-guided focus-stack reconstruction framework that combines DFF/GADFF priors, focal-difference representation, glare-aware cues, and learning-based correction.
3. We establish a two-level evaluation protocol that separates synthetic absolute error evaluation from real no-reference morphology validation.
4. We analyze representative reflective defect cases and identify remaining challenges in high-risk regions such as glare, weak texture, and sharp boundaries.

Version after additional experiments:

1. Add external DFV/DDFFNet comparison if completed.
2. Add module ablation contribution if completed.
3. Add calibrated real subset contribution only if real GT is collected.

## 7. Claims to Avoid in Introduction

1. Do not claim real absolute metrology accuracy.
2. Do not claim superiority over external deep DFF SOTA before DFV/DDFFNet experiments.
3. Do not claim all modules are proven effective before ablation.
4. Do not describe Depth Anything V2 as a direct focus-stack baseline.
5. Do not generalize from current samples to all industrial reflective surfaces.

## 8. Transition to Related Work

Suggested final transition sentence:

The following sections review classical focus-based shape recovery, learning-based depth from focus, simulation-to-real defocus learning, and industrial surface morphology reconstruction, which together define the technical context of the proposed framework.

## 9. Transition to Method

Suggested final sentence before Method:

Based on these observations, we formulate reflective surface morphology reconstruction as a prior-guided focus-stack correction problem, where synthetic height supervision provides quantitative training signals and traditional focus-based priors provide interpretable axial cues.
