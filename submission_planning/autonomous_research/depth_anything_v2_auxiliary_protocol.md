# Depth Anything V2 Auxiliary Protocol

Updated: 2026-06-18

Purpose: define how Depth Anything V2 can be used in this project as a foundation-depth auxiliary reference without confusing it with focus-stack SOTA baselines.

## 1. Positioning

Depth Anything V2, arXiv:2406.09414v1, is a monocular depth estimation foundation model. The paper emphasizes three practices that are directly useful for this project narrative: precise synthetic depth labels, large-scale pseudo-labeled real images, and teacher-student distillation. Its input is a single RGB image, so it lacks the axial focus response used by DFF, SFF, DFV, and this project's focus-stack model.

Recommended role in this project:

| Role | Use | Manuscript location |
|---|---|---|
| training-strategy evidence | support the value of high-quality synthetic labels and pseudo-labeled real images | Related Work, Discussion |
| auxiliary prior | test whether a single-frame foundation model gives plausible global shape hints | optional qualitative figure |
| future extension | motivate teacher-student pseudo-depth or pseudo-confidence for real unlabeled stacks | Future Work |

Disallowed role:

| Disallowed use | Reason |
|---|---|
| main synthetic SOTA table | single-image MDE does not use focal-stack information and output scale is not native height |
| claim of real metrology accuracy | real samples do not have calibrated height GT |
| direct replacement for DFV/DDFFNet | DFV/DDFFNet are focus-stack baselines; Depth Anything V2 is a monocular auxiliary model |

## 2. Candidate Inputs

The minimum non-destructive input plan is:

| Input type | Source | Use |
|---|---|---|
| center-focus frame | `tmp/external_baseline_data/samples/<sample_id>/stack/<center>.png` | first smoke input, no image copying needed |
| best-focus frame | future focus-measure selection from the exported stack | stronger single-frame input |
| all-in-focus image | future AiF fusion from the exported stack | most suitable qualitative input |
| real best-focus or AiF frame | future real sample preprocessing under `tmp/` | optional real-domain visual comparison |

The scaffold intentionally records paths only. It does not copy images into the auxiliary workspace.

## 3. Output Boundary

All future outputs must stay under:

```text
tmp/foundation_depth_auxiliary/DepthAnythingV2/
```

Allowed future files:

| Path | Content |
|---|---|
| `input_manifest.csv` | single-frame inputs and expected output paths |
| `run_config.json` | repository, model, status, scale, and claim boundary |
| `logs/<date>_protocol.md` | environment, model checkpoint, input choice, and failure notes |
| `predictions/` | future relative depth outputs, if the model is actually run |
| `visualizations/` | future qualitative side-by-side figures |

No model weights, downloaded repositories, or generated figures are created by the scaffold.

## 4. Qualitative Figure Design

If the model is later run, the safest figure is a side-by-side qualitative comparison:

| Panel | Content | Claim boundary |
|---|---|---|
| A | single-frame input used by Depth Anything V2 | shows input condition |
| B | Depth Anything V2 relative depth | auxiliary prior / global structure only |
| C | DFF or GADFF depth prior | focus-stack physics-based reference |
| D | Proposed S2R-FocusNet output | project method output |
| E | synthetic GT if available, or real input image if no GT | context, not real absolute accuracy |

For synthetic P10 only, a scale-aligned MAE can be computed as a diagnostic sanity check, but it should stay outside the main SOTA table. For real samples, only visual plausibility and no-reference morphology discussion are allowed.

## 5. Evidence Gates

| Gate | Required evidence | If missing |
|---|---|---|
| G1 input manifest | valid single-frame input paths | protocol remains planning-only |
| G2 model run log | model source, checkpoint, command, environment, input preprocessing | no prediction can be cited |
| G3 output format | saved relative depth and visualization under `tmp/` | no figure can be used |
| G4 claim review | explicit statement that output is monocular relative depth | cannot enter manuscript |
| G5 optional synthetic diagnostic | scale-alignment method and metric code | cannot report any number |

Current status: scaffold-only auxiliary protocol. No Depth Anything V2 model has been downloaded, run, or evaluated.

## 6. Recommended Manuscript Sentence

Depth Anything V2 is best cited as evidence that precise synthetic labels and pseudo-labeled real images can improve depth-model generalization. In this work, that observation motivates the use of simulation-derived height ground truth and suggests a future teacher-student bridge for unlabeled real focus stacks, while the main experimental comparison remains centered on focus-stack DFF/SFF methods.

## 7. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/scaffold_depth_anything_v2_auxiliary_workspace.py
```

Default output:

```text
tmp/foundation_depth_auxiliary/DepthAnythingV2/run_config.json
tmp/foundation_depth_auxiliary/DepthAnythingV2/input_manifest.csv
tmp/foundation_depth_auxiliary/DepthAnythingV2/logs/<date>_protocol.md
```

## 8. Sources

- Depth Anything V2 arXiv: https://arxiv.org/abs/2406.09414
- Depth Anything V2 project page: https://depth-anything-v2.github.io/
- Depth Anything V2 GitHub: https://github.com/DepthAnything/Depth-Anything-V2
