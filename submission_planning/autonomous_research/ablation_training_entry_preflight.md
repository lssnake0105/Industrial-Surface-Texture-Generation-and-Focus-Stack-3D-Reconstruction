# Ablation Training Entry Preflight

Updated: 2026-06-18

Purpose: verify the local training-entry and run-matrix assumptions for ABL-00 to ABL-04 before any ablation training starts. This preflight records code mapping, feature-space decisions, and plan corrections without modifying source files or launching training.

## 1. Boundary

This preflight does not:

1. train any model;
2. import heavy training modules;
3. create checkpoints;
4. generate predictions or figures;
5. modify `src/`, existing result folders, manuscript source, or original project assets.

All reports stay under:

```text
tmp/ablation_results/preflight/
```

## 2. Why This Gate Is Needed

Earlier scaffold files used `src/final_dataset_training.py` as the generic training entry. Current code inspection shows a more precise distinction:

| Script | Role |
|---|---|
| `src/final_dataset_training.py` | dataset construction, base 22-channel features, TinyDepthNet-style training and metrics |
| `src/train_focus_resunet_loss_experiment.py` | Focus-ResUNet final-method candidate with 38-channel upgraded features and hybrid loss |
| `src/simulate_antiglare_highres_samples.py` | generator, base features, metrics, DFF/GADFF priors |

Therefore, ABL-00 and ABL-03 should be anchored to the Focus-ResUNet upgraded feature path. ABL-01, ABL-02, and ABL-04 can be first tested as input-mask variants, but their final training script should still be derived from the Focus-ResUNet training entry if the paper's final method is S2R-FocusNet.

## 3. Checks

| Check | Pass condition |
|---|---|
| source files exist | all three scripts above exist |
| source syntax | all three scripts parse as Python AST |
| Focus-ResUNet entry symbols | `FocusResUNet`, `HybridDFFLoss`, `augment_features`, `train_model`, `evaluate_split` exist |
| base feature symbols | `features_for_model`, `feature_channel_count`, `generate_sample_arrays`, `metrics` exist |
| run matrix coverage | ABL-00 to ABL-04 exist in `ablation_run_matrix_template.csv` |
| workspace coverage | ABL-00 to ABL-04 directories and run configs exist |
| mask evidence | schema audit, ABL-03 audit, and mask smoke reports exist |
| output risk | no checkpoint, prediction, or figure files are present in ABL-00 to ABL-04 |

## 4. Current Plan Correction

The corrected ablation-entry interpretation is:

| Run ID | Corrected entry interpretation |
|---|---|
| ABL-00 | Full Focus-ResUNet / S2R-FocusNet candidate through `train_focus_resunet_loss_experiment.py` |
| ABL-01 | Direct image-to-depth as focus-stack-only or lower-prior variant; needs a derived training runner |
| ABL-02 | w/o DFF/GADFF prior; can use mask 18-21 in base or corresponding prior channels in upgraded features |
| ABL-03 | w/o focal-difference input signal; zero upgraded channels 17-32 |
| ABL-04 | w/o glare cue; current safe mask zeroes base risk channel 17, while GADFF-derived glare effects need a separate decision |

## 5. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/preflight_ablation_training_entry.py
```

Default outputs:

```text
tmp/ablation_results/preflight/ablation_training_entry_preflight.json
tmp/ablation_results/preflight/ablation_training_entry_preflight.md
```

## 6. Interpretation

Passing this preflight means the code and run-matrix assumptions are coherent enough to design a minimal ablation runner. It does not mean ablation training has run, module contributions are validated, or `claim_eligible` can be set to true.
