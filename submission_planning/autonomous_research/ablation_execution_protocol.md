# Ablation Execution Protocol

Updated: 2026-06-18

Purpose: convert the ablation study from a conceptual design into an executable, auditable experiment plan without changing current training code or overwriting existing results.

## 1. Current Boundary

The current manuscript can state that ablation experiments are required, but it cannot claim that each module has been independently validated. Existing evidence supports the final internal comparison result for Focus-ResUNet, while the following component-level claims remain pending:

1. DFF/GADFF priors reduce learning burden;
2. focal-difference representation improves axial focus-response modeling;
3. glare-aware cues improve high-risk reflective regions;
4. domain randomization improves real-sample morphology stability;
5. residual or bounded correction suppresses unstable spikes.

## 2. Run Matrix

| Run ID | Variant | DFF/GADFF Prior | Focal Difference | Glare Cue | Domain Randomization | Residual / Bounded | Current Status | Manuscript Use |
|---|---|---|---|---|---|---|---|---|
| ABL-00 | Full S2R-FocusNet | yes | yes | yes | yes | yes | existing candidate | final method if config is verified |
| ABL-01 | Direct image-to-depth | no | no | no | yes | no | pending | prior-guided necessity |
| ABL-02 | w/o DFF/GADFF prior | no | yes | yes | yes | yes | pending | focus-prior contribution |
| ABL-03 | w/o focal difference | yes | no | yes | yes | yes | pending | axial focus-response contribution |
| ABL-04 | w/o glare cue | yes | yes | no | yes | yes | pending | high-risk / glare modeling contribution |
| ABL-05 | w/o domain randomization | yes | yes | yes | no | yes | pending | simulation-to-real robustness |
| ABL-06 | unbounded prediction | yes | yes | yes | yes | no | optional | spike-control and residual-bound analysis |

## 3. Required Evidence Per Run

Each ablation run should produce:

```text
tmp/ablation_results/<run_id>/
  run_config.json
  logs/<date>_run_log.md
  metrics/synthetic_metrics.csv
  metrics/real_no_reference_metrics.csv
  predictions/
  figures/
```

Minimum metadata:

| Field | Required |
|---|---|
| run id and variant name | yes |
| code source and commit / local status | yes |
| training split and test split | yes |
| enabled input channels | yes |
| loss function and supervision | yes |
| random seed | yes |
| model checkpoint path | yes, if trained |
| synthetic metric file | yes |
| real no-reference metric file | yes, if real inference is run |
| failed samples | yes |

## 4. Metrics

Synthetic metrics:

1. mean MAE;
2. edge MAE;
3. high-risk MAE;
4. P90 error;
5. spike count or equivalent tail-error indicator.

Real no-reference metrics:

1. roughness;
2. low-confidence spike count;
3. relative dynamic range;
4. edge retention;
5. profile continuity or visual profile evidence.

## 5. Inference Rules

| Claim | Required evidence |
|---|---|
| prior-guided correction is useful | ABL-00 outperforms ABL-01 and ABL-02 under the same split |
| focal-difference is useful | ABL-00 outperforms ABL-03, especially on edge / periodic / P10 samples |
| glare cue is useful | ABL-00 outperforms ABL-04 on high-risk MAE or real spike maps |
| domain randomization helps real transfer | ABL-00 shows better real no-reference stability than ABL-05 without harming synthetic metrics substantially |
| residual bound suppresses spikes | ABL-00 improves spike count / tail error compared with ABL-06 |

If a comparison is mixed, the paper should report it as a limitation or conditional effect rather than forcing a positive contribution claim.

## 6. Execution Order

Recommended order:

1. ABL-00 config verification;
2. ABL-01 direct image-to-depth;
3. ABL-02 w/o DFF/GADFF prior;
4. ABL-03 w/o focal difference;
5. ABL-04 w/o glare cue;
6. ABL-05 w/o domain randomization;
7. ABL-06 unbounded prediction only if spike behavior remains central.

## 7. Current Code Mapping Notes

Current code inspection suggests these likely mapping points:

| Component | Candidate source location | Current interpretation |
|---|---|---|
| training entry | `src/final_dataset_training.py` | main synthetic training and validation workflow |
| model baseline | `TinyDepthNet` from `src/simulate_antiglare_prototype.py` | direct or lightweight learning baseline candidate |
| synthetic arrays | `generate_sample_arrays()` from `src/simulate_antiglare_highres_samples.py` | provides stack, truth, DFF/GADFF, risk, and feature arrays |
| metric function | `metrics()` from `src/simulate_antiglare_highres_samples.py` | source for MAE, edge, high-risk metrics |
| comparison summary | `src/paper_algorithm_comparison.py` | current internal comparison aggregation |

These notes are not a code-change instruction. Before running ablations, inspect the exact feature channel order and training function inputs, then create a separate temporary run script or config under `tmp/` or `submission_planning/tools/`.

## 7.1 Feature Schema Audit Result

The current feature schema audit on P10 reports a 22-channel tensor:

| Channel range | Meaning |
|---|---|
| 0-16 | raw 17-layer focus stack |
| 17 | risk / high-risk map |
| 18 | DFF depth prior |
| 19 | DFF focus confidence |
| 20 | GADFF depth prior |
| 21 | GADFF confidence |

This means ABL-01, ABL-02, and ABL-04 can be defined as channel-masking variants at the base-feature input level. ABL-03 should instead use the upgraded Focus-ResUNet feature path:

```text
src/train_focus_resunet_loss_experiment.py::augment_features()
```

The ABL-03 implementation audit confirms that this function converts the 22-channel base tensor into a 38-channel tensor:

| Channel range | Meaning |
|---|---|
| 0-16 | raw 17-layer focus stack |
| 17-32 | 16 adjacent focal-difference channels |
| 33-37 | 5 prior channels |

The recommended ABL-03 action is to zero channels 17-32 while keeping the 38-channel Focus-ResUNet architecture unchanged. This should be described as "w/o focal-difference input signal" unless a later implementation removes the difference branch architecturally.

ABL-05 and ABL-06 also require implementation beyond channel masking: ABL-05 controls data-generation randomization, and ABL-06 controls the model output or loss design.

## 7.2 Mask Smoke Test Result

The ablation mask smoke test verifies that the current mask definitions operate on the intended feature spaces:

| Run ID | Feature space | Masked channels | Smoke result |
|---|---|---|---|
| ABL-01 | base 22-channel features | 17-21 | pass |
| ABL-02 | base 22-channel features | 18-21 | pass |
| ABL-03 | upgraded 38-channel Focus-ResUNet features | 17-32 | pass |
| ABL-04 | base 22-channel features | 17 | pass |

The smoke test only verifies input masking. It does not train any model or support module-effectiveness claims.

## 7.3 Training Entry Preflight Result

The ablation training-entry preflight reports `status=pass` and verifies that the current scaffold, schema audit, ABL-03 focal-difference audit, and mask smoke test are traceable before training. It also corrects the training-entry assumptions for the core ablations:

| Run ID | Corrected entry assumption | Feature space | Next implementation action |
|---|---|---|---|
| ABL-00 | Use `src/train_focus_resunet_loss_experiment.py` as the final-method anchor | upgraded 38-channel Focus-ResUNet features | verify full config and metrics write path |
| ABL-01 | Derive a lower-prior image-to-depth runner from the final-method path | to be fixed by runner design | avoid treating it as a simple channel mask on the final model |
| ABL-02 | Derive from the final-method path and zero prior channels after feature augmentation | upgraded 38-channel features recommended | define whether DFF/GADFF-derived channels are all disabled |
| ABL-03 | Use the final-method path and zero focal-difference channels 17-32 | upgraded 38-channel Focus-ResUNet features | keep architecture stable and remove the input signal |
| ABL-04 | Derive from the final-method path and disable glare/risk cue consistently | upgraded 38-channel features recommended | decide whether only risk is removed or GADFF-derived glare information is also separated |

The earlier `src/final_dataset_training.py` mapping remains useful for dataset and base training references, but the final-method ablation plan should now be built around `src/train_focus_resunet_loss_experiment.py`. The next safe action is to create a minimal ablation runner and run shape/config checks before launching any training job.

## 7.4 Minimal Runner Smoke Result

The minimal runner smoke test reports `status=pass` on 2026-06-19. It builds a P10 64 x 64 center patch, applies the corrected masks in the upgraded 38-channel feature space, and runs the randomly initialized Focus-ResUNet in `eval()` mode with `torch.no_grad()`.

| Run ID | Smoke action | Result |
|---|---|---|
| ABL-00 | full upgraded 38-channel input | `[1, 38, 64, 64] -> [1, 1, 64, 64]` shape smoke passed |
| ABL-01 | raw 17-channel focus stack only | no forward pass; lower-prior runner design recorded |
| ABL-02 | zero upgraded prior channels 34-37 | `[1, 38, 64, 64] -> [1, 1, 64, 64]` shape smoke passed |
| ABL-03 | zero upgraded focal-difference channels 17-32 | `[1, 38, 64, 64] -> [1, 1, 64, 64]` shape smoke passed |
| ABL-04 | zero upgraded risk channel 33 | `[1, 38, 64, 64] -> [1, 1, 64, 64]` shape smoke passed |

This result confirms the runner interface and channel-control plan for ABL-00/02/03/04. ABL-01 remains a separate architectural decision because a direct image-to-depth baseline should not be approximated only by masking the final model's prior channels.

The smoke diagnostic losses are finite-forward checks only. They are not metrics and must not enter the ablation table.

## 7.5 Training Runner Preflight Result

The training-runner preflight reports `status=pass` on 2026-06-19. It verifies that future ablation training should reuse components from `src/train_focus_resunet_loss_experiment.py`, but should not call output-writing functions from the original experiment script.

The following functions are gated from direct use in the ablation runner:

```text
main()
evaluate_split()
write_metric_plots()
write_report()
```

These functions write to the original delivery-package output path. The future ablation runner must write all artifacts under:

```text
tmp/ablation_results/<run_id>/
```

Current trainable status after runner implementation:

| Run ID | Runner mode | Status |
|---|---|---|
| ABL-00 | `focus_resunet_upgraded` | trainable after runner implementation |
| ABL-01 | `lower_prior_focus_stack_only` | gated by architecture decision |
| ABL-02 | `focus_resunet_upgraded_masked` | trainable after runner implementation |
| ABL-03 | `focus_resunet_upgraded_masked` | trainable after runner implementation |
| ABL-04 | `focus_resunet_upgraded_masked` | trainable after runner implementation |

ABL-01 has a separate decision log at `tmp/ablation_results/ABL-01/logs/2026-06-19_lower_prior_architecture_decision.md`. The preferred direction is a 17-channel focus-stack-only lower-prior network, with TinyDepthNet retained as an internal baseline rather than the core ABL-01 variant.

## 7.6 Training Runner Dry-Run Result

The default training-runner dry run reports `status=pass` on 2026-06-19 for ABL-00 and ABL-03.

| Run ID | Action | Result |
|---|---|---|
| ABL-00 | full final-method dry run | `[1, 38, 64, 64] -> [1, 1, 64, 64]` forward/loss interface passed |
| ABL-03 | zero focal-difference channels 17-32 | `[1, 38, 64, 64] -> [1, 1, 64, 64]` forward/loss interface passed |

The dry run writes:

```text
tmp/ablation_results/training_runner_dry_run/ablation_training_runner_dry_run_summary.md
tmp/ablation_results/ABL-00/logs/2026-06-19_training_runner_dry_run.md
tmp/ablation_results/ABL-03/logs/2026-06-19_training_runner_dry_run.md
```

It does not create an optimizer, run backpropagation, save checkpoints, save predictions, write metric results, or update `claim_eligible`. The next step is an explicitly guarded small-training mode.

## 7.7 Small-Training Debug Result

The explicitly guarded small-training debug run reports `status=pass` on 2026-06-19 for ABL-00 and ABL-03.

Command:

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-id ABL-00 --run-id ABL-03 --max-epochs 1 --train-patches 8 --val-patches 4 --batch-size 1
```

| Run ID | Status | Debug checkpoint | Claim state |
|---|---|---|---|
| ABL-00 | small training debug completed | `tmp/ablation_results/ABL-00/checkpoints/2026-06-19_small_training_debug.pt` | `claim_eligible=false` |
| ABL-03 | small training debug completed | `tmp/ablation_results/ABL-03/checkpoints/2026-06-19_small_training_debug.pt` | `claim_eligible=false` |

This run only proves that the protected training runner can execute the minimal optimizer/backward/update chain and write debug artifacts under `tmp/ablation_results/`. It is not a valid ablation result because it uses a single P10 patch-sampling setup, 1 epoch, 8 training patches, and 4 validation patches.

The next experimental step should be a controlled pilot for ABL-00/02/03/04 with longer but still explicitly marked debug settings, followed by a claim eligibility audit.

## 8. Manuscript Wording Rule

Before ablation results exist:

> Ablation experiments are planned to isolate the effects of focus-based priors, focal-difference representation, glare-aware cues, and domain randomization.

After valid results exist:

> The ablation table reports [specific measured difference] under the same synthetic split and no-reference real evaluation protocol.

Do not write component effectiveness as a confirmed conclusion until the corresponding run has metrics and logs.
