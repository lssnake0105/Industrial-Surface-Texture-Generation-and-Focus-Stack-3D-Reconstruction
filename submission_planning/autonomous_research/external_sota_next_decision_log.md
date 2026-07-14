# External SOTA Next-Step Decision Log

Updated: 2026-06-18

Purpose: preserve the current research decision boundary for external SOTA comparison before running DFV or DDFFNet. This file records what is ready, what remains unproven, and which evidence is required before any external method can enter the manuscript table.

## 1. Current Decision

The project should prioritize DFV first, then DDFFNet. DFV is the closer conceptual match because it uses differential focus volume, which aligns with the current project story around focal-difference information. DDFFNet remains useful as an earlier learning-based DFF baseline and can help show that the proposed method is evaluated against both early and modern deep DFF.

Current external baseline infrastructure is ready for a single-sample adapter smoke test and future prediction evaluation. It is not yet ready for a manuscript SOTA claim because no DFV or DDFFNet prediction has been produced under the fixed synthetic evaluation split.

## 2. Evidence Already Available

| Evidence | Status | Interpretation |
|---|---|---|
| P10 focal-stack export | ready | one synthetic sample can be regenerated into a temporary external-baseline package |
| Dataloader smoke test | pass | the exported sample can be read as focal-stack tensors |
| Single prediction evaluator | pass | future `.npy` predictions can be evaluated with project-aligned masks and metrics |
| Batch prediction evaluator | pass | method-level summaries can be generated once predictions exist |
| DFV/DDFFNet integration protocol | ready | the next execution gates are specified |

## 3. Claims Allowed Now

1. The project has identified DFV and DDFFNet as priority external deep DFF baselines.
2. A temporary P10 external-baseline package can be exported without modifying original project data.
3. The evaluation pipeline for future external `.npy` predictions is prepared and metric-aligned with the project rules.
4. Depth Anything V2 is relevant as a simulation-to-real training-strategy reference and auxiliary single-image prior.
5. The current manuscript should present external SOTA comparison as pending until real external predictions exist.

## 4. Claims Not Allowed Now

1. DFV/DDFFNet comparison superiority for the proposed method.
2. The current P10 smoke test represents the full synthetic test split.
3. The current evaluator itself provides external SOTA results.
4. Real-sample no-reference metrics prove calibrated real height accuracy.
5. Depth Anything V2 is a fair direct focus-stack numerical baseline.

## 5. Next Execution Gate

The next meaningful experiment is a DFV repository and dataloader smoke test under `tmp/`.

Required output:

```text
tmp/external_baseline_results/DFV/logs/<date>_inventory.md
tmp/external_baseline_results/DFV/logs/<date>_p10_loader_or_inference.md
```

Minimum pass condition:

| Gate | Requirement |
|---|---|
| location | all external code, logs, and outputs stay under `tmp/` |
| input | DFV adapter reads the exported P10 stack |
| frame order | frame order is documented |
| tensor shape | method input tensor shape is recorded |
| prediction | if inference runs, one `[540, 960]` `.npy` prediction is exported |
| evaluation | prediction is evaluated by `evaluate_external_prediction.py` |
| failure handling | if blocked, exact dependency or interface issue is logged |

## 6. Fallback Rule

If DFV cannot run quickly because of dependency or dataset-specific assumptions, the project should not spend all remaining effort on environment repair. The fallback is:

1. keep DFV as the most important pending external baseline;
2. attempt DDFFNet loader adaptation because its model may be simpler;
3. continue manuscript strengthening through ablation design, claim audit, and failure analysis;
4. report external SOTA as planned or future work unless at least one valid external prediction summary exists.

## 7. Manuscript Table Rule

An external method can enter the main synthetic table only after:

1. predictions exist for the fixed test split or skipped samples are explicitly logged;
2. scale alignment is documented;
3. batch evaluator outputs `method_summary_metrics.csv`;
4. training or zero-shot setting is stated;
5. no test ground truth is used for training or manual tuning.
