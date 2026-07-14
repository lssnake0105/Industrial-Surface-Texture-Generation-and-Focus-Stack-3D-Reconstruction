# Ablation Training Runner Dry Run

Updated: 2026-06-19

Purpose: validate the default dry-run behavior of the future ablation training runner before any real training is enabled.

## 1. Boundary

This dry run does not:

1. create an optimizer;
2. run backward propagation;
3. update model weights;
4. save checkpoint files;
5. save prediction arrays or figures;
6. write real metric results;
7. update `claim_eligible`.

The generated logs stay under:

```text
tmp/ablation_results/
```

## 2. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/run_ablation_variant_training.py
```

By default, the tool runs only ABL-00 and ABL-03 in dry-run mode. Real training is intentionally disabled; passing `--execute-training` currently exits with an error.

## 3. Current Result

The 2026-06-19 dry run reports `status=pass` with 20 checks, 0 errors, and 0 warnings.

| Run | Status | Input | Output | Interpretation |
|---|---|---|---|---|
| ABL-00 | dry-run passed | `[1, 38, 64, 64]` | `[1, 1, 64, 64]` | full final-method runner interface is reachable |
| ABL-03 | dry-run passed | `[1, 38, 64, 64]` | `[1, 1, 64, 64]` | w/o focal-difference mask can enter the same runner interface |

Report files:

```text
tmp/ablation_results/training_runner_dry_run/ablation_training_runner_dry_run_summary.md
tmp/ablation_results/ABL-00/logs/2026-06-19_training_runner_dry_run.md
tmp/ablation_results/ABL-03/logs/2026-06-19_training_runner_dry_run.md
```

## 4. Interpretation

Passing this dry run means the future training runner can safely reach the ABL-00 and ABL-03 data, mask, model, and loss interfaces while writing only to `tmp/ablation_results/`.

It does not mean ABL-00 or ABL-03 has been trained. The diagnostic losses are only finite-forward checks on randomly initialized models and must not be used as performance metrics.

## 5. Next Step

The next implementation step is to add an explicitly controlled small-scale training mode, still writing only under `tmp/ablation_results/<run_id>/`, with conservative parameters such as:

```text
--run-id ABL-00 --max-epochs 1 --train-patches 8 --val-patches 4
--run-id ABL-03 --max-epochs 1 --train-patches 8 --val-patches 4
```

Before any real training output is used in the manuscript, each run must have metrics, logs, code-state notes, and a claim eligibility audit.
