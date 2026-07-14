# External Baseline Data Preflight Plan

Updated: 2026-06-18

Purpose: move the DFV/DDFFNet comparison plan from literature-level planning toward executable data readiness checks, while keeping all generated artifacts outside the core project resources.

## Key Finding

The current project contains split metadata for 24 synthetic samples and saved height maps for several generated surface examples, but the checked generated-sample folders do not expose complete 17-frame focus stacks in a stable external-baseline format. Therefore, DFV/DDFFNet cannot be fairly run directly from the current presentation package alone. The next practical step is a controlled regeneration/export into `tmp/external_baseline_data/`.

## Added Tool

`submission_planning/tools/preflight_external_baseline_data.py`

Default behavior:

```powershell
python -X utf8 submission_planning/tools/preflight_external_baseline_data.py --json
```

Default output:

```text
tmp/external_baseline_preflight/
  README.md
  preflight_manifest.csv
  preflight_manifest.json
```

The script only reads existing metadata and generated-sample folders. It does not export large stacks, overwrite project files, download external code, or run training.

On Windows PowerShell, read generated Markdown with UTF-8 explicitly if Chinese paths are displayed incorrectly:

```powershell
Get-Content -Encoding UTF8 tmp/external_baseline_preflight/README.md
```

## What It Checks

| Check | Reason |
|---|---|
| split row exists | keeps train/validation/test protocol fixed |
| matching generated-sample folder exists | identifies whether local assets can be inspected |
| `_depth_um.npy` exists | verifies synthetic height GT availability |
| `_depth_norm.npy` exists | verifies normalized preview/analysis data availability |
| candidate stack frame count | checks whether an external focal-stack model can read frames |
| readiness category | prevents premature SOTA claims |

## Readiness Categories

| Category | Meaning | Action |
|---|---|---|
| `ready_for_stack_baseline` | stack frames and height GT appear available | eligible for one-sample DFV/DDFFNet smoke test |
| `gt_only` | height GT exists but focus-stack frames were not found | regenerate stack frames into `tmp/` |
| `metadata_only` | folder exists but GT and complete stack are missing | regenerate full sample |
| `missing_sample_folder` | no matching folder found | map sample ID to generator scenario |

## Baseline Consequence

For the manuscript, this means the external SOTA comparison should stay in planned/future status until a temporary export is produced and a model can read it. It is still valid to discuss DFV/DDFFNet as priority baselines and Depth Anything V2 as a training-strategy reference, but the paper should not report external numerical superiority yet.

## Recommended Next Step

Create a separate exporter that calls `src/simulate_antiglare_highres_samples.py::generate_sample_arrays` for one test sample, writes stack frames, `height_gt.npy`, risk masks, and priors into `tmp/external_baseline_data/`, then runs a dataloader-only smoke test. The first target should be `test_V谷_P10_宽谷粗糙平底` because it is already central to the failure analysis.

## One-Sample Export Tool

`submission_planning/tools/export_one_external_baseline_sample.py`

Default smoke-test export:

```powershell
python -X utf8 submission_planning/tools/export_one_external_baseline_sample.py --overwrite
```

Explicit larger exports:

```powershell
python -X utf8 submission_planning/tools/export_one_external_baseline_sample.py --split test --overwrite
python -X utf8 submission_planning/tools/export_one_external_baseline_sample.py --all --overwrite
```

Use `--split test` or `--all` only when the temporary data volume is acceptable and an external baseline run is ready. The default command remains single-sample P10 export.

Default output:

```text
tmp/external_baseline_data/
  manifest.csv
  run_config_template.json
  samples/test_V谷_P10_宽谷粗糙平底/
    meta.json
    height_gt.npy
    focus_positions_norm.npy
    stack/000.png ... 016.png
    masks/high_risk_mask.npy
    masks/high_risk_mask.png
    masks/risk_layers.npy
    priors/dff_depth.npy
    priors/gadff_depth.npy
    priors/focus_confidence.npy
    priors/gadff_confidence.npy
    previews/
```

This export is only a dataloader smoke-test artifact. It is not eligible for the manuscript main results table until the full train/validation/test export, external method run configuration, and scale-aligned metrics are completed.

## Dataloader-Only Smoke Test

`submission_planning/tools/smoke_test_external_baseline_package.py`

Command:

```powershell
python -X utf8 submission_planning/tools/smoke_test_external_baseline_package.py
```

Result on 2026-06-18:

| Check | Result |
|---|---|
| Frame tensor | `[17, 540, 960]` |
| Height GT | `[540, 960]` |
| Risk layers | `[17, 540, 960]` |
| PyTorch-like grayscale batch | `[1, 17, 1, 540, 960]` |
| PyTorch-like RGB batch | `[1, 17, 3, 540, 960]` |
| Status | pass |

This proves the temporary P10 package is readable as a focal-stack sample. It still does not provide DFV/DDFFNet numerical results or support a SOTA claim.
