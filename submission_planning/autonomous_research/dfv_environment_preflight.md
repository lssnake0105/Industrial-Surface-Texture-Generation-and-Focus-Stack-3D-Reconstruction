# DFV Environment Preflight

Updated: 2026-06-18

Purpose: define the local readiness gate before downloading or running the DFV external baseline. This preflight keeps the project in scaffold-only mode and records whether the current machine and temporary data package are ready for a later DFV repository smoke test.

## 1. Boundary

This preflight does not:

1. download the DFV repository;
2. install dependencies;
3. import external model code;
4. load model weights;
5. generate predictions;
6. modify source data, manuscript assets, `src/`, `results/`, `output/`, or `论文与PPT制作项目包/`.

All outputs stay under:

```text
tmp/external_baseline_results/DFV/logs/
tmp/external_baseline_results/DFV/preflight/
```

## 2. Checks

| Check | Pass condition | Interpretation |
|---|---|---|
| workspace | `tmp/external_baseline_results/DFV/` exists | external result bookkeeping is ready |
| repository path | `tmp/external_repos/DFV/` exists or is explicitly missing | missing means repository download is the next setup step |
| P10 data package | exported P10 stack, GT, masks, priors, and manifest exist | loader smoke can be attempted after code exists |
| Python | local Python executable and version recorded | environment can be documented |
| NumPy / PIL | importable | local data package checks can run |
| PyTorch | importable or missing recorded | missing blocks most DFV execution |
| CUDA | available or unavailable recorded | CPU-only may still support loader checks |
| output directories | `predictions/`, `evaluation/`, `batch_evaluation/`, `logs/` exist | later inference has isolated paths |
| accidental outputs | no `.npy` predictions under DFV predictions | scaffold remains result-free |

## 3. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/preflight_dfv_environment.py
```

Default outputs:

```text
tmp/external_baseline_results/DFV/preflight/dfv_environment_preflight.json
tmp/external_baseline_results/DFV/preflight/dfv_environment_preflight.md
tmp/external_baseline_results/DFV/logs/<date>_dfv_environment_preflight.md
```

## 4. Decision Rule

| Result | Next action |
|---|---|
| pass with repository missing | ask for or perform repository download under `tmp/external_repos/DFV/` when network approval is available |
| pass with repository present | run code inventory and P10 loader adapter |
| fail due to missing P10 package | regenerate `tmp/external_baseline_data/` first |
| fail due to missing PyTorch | record dependency blocker and avoid running model inference |

## 5. Manuscript Boundary

Passing this preflight supports only this statement: the local temporary data and bookkeeping paths are ready for a future DFV adapter test. It does not support a DFV result, SOTA comparison, or main-table entry.
