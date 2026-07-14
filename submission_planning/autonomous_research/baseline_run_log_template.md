# Baseline Run Log Template

更新日期：2026-06-18  
用途：记录外部 baseline 复现过程，保证后续结果可追溯、可解释、可判断是否进入论文主表。  
边界：这是日志模板，不包含实际运行结果。

## 1. Run Metadata

| Field | Value |
|---|---|
| run_id |  |
| date |  |
| method |  |
| method category | DFV / DDFFNet / HybridDepth / DDFS / other |
| code source |  |
| code version / commit |  |
| operator |  |
| workspace |  |
| external repo path |  |
| output path |  |

## 2. Environment

| Field | Value |
|---|---|
| OS |  |
| Python version |  |
| PyTorch version |  |
| CUDA version |  |
| GPU / CPU |  |
| key dependencies |  |
| environment isolation | yes / no |
| notes |  |

## 3. Data Configuration

| Field | Value |
|---|---|
| dataset split | train / validation / test / real |
| sample count |  |
| sample ids |  |
| input frame count |  |
| frame sampling | all / uniform / center / custom |
| image resolution |  |
| image channels | grayscale / RGB |
| normalization |  |
| resize / crop |  |
| height unit | um / relative / unknown |
| valid mask | yes / no |
| edge mask | yes / no |
| high-risk mask | yes / no |

## 4. Training / Inference Setting

| Field | Value |
|---|---|
| setting | zero-shot / pretrained / synthetic retrain / fine-tune |
| pretrained weights |  |
| training split |  |
| validation split |  |
| test split |  |
| epochs |  |
| batch size |  |
| learning rate |  |
| loss function |  |
| random seed |  |
| runtime |  |

## 5. Output Interpretation

| Field | Value |
|---|---|
| output type | depth / disparity / focus index / probability / relative depth |
| output shape |  |
| output unit |  |
| scale alignment | raw_um / affine / min-max / none |
| conversion formula |  |
| can enter synthetic main table | yes / no |
| can enter real no-reference table | yes / no |
| qualitative only | yes / no |

## 6. Synthetic Metrics

```csv
method,run_id,sample_id,mae_um,edge_mae_um,high_risk_mae_um,p90_um,scale_alignment,notes
```

Summary:

| Metric | Value |
|---|---:|
| mean MAE |  |
| mean edge MAE |  |
| mean high-risk MAE |  |
| mean P90 |  |
| failed samples |  |

## 7. Real No-Reference Metrics

```csv
method,run_id,sample_id,roughness,edge_retention,relative_dynamic_range,low_conf_spike_count,notes
```

Summary:

| Metric | Value |
|---|---:|
| mean roughness |  |
| mean edge retention |  |
| mean dynamic range |  |
| mean spike count |  |
| failed samples |  |

## 8. Failure / Issue Log

| Issue | Cause | Decision |
|---|---|---|
|  |  |  |

Possible decisions:

1. continue full evaluation;
2. restrict to qualitative result;
3. move to Related Work only;
4. stop due to unfair comparison;
5. retry in isolated environment.

## 9. Main-Table Eligibility Audit

| Requirement | Pass / Fail | Evidence |
|---|---|---|
| same synthetic test split |  |  |
| no test GT leakage |  |  |
| output scale documented |  |  |
| scale alignment documented |  |  |
| training setting documented |  |  |
| input frame count documented |  |  |
| failed samples documented |  |  |
| real results treated as no-reference |  |  |

Final decision:

```text
Main synthetic table: yes / no
Auxiliary synthetic table: yes / no
Real no-reference table: yes / no
Qualitative figure only: yes / no
Related Work only: yes / no
```

## 10. Paper Wording

If eligible for main table:

> [Method] is evaluated under the same synthetic test split with [training setting]. Its output is aligned using [scale alignment], and the same MAE, edge MAE, and high-risk MAE metrics are reported.

If qualitative only:

> [Method] is included as an auxiliary qualitative reference because its output scale is not directly comparable with the synthetic height ground truth.

If Related Work only:

> [Method] is discussed as a relevant external approach, but it is not included in the quantitative table because [reason].

## 11. Artifact Checklist

| Artifact | Path | Exists |
|---|---|---|
| run_config.json |  |  |
| metrics.csv |  |  |
| real_metrics.csv |  |  |
| prediction examples |  |  |
| log file |  |  |
| environment notes |  |  |
| failure notes |  |  |
