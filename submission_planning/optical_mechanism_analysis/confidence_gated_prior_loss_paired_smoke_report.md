# Paired Smoke: Baseline vs Confidence-Gated Prior Loss

- Date: 2026-06-22
- Run id: `ABL-07`
- Tag: `2026-06-22_paired_loss_smoke`
- Artifact level: `smoke_only_claim_ineligible`
- Paired design: same model seed, same patch RNG seed, same train/validation samples, same optimizer budget

## Last-Epoch Smoke Metrics

| Variant | Val MAE Norm | Val Loss | Val Prior Loss | Val Data Weight | Val Prior Weight |
|---|---:|---:|---:|---:|---:|
| Baseline HybridDFFLoss | 0.21079206 | 0.23324849 | 0.27500099 | 1.00000000 | 0.26715106 |
| Confidence-gated prior loss | 0.21210571 | 0.23440468 | 0.27148971 | 1.00000000 | 0.17544076 |

## Paired Delta

- Delta val MAE norm, gated minus baseline: `0.00131364`
- Relative delta val MAE: `0.62%`
- Delta val loss, gated minus baseline: `0.00115620`
- Preferred by this smoke: `baseline_hybrid`

## Mechanism Interpretation

The paired smoke isolates loss behavior rather than model capacity. The confidence-gated design removes direct glare-based upweighting from the supervised data term and moves the mechanism emphasis to DFF/GADFF prior reliability. A full candidate run still needs high-risk, low-confidence, and normal-region stratified evaluation before the result can support a manuscript claim.

## Artifacts

| Artifact | Path |
|---|---|
| summary_json | `tmp\ablation_results\ABL-07\comparisons\2026-06-22_paired_loss_smoke_summary.json` |
| summary_md | `tmp\ablation_results\ABL-07\comparisons\2026-06-22_paired_loss_smoke_summary.md` |
| metrics_csv | `tmp\ablation_results\ABL-07\comparisons\2026-06-22_paired_loss_smoke_metrics.csv` |
| mechanism_report | `submission_planning\optical_mechanism_analysis\confidence_gated_prior_loss_paired_smoke_report.md` |
| baseline_checkpoint | `tmp\ablation_results\ABL-07\checkpoints\2026-06-22_paired_loss_smoke_baseline_hybrid.pt` |
| confidence_gated_checkpoint | `tmp\ablation_results\ABL-07\checkpoints\2026-06-22_paired_loss_smoke_confidence_gated.pt` |
| run_config | `tmp\ablation_results\ABL-07\run_config.json` |
