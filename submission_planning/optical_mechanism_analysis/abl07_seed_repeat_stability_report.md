# ABL-07 Seed-Repeat Stability Report

- Date: 2026-06-22
- Variant: Confidence-gated DFF/GADFF prior loss
- First full candidate seed: 20260622
- Repeat seed: 20260623
- Status: two full train/eval runs completed; claim-ineligible until audit and real-stack alignment

## Executive Finding

ABL-07 remains stronger than the previous best longer-repeat baseline under a different seed. The repeat reaches 66.10 um mean MAE on the fixed 7-sample synthetic test split, still below the old best ABL-04 result of 75.46 um. The first ABL-07 run was stronger at 55.93 um, so the effect is promising but seed-sensitive.

## Stability Table

| Run | Seed | Mean MAE um | Edge MAE um | High-Risk MAE um | Gain vs DFF | Win Rate vs DFF |
|---|---:|---:|---:|---:|---:|---:|
| ABL-04 longer repeat | n/a | 75.4572 | 126.8816 | 60.7381 | n/a | n/a |
| ABL-07 full candidate | 20260622 | 55.9286 | 123.6665 | 41.7552 | 44.38% | 0.8571 |
| ABL-07 seed repeat | 20260623 | 66.0954 | 119.4014 | 48.3307 | 34.27% | 0.5714 |

## Stratified Stability

| Stratum | First ABL-07 MAE um | Repeat MAE um | Repeat Gain vs DFF | Repeat Win Rate |
|---|---:|---:|---:|---:|
| high_risk | 44.5185 | 50.0125 | 4.17% | 0.4286 |
| low_confidence | 67.0181 | 75.9363 | 49.74% | 0.8571 |
| normal | 70.2489 | 83.0684 | 23.14% | 0.4286 |

## Mechanism Interpretation

The repeat supports the main mechanism direction: confidence-gated prior consistency is beneficial, and the most stable gain remains in low-confidence focus regions. The high-risk region improves in aggregate but has only a 3/7 win rate against DFF in both runs, so glare/risk should remain a contextual cue rather than the core reliability target.

## Decision

ABL-07 should replace the previous full model as the main synthetic candidate for the manuscript story. The claim should be framed around confidence-aware DFF/GADFF prior gating, not around a generic glare module. Before using it as final paper evidence, the project still needs:

1. an eligibility audit over training/evaluation artifacts;
2. a manuscript-claim safety audit;
3. real-stack no-reference diagnostic alignment;
4. a concise method rename that reflects the stable mechanism.

## Suggested Paper Wording

Use a cautious statement:

> Confidence-gated consistency improves synthetic held-out performance across two seeds and is most stable in low-confidence focus regions, suggesting that DFF/GADFF priors should be treated as reliability-weighted observations rather than fixed pseudo-labels.

Avoid a stronger statement until real-stack diagnostics and artifact audits are complete.
