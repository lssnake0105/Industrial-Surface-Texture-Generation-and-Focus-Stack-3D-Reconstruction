# ABL-07 Full Candidate Evidence Report

- Date: 2026-06-22
- Run: ABL-07
- Variant: Confidence-gated DFF/GADFF prior loss
- Status: full matched candidate and one seed repeat trained/evaluated; claim-ineligible until audit and real-stack alignment

## Executive Finding

ABL-07 is the strongest synthetic test-split candidate so far. On the fixed 7-sample matched test split, the first full candidate reaches a mean MAE of 55.93 um, and a second seed repeat reaches 66.10 um. Both improve over the previous longer-repeat best ABL-04 at 75.46 um and the previous full model ABL-00 at 109.22 um.

## Comparison Against Existing Longer-Repeat Ablations

| Run | Variant | Test Samples | Mean MAE um | Edge MAE um | High-Risk MAE um |
|---|---|---:|---:|---:|---:|
| ABL-00 | Full S2R-FocusNet | 7 | 109.2209 | 153.0310 | 86.6455 |
| ABL-02 | w/o DFF/GADFF prior | 7 | 133.4808 | 181.9107 | 121.3387 |
| ABL-03 | w/o focal difference | 7 | 90.4542 | 158.5932 | 57.2526 |
| ABL-04 | w/o glare cue | 7 | 75.4572 | 126.8816 | 60.7381 |
| ABL-07 | Confidence-gated DFF/GADFF prior loss | 7 | 55.9286 | 123.6665 | 41.7552 |
| ABL-07-repeat | Confidence-gated DFF/GADFF prior loss | 7 | 66.0954 | 119.4014 | 48.3307 |

## ABL-07 Versus Classical DFF/GADFF

| Metric | ABL-07 | DFF | GADFF | Interpretation |
|---|---:|---:|---:|---|
| Mean MAE um | 55.9286 | 100.5533 | 105.8327 | ABL-07 improves the ratio-of-means MAE over DFF by 44.38%. |
| Win rate vs DFF | 0.8571 | - | - | ABL-07 beats DFF on 6 of 7 held-out synthetic samples. |
| Mean high-risk MAE um | 41.7552 | - | - | This is lower than all previous longer-repeat neural variants. |

The seed repeat reaches 66.0954 um mean MAE, with a 34.27% ratio-of-means gain over DFF and a 4/7 win rate. It is weaker than the first run but still better than the old ABL-04 threshold.

## Stratified Reading

| Stratum | Risk | Focus Conf | ABL-07 MAE um | DFF MAE um | Gain vs DFF | Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| high_risk | 1.0000 | 0.3733 | 44.5185 | 52.1882 | 14.70% | 0.4286 |
| low_confidence | 0.3249 | 0.0789 | 67.0181 | 151.0754 | 55.64% | 0.8571 |
| normal | 0.0199 | 0.5574 | 70.2489 | 108.0737 | 35.00% | 0.4286 |

The strongest and most stable mechanism signal is in low-confidence focus regions. High-risk regions improve in aggregate but only win on 3 of 7 samples, which means glare/risk remains heterogeneous and should not be treated as a uniformly reliable training target.

## Mechanism Implication

The evidence supports a revised story: DFF/GADFF should be used as a confidence-aware observation, with prior consistency gated by focus reliability. The previous glare cue harmed the full model when applied too directly, while ABL-07 preserves the cue as context and removes direct glare-based upweighting from the supervised data term. This better matches the optical diagnosis that focus-curve reliability is the primary failure indicator and glare risk is a contextual condition.

## Current Claim Boundary

ABL-07 should still be marked outside manuscript tables. It now has two full synthetic train/eval runs, but no eligibility audit, no manuscript-claim safety audit, and no real-stack aligned evaluation. The result is strong enough to become the main candidate direction, not strong enough to finalize a paper claim.

## Recommended Next Step

Run artifact audits and real-stack alignment next:

```powershell
python -B submission_planning/tools/audit_ablation_matched_training_eligibility.py
python -B submission_planning/tools/audit_manuscript_claim_safety.py
```

If audits pass after adapting them to ABL-07, the next writing step is to rename the method around confidence-gated prior consistency and move the story away from a generic glare module.
