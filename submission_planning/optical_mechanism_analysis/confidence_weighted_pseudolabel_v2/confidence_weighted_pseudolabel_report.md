# Confidence-Weighted Pseudo-Label Study

## Research Question

This controlled synthetic study asks when a focus-confidence prior makes DFF pseudo-label supervision more useful. Unlike the real-stack probes, this experiment has ground-truth height, so it can evaluate both effective pseudo-label noise and the error of a lightweight spatial student fitted to pseudo labels.

## Setup

Total conditions: 96. Each condition synthesizes a micro-surface, focus stack, DFF depth estimate, pseudo-label degradation, confidence weight map, and two ridge-regression student reconstructions. The uniform student fits all pseudo labels equally. The confidence-weighted student down-weights low-margin, low-peak-strength, and glare-risk regions. The student only sees DFF-derived and image-derived features, not ground-truth height.

Degradation modes:

- `mixed`: smooth bias plus glare-dominated early-layer corruption.
- `glare`: stronger specular/glare corruption.
- `weak_texture`: low-texture smoothing and random perturbation.

## Overall Results

| Metric | Value |
|---|---:|
| Conditions | 96 |
| Uniform student MAE | 0.0883 |
| Confidence-weighted student MAE | 0.0857 |
| Mean MAE reduction | 0.0027 |
| Mean relative improvement | 2.6% |
| Win rate | 71.9% |
| Unweighted pseudo-label noise | 0.1198 |
| Confidence-weighted pseudo-label noise | 0.1195 |
| Effective noise reduction | 0.0003 |

## Results by Degradation Mode

| Mode | N | Uniform MAE | Weighted MAE | Relative improvement | Win rate | Noise reduction |
|---|---:|---:|---:|---:|---:|---:|
| glare | 32 | 0.1128 | 0.1121 | 0.6% | 62.5% | 0.0004 |
| mixed | 32 | 0.0993 | 0.0921 | 7.2% | 100.0% | 0.0004 |
| weak_texture | 32 | 0.0528 | 0.0528 | -0.0% | 53.1% | 0.0001 |

## Interpretation

The evidence is conditional rather than uniform. Across all 96 conditions, confidence weighting reduces student MAE from 0.0883 to 0.0857, with a 71.9% win rate. The gain is concentrated in the mixed degradation mode, where relative improvement reaches 7.2% with a 100.0% win rate. The glare mode shows a smaller 0.6% improvement, while weak-texture degradation is essentially unchanged. The effective pseudo-label noise measured by weighted mean absolute error changes only slightly, so the main benefit is not a large global noise reduction; it is that the weighting changes which corrupted regions dominate the fitted student.

## Simulation-to-Real Implication

The real focus-curve morphology probe showed that real stacks contain flat ambiguous responses, multi-peak competition, local peak-layer spikes, saturated highlights, and dark low-signal regions. This synthetic study connects that observation to training design: confidence weighting helps most when several degradation sources are mixed, which resembles the real-stack setting more than an isolated weak-texture condition. Thus the simulator should output not only synthetic images and DFF depth, but also a focus-confidence map used for sample weighting, auxiliary confidence prediction, or uncertainty-aware loss.

## Paper-Ready Statement

CN: 在受控合成实验中，我们将 DFF 深度视为伪标签，并比较均匀监督与置信度加权监督。96 个条件下，置信度加权学生模型的平均 MAE 从 0.0883 降至 0.0857，胜率为 71.9%。收益主要集中在 mixed 退化模式，相对改善为 7.2%，胜率为 100.0%；glare 模式仅有轻微改善，weak-texture 模式基本不变。这说明置信度策略更适合作为按退化类型调节伪标签可信度的训练机制，而不是简单宣称对所有场景都有同等收益。

EN: In a controlled synthetic experiment, we treat DFF depth as a pseudo label and compare uniform supervision with confidence-weighted supervision. Over 96 conditions, the average MAE of the confidence-weighted student decreases from 0.0883 to 0.0857, with a 71.9% win rate. The benefit is concentrated in the mixed degradation mode, where relative improvement reaches 7.2% with a 100.0% win rate; glare shows a small gain, while weak-texture degradation is nearly unchanged. This supports confidence-aware pseudo-label weighting as a degradation-dependent training mechanism rather than a uniformly beneficial heuristic.

## Limitations

This is a controlled synthetic proxy experiment, not a full FocusResUNet training run. The student model is intentionally lightweight so that the effect of pseudo-label weighting is auditable. The result should be used to justify confidence-aware training design, while final model claims still require full neural training, seed repeats, and real-domain validation.
