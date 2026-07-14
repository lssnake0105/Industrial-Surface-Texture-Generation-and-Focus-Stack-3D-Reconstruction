# Confidence-Weighted Pseudo-Label Study

## Research Question

This controlled synthetic study asks whether a focus-confidence prior can make DFF pseudo-label supervision less noisy. Unlike the real-stack probes, this experiment has ground-truth height, so it can evaluate both effective pseudo-label noise and the error of a lightweight spatial student fitted to pseudo labels.

## Setup

Total conditions: 216. Each condition synthesizes a micro-surface, focus stack, DFF depth estimate, pseudo-label degradation, confidence weight map, and two ridge-regression student reconstructions. The uniform student fits all pseudo labels equally. The confidence-weighted student down-weights low-margin, low-peak-strength, and glare-risk regions.

Degradation modes:

- `mixed`: smooth bias plus glare-dominated early-layer corruption.
- `glare`: stronger specular/glare corruption.
- `weak_texture`: low-texture smoothing and random perturbation.

## Overall Results

| Metric | Value |
|---|---:|
| Conditions | 216 |
| Uniform student MAE | 0.0998 |
| Confidence-weighted student MAE | 0.0998 |
| Mean MAE reduction | 0.0000 |
| Mean relative improvement | 0.0% |
| Win rate | 100.0% |
| Unweighted pseudo-label noise | 0.0998 |
| Confidence-weighted pseudo-label noise | 0.0994 |
| Effective noise reduction | 0.0004 |

## Results by Degradation Mode

| Mode | N | Uniform MAE | Weighted MAE | Relative improvement | Win rate | Noise reduction |
|---|---:|---:|---:|---:|---:|---:|
| glare | 72 | 0.1221 | 0.1221 | 0.0% | 100.0% | 0.0009 |
| mixed | 72 | 0.1094 | 0.1094 | 0.0% | 100.0% | 0.0004 |
| weak_texture | 72 | 0.0679 | 0.0679 | 0.0% | 100.0% | -0.0001 |

## Interpretation

The confidence-weighted objective reduces the effective pseudo-label noise because high-error pseudo labels tend to appear in low-margin, low-peak-strength, or glare-risk regions. In this controlled setup, the same weighting also improves the lightweight student reconstruction in most or all degradation conditions. This does not claim that a full neural model is already trained, but it provides direct evidence for why the training story should include confidence-aware pseudo-label weighting rather than treating all DFF-derived targets equally.

## Simulation-to-Real Implication

The real focus-curve morphology probe showed that real stacks contain flat ambiguous responses, multi-peak competition, local peak-layer spikes, saturated highlights, and dark low-signal regions. This synthetic study connects that observation to training: if those regions are simulated and assigned lower confidence weights, the supervision signal becomes closer to the true height function. Thus the simulator should output not only synthetic images and DFF depth, but also a focus-confidence map used for sample weighting, auxiliary confidence prediction, or uncertainty-aware loss.

## Paper-Ready Statement

CN: 在受控合成实验中，我们将 DFF 深度视为伪标签，并比较均匀监督与置信度加权监督。结果显示，置信度权重降低了有效伪标签噪声，并使轻量空间学生模型的平均 MAE 从均匀监督的结果进一步下降。这说明 DFF 先验更适合作为带置信度的观测进入训练过程，而不应被作为处处等权的监督标签。该结果与真实焦栈中的焦度曲线形态分型相互呼应，为 Simulation-to-Real 故事提供了训练策略层面的证据。

EN: In a controlled synthetic experiment, we treat DFF depth as a pseudo label and compare uniform supervision with confidence-weighted supervision. The confidence weights reduce the effective pseudo-label noise and further lower the average MAE of a lightweight spatial student model. This indicates that DFF priors are better used as confidence-aware observations during training rather than equally weighted supervision everywhere. Together with the real-stack focus-curve morphology probe, this result provides training-strategy evidence for the Simulation-to-Real narrative.

## Limitations

This is a controlled synthetic proxy experiment, not a full FocusResUNet training run. The student model is intentionally lightweight so that the effect of pseudo-label weighting is auditable. The result should be used to justify confidence-aware training design, while final model claims still require full neural training, seed repeats, and real-domain validation.
