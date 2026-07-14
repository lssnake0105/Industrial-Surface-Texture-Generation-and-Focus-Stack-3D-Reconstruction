# Real Focus-Curve Morphology Probe

Data root: `论文与PPT制作项目包\06_Samples\real_focus_stacks`
Analyzed stacks: 7
Skipped stacks: 微孔(4), 微孔-竞赛时(0)

## Sample IDs

| ID | Stack |
|---|---|
| S1 | 1124 |
| S2 | 3D层纹 |
| S3 | 3D表面 |
| S4 | 圆孔50um |
| S5 | 磕碰孔5um |
| S6 | 钥匙尖头50um |
| S7 | 钥匙纹路100um |

## Diagnostic Classes

| Class | Operational meaning | Paper use |
|---|---|---|
| confident_single_peak | A clear dominant focus maximum with lower entropy and stronger peak response. | Treat as regions where a DFF-derived depth cue is internally reliable. |
| flat_ambiguous | Broad or low-margin focus response without a unique peak. | Indicates that the focus stack supplies weak depth evidence even before model learning. |
| multi_peak | Two strong focus candidates separated across layers. | Captures competing depth hypotheses caused by texture, reflection, or repeated structures. |
| local_peak_spike | The selected DFF peak layer is locally inconsistent with neighboring pixels. | Marks regions where depth maps may contain spatially isolated layer jumps. |
| saturated_highlight | Saturated or persistently bright response across layers. | Separates glare-dominated optical failure from generic low-confidence texture. |
| dark_low_signal | Low intensity and weak focus peak. | Captures signal-poor regions where defocus cues are underdetermined. |

## Aggregate Results

| Class | Mean fraction | Median fraction | Max fraction | Present stacks |
|---|---:|---:|---:|---:|
| confident_single_peak | 49.5% | 50.2% | 53.8% | 7/7 |
| flat_ambiguous | 25.5% | 22.9% | 37.3% | 7/7 |
| multi_peak | 4.2% | 5.0% | 7.8% | 6/7 |
| local_peak_spike | 3.8% | 4.2% | 4.3% | 7/7 |
| saturated_highlight | 1.8% | 0.2% | 6.5% | 3/7 |
| dark_low_signal | 15.2% | 15.6% | 18.0% | 7/7 |

## Per-Stack Fractions

| ID | Stack | Confident | Flat ambiguous | Multi-peak | Local spike | Saturated highlight | Dark low signal |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | 1124 | 50.2% | 22.9% | 1.6% | 3.6% | 6.5% | 15.3% |
| S2 | 3D层纹 | 53.8% | 22.6% | 7.3% | 4.2% | 0.0% | 12.1% |
| S3 | 3D表面 | 49.0% | 22.2% | 7.8% | 4.2% | 0.0% | 16.7% |
| S4 | 圆孔50um | 53.2% | 23.9% | 7.1% | 4.2% | 0.0% | 11.6% |
| S5 | 磕碰孔5um | 47.1% | 32.0% | 0.3% | 3.3% | 0.2% | 17.2% |
| S6 | 钥匙尖头50um | 41.0% | 37.3% | 0.1% | 2.9% | 0.7% | 18.0% |
| S7 | 钥匙纹路100um | 52.5% | 17.4% | 5.0% | 4.3% | 5.2% | 15.6% |

## Interpretation

The real stacks do not show a single failure mode. Their focus responses split into low-margin flat curves, separated multi-peak curves, spatial peak-layer spikes, persistent highlights, and dark low-signal areas. This supports a training strategy that predicts or consumes focus confidence rather than treating every DFF-derived target as equally reliable.

For the Simulation-to-Real story, the useful abstraction is not only to synthesize surface height. The simulator should also reproduce the distribution of focus-curve morphologies: unique peaks, broad ambiguous peaks, multi-peak competition, saturated highlights, and low-signal regions. These morphology classes can guide data augmentation, sample weighting, or an auxiliary confidence head.

## Paper-Ready Statement

CN: 我们进一步对真实焦栈的逐像素焦度曲线进行无标注形态分型。结果显示，真实域中的不可靠区域并非单一来源，而是由低峰值间隔的平坦响应、跨层多峰竞争、局部 peak-layer 跳变、持续高亮饱和以及暗弱信号共同构成。因此，DFF 先验更适合作为带置信度的观测，而不应被等价地视为处处可靠的监督标签。

EN: We further perform an unsupervised morphology analysis of per-pixel focus-response curves in real focus stacks. The unreliable regions are not governed by a single failure source; instead, they consist of flat low-margin responses, separated multi-peak competition, local peak-layer spikes, persistent saturated highlights, and dark low-signal regions. This suggests that DFF priors should be treated as confidence-aware observations rather than uniformly reliable supervision.

## Limitations

The classes are no-reference diagnostic categories derived from focus responses and image intensity statistics. They do not replace ground-truth height evaluation. Thresholds are adaptive per stack, so the current result supports mechanism analysis and training design, while final quantitative claims still require labeled depth or repeated acquisition validation.
