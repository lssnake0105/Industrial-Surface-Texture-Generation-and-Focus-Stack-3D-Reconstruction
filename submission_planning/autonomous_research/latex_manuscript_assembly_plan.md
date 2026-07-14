# LaTeX Manuscript Assembly Plan

更新日期：2026-06-18  
用途：将当前英文大纲和研究材料组装成普通 `article` 风格 LaTeX 投稿草稿。  
边界：本文档是组稿计划，不直接修改现有 `Updated_English_Project_Paper.tex` 或论文包模板。

## 1. Target Output

建议后续生成一个独立草稿文件：

```text
submission_planning/manuscript_draft/s2r_focus_stack_manuscript.tex
```

该文件应使用普通 LaTeX article 格式，不依赖 zjuthesis 模板。投稿前可再迁移到具体期刊模板。

## 2. Source Materials

| Section | Source File |
|---|---|
| Title / Abstract | `abstract_draft_en.md` |
| Introduction | `introduction_outline_en.md` |
| Related Work | `related_work_draft_cn.md` + `sota_comparison_chinese.md` |
| Method | `method_outline_en.md` |
| Experiments | `experiments_outline_en.md` |
| Discussion | `discussion_outline_en.md` |
| Figures / Tables | `figure_table_plan.md` |
| Claims / Limitations | `claim_evidence_matrix.md` |
| Baseline protocol | `baseline_adapter_spec.md`, `baseline_reproduction_decision_table.md` |

## 3. Proposed LaTeX Structure

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{siunitx}
\usepackage{caption}
\usepackage{subcaption}

\title{Simulation-to-Real Focus-Stack Reconstruction for Reflective Surface Defect Morphology Using Prior-Guided Deep Correction}
\author{...}
\date{}

\begin{document}
\maketitle
\begin{abstract}
...
\end{abstract}

\section{Introduction}
\section{Related Work}
\section{Method}
\section{Experiments}
\section{Discussion}
\section{Conclusion}
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

## 4. Section Assembly Order

### 4.1 Abstract

Use Version A from `abstract_draft_en.md` until external DFV/DDFFNet results exist.

Rules:

1. Keep real-sample claims no-reference.
2. Do not mention external SOTA superiority.
3. Use `\SI{53.22}{\micro\meter}` only if `siunitx` setup is confirmed; otherwise use `53.22~\textmu m`.

### 4.2 Introduction

Convert the five-paragraph outline into full prose:

1. industrial 2D inspection to 3D morphology;
2. focus-stack value and DFF limitations;
3. real GT scarcity and simulation-to-real strategy;
4. proposed prior-guided correction framework;
5. contributions.

### 4.3 Related Work

Recommended subsections:

```latex
\subsection{Classical Shape and Depth from Focus}
\subsection{Learning-Based Depth from Focus}
\subsection{Simulation-to-Real and Defocus-Based Learning}
\subsection{Industrial Surface Morphology Reconstruction}
```

Depth Anything V2 should be placed in simulation-to-real / foundation depth discussion, not in the direct DFF baseline paragraph.

### 4.4 Method

Recommended subsections:

```latex
\subsection{Problem Formulation}
\subsection{Synthetic Focus-Stack Data Construction}
\subsection{Focus-Based Priors}
\subsection{Focal-Difference Representation}
\subsection{Prior-Guided Correction Network}
\subsection{Training and Inference}
```

Only implemented loss terms should be written as part of the main method. Candidate losses should move to Discussion or Future Work.

### 4.5 Experiments

Recommended subsections:

```latex
\subsection{Dataset and Evaluation Protocol}
\subsection{Baselines and Metrics}
\subsection{Synthetic Quantitative Results}
\subsection{Real No-Reference Morphology Evaluation}
\subsection{Ablation and Failure Analysis}
```

If ablation is not completed, use `Ablation Plan and Failure Analysis` or move ablation to Discussion.

### 4.6 Discussion

Recommended subsections:

```latex
\subsection{Role of Simulation-Based Supervision}
\subsection{Effect of Prior-Guided Correction}
\subsection{Limitations of Real-Sample Evaluation}
\subsection{Relation to Foundation Depth Models}
\subsection{Limitations and Future Work}
```

## 5. Figure Placement

| Figure | LaTeX Label | Source | Placement |
|---|---|---|---|
| Pipeline | `fig:pipeline` | `output/imagegen/focus_stack_dff_pipeline_v2.png` | Introduction / Method |
| Network | `fig:architecture` | `output/imagegen/focus-resunet-industrial-network-diagram-final-2048x1152.png` | Method |
| Synthetic examples | `fig:synthetic_examples` | project sample previews | Method / Experiments |
| Synthetic comparison | `fig:synthetic_results` | `results/figures/paper_comparison_mean_mae.png` | Results |
| Difficult sample | `fig:difficult_sample` | `results/figures/simulation_multisample_algorithm_panel.png` | Results |
| Real comparison | `fig:real_results` | `results/figures/real_midterm_multisample_panel.png` | Results |
| Ablation/failure | `fig:ablation_failure` | future outputs | Ablation / Discussion |

## 6. Table Placement

| Table | LaTeX Label | Source |
|---|---|---|
| Dataset split | `tab:dataset_split` | `dataset_split.csv` |
| Synthetic comparison | `tab:synthetic_comparison` | `paper_algorithm_comparison_metrics.csv` |
| Real no-reference metrics | `tab:real_metrics` | `real_midterm_method_summary.csv` |
| Ablation | `tab:ablation` | future ablation results |
| Baseline fairness | `tab:baseline_fairness` | future baseline logs |

## 7. Citation Plan

Minimum groups:

1. classical SFF/DFF: Nayar, Pertuz, Lee2013, Li2019;
2. deep DFF: DDFFNet, AiFDepthNet, DFV, DfF in the Wild, DDFS;
3. recent methods: HybridDepth, FAD, DualFocus, DDL-Recurrent SFF;
4. sim-to-real / self-supervised: Focus on Defocus, DEReD, Depth Anything V2;
5. industrial morphology: surface defect detection review, microscopy/structured illumination 3D reconstruction.

Use existing BibTeX files when possible:

```text
output/related_work_core_15.bib
output/related_work_zotero_export.bib
```

## 8. Claim Safety Checklist

Before compiling a LaTeX draft:

1. Abstract does not claim real absolute height accuracy.
2. Results separate synthetic absolute metrics and real no-reference metrics.
3. External SOTA superiority appears only after DFV/DDFFNet results exist.
4. Ablation conclusions appear only after ablation results exist.
5. Depth Anything V2 is described as a foundation-depth / training-strategy reference.
6. Final method name is consistent.

## 9. Recommended Next Draft Step

Create a first LaTeX draft only after deciding the final method name:

| Option | Consequence |
|---|---|
| `S2R-FocusNet` | emphasizes simulation-to-real story |
| `Prior-Guided Focus-ResUNet` | emphasizes architecture and prior-guided correction |

If the name is not decided, use a placeholder macro:

```latex
\newcommand{\method}{S2R-FocusNet}
```

This allows global replacement later.
