from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from focus_confidence_risk_study import (
    auc_score,
    block_average,
    block_max,
    box_blur,
    confidence_maps,
    focus_volume,
    normalize01,
    normals,
    point_biserial,
    specular_risk,
    top_fraction_mask,
)


def make_parametric_surface(size: int, seed: int, roughness: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]

    ramp = 0.48 + 0.22 * x
    pit = -0.22 * np.exp(-(((x + 0.25) / 0.25) ** 2 + ((y - 0.20) / 0.30) ** 2))
    ridge = 0.18 * np.exp(-((x - 0.46) / 0.10) ** 2)
    notch = -0.08 * np.exp(-(((x - 0.28) / 0.18) ** 2 + ((y + 0.42) / 0.20) ** 2))
    scratch = (0.030 + 0.055 * roughness) * np.sin(62 * y + 12 * np.sin(7 * x + 0.3 * seed))

    noise = rng.normal(size=(size, size))
    for _ in range(5):
        noise = box_blur(noise, 1)
    fine = rng.normal(size=(size, size))
    for _ in range(2):
        fine = box_blur(fine, 1)

    height = ramp + pit + ridge + notch + scratch
    height += (0.030 + 0.055 * roughness) * normalize01(noise)
    height += 0.018 * roughness * normalize01(fine)
    return normalize01(height)


def synthesize_stack_param(
    height: np.ndarray,
    risk: np.ndarray,
    layers: int,
    exposure: float,
    specular_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    focus_planes = np.linspace(0.0, 1.0, layers)
    texture = 0.42 + 0.20 * normalize01(np.sin(55 * height) + 0.5 * np.cos(93 * height))
    stack = []
    for z in focus_planes:
        defocus = np.abs(height - z)
        sharp = np.exp(-(defocus / 0.105) ** 2)
        halo_radius = max(1, int(round(1.0 + 6.0 * float(np.mean(defocus)))))
        halo = box_blur(texture + specular_strength * risk, halo_radius)
        weak_texture = 0.18 + 0.82 * texture
        image = 0.17 + 0.55 * sharp * weak_texture + (0.16 + 0.84 * risk) * halo
        image = np.clip(image * exposure, 0.0, 1.0)
        stack.append(image.astype(np.float32))
    return np.stack(stack, axis=0), focus_planes


def evaluate_case(
    seed: int,
    roughness: float,
    exposure: float,
    specular_strength: float,
    factor: int,
    low_size: int,
    layers: int,
) -> list[dict[str, float | int | str]]:
    high_size = low_size * factor
    height_hr = make_parametric_surface(high_size, seed=seed, roughness=roughness)
    risk_hr = specular_risk(normals(height_hr))
    stack_hr, planes = synthesize_stack_param(
        height_hr,
        risk_hr,
        layers=layers,
        exposure=exposure,
        specular_strength=specular_strength,
    )

    stack = np.stack([block_average(im, factor) for im in stack_hr], axis=0)
    height = block_average(height_hr, factor)
    risk_mean = block_average(risk_hr, factor)
    risk_max = block_max(risk_hr, factor)
    sat_persistence = np.mean(stack >= 0.98, axis=0)
    bright_persistence = np.mean(stack >= 0.90, axis=0)

    fv = focus_volume(stack)
    peak_idx = np.argmax(fv, axis=0)
    pred = planes[peak_idx]
    abs_error = np.abs(pred - height)
    conf = confidence_maps(fv)
    failure = abs_error >= np.quantile(abs_error, 0.90)

    scores = {
        "risk_mean": risk_mean,
        "risk_max": risk_max,
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "low_margin": 1.0 - conf["confidence_margin"],
        "focus_entropy": conf["focus_entropy"],
        "low_peak_strength": 1.0 - conf["confidence_peak_strength"],
        "hybrid_confidence": 0.60 * normalize01(conf["focus_entropy"]) + 0.40 * normalize01(1.0 - conf["confidence_margin"]),
        "hybrid_risk_confidence": (
            0.46 * normalize01(conf["focus_entropy"])
            + 0.30 * normalize01(1.0 - conf["confidence_margin"])
            + 0.14 * normalize01(risk_max)
            + 0.10 * normalize01(sat_persistence)
        ),
    }

    rows: list[dict[str, float | int | str]] = []
    for score_name, score in scores.items():
        top10 = top_fraction_mask(score, 0.10)
        rows.append(
            {
                "seed": seed,
                "roughness": roughness,
                "exposure": exposure,
                "specular_strength": specular_strength,
                "score": score_name,
                "auc_failure_top10": auc_score(score, failure),
                "point_biserial_failure": point_biserial(score, failure),
                "mean_error_top10_score": float(abs_error[top10].mean()),
                "mean_error_rest_top10": float(abs_error[~top10].mean()),
                "error_lift_top10": float(abs_error[top10].mean() - abs_error[~top10].mean()),
                "failure_rate_top10_score": float(np.mean(failure[top10])),
                "failure_base_rate": float(np.mean(failure)),
                "sat_fraction": float(np.mean(stack >= 0.98)),
                "bright_fraction": float(np.mean(stack >= 0.90)),
                "global_mae": float(np.mean(abs_error)),
                "global_rmse": float(np.sqrt(np.mean(abs_error * abs_error))),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str | int]]:
    scores = sorted({str(r["score"]) for r in rows})
    summary: list[dict[str, float | str | int]] = []
    for score in scores:
        group = [r for r in rows if r["score"] == score]
        aucs = np.array([float(r["auc_failure_top10"]) for r in group], dtype=float)
        lifts = np.array([float(r["error_lift_top10"]) for r in group], dtype=float)
        fail_rates = np.array([float(r["failure_rate_top10_score"]) for r in group], dtype=float)
        summary.append(
            {
                "score": score,
                "n_cases": len(group),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "auc_min": float(np.min(aucs)),
                "auc_max": float(np.max(aucs)),
                "auc_gt_0p5_rate": float(np.mean(aucs > 0.5)),
                "auc_gt_0p55_rate": float(np.mean(aucs > 0.55)),
                "error_lift_mean": float(np.mean(lifts)),
                "error_lift_std": float(np.std(lifts, ddof=1)) if len(lifts) > 1 else 0.0,
                "positive_lift_rate": float(np.mean(lifts > 0.0)),
                "failure_rate_top10_mean": float(np.mean(fail_rates)),
                "failure_rate_top10_std": float(np.std(fail_rates, ddof=1)) if len(fail_rates) > 1 else 0.0,
            }
        )
    return sorted(summary, key=lambda r: float(r["auc_mean"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(out: Path, summary: list[dict[str, float | str | int]]) -> None:
    names = [str(r["score"]) for r in summary]
    auc_mean = np.array([float(r["auc_mean"]) for r in summary])
    auc_std = np.array([float(r["auc_std"]) for r in summary])
    lift = np.array([float(r["error_lift_mean"]) for r in summary])
    fail_rate = np.array([float(r["failure_rate_top10_mean"]) for r in summary])
    x = np.arange(len(names))

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True)
    axes[0].bar(x, auc_mean, yerr=auc_std, color="#4C78A8", capsize=3)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("AUC mean")
    axes[0].set_title("Robustness across seeds, roughness, exposure, and specular strength")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, lift, color="#F58518")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("top10 error lift")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, fail_rate, color="#54A24B")
    axes[2].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("failure rate in top10")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=32, ha="right")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "confidence_risk_robustness_bars.png", dpi=180)
    plt.close(fig)


def plot_condition_heatmap(out: Path, rows: list[dict[str, float | int | str]], target_scores: list[str]) -> None:
    exposures = sorted({float(r["exposure"]) for r in rows})
    strengths = sorted({float(r["specular_strength"]) for r in rows})
    fig, axes = plt.subplots(
        1,
        len(target_scores),
        figsize=(4.9 * len(target_scores) + 1.2, 4.4),
        sharey=True,
        constrained_layout=True,
    )
    if len(target_scores) == 1:
        axes = [axes]
    for ax, score in zip(axes, target_scores):
        mat = np.full((len(exposures), len(strengths)), np.nan)
        for i, exposure in enumerate(exposures):
            for j, strength in enumerate(strengths):
                vals = [
                    float(r["auc_failure_top10"])
                    for r in rows
                    if r["score"] == score
                    and float(r["exposure"]) == exposure
                    and float(r["specular_strength"]) == strength
                ]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        im = ax.imshow(mat, cmap="viridis", vmin=0.40, vmax=0.70)
        ax.set_title(score)
        ax.set_xticks(np.arange(len(strengths)))
        ax.set_xticklabels([f"{v:.2f}" for v in strengths])
        ax.set_yticks(np.arange(len(exposures)))
        ax.set_yticklabels([f"{v:.2f}" for v in exposures])
        ax.set_xlabel("specular strength")
        ax.set_ylabel("exposure")
        for i in range(len(exposures)):
            for j in range(len(strengths)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] < 0.56 else "black", fontsize=8)
    fig.colorbar(im, ax=list(axes), location="right", shrink=0.88, pad=0.025, label="mean AUC")
    fig.savefig(out / "confidence_risk_condition_heatmap.png", dpi=180)
    plt.close(fig)


def write_markdown_summary(out: Path, summary: list[dict[str, float | str | int]], rows: list[dict[str, float | int | str]]) -> None:
    n_cases = len({(r["seed"], r["roughness"], r["exposure"], r["specular_strength"]) for r in rows})
    lines = [
        "# Focus Confidence Robustness Summary",
        "",
        f"Cases: {n_cases}",
        "",
        "| Score | AUC mean | AUC std | AUC>0.5 | AUC>0.55 | Error lift mean | Failure rate top10 mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            "| {score} | {auc_mean:.4f} | {auc_std:.4f} | {auc_gt_0p5_rate:.2f} | {auc_gt_0p55_rate:.2f} | {error_lift_mean:.4f} | {failure_rate_top10_mean:.4f} |".format(
                **r
            )
        )
    best = summary[0]
    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            f"The best mean AUC is `{best['score']}` ({float(best['auc_mean']):.4f} +/- {float(best['auc_std']):.4f}).",
            "Focus-confidence diagnostics should be treated as probabilistic quality cues rather than deterministic failure masks.",
        ]
    )
    (out / "confidence_risk_robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/confidence_risk_robustness")
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--low-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=25)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 17, 23, 31, 43])
    parser.add_argument("--roughness", nargs="*", type=float, default=[0.65, 1.00, 1.35])
    parser.add_argument("--exposures", nargs="*", type=float, default=[1.18, 1.38, 1.58])
    parser.add_argument("--specular-strengths", nargs="*", type=float, default=[0.95, 1.35, 1.75])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for roughness in args.roughness:
            for exposure in args.exposures:
                for specular_strength in args.specular_strengths:
                    rows.extend(
                        evaluate_case(
                            seed=seed,
                            roughness=roughness,
                            exposure=exposure,
                            specular_strength=specular_strength,
                            factor=args.factor,
                            low_size=args.low_size,
                            layers=args.layers,
                        )
                    )

    summary = summarize(rows)
    write_csv(out / "confidence_risk_robustness_case_metrics.csv", rows)
    write_csv(out / "confidence_risk_robustness_score_summary.csv", summary)
    plot_summary(out, summary)
    plot_condition_heatmap(out, rows, ["focus_entropy", "low_margin", "hybrid_confidence", "risk_max"])
    write_markdown_summary(out, summary, rows)
    print(out)


if __name__ == "__main__":
    main()
