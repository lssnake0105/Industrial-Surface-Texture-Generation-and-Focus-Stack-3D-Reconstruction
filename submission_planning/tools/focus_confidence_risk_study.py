from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize01(a: np.ndarray) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def block_average(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return a[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def block_max(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return a[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).max(axis=(1, 3))


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return a
    out = a.copy()
    for _ in range(radius):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5.0
    return out


def make_micro_surface(size: int, seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    ramp = 0.50 + 0.26 * x
    pit = -0.22 * np.exp(-(((x + 0.25) / 0.27) ** 2 + ((y - 0.22) / 0.32) ** 2))
    ridge = 0.22 * np.exp(-((x - 0.48) / 0.10) ** 2)
    scratch = 0.055 * np.sin(76 * y + 10 * np.sin(8 * x))
    weak_texture_zone = -0.06 * np.exp(-(((x + 0.05) / 0.42) ** 2 + ((y + 0.42) / 0.24) ** 2))
    noise = rng.normal(size=(size, size))
    for _ in range(5):
        noise = box_blur(noise, 1)
    height = ramp + pit + ridge + scratch + weak_texture_zone + 0.06 * normalize01(noise)
    return normalize01(height)


def normals(height: np.ndarray, lateral_scale_um: float = 300.0, height_scale_um: float = 80.0) -> np.ndarray:
    dzdy, dzdx = np.gradient(
        height * height_scale_um,
        lateral_scale_um / height.shape[0],
        lateral_scale_um / height.shape[1],
    )
    n = np.dstack([-dzdx, -dzdy, np.ones_like(height)])
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    return n


def specular_risk(n: np.ndarray, na: float = 0.4) -> np.ndarray:
    light = np.array([0.0, 0.0, -1.0])
    view = np.array([0.0, 0.0, 1.0])
    dot = np.sum(n * light[None, None, :], axis=2, keepdims=True)
    reflected = light[None, None, :] - 2.0 * dot * n
    reflected /= np.maximum(np.linalg.norm(reflected, axis=2, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(np.sum(reflected * view[None, None, :], axis=2), -1.0, 1.0))
    acceptance = np.arcsin(np.clip(na, 0.0, 0.99))
    hard = (theta <= acceptance).astype(float)
    soft = np.exp(-(theta / max(acceptance, 1e-6)) ** 2)
    return np.maximum(0.75 * hard, soft)


def synthesize_stack(height: np.ndarray, risk: np.ndarray, layers: int, exposure: float = 1.48) -> tuple[np.ndarray, np.ndarray]:
    focus_planes = np.linspace(0.0, 1.0, layers)
    texture = 0.42 + 0.20 * normalize01(np.sin(55 * height) + 0.5 * np.cos(93 * height))
    stack = []
    for z in focus_planes:
        defocus = np.abs(height - z)
        sharp = np.exp(-(defocus / 0.105) ** 2)
        halo_radius = max(1, int(round(1.0 + 6.0 * float(np.mean(defocus)))))
        halo = box_blur(texture + 1.35 * risk, halo_radius)
        weak_texture = 0.18 + 0.82 * texture
        im = 0.17 + 0.55 * sharp * weak_texture + (0.16 + 0.84 * risk) * halo
        im = np.clip(im * exposure, 0.0, 1.0)
        stack.append(im.astype(np.float32))
    return np.stack(stack, axis=0), focus_planes


def focus_volume(stack: np.ndarray) -> np.ndarray:
    vol = []
    for im in stack:
        c = im[1:-1, 1:-1]
        lap = np.abs(im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:] - 4.0 * c)
        vol.append(box_blur(np.pad(lap, 1, mode="edge"), 2))
    return np.stack(vol, axis=0)


def confidence_maps(fv: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-8
    sorted_fv = np.sort(fv, axis=0)
    peak = sorted_fv[-1]
    second = sorted_fv[-2]
    margin = (peak - second) / np.maximum(peak, eps)
    peak_strength = normalize01(peak)
    prob = fv / np.maximum(np.sum(fv, axis=0, keepdims=True), eps)
    entropy = -np.sum(prob * np.log(np.maximum(prob, eps)), axis=0) / np.log(fv.shape[0])
    return {
        "confidence_margin": margin.astype(np.float32),
        "confidence_peak_strength": peak_strength.astype(np.float32),
        "focus_entropy": entropy.astype(np.float32),
    }


def point_biserial(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    s1 = scores[labels]
    s0 = scores[~labels]
    return float((s1.mean() - s0.mean()) / (scores.std() + 1e-8))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    scores = scores.ravel()
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-based AUC with average ranks for ties approximated by stable argsort.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    flat = score.ravel()
    n = max(1, int(round(flat.size * fraction)))
    order = np.argsort(flat, kind="mergesort")
    mask_flat = np.zeros(flat.size, dtype=bool)
    mask_flat[order[-n:]] = True
    return mask_flat.reshape(score.shape)


def run(out: Path, factor: int = 8, low_size: int = 160, layers: int = 25) -> None:
    high_size = low_size * factor
    height_hr = make_micro_surface(high_size)
    risk_hr = specular_risk(normals(height_hr))
    stack_hr, planes = synthesize_stack(height_hr, risk_hr, layers=layers)
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
        "hybrid_risk_entropy": normalize01(risk_max) * 0.45 + normalize01(conf["focus_entropy"]) * 0.35 + normalize01(sat_persistence) * 0.20,
    }

    rows = []
    for name, score in scores.items():
        q90 = top_fraction_mask(score, 0.10)
        q80 = top_fraction_mask(score, 0.20)
        rows.append(
            {
                "score": name,
                "auc_failure_top10": auc_score(score, failure),
                "point_biserial_failure": point_biserial(score, failure),
                "mean_error_top10_score": float(abs_error[q90].mean()),
                "mean_error_rest_top10": float(abs_error[~q90].mean()),
                "mean_error_top20_score": float(abs_error[q80].mean()),
                "mean_error_rest_top20": float(abs_error[~q80].mean()),
                "failure_recall_top10_score": float(np.mean(failure[q90])),
                "failure_base_rate": float(np.mean(failure)),
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "confidence_risk_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 4, figsize=(13.0, 9.4))
    maps = [
        ("GT height", height, "viridis", 0, 1),
        ("DFF depth", pred, "viridis", 0, 1),
        ("abs error", abs_error, "magma", 0, float(np.quantile(abs_error, 0.99))),
        ("failure top10%", failure.astype(float), "gray", 0, 1),
        ("risk mean", risk_mean, "inferno", 0, float(np.quantile(risk_mean, 0.99))),
        ("risk max", risk_max, "inferno", 0, 1),
        ("sat persistence", sat_persistence, "inferno", 0, max(float(sat_persistence.max()), 1e-6)),
        ("bright persistence", bright_persistence, "inferno", 0, max(float(bright_persistence.max()), 1e-6)),
        ("confidence margin", conf["confidence_margin"], "magma", 0, float(np.quantile(conf["confidence_margin"], 0.99))),
        ("focus entropy", conf["focus_entropy"], "magma", float(np.quantile(conf["focus_entropy"], 0.01)), 1),
        ("peak strength", conf["confidence_peak_strength"], "magma", 0, 1),
        ("hybrid risk entropy", scores["hybrid_risk_entropy"], "magma", 0, 1),
    ]
    for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), maps):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Focus confidence and glare-risk diagnostics")
    fig.tight_layout()
    fig.savefig(out / "confidence_risk_maps.png", dpi=180)
    plt.close(fig)

    sorted_rows = sorted(rows, key=lambda r: r["auc_failure_top10"], reverse=True)
    names = [r["score"] for r in sorted_rows]
    aucs = [r["auc_failure_top10"] for r in sorted_rows]
    top10_err = [r["mean_error_top10_score"] for r in sorted_rows]
    rest_err = [r["mean_error_rest_top10"] for r in sorted_rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    axes[0].bar(x, aucs, color="#4C78A8")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("AUC for top-10% DFF failures")
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.38
    axes[1].bar(x - width / 2, top10_err, width=width, label="top 10% score", color="#F58518")
    axes[1].bar(x + width / 2, rest_err, width=width, label="rest", color="#54A24B")
    axes[1].set_ylabel("mean abs error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "confidence_risk_auc_bars.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Focus Confidence and Glare-Risk Study Summary",
        "",
        "This branch evaluates whether focus-confidence and glare-risk maps can identify DFF failure regions in a synthetic super-resolution-integrated focal stack.",
        "",
        "| Score | AUC failure top10 | Effect size | Mean error top10 score | Mean error rest | Failure rate in top10 score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted_rows:
        lines.append(
            "| {score} | {auc_failure_top10:.4f} | {point_biserial_failure:.4f} | {mean_error_top10_score:.4f} | {mean_error_rest_top10:.4f} | {failure_recall_top10_score:.4f} |".format(
                **r
            )
        )
    best = sorted_rows[0]
    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            f"The best single diagnostic in this run is `{best['score']}`, with AUC={best['auc_failure_top10']:.4f} for identifying the top-10% DFF error pixels.",
            "Hybrid risk/confidence scores are candidates for training masks, loss weighting, and no-reference real-sample diagnostics.",
        ]
    )
    (out / "confidence_risk_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/confidence_risk_branch")
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--low-size", type=int, default=160)
    parser.add_argument("--layers", type=int, default=25)
    args = parser.parse_args()
    run(Path(args.out), factor=args.factor, low_size=args.low_size, layers=args.layers)


if __name__ == "__main__":
    main()
