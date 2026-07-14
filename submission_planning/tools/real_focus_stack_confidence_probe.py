from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from focus_confidence_risk_study import auc_score, box_blur, confidence_maps, focus_volume, normalize01, top_fraction_mask


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def image_files(stack_dir: Path) -> list[Path]:
    return sorted(
        [p for p in stack_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=numeric_key,
    )


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name)


def display_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "real_stack"


def read_gray(path: Path, max_side: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def roi_from_fraction(shape: tuple[int, int], frac: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = int(round(frac[0] * w))
    y0 = int(round(frac[1] * h))
    x1 = int(round(frac[2] * w))
    y1 = int(round(frac[3] * h))
    x0 = max(0, min(w - 2, x0))
    y0 = max(0, min(h - 2, y0))
    x1 = max(x0 + 2, min(w, x1))
    y1 = max(y0 + 2, min(h, y1))
    return x0, y0, x1, y1


def crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def local_mean(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    pad = np.pad(arr, radius, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += pad[radius + dy : radius + dy + arr.shape[0], radius + dx : radius + dx + arr.shape[1]]
            count += 1
    return out / float(count)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    flat = x.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    return ranks.reshape(x.shape)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata_simple(a), rankdata_simple(b))


def choose_rois(stack: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    h, w = stack.shape[1:]
    max_map = stack.max(axis=0)
    std_map = stack.std(axis=0)
    mean_map = stack.mean(axis=0)
    rois = {
        "highlight_edge": roi_from_fraction((h, w), (0.84, 0.18, 0.98, 0.86)),
        "ordinary_texture": roi_from_fraction((h, w), (0.34, 0.32, 0.52, 0.56)),
        "dark_region": roi_from_fraction((h, w), (0.56, 0.34, 0.76, 0.62)),
    }
    bright_score = max_map + std_map
    y, x = np.unravel_index(int(np.argmax(bright_score)), bright_score.shape)
    half_w = max(32, w // 14)
    half_h = max(48, h // 8)
    rois["highlight_edge"] = (
        max(0, x - half_w),
        max(0, y - half_h),
        min(w, x + half_w),
        min(h, y + half_h),
    )
    if float(np.mean(crop(max_map, rois["ordinary_texture"]))) > 0.75:
        rois["ordinary_texture"] = roi_from_fraction((h, w), (0.25, 0.28, 0.45, 0.52))
    if float(np.mean(crop(mean_map, rois["dark_region"]))) > 0.30:
        rois["dark_region"] = roi_from_fraction((h, w), (0.52, 0.50, 0.75, 0.76))
    return rois


def draw_maps(
    stack_name: str,
    stack: np.ndarray,
    maps: dict[str, np.ndarray],
    rois: dict[str, tuple[int, int, int, int]],
    out: Path,
) -> None:
    layers = stack.shape[0]
    panel = [
        ("first layer", stack[0], "gray", 0, 1),
        ("middle layer", stack[layers // 2], "gray", 0, 1),
        ("max intensity", stack.max(axis=0), "gray", 0, 1),
        ("sat persistence", maps["sat_persistence"], "inferno", 0, max(float(maps["sat_persistence"].max()), 1e-6)),
        ("DFF peak layer", maps["peak_layer"], "turbo", 1, layers),
        ("low margin", maps["low_margin"], "magma", 0, float(np.quantile(maps["low_margin"], 0.995))),
        ("focus entropy", maps["focus_entropy"], "magma", float(np.quantile(maps["focus_entropy"], 0.005)), 1),
        ("spike proxy", maps["spike_proxy"], "magma", 0, float(np.quantile(maps["spike_proxy"], 0.995))),
        ("quality proxy", maps["quality_proxy"], "magma", 0, 1),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.6, 10.2))
    colors = {"highlight_edge": "#E41A1C", "ordinary_texture": "#377EB8", "dark_region": "#4DAF4A"}
    for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), panel):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        if title in {"first layer", "low margin", "spike proxy", "quality proxy"}:
            for name, (x0, y0, x1, y1) in rois.items():
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colors[name], linewidth=1.5)
                ax.add_patch(rect)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{display_name(stack_name)}: real focus-stack confidence diagnostics")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def draw_roi_diagnostics(
    stack_name: str,
    rois: dict[str, tuple[int, int, int, int]],
    maps: dict[str, np.ndarray],
    out: Path,
) -> None:
    names = list(rois.keys())
    metrics = ["peak_layer", "low_margin", "focus_entropy", "sat_persistence", "spike_proxy", "quality_proxy"]
    fig, axes = plt.subplots(len(names), len(metrics), figsize=(2.15 * len(metrics), 2.05 * len(names)))
    for i, name in enumerate(names):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            patch = crop(maps[metric], rois[name])
            cmap = "turbo" if metric == "peak_layer" else "magma"
            ax.imshow(patch, cmap=cmap)
            if i == 0:
                ax.set_title(metric, fontsize=8)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{display_name(stack_name)}: ROI diagnostic crops", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def summarize_top_score(score: np.ndarray, targets: dict[str, np.ndarray], fraction: float = 0.10) -> dict[str, float]:
    mask = top_fraction_mask(score, fraction)
    row: dict[str, float] = {}
    for name, target in targets.items():
        row[f"{name}_mean_top10"] = float(target[mask].mean())
        row[f"{name}_mean_rest"] = float(target[~mask].mean())
        labels = top_fraction_mask(target, fraction)
        row[f"auc_for_{name}_top10"] = auc_score(score, labels)
    return row


def analyze(stack_dir: Path, out_dir: Path, max_side: int = 640) -> None:
    files = image_files(stack_dir)
    if not files:
        raise FileNotFoundError(f"No image files in {stack_dir}")
    arrays = [read_gray(p, max_side=max_side) for p in files]
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    stack = np.stack([a[:h, :w] for a in arrays], axis=0)
    rois = choose_rois(stack)

    fv = focus_volume(stack)
    conf = confidence_maps(fv)
    peak_idx = np.argmax(fv, axis=0).astype(np.float32)
    peak_layer = peak_idx + 1.0
    low_margin = 1.0 - conf["confidence_margin"]
    low_peak_strength = 1.0 - conf["confidence_peak_strength"]
    sat_persistence = np.mean(stack >= 0.98, axis=0)
    bright_persistence = np.mean(stack >= 0.90, axis=0)
    local_peak_mean = local_mean(peak_idx, radius=2)
    spike_proxy = normalize01(np.abs(peak_idx - local_peak_mean))
    quality_proxy = normalize01(0.68 * normalize01(low_margin) + 0.22 * normalize01(spike_proxy) + 0.10 * normalize01(sat_persistence))

    maps = {
        "peak_layer": peak_layer,
        "low_margin": low_margin,
        "focus_entropy": conf["focus_entropy"],
        "low_peak_strength": low_peak_strength,
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "spike_proxy": spike_proxy,
        "quality_proxy": quality_proxy,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(stack_dir.name)

    roi_rows: list[dict[str, float | str | int]] = []
    for roi_name, box in rois.items():
        row: dict[str, float | str | int] = {"stack": stack_dir.name, "roi": roi_name, "box": str(box)}
        for name, data in maps.items():
            patch = crop(data, box)
            row[f"{name}_mean"] = float(np.mean(patch))
            row[f"{name}_p90"] = float(np.quantile(patch, 0.90))
            row[f"{name}_max"] = float(np.max(patch))
        roi_rows.append(row)
    with (out_dir / f"{safe}_real_roi_confidence_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
        writer.writeheader()
        writer.writerows(roi_rows)

    targets = {
        "spike_proxy": spike_proxy,
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "early_peak": (peak_layer <= 3).astype(np.float32),
    }
    score_maps = {
        "low_margin": low_margin,
        "focus_entropy": conf["focus_entropy"],
        "low_peak_strength": low_peak_strength,
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "quality_proxy": quality_proxy,
    }
    assoc_rows: list[dict[str, float | str]] = []
    for score_name, score in score_maps.items():
        row: dict[str, float | str] = {"score": score_name}
        for target_name, target in targets.items():
            row[f"pearson_{target_name}"] = pearson(score, target)
            row[f"spearman_{target_name}"] = spearman(score, target)
        row.update(summarize_top_score(score, targets))
        assoc_rows.append(row)
    with (out_dir / f"{safe}_real_proxy_associations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(assoc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(assoc_rows)

    draw_maps(stack_dir.name, stack, maps, rois, out_dir / f"{safe}_real_confidence_maps.png")
    draw_roi_diagnostics(stack_dir.name, rois, maps, out_dir / f"{safe}_real_roi_diagnostic_crops.png")

    lines = [
        f"# Real Focus-Stack Confidence Probe: {stack_dir.name}",
        "",
        f"Layers: {stack.shape[0]}, resized shape: {stack.shape[1]} x {stack.shape[2]}",
        "",
        "## ROI Metrics",
        "",
        "| ROI | peak layer mean | low margin mean | focus entropy mean | sat persistence mean | spike proxy mean | quality proxy mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in roi_rows:
        lines.append(
            "| {roi} | {peak_layer_mean:.2f} | {low_margin_mean:.4f} | {focus_entropy_mean:.4f} | {sat_persistence_mean:.4f} | {spike_proxy_mean:.4f} | {quality_proxy_mean:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Proxy Association",
            "",
            "| Score | Spearman spike | AUC spike top10 | AUC saturation top10 | AUC early-peak top10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in assoc_rows:
        lines.append(
            "| {score} | {spearman_spike_proxy:.4f} | {auc_for_spike_proxy_top10:.4f} | {auc_for_sat_persistence_top10:.4f} | {auc_for_early_peak_top10:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These values are no-reference proxy diagnostics, not absolute reconstruction errors.",
            "A high association with spike, saturation, or early-peak proxies indicates that the score identifies internally unstable or glare-dominated DFF regions in the real stack.",
        ]
    )
    (out_dir / f"{safe}_real_confidence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_dir / f"{safe}_real_confidence_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="论文与PPT制作项目包/06_Samples/real_focus_stacks/钥匙纹路100um")
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/real_confidence_probe")
    parser.add_argument("--max-side", type=int, default=640)
    args = parser.parse_args()
    analyze(Path(args.stack), Path(args.out), max_side=args.max_side)


if __name__ == "__main__":
    main()
