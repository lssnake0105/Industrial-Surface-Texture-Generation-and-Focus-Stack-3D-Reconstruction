from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from focus_confidence_risk_study import confidence_maps, focus_volume, normalize01


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


CLASS_ORDER = [
    "confident_single_peak",
    "flat_ambiguous",
    "multi_peak",
    "local_peak_spike",
    "saturated_highlight",
    "dark_low_signal",
]

CLASS_COLORS = {
    "confident_single_peak": "#4DAF4A",
    "flat_ambiguous": "#984EA3",
    "multi_peak": "#377EB8",
    "local_peak_spike": "#FF7F00",
    "saturated_highlight": "#E41A1C",
    "dark_low_signal": "#222222",
}


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
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name).strip("_")
    return cleaned or "real_stack"


def display_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or safe_name(name)


def read_gray(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def load_stack(stack_dir: Path, max_side: int) -> np.ndarray:
    files = image_files(stack_dir)
    if not files:
        raise FileNotFoundError(f"No image files found in {stack_dir}")
    arrays = [read_gray(path, max_side=max_side) for path in files]
    h = min(arr.shape[0] for arr in arrays)
    w = min(arr.shape[1] for arr in arrays)
    return np.stack([arr[:h, :w] for arr in arrays], axis=0)


def local_mean(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    pad = np.pad(arr, radius, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += pad[radius + dy : radius + dy + arr.shape[0], radius + dx : radius + dx + arr.shape[1]]
            count += 1
    return out / float(count)


def quantile_threshold(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values[np.isfinite(values)], q))


def top_two_indices(fv: np.ndarray, peak_idx: np.ndarray) -> np.ndarray:
    masked = fv.copy()
    yy, xx = np.indices(peak_idx.shape)
    masked[peak_idx, yy, xx] = -np.inf
    return np.argmax(masked, axis=0).astype(np.int16)


def normalized_curves(fv: np.ndarray) -> np.ndarray:
    curves = fv.astype(np.float32)
    lo = np.min(curves, axis=0, keepdims=True)
    hi = np.max(curves, axis=0, keepdims=True)
    return (curves - lo) / np.maximum(hi - lo, 1e-8)


def classify_pixels(stack: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fv = focus_volume(stack)
    conf = confidence_maps(fv)
    peak_idx = np.argmax(fv, axis=0).astype(np.int16)
    second_idx = top_two_indices(fv, peak_idx)
    top2_sep = np.abs(peak_idx.astype(np.float32) - second_idx.astype(np.float32))
    peak_layer = peak_idx.astype(np.float32) + 1.0
    margin = conf["confidence_margin"]
    low_margin = 1.0 - margin
    entropy = conf["focus_entropy"]
    peak_strength = conf["confidence_peak_strength"]
    sat_persistence = np.mean(stack >= 0.98, axis=0).astype(np.float32)
    bright_persistence = np.mean(stack >= 0.90, axis=0).astype(np.float32)
    mean_intensity = np.mean(stack, axis=0).astype(np.float32)
    max_intensity = np.max(stack, axis=0).astype(np.float32)
    spike_proxy = normalize01(np.abs(peak_idx.astype(np.float32) - local_mean(peak_idx.astype(np.float32), radius=2)))
    broad_width = np.mean(fv >= (np.max(fv, axis=0, keepdims=True) * 0.75), axis=0).astype(np.float32)

    h, w = peak_idx.shape
    classes = np.full((h, w), "confident_single_peak", dtype=object)

    q_low_margin_60 = quantile_threshold(low_margin, 0.60)
    q_low_margin_70 = quantile_threshold(low_margin, 0.70)
    q_low_margin_78 = quantile_threshold(low_margin, 0.78)
    q_entropy_70 = quantile_threshold(entropy, 0.70)
    q_peak_strength_35 = quantile_threshold(peak_strength, 0.35)
    q_peak_strength_50 = quantile_threshold(peak_strength, 0.50)
    q_mean_18 = quantile_threshold(mean_intensity, 0.18)
    q_bright_95 = quantile_threshold(bright_persistence, 0.95)
    q_max_90 = quantile_threshold(max_intensity, 0.90)
    q_spike_90 = quantile_threshold(spike_proxy, 0.90)

    layers = stack.shape[0]
    sep_gate = max(3.0, layers * 0.16)

    saturated = (sat_persistence >= 0.05) | ((bright_persistence >= q_bright_95) & (max_intensity >= max(0.82, q_max_90)))
    dark_low = (mean_intensity <= q_mean_18) & (peak_strength <= q_peak_strength_50)
    multi_peak = (top2_sep >= sep_gate) & (low_margin >= q_low_margin_60) & (entropy >= q_entropy_70)
    local_spike = (spike_proxy >= q_spike_90) & (low_margin >= q_low_margin_60)
    flat_ambiguous = (
        (low_margin >= q_low_margin_78)
        | ((entropy >= q_entropy_70) & (peak_strength <= q_peak_strength_35))
        | (broad_width >= quantile_threshold(broad_width, 0.80))
    )

    classes[flat_ambiguous] = "flat_ambiguous"
    classes[multi_peak] = "multi_peak"
    classes[local_spike] = "local_peak_spike"
    classes[dark_low] = "dark_low_signal"
    classes[saturated] = "saturated_highlight"

    maps = {
        "focus_volume": fv,
        "normalized_curves": normalized_curves(fv),
        "peak_layer": peak_layer,
        "low_margin": low_margin.astype(np.float32),
        "focus_entropy": entropy.astype(np.float32),
        "peak_strength": peak_strength.astype(np.float32),
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "top2_separation": top2_sep.astype(np.float32),
        "spike_proxy": spike_proxy.astype(np.float32),
        "broad_width": broad_width,
    }
    return {"class": classes}, maps


def class_id_map(classes: np.ndarray) -> np.ndarray:
    ids = np.zeros(classes.shape, dtype=np.float32)
    for i, name in enumerate(CLASS_ORDER):
        ids[classes == name] = float(i)
    return ids


def sample_indices(mask: np.ndarray, limit: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(mask)
    if yy.size <= limit:
        return yy, xx
    rng = np.random.default_rng(42)
    idx = rng.choice(yy.size, size=limit, replace=False)
    return yy[idx], xx[idx]


def median_curve(curves: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = sample_indices(mask)
    if yy.size == 0:
        n = curves.shape[0]
        nan = np.full(n, np.nan, dtype=np.float32)
        return nan, nan, nan
    selected = curves[:, yy, xx]
    return (
        np.nanpercentile(selected, 10, axis=1),
        np.nanmedian(selected, axis=1),
        np.nanpercentile(selected, 90, axis=1),
    )


def draw_stack_figure(stack_name: str, stack: np.ndarray, classes: np.ndarray, maps: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.8))
    cmap = plt.matplotlib.colors.ListedColormap([CLASS_COLORS[name] for name in CLASS_ORDER])
    layers = stack.shape[0]

    panels = [
        ("mid layer", stack[layers // 2], "gray", 0.0, 1.0),
        ("focus class map", class_id_map(classes), cmap, -0.5, len(CLASS_ORDER) - 0.5),
        ("peak layer", maps["peak_layer"], "turbo", 1.0, float(layers)),
        ("low margin", maps["low_margin"], "magma", 0.0, quantile_threshold(maps["low_margin"], 0.995)),
        ("top-2 layer separation", maps["top2_separation"], "viridis", 0.0, float(layers)),
    ]

    for ax, (title, data, cm, vmin, vmax) in zip(axes.ravel()[:5], panels):
        ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes.ravel()[5]
    x = np.arange(1, layers + 1)
    for name in CLASS_ORDER:
        mask = classes == name
        frac = float(mask.mean())
        if frac <= 0.002:
            continue
        p10, med, p90 = median_curve(maps["normalized_curves"], mask)
        ax.plot(x, med, label=f"{name} ({frac * 100:.1f}%)", color=CLASS_COLORS[name], linewidth=1.8)
        ax.fill_between(x, p10, p90, color=CLASS_COLORS[name], alpha=0.14, linewidth=0)
    ax.set_title("median normalized focus curves")
    ax.set_xlabel("layer")
    ax.set_ylabel("normalized focus response")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle(f"{display_name(stack_name)}: focus-curve morphology")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def stack_summary_rows(stack_name: str, classes: np.ndarray, maps: dict[str, np.ndarray]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    pixels = classes.size
    for name in CLASS_ORDER:
        mask = classes == name
        row: dict[str, float | str | int] = {
            "stack": stack_name,
            "class": name,
            "pixel_fraction": float(mask.mean()),
            "pixel_count": int(mask.sum()),
        }
        if mask.any():
            for metric in [
                "peak_layer",
                "low_margin",
                "focus_entropy",
                "peak_strength",
                "sat_persistence",
                "bright_persistence",
                "mean_intensity",
                "top2_separation",
                "spike_proxy",
                "broad_width",
            ]:
                row[f"{metric}_mean"] = float(np.mean(maps[metric][mask]))
                row[f"{metric}_p90"] = float(np.quantile(maps[metric][mask], 0.90))
        else:
            for metric in [
                "peak_layer",
                "low_margin",
                "focus_entropy",
                "peak_strength",
                "sat_persistence",
                "bright_persistence",
                "mean_intensity",
                "top2_separation",
                "spike_proxy",
                "broad_width",
            ]:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_p90"] = float("nan")
        row["total_pixels"] = pixels
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def discover_stacks(root: Path, min_layers: int) -> list[Path]:
    candidates = []
    for path in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        count = len(image_files(path))
        if count >= min_layers:
            candidates.append(path)
    return candidates


def sample_ids(stack_names: list[str]) -> dict[str, str]:
    return {name: f"S{i + 1}" for i, name in enumerate(stack_names)}


def draw_aggregate_fractions(rows: list[dict[str, float | str | int]], out: Path) -> None:
    stacks = []
    for row in rows:
        stack = str(row["stack"])
        if stack not in stacks:
            stacks.append(stack)
    ids = sample_ids(stacks)
    x = np.arange(len(stacks))
    bottom = np.zeros(len(stacks), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for cls in CLASS_ORDER:
        vals = []
        for stack in stacks:
            val = next(float(r["pixel_fraction"]) for r in rows if r["stack"] == stack and r["class"] == cls)
            vals.append(val)
        vals_arr = np.array(vals, dtype=np.float32)
        ax.bar(x, vals_arr, bottom=bottom, label=cls, color=CLASS_COLORS[cls])
        bottom += vals_arr
    ax.set_xticks(x)
    ax.set_xticklabels([ids[s] for s in stacks])
    ax.set_ylabel("pixel fraction")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def aggregate_table(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str]]:
    out = []
    for cls in CLASS_ORDER:
        cls_rows = [r for r in rows if r["class"] == cls]
        fractions = np.array([float(r["pixel_fraction"]) for r in cls_rows], dtype=np.float64)
        out.append(
            {
                "class": cls,
                "mean_fraction": float(np.mean(fractions)),
                "median_fraction": float(np.median(fractions)),
                "max_fraction": float(np.max(fractions)),
                "present_in_stacks": int(np.sum(fractions > 0.002)),
            }
        )
    return out


def fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{100.0 * x:.1f}%"


def write_summary(out_dir: Path, root: Path, stack_rows: list[dict[str, float | str | int]], skipped: list[str]) -> None:
    agg = aggregate_table(stack_rows)
    write_csv(out_dir / "real_focus_curve_morphology_aggregate.csv", agg)
    stack_names = []
    for row in stack_rows:
        name = str(row["stack"])
        if name not in stack_names:
            stack_names.append(name)
    ids = sample_ids(stack_names)

    lines = [
        "# Real Focus-Curve Morphology Probe",
        "",
        f"Data root: `{root}`",
        f"Analyzed stacks: {len(stack_names)}",
        f"Skipped stacks: {', '.join(skipped) if skipped else 'none'}",
        "",
        "## Sample IDs",
        "",
        "| ID | Stack |",
        "|---|---|",
    ]
    for name in stack_names:
        lines.append(f"| {ids[name]} | {name} |")
    lines.extend(
        [
            "",
            "## Diagnostic Classes",
            "",
            "| Class | Operational meaning | Paper use |",
            "|---|---|---|",
            "| confident_single_peak | A clear dominant focus maximum with lower entropy and stronger peak response. | Treat as regions where a DFF-derived depth cue is internally reliable. |",
            "| flat_ambiguous | Broad or low-margin focus response without a unique peak. | Indicates that the focus stack supplies weak depth evidence even before model learning. |",
            "| multi_peak | Two strong focus candidates separated across layers. | Captures competing depth hypotheses caused by texture, reflection, or repeated structures. |",
            "| local_peak_spike | The selected DFF peak layer is locally inconsistent with neighboring pixels. | Marks regions where depth maps may contain spatially isolated layer jumps. |",
            "| saturated_highlight | Saturated or persistently bright response across layers. | Separates glare-dominated optical failure from generic low-confidence texture. |",
            "| dark_low_signal | Low intensity and weak focus peak. | Captures signal-poor regions where defocus cues are underdetermined. |",
            "",
            "## Aggregate Results",
            "",
            "| Class | Mean fraction | Median fraction | Max fraction | Present stacks |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in agg:
        lines.append(
            f"| {row['class']} | {fmt_pct(float(row['mean_fraction']))} | {fmt_pct(float(row['median_fraction']))} | {fmt_pct(float(row['max_fraction']))} | {int(row['present_in_stacks'])}/{len(stack_names)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Stack Fractions",
            "",
            "| ID | Stack | Confident | Flat ambiguous | Multi-peak | Local spike | Saturated highlight | Dark low signal |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stack in stack_names:
        fractions = {str(r["class"]): float(r["pixel_fraction"]) for r in stack_rows if r["stack"] == stack}
        lines.append(
            "| {sample_id} | {stack} | {conf} | {flat} | {multi} | {spike} | {sat} | {dark} |".format(
                sample_id=ids[stack],
                stack=stack,
                conf=fmt_pct(fractions.get("confident_single_peak", float("nan"))),
                flat=fmt_pct(fractions.get("flat_ambiguous", float("nan"))),
                multi=fmt_pct(fractions.get("multi_peak", float("nan"))),
                spike=fmt_pct(fractions.get("local_peak_spike", float("nan"))),
                sat=fmt_pct(fractions.get("saturated_highlight", float("nan"))),
                dark=fmt_pct(fractions.get("dark_low_signal", float("nan"))),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The real stacks do not show a single failure mode. Their focus responses split into low-margin flat curves, separated multi-peak curves, spatial peak-layer spikes, persistent highlights, and dark low-signal areas. This supports a training strategy that predicts or consumes focus confidence rather than treating every DFF-derived target as equally reliable.",
            "",
            "For the Simulation-to-Real story, the useful abstraction is not only to synthesize surface height. The simulator should also reproduce the distribution of focus-curve morphologies: unique peaks, broad ambiguous peaks, multi-peak competition, saturated highlights, and low-signal regions. These morphology classes can guide data augmentation, sample weighting, or an auxiliary confidence head.",
            "",
            "## Paper-Ready Statement",
            "",
            "CN: 我们进一步对真实焦栈的逐像素焦度曲线进行无标注形态分型。结果显示，真实域中的不可靠区域并非单一来源，而是由低峰值间隔的平坦响应、跨层多峰竞争、局部 peak-layer 跳变、持续高亮饱和以及暗弱信号共同构成。因此，DFF 先验更适合作为带置信度的观测，而不应被等价地视为处处可靠的监督标签。",
            "",
            "EN: We further perform an unsupervised morphology analysis of per-pixel focus-response curves in real focus stacks. The unreliable regions are not governed by a single failure source; instead, they consist of flat low-margin responses, separated multi-peak competition, local peak-layer spikes, persistent saturated highlights, and dark low-signal regions. This suggests that DFF priors should be treated as confidence-aware observations rather than uniformly reliable supervision.",
            "",
            "## Limitations",
            "",
            "The classes are no-reference diagnostic categories derived from focus responses and image intensity statistics. They do not replace ground-truth height evaluation. Thresholds are adaptive per stack, so the current result supports mechanism analysis and training design, while final quantitative claims still require labeled depth or repeated acquisition validation.",
        ]
    )
    (out_dir / "real_focus_curve_morphology_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(root: Path, out_dir: Path, max_side: int, min_layers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stacks = discover_stacks(root, min_layers=min_layers)
    skipped = []
    for path in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if path not in stacks:
            skipped.append(f"{path.name}({len(image_files(path))})")

    all_rows: list[dict[str, float | str | int]] = []
    for stack_dir in stacks:
        stack = load_stack(stack_dir, max_side=max_side)
        labels, maps = classify_pixels(stack)
        classes = labels["class"]
        safe = safe_name(stack_dir.name)
        draw_stack_figure(stack_dir.name, stack, classes, maps, out_dir / f"{safe}_focus_curve_morphology.png")
        all_rows.extend(stack_summary_rows(stack_dir.name, classes, maps))

    write_csv(out_dir / "real_focus_curve_morphology_by_stack.csv", all_rows)
    draw_aggregate_fractions(all_rows, out_dir / "real_focus_curve_morphology_fractions.png")
    write_summary(out_dir, root, all_rows, skipped)
    print(out_dir / "real_focus_curve_morphology_report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("论文与PPT制作项目包") / "06_Samples" / "real_focus_stacks")
    parser.add_argument("--out", type=Path, default=Path("submission_planning") / "optical_mechanism_analysis" / "real_focus_curve_morphology")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--min-layers", type=int, default=8)
    args = parser.parse_args()
    analyze(args.root, args.out, max_side=args.max_side, min_layers=args.min_layers)


if __name__ == "__main__":
    main()
