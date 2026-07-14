"""Run ABL-07 on real focus stacks and compute no-reference alignment diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from focus_confidence_risk_study import confidence_maps, focus_volume, normalize01, top_fraction_mask  # noqa: E402
from real_focus_curve_morphology_probe import CLASS_COLORS, CLASS_ORDER, classify_pixels  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, dff_depth, features_for_model  # noqa: E402
from train_focus_resunet_loss_experiment import FocusResUNet, augment_features, predict_tiled_upgraded  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ABL_ROOT = ROOT / "tmp" / "ablation_results"
RUN_ID = "ABL-07"
DEFAULT_STACK_ROOT = ROOT / "archive_local" / "original_workspace" / "DFFcode" / "ALL_IMAGES"
DEFAULT_OUT = ROOT / "submission_planning" / "optical_mechanism_analysis" / "abl07_real_stack_alignment"
DEFAULT_CHECKPOINTS = [
    "2026-06-22_confidence_gated_prior_full_candidate",
    "2026-06-22_confidence_gated_prior_seed_repeat",
]


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def image_files(stack_dir: Path) -> list[Path]:
    return sorted([p for p in stack_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=numeric_key)


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
        raise FileNotFoundError(stack_dir)
    arrays = [read_gray(path, max_side=max_side) for path in files]
    h = min(arr.shape[0] for arr in arrays)
    w = min(arr.shape[1] for arr in arrays)
    return np.stack([arr[:h, :w] for arr in arrays], axis=0).astype(np.float32)


def resample_layers(stack: np.ndarray, target_layers: int = DEFAULT_STACK_LAYERS) -> np.ndarray:
    if stack.shape[0] == target_layers:
        return stack.astype(np.float32)
    pos = np.linspace(0.0, stack.shape[0] - 1, target_layers)
    out = []
    for p in pos:
        lo = int(np.floor(p))
        hi = int(np.ceil(p))
        if lo == hi:
            out.append(stack[lo])
        else:
            alpha = float(p - lo)
            out.append((1.0 - alpha) * stack[lo] + alpha * stack[hi])
    return np.stack(out, axis=0).astype(np.float32)


def local_mean(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    pad = np.pad(arr, radius, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += pad[radius + dy : radius + dy + arr.shape[0], radius + dx : radius + dx + arr.shape[1]]
            count += 1
    return out / float(count)


def risk_layers_from_stack(stack17: np.ndarray) -> np.ndarray:
    layers = []
    for layer in stack17:
        local = local_mean(layer, radius=5)
        local_excess = layer - local
        risk = ((layer >= 0.98) | ((layer >= 0.90) & (local_excess >= 0.045)) | ((layer >= 0.82) & (local_excess >= 0.08))).astype(np.float32)
        risk = local_mean(risk, radius=2)
        layers.append(np.clip(risk, 0, 1).astype(np.float32))
    return np.stack(layers, axis=0)


def local_deviation(depth: np.ndarray, radius: int = 2) -> np.ndarray:
    return np.abs(depth - local_mean(depth.astype(np.float32), radius=radius)).astype(np.float32)


def pearson_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def reduction_percent(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    ref = float(np.mean(reference[mask]))
    cand = float(np.mean(candidate[mask]))
    return (ref - cand) / max(ref, 1e-8) * 100.0


def make_real_features(stack_full: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stack17 = resample_layers(stack_full, DEFAULT_STACK_LAYERS)
    risk_layers = risk_layers_from_stack(stack17)
    risk = np.clip(np.mean(risk_layers, axis=0), 0, 1).astype(np.float32)
    dff, conf = dff_depth(stack17)
    gadff, ga_conf = dff_depth(stack17, risk_layers)
    base = features_for_model(stack17, risk, dff, conf, gadff, ga_conf)

    fv = focus_volume(stack17)
    focus_conf = confidence_maps(fv)
    peak_idx = np.argmax(fv, axis=0).astype(np.float32)
    low_margin = 1.0 - focus_conf["confidence_margin"]
    sat_persistence = np.mean(stack17 >= 0.98, axis=0).astype(np.float32)
    bright_persistence = np.mean(stack17 >= 0.90, axis=0).astype(np.float32)
    spike_proxy = normalize01(np.abs(peak_idx - local_mean(peak_idx, radius=2)))
    quality_proxy = normalize01(0.68 * normalize01(low_margin) + 0.22 * normalize01(spike_proxy) + 0.10 * normalize01(sat_persistence))
    maps = {
        "stack17": stack17,
        "risk": risk,
        "risk_layers": risk_layers,
        "dff": dff,
        "conf": conf,
        "gadff": gadff,
        "ga_conf": ga_conf,
        "low_margin": low_margin.astype(np.float32),
        "focus_entropy": focus_conf["focus_entropy"].astype(np.float32),
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "spike_proxy": spike_proxy.astype(np.float32),
        "quality_proxy": quality_proxy.astype(np.float32),
    }
    return base, maps


def checkpoint_for_tag(tag: str) -> Path:
    cfg_path = ABL_ROOT / RUN_ID / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        latest = cfg.get("latest_smoke", {})
        if isinstance(latest, dict) and latest.get("tag") == tag and latest.get("checkpoint"):
            return Path(str(latest["checkpoint"]))
    return ABL_ROOT / RUN_ID / "checkpoints" / f"{tag}.pt"


def load_model(tag: str, device: str) -> FocusResUNet:
    checkpoint = checkpoint_for_tag(tag)
    payload = torch.load(checkpoint, map_location=device)
    model = FocusResUNet().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def class_id_map(classes: np.ndarray) -> np.ndarray:
    ids = np.zeros(classes.shape, dtype=np.float32)
    for i, name in enumerate(CLASS_ORDER):
        ids[classes == name] = float(i)
    return ids


def diagnostic_masks(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    low_margin = maps["low_margin"]
    spike = maps["spike_proxy"]
    sat = maps["sat_persistence"]
    conf = 1.0 - low_margin
    low_confidence = low_margin >= float(np.quantile(low_margin, 0.75))
    spike_top10 = top_fraction_mask(spike, 0.10)
    saturated = sat >= max(float(np.quantile(sat, 0.95)), 0.05)
    confident = (low_margin <= float(np.quantile(low_margin, 0.35))) & (spike <= float(np.quantile(spike, 0.65))) & (sat < 0.05)
    quality_top10 = top_fraction_mask(maps["quality_proxy"], 0.10)
    return {
        "low_confidence": low_confidence,
        "spike_top10": spike_top10,
        "saturated": saturated,
        "confident": confident,
        "quality_top10": quality_top10,
        "focus_confident_by_margin": conf >= float(np.quantile(conf, 0.65)),
    }


def summarize_prediction(
    stack_name: str,
    checkpoint_tag: str,
    stack_full: np.ndarray,
    maps: dict[str, np.ndarray],
    classes: np.ndarray,
    pred: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dff = maps["dff"]
    dff_dev = local_deviation(dff)
    pred_dev = local_deviation(pred)
    masks = diagnostic_masks(maps)
    row: dict[str, Any] = {
        "stack": stack_name,
        "checkpoint_tag": checkpoint_tag,
        "original_layers": int(stack_full.shape[0]),
        "height": int(stack_full.shape[1]),
        "width": int(stack_full.shape[2]),
        "dff_dev_mean": float(np.mean(dff_dev)),
        "model_dev_mean": float(np.mean(pred_dev)),
        "dff_depth_std": float(np.std(dff)),
        "model_depth_std": float(np.std(pred)),
        "all_region_pearson_model_dff": pearson_masked(pred, dff, np.ones_like(dff, dtype=bool)),
    }
    for name, mask in masks.items():
        row[f"{name}_fraction"] = float(np.mean(mask))
        row[f"{name}_dff_dev_mean"] = float(np.mean(dff_dev[mask])) if np.any(mask) else float("nan")
        row[f"{name}_model_dev_mean"] = float(np.mean(pred_dev[mask])) if np.any(mask) else float("nan")
        row[f"{name}_model_dev_reduction_percent"] = reduction_percent(dff_dev, pred_dev, mask)
        row[f"{name}_pearson_model_dff"] = pearson_masked(pred, dff, mask)
        row[f"{name}_model_std_over_dff_std"] = float(np.std(pred[mask]) / max(float(np.std(dff[mask])), 1e-8)) if np.any(mask) else float("nan")

    class_rows: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        mask = classes == class_name
        if not np.any(mask):
            continue
        class_rows.append(
            {
                "stack": stack_name,
                "checkpoint_tag": checkpoint_tag,
                "class": class_name,
                "pixel_fraction": float(np.mean(mask)),
                "dff_dev_mean": float(np.mean(dff_dev[mask])),
                "model_dev_mean": float(np.mean(pred_dev[mask])),
                "model_dev_reduction_percent": reduction_percent(dff_dev, pred_dev, mask),
                "pearson_model_dff": pearson_masked(pred, dff, mask),
                "model_std_over_dff_std": float(np.std(pred[mask]) / max(float(np.std(dff[mask])), 1e-8)),
            }
        )
    return row, class_rows


def draw_alignment_panel(stack_name: str, checkpoint_tag: str, stack17: np.ndarray, maps: dict[str, np.ndarray], classes: np.ndarray, pred: np.ndarray, out: Path) -> None:
    dff = maps["dff"]
    dff_dev = local_deviation(dff)
    pred_dev = local_deviation(pred)
    reduction = dff_dev - pred_dev
    panels = [
        ("mid layer", stack17[stack17.shape[0] // 2], "gray", 0, 1),
        ("DFF depth", dff, "viridis", 0, 1),
        ("ABL-07 depth", pred, "viridis", 0, 1),
        ("low margin", maps["low_margin"], "magma", 0, float(np.quantile(maps["low_margin"], 0.995))),
        ("spike proxy", maps["spike_proxy"], "magma", 0, float(np.quantile(maps["spike_proxy"], 0.995))),
        ("sat persistence", maps["sat_persistence"], "inferno", 0, max(float(np.max(maps["sat_persistence"])), 1e-6)),
        ("DFF local deviation", dff_dev, "magma", 0, float(np.quantile(dff_dev, 0.995))),
        ("ABL-07 local deviation", pred_dev, "magma", 0, float(np.quantile(dff_dev, 0.995))),
        ("DFF dev - ABL dev", reduction, "coolwarm", -float(np.quantile(np.abs(reduction), 0.995)), float(np.quantile(np.abs(reduction), 0.995))),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 10.2))
    for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{display_name(stack_name)} / {checkpoint_tag}: no-reference alignment")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)

    class_out = out.with_name(out.stem + "_class_map.png")
    cmap = plt.matplotlib.colors.ListedColormap([CLASS_COLORS[name] for name in CLASS_ORDER])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.imshow(class_id_map(classes), cmap=cmap, vmin=-0.5, vmax=len(CLASS_ORDER) - 0.5)
    ax.set_title(f"{display_name(stack_name)} focus-curve morphology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(class_out, dpi=170)
    plt.close(fig)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tag in sorted({row["checkpoint_tag"] for row in rows}):
        part = [row for row in rows if row["checkpoint_tag"] == tag]
        agg: dict[str, Any] = {"checkpoint_tag": tag, "stack_count": len(part)}
        for key in [
            "model_dev_mean",
            "dff_dev_mean",
            "low_confidence_model_dev_reduction_percent",
            "spike_top10_model_dev_reduction_percent",
            "saturated_model_dev_reduction_percent",
            "quality_top10_model_dev_reduction_percent",
            "confident_pearson_model_dff",
            "confident_model_std_over_dff_std",
            "all_region_pearson_model_dff",
        ]:
            vals = [float(row[key]) for row in part if np.isfinite(float(row[key]))]
            agg[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, stack_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], checkpoint_tags: list[str]) -> None:
    lines = [
        "# ABL-07 Real Focus-Stack Alignment Diagnostics",
        "",
        "- Date: 2026-06-22",
        f"- Checkpoints: `{checkpoint_tags}`",
        "- Evidence type: no-reference real-stack diagnostic",
        "- Claim boundary: no real calibrated height ground truth is used here.",
        "",
        "## Aggregate Summary",
        "",
        "| Checkpoint | Stacks | Low-conf dev reduction | Spike-top10 dev reduction | Saturated dev reduction | Confident corr. | Confident std ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['checkpoint_tag']} | {row['stack_count']} | "
            f"{row['mean_low_confidence_model_dev_reduction_percent']:.2f}% | "
            f"{row['mean_spike_top10_model_dev_reduction_percent']:.2f}% | "
            f"{row['mean_saturated_model_dev_reduction_percent']:.2f}% | "
            f"{row['mean_confident_pearson_model_dff']:.4f} | "
            f"{row['mean_confident_model_std_over_dff_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Stack Core Metrics",
            "",
            "| Stack | Checkpoint | Low-conf reduction | Spike-top10 reduction | Quality-top10 reduction | Confident corr. |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in stack_rows:
        lines.append(
            f"| {row['stack']} | {row['checkpoint_tag']} | "
            f"{row['low_confidence_model_dev_reduction_percent']:.2f}% | "
            f"{row['spike_top10_model_dev_reduction_percent']:.2f}% | "
            f"{row['quality_top10_model_dev_reduction_percent']:.2f}% | "
            f"{row['confident_pearson_model_dff']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Positive local-deviation reduction means ABL-07 is smoother than DFF in regions that DFF diagnostics mark as unreliable. This is useful only when the model still tracks DFF in confident regions. The confident-region correlation and standard-deviation ratio are therefore included as structure-retention checks.",
            "",
            "This analysis does not prove absolute real-height accuracy. It only tests whether ABL-07 aligns with the no-reference real-stack failure diagnosis developed from focus-margin, spike, saturation, and morphology probes.",
        ]
    )
    (out_dir / "abl07_real_stack_alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stack_dirs(root: Path, explicit: list[Path] | None, min_layers: int) -> list[Path]:
    if explicit:
        return explicit
    dirs = []
    for path in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if len(image_files(path)) >= min_layers:
            dirs.append(path)
    return dirs


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tags = args.checkpoint_tag or DEFAULT_CHECKPOINTS
    models = {tag: load_model(tag, device) for tag in checkpoint_tags}
    selected_stacks = stack_dirs(args.stack_root, args.stack, args.min_layers)
    stack_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for stack_dir in selected_stacks:
        stack_full = load_stack(stack_dir, args.max_side)
        base, maps = make_real_features(stack_full)
        classes_info, _morph_maps = classify_pixels(maps["stack17"])
        classes = classes_info["class"]
        upgraded = augment_features(base)
        for tag, model in models.items():
            pred = predict_tiled_upgraded(model, upgraded, device, tile=args.tile, overlap=args.overlap)
            row, per_class = summarize_prediction(stack_dir.name, tag, stack_full, maps, classes, pred)
            stack_rows.append(row)
            class_rows.extend(per_class)
            if args.figures:
                fig_name = f"{safe_name(stack_dir.name)}_{safe_name(tag)}_alignment.png"
                draw_alignment_panel(stack_dir.name, tag, maps["stack17"], maps, classes, pred, out_dir / fig_name)
    aggregate_rows = aggregate(stack_rows)
    write_csv(out_dir / "abl07_real_stack_alignment_stack_metrics.csv", stack_rows)
    write_csv(out_dir / "abl07_real_stack_alignment_class_metrics.csv", class_rows)
    write_csv(out_dir / "abl07_real_stack_alignment_aggregate.csv", aggregate_rows)
    payload = {
        "date": "2026-06-22",
        "status": "pass",
        "stack_count": len(selected_stacks),
        "checkpoint_tags": checkpoint_tags,
        "output_dir": str(out_dir),
        "claim_boundary": "No calibrated real height ground truth. No absolute real-height accuracy claim.",
        "aggregate": aggregate_rows,
    }
    (out_dir / "abl07_real_stack_alignment_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, stack_rows, class_rows, aggregate_rows, checkpoint_tags)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--stack", action="append", type=Path, help="Explicit stack directory. May be repeated.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-tag", action="append", help="ABL-07 checkpoint tag. May be repeated.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "stack_count": payload["stack_count"], "output_dir": payload["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
