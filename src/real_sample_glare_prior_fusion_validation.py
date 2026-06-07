from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from final_dataset_training import build_dataset
from simulate_antiglare_highres_samples import generate_sample_arrays, metrics, save_float_image
from train_focus_resunet_loss_experiment import FocusResUNet, augment_features, predict_tiled_upgraded


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "结题交付包"
OUT = PACKAGE / "05_图表与结果" / "真实样本_原算法与眩光先验融合验证"
MODEL_PATH = PACKAGE / "05_图表与结果" / "模型与损失函数升级实验" / "model" / "focus_resunet_hybrid_loss.pt"
STACK_LAYERS = 17


SAMPLES = [
    {"key": "01_3D层纹", "name": "3D打印层纹", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "3D层纹", "height_um": 100.0},
    {"key": "02_3D表面", "name": "3D打印表面", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "3D表面", "height_um": 100.0},
    {"key": "03_磕碰孔5um", "name": "金属表面磕碰孔", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "磕碰孔5um", "height_um": 5.0},
    {"key": "04_钥匙纹路100um", "name": "钥匙纹路", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "钥匙纹路100um", "height_um": 100.0},
    {"key": "05_圆孔50um", "name": "圆孔50um", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "圆孔50um", "height_um": 50.0},
    {"key": "06_钥匙尖头50um", "name": "钥匙尖头50um", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "钥匙尖头50um", "height_um": 50.0},
]


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        files.extend(folder.glob(ext))
    return sorted(files, key=natural_key)


def imread_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def normalize01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def normalize_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(normalize01(arr) * 255, 0, 255).astype(np.uint8)


def resize_stack_to_first(images: list[np.ndarray]) -> list[np.ndarray]:
    h, w = images[0].shape
    return [cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape != (h, w) else img for img in images]


def resample_stack(images: list[np.ndarray], layers: int = STACK_LAYERS) -> list[np.ndarray]:
    if len(images) == layers:
        return images
    if len(images) < 2:
        return images * layers
    positions = np.linspace(0, len(images) - 1, layers)
    chosen: list[np.ndarray] = []
    for pos in positions:
        lo = int(math.floor(float(pos)))
        hi = min(lo + 1, len(images) - 1)
        alpha = float(pos - lo)
        blended = (1.0 - alpha) * images[lo].astype(np.float32) + alpha * images[hi].astype(np.float32)
        chosen.append(np.clip(blended, 0, 255).astype(np.uint8))
    return chosen


def focus_maps_from_stack(stack: np.ndarray) -> np.ndarray:
    maps: list[np.ndarray] = []
    for layer in stack:
        u8 = np.clip(layer * 255, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(u8, (3, 3), 0)
        lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
        sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = sx * sx + sy * sy
        fm = cv2.boxFilter(lap, -1, (7, 7), normalize=True) + 0.0018 * cv2.boxFilter(tenengrad, -1, (7, 7), normalize=True)
        maps.append(fm.astype(np.float32))
    return np.stack(maps, axis=0)


def risk_layers_from_stack(stack: np.ndarray) -> np.ndarray:
    layers: list[np.ndarray] = []
    h, w = stack.shape[1:]
    for layer in stack:
        local_med = cv2.medianBlur(np.clip(layer * 255, 0, 255).astype(np.uint8), 21).astype(np.float32) / 255.0
        local_excess = layer - local_med
        high = layer > 0.94
        local_high = (layer > 0.78) & (local_excess > 0.075)
        bloom = cv2.GaussianBlur(np.maximum(layer - np.percentile(layer, 96.0), 0).astype(np.float32), (0, 0), max(1.2, 0.006 * max(h, w)))
        bloom = normalize01(bloom) > 0.14
        risk = (high | local_high | bloom).astype(np.float32)
        risk = cv2.GaussianBlur(risk, (0, 0), max(1.0, 0.0035 * max(h, w)))
        layers.append(np.clip(risk, 0, 1).astype(np.float32))
    return np.stack(layers, axis=0)


def dff_depth_from_focus(focus: np.ndarray, risk_layers: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if risk_layers is not None:
        focus = focus * np.clip(1.0 - 0.70 * risk_layers, 0.20, 1.0)
    idx = np.argmax(focus, axis=0)
    depth = idx.astype(np.float32) / max(focus.shape[0] - 1, 1)
    sorted_focus = np.sort(focus, axis=0)
    peak = sorted_focus[-1]
    second = sorted_focus[-2] if focus.shape[0] > 1 else np.zeros_like(peak)
    conf_raw = (peak - second) / (peak + 1e-6)
    conf = np.clip(conf_raw / (np.percentile(conf_raw, 98.5) + 1e-6), 0, 1)
    return depth.astype(np.float32), conf.astype(np.float32)


def postprocess_depth(
    depth: np.ndarray,
    median_kernel: int = 5,
    gaussian_kernel: int = 9,
    morph_kernel: int = 5,
    order: str = "median_morph_gaussian",
) -> np.ndarray:
    depth = np.clip(depth.astype(np.float32), 0.0, 1.0)
    if float(np.max(depth) - np.min(depth)) < 1e-6:
        return depth
    u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    if median_kernel > 1:
        if median_kernel % 2 == 0:
            median_kernel += 1
        u8 = cv2.medianBlur(u8, median_kernel)
    if gaussian_kernel % 2 == 0:
        gaussian_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)) if morph_kernel > 1 else None
    if order == "median_gaussian_morph":
        if gaussian_kernel > 1:
            u8 = cv2.GaussianBlur(u8, (gaussian_kernel, gaussian_kernel), 1.0)
        if kernel is not None:
            u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)
    else:
        if kernel is not None:
            u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)
        if gaussian_kernel > 1:
            u8 = cv2.GaussianBlur(u8, (gaussian_kernel, gaussian_kernel), 0)
    return np.clip(u8.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)


def original_full_postprocess(depth: np.ndarray) -> np.ndarray:
    """Match laplace_3dreconstruction_0.2.py: median(5) -> Gaussian(15, sigma=1) -> morphology open(15)."""
    return postprocess_depth(depth, median_kernel=5, gaussian_kernel=15, morph_kernel=15, order="median_gaussian_morph")


def edge_retention(depth: np.ndarray, reference: np.ndarray) -> float:
    dep_edge = normalize01(np.abs(cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F)))
    ref_edge = normalize01(np.abs(cv2.Laplacian(reference.astype(np.float32), cv2.CV_32F)))
    a = dep_edge.flatten()
    b = ref_edge.flatten()
    if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def roughness(depth: np.ndarray) -> float:
    gx, gy = np.gradient(depth.astype(np.float32))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def jump_strength(depth: np.ndarray, mask: np.ndarray) -> float:
    gx, gy = np.gradient(depth.astype(np.float32))
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag[mask])) if np.any(mask) else float("nan")


def count_spikes(depth: np.ndarray, mask: np.ndarray) -> int:
    local = cv2.medianBlur(normalize_u8(depth), 9).astype(np.float32) / 255.0
    diff = np.abs(normalize01(depth) - local)
    threshold = max(float(np.percentile(diff, 99.0)), 0.12)
    return int(np.sum((diff > threshold) & mask))


def save_3d_preview(depth: np.ndarray, title: str, out: Path) -> None:
    small = cv2.resize(normalize_u8(depth), (220, 180), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]
    fig = plt.figure(figsize=(6.6, 5.0), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, small.astype(np.float32), cmap="viridis", linewidth=0, antialiased=True)
    ax.view_init(elev=42, azim=-58)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("relative height")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def load_focus_resunet(device: str) -> FocusResUNet:
    model = FocusResUNet().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def build_real_features(files: list[Path]) -> dict[str, np.ndarray]:
    images = resize_stack_to_first([imread_gray(f) for f in files])
    selected = resample_stack(images, STACK_LAYERS)
    stack = np.stack([img.astype(np.float32) / 255.0 for img in selected], axis=0)
    focus = focus_maps_from_stack(stack)
    risk_layers = risk_layers_from_stack(stack)
    risk = np.clip(np.mean(risk_layers, axis=0), 0, 1).astype(np.float32)
    dff, conf = dff_depth_from_focus(focus)
    gadff, ga_conf = dff_depth_from_focus(focus, risk_layers)
    base_features = np.concatenate(
        [stack, risk[None], dff[None], conf[None], gadff[None], ga_conf[None]],
        axis=0,
    ).astype(np.float32)
    return {
        "stack": stack,
        "focus": focus,
        "risk_layers": risk_layers,
        "risk": risk,
        "dff": dff,
        "confidence": conf,
        "gadff": gadff,
        "ga_confidence": ga_conf,
        "features": base_features,
    }


def choose_roi_centers(risk: np.ndarray, conf: np.ndarray, ref: np.ndarray) -> list[tuple[str, int, int]]:
    h, w = risk.shape
    edge = normalize01(np.abs(cv2.Laplacian(ref.astype(np.float32), cv2.CV_32F)))
    candidates = [
        ("正常纹理区", (1.0 - risk) * conf * edge),
        ("眩光高亮区", risk),
        ("凹坑/沟槽边缘", edge * (1.0 - 0.5 * risk)),
        ("平坦区", (1.0 - edge) * (1.0 - risk) * conf),
        ("低纹理/低置信区", (1.0 - conf) * (1.0 - 0.4 * risk)),
    ]
    centers: list[tuple[str, int, int]] = []
    margin = max(16, min(h, w) // 12)
    for label, score in candidates:
        masked = score.copy()
        masked[:margin, :] = -1
        masked[-margin:, :] = -1
        masked[:, :margin] = -1
        masked[:, -margin:] = -1
        y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
        centers.append((label, int(y), int(x)))
    return centers


def save_focus_curve_diagnostics(sample_name: str, arrays: dict[str, np.ndarray], out_dir: Path) -> list[dict[str, object]]:
    focus = arrays["focus"]
    risk = arrays["risk"]
    conf = arrays["confidence"]
    stack = arrays["stack"]
    ref = stack[len(stack) // 2]
    centers = choose_roi_centers(risk, conf, ref)
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), dpi=150)
    axes = axes.ravel()
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title(f"{sample_name} ROI位置")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    radius = 9
    for i, (label, y, x) in enumerate(centers):
        axes[0].scatter([x], [y], s=42, c=colors[i], label=label)
        y0, y1 = max(0, y - radius), min(focus.shape[1], y + radius + 1)
        x0, x1 = max(0, x - radius), min(focus.shape[2], x + radius + 1)
        curve = np.mean(focus[:, y0:y1, x0:x1], axis=(1, 2))
        sorted_curve = np.sort(curve)
        peak = float(sorted_curve[-1])
        second = float(sorted_curve[-2]) if len(sorted_curve) > 1 else 0.0
        best_idx = int(np.argmax(curve))
        peak_gap = float((peak - second) / (peak + 1e-6))
        ax = axes[i + 1]
        ax.plot(np.arange(len(curve)), normalize01(curve), marker="o", color=colors[i])
        ax.axvline(best_idx, color=colors[i], linestyle="--", alpha=0.7)
        ax.set_title(f"{label}\npeak={best_idx}, gap={peak_gap:.3f}, risk={float(np.mean(risk[y0:y1,x0:x1])):.3f}")
        ax.set_xlabel("焦平面层号")
        ax.set_ylabel("归一化聚焦响应")
        ax.grid(alpha=0.25)
        rows.append(
            {
                "roi": label,
                "y": y,
                "x": x,
                "best_focus_index": best_idx,
                "peak_second_gap": peak_gap,
                "dff_confidence_mean": float(np.mean(conf[y0:y1, x0:x1])),
                "glare_risk_mean": float(np.mean(risk[y0:y1, x0:x1])),
                "curve_values": ";".join(f"{float(v):.6g}" for v in curve),
            }
        )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "focus_curve_diagnostic_panel.png", bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "focus_curve_diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_real_panel(sample_name: str, arrays: dict[str, np.ndarray], outputs: dict[str, np.ndarray], out: Path) -> None:
    panels = [
        ("代表帧", arrays["stack"][len(arrays["stack"]) // 2], "gray"),
        ("眩光风险 R", arrays["risk"], "magma"),
        ("原算法 raw", outputs["original_raw"], "viridis"),
        ("原算法 post", outputs["original_post"], "viridis"),
        ("模型 raw", outputs["model_raw"], "viridis"),
        ("模型 post", outputs["model_post"], "viridis"),
        ("融合权重", outputs["fusion_weight"], "magma"),
        ("保守融合", outputs["fused"], "viridis"),
        ("融合-原算法变化", outputs["fused"] - outputs["original_post"], "coolwarm"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.3, 10.2), dpi=150)
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(f"{sample_name}: 原算法、模型与保守融合无真值比较", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def evaluate_real_sample(
    sample: dict[str, object],
    model: FocusResUNet | None,
    device: str,
    invert_model_output: bool = False,
) -> dict[str, object] | None:
    files = list_images(sample["folder"])  # type: ignore[index]
    if not files:
        return None
    out_dir = OUT / "真实样本" / str(sample["key"])
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_real_features(files)
    base_features = arrays["features"]
    original_raw = arrays["dff"]
    original_post = postprocess_depth(original_raw, median_kernel=5, gaussian_kernel=9, morph_kernel=5)
    gadff_post = postprocess_depth(arrays["gadff"], median_kernel=5, gaussian_kernel=9, morph_kernel=5)

    if model is not None:
        model_features = augment_features(base_features)
        model_raw = predict_tiled_upgraded(model, model_features, device, tile=256, overlap=80)
        if invert_model_output:
            model_raw = 1.0 - model_raw
    else:
        model_raw = arrays["gadff"]
    model_post = postprocess_depth(model_raw, median_kernel=3, gaussian_kernel=7, morph_kernel=3)
    fusion_weight = np.clip(arrays["risk"] * (1.0 - arrays["confidence"]), 0, 1).astype(np.float32)
    fused = ((1.0 - fusion_weight) * original_post + fusion_weight * model_post).astype(np.float32)

    outputs = {
        "original_raw": original_raw,
        "original_post": original_post,
        "gadff_post": gadff_post,
        "model_raw": model_raw,
        "model_post": model_post,
        "fusion_weight": fusion_weight,
        "fused": fused,
    }
    for name, arr in outputs.items():
        save_float_image(out_dir / f"{name}_height.png", arr, cv2.COLORMAP_VIRIDIS if "weight" not in name else cv2.COLORMAP_MAGMA)
    imwrite(out_dir / "representative_frame.png", normalize_u8(arrays["stack"][len(arrays["stack"]) // 2]))
    save_float_image(out_dir / "glare_risk.png", arrays["risk"], cv2.COLORMAP_MAGMA)
    save_3d_preview(model_raw, f"{sample['name']} model raw", out_dir / "model_raw_3d_preview.png")
    save_3d_preview(model_post, f"{sample['name']} model post", out_dir / "model_post_3d_preview.png")
    save_3d_preview(fused, f"{sample['name']} fused", out_dir / "fused_3d_preview.png")
    save_real_panel(str(sample["name"]), arrays, outputs, out_dir / "real_sample_comparison_panel.png")
    roi_rows = save_focus_curve_diagnostics(str(sample["name"]), arrays, out_dir)

    high_risk = arrays["risk"] > max(float(np.percentile(arrays["risk"], 84)), 0.08)
    low_conf = arrays["confidence"] < min(float(np.percentile(arrays["confidence"], 35)), 0.35)
    ref = arrays["stack"][len(arrays["stack"]) // 2]
    row = {
        "sample": sample["name"],
        "frames_original": len(files),
        "frames_used": STACK_LAYERS,
        "image_shape": f"{arrays['stack'].shape[2]}x{arrays['stack'].shape[1]}",
        "risk_area_percent": float(np.mean(arrays["risk"] > 0.08) * 100),
        "low_confidence_percent": float(np.mean(low_conf) * 100),
        "original_post_roughness": roughness(original_post),
        "model_post_roughness": roughness(model_post),
        "fused_roughness": roughness(fused),
        "original_high_risk_jump": jump_strength(original_post, high_risk),
        "model_high_risk_jump": jump_strength(model_post, high_risk),
        "fused_high_risk_jump": jump_strength(fused, high_risk),
        "original_low_conf_spikes": count_spikes(original_post, low_conf),
        "model_low_conf_spikes": count_spikes(model_post, low_conf),
        "fused_low_conf_spikes": count_spikes(fused, low_conf),
        "original_edge_retention": edge_retention(original_post, ref),
        "model_edge_retention": edge_retention(model_post, ref),
        "fused_edge_retention": edge_retention(fused, ref),
        "mean_fusion_weight": float(np.mean(fusion_weight)),
        "mean_abs_fused_change": float(np.mean(np.abs(fused - original_post))),
        "judgment": judge_real_result(original_post, model_post, fused, arrays["risk"], arrays["confidence"], ref),
        "panel": str(out_dir / "real_sample_comparison_panel.png"),
        "focus_curve_panel": str(out_dir / "focus_curve_diagnostic_panel.png"),
    }
    with (out_dir / "real_sample_no_gt_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    (out_dir / "real_sample_summary.json").write_text(
        json.dumps({"metrics": row, "roi_diagnostics": roi_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return row


def judge_real_result(original: np.ndarray, model: np.ndarray, fused: np.ndarray, risk: np.ndarray, conf: np.ndarray, ref: np.ndarray) -> str:
    high = risk > max(float(np.percentile(risk, 84)), 0.08)
    low = conf < min(float(np.percentile(conf, 35)), 0.35)
    orig_jump = jump_strength(original, high)
    fused_jump = jump_strength(fused, high)
    orig_spikes = count_spikes(original, low)
    fused_spikes = count_spikes(fused, low)
    orig_edge = edge_retention(original, ref)
    fused_edge = edge_retention(fused, ref)
    if np.isfinite(orig_jump) and fused_jump < 0.92 * orig_jump and fused_spikes <= orig_spikes and fused_edge > 0.75 * orig_edge:
        return "改善：高风险区跳变降低且边缘未明显损失"
    if fused_spikes > orig_spikes * 1.15 or fused_edge < 0.65 * orig_edge:
        return "退化风险：可能引入过平滑或结构漂移"
    return "持平/不确定：无真值条件下仅能作为可视化参考"


def laplace_raw_from_stack(stack: np.ndarray, truth: np.ndarray | None = None) -> np.ndarray:
    focus = focus_maps_from_stack(stack)
    depth, _ = dff_depth_from_focus(focus)
    if truth is not None:
        flipped = 1.0 - depth
        mae = float(np.mean(np.abs(depth - truth)))
        flipped_mae = float(np.mean(np.abs(flipped - truth)))
        if flipped_mae < mae:
            return flipped.astype(np.float32)
    return depth


def edge_mae(pred: np.ndarray, truth: np.ndarray, depth_range_um: float) -> float:
    edge = normalize01(np.abs(cv2.Laplacian(truth.astype(np.float32), cv2.CV_32F)))
    mask = edge > np.percentile(edge, 88)
    return float(np.mean(np.abs(pred[mask] - truth[mask])) * depth_range_um)


def calibrate_model_direction(model: FocusResUNet | None, device: str) -> bool:
    if model is None:
        return False
    category, scenario = build_dataset()["validation"][0]
    arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
    pred = predict_tiled_upgraded(model, augment_features(arrays["features"]), device, tile=256, overlap=80)
    truth = arrays["truth"]
    assert isinstance(truth, np.ndarray)
    direct = float(np.mean(np.abs(pred - truth)))
    inverted = float(np.mean(np.abs((1.0 - pred) - truth)))
    return inverted < direct


def evaluate_simulation_baselines(model: FocusResUNet | None, device: str, invert_model_output: bool = False) -> list[dict[str, object]]:
    dataset = build_dataset()
    rows: list[dict[str, object]] = []
    out_root = OUT / "仿真公平对比"
    out_root.mkdir(parents=True, exist_ok=True)
    for category, scenario in dataset["test"]:
        sample_dir = out_root / scenario.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
        stack = arrays["stack"]
        truth = arrays["truth"]
        risk = arrays["risk"]
        dff = arrays["dff"]
        gadff = arrays["gadff"]
        features = arrays["features"]
        assert isinstance(stack, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(dff, np.ndarray)
        assert isinstance(gadff, np.ndarray)
        assert isinstance(features, np.ndarray)
        original_raw = laplace_raw_from_stack(stack, truth)
        original_post = original_full_postprocess(original_raw)
        gadff_post = postprocess_depth(gadff, median_kernel=5, gaussian_kernel=9, morph_kernel=5)
        gadff_full_post = original_full_postprocess(gadff)
        if model is not None:
            model_raw = predict_tiled_upgraded(model, augment_features(features), device, tile=256, overlap=80)
            if invert_model_output:
                model_raw = 1.0 - model_raw
        else:
            model_raw = gadff
        model_post = postprocess_depth(model_raw, median_kernel=3, gaussian_kernel=7, morph_kernel=3)
        model_full_post = original_full_postprocess(model_raw)
        _, conf = dff_depth_from_focus(focus_maps_from_stack(stack))
        fusion_weight = np.clip(risk * (1.0 - conf), 0, 1).astype(np.float32)
        fused = ((1.0 - fusion_weight) * original_post + fusion_weight * model_post).astype(np.float32)
        fused_full_post = ((1.0 - fusion_weight) * original_post + fusion_weight * model_full_post).astype(np.float32)
        outputs = {
            "truth": truth,
            "original_raw": original_raw,
            "original_post": original_post,
            "gadff": gadff,
            "gadff_post": gadff_post,
            "gadff_full_post": gadff_full_post,
            "model_raw": model_raw,
            "model_post": model_post,
            "model_full_post": model_full_post,
            "fused": fused,
            "fused_full_post": fused_full_post,
        }
        for name, arr in outputs.items():
            save_float_image(sample_dir / f"{name}.png", arr, cv2.COLORMAP_VIRIDIS)
        save_sim_panel(scenario.name, stack, risk, outputs, sample_dir / "simulation_fair_comparison_panel.png")
        method_metrics: dict[str, dict[str, float]] = {}
        for method, pred in outputs.items():
            if method == "truth":
                continue
            m = metrics(pred, truth, risk, scenario.depth_range_um)
            method_metrics[method] = {
                "mae_um": float(m["mae_um"]),
                "p90_um": float(m["p90_norm"] * scenario.depth_range_um),
                "high_risk_mae_um": float(m["high_risk_mae_um"]),
                "edge_mae_um": edge_mae(pred, truth, scenario.depth_range_um),
                "edge_retention": edge_retention(pred, truth),
            }
        row: dict[str, object] = {
            "sample": scenario.name,
            "category": category,
            "depth_range_um": scenario.depth_range_um,
            "risk_area_percent": float(np.mean(risk > 0.08) * 100),
            "best_by_mae": min(method_metrics.items(), key=lambda kv: kv[1]["mae_um"])[0],
            "panel": str(sample_dir / "simulation_fair_comparison_panel.png"),
        }
        for method, vals in method_metrics.items():
            for key, val in vals.items():
                row[f"{method}_{key}"] = val
        rows.append(row)
    with (out_root / "simulation_fair_comparison_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "simulation_fair_comparison_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def save_sim_panel(sample_name: str, stack: np.ndarray, risk: np.ndarray, outputs: dict[str, np.ndarray], out: Path) -> None:
    panels = [
        ("代表帧", stack[len(stack) // 2], "gray"),
        ("真值", outputs["truth"], "viridis"),
        ("眩光风险", risk, "magma"),
        ("原算法 raw", outputs["original_raw"], "viridis"),
        ("原算法 post", outputs["original_post"], "viridis"),
        ("GADFF + full post", outputs["gadff_full_post"], "viridis"),
        ("模型 raw", outputs["model_raw"], "viridis"),
        ("模型 + full post", outputs["model_full_post"], "viridis"),
        ("full-post 融合", outputs["fused_full_post"], "viridis"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.0), dpi=150)
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(f"{sample_name}: 仿真真值公平对比", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_report(real_rows: list[dict[str, object]], sim_rows: list[dict[str, object]], model_loaded: bool) -> None:
    out = OUT / "原算法与眩光先验融合验证报告.md"
    avg = lambda key: float(np.nanmean([float(r[key]) for r in sim_rows if key in r and str(r[key]) != "nan"]))
    lines = [
        "# 原算法、眩光先验模型与真实样本融合验证报告",
        "",
        "## 1. 结论摘要",
        "",
        "本次验证不把眩光先验模型作为中期原算法的直接替代，而是将其定位为对传统 DFF 的眩光风险提示、低置信区域校正和保守融合模块。中期原算法 `laplace_3dreconstruction_0.2.py` 被固定为工程基线：它包含自然排序、方向映射、Laplacian 聚焦评价、median/Gaussian/形态学开运算后处理和三维可视化。",
        "",
        "真实样本没有 ground truth，因此真实结果不能用 MAE 证明精度提升。本报告只给出聚焦曲线诊断、无真值稳定性指标和可视化比较；定量精度依据来自仿真样本。",
        "",
        f"模型权重加载状态：{'已加载 FocusResUNet' if model_loaded else '未加载模型，退化为 GADFF 参考输出'}。",
        "",
        "## 2. 为什么不是简单抛弃原算法",
        "",
        "原算法的优势是后处理稳定，真实样本三维表面更连续；不足是前端聚焦评价仍依赖 `argmax F_k`，当眩光或杂散光制造虚假峰值时，后处理只能平滑结果，不能判断峰值是否来自真实聚焦。眩光先验模型的意义是把 `眩光风险 R`、`DFF 深度`、`DFF 置信度`、`眩光感知 DFF` 和 `眩光感知置信度` 显式交给模型，使其在高风险低置信区域进行有约束的校正。",
        "",
        "因此推荐路线是：原算法后处理作为稳定基线，眩光先验模型作为风险区域校正，最终通过保守融合避免模型破坏可靠区域。",
        "",
        "## 3. 仿真样本公平对比",
        "",
        "仿真样本有真实高度，可以计算 MAE、P90 error、high-risk MAE 和 edge MAE。本次新增了 `original_post`，即中期原算法完整后处理版本，用于避免只拿 raw DFF 与模型比较。",
        "",
        f"- 原算法 post 平均 MAE：`{avg('original_post_mae_um'):.2f} um`。",
        f"- GADFF 平均 MAE：`{avg('gadff_mae_um'):.2f} um`。",
        f"- GADFF + 原方案完整后处理平均 MAE：`{avg('gadff_full_post_mae_um'):.2f} um`。",
        f"- 模型 raw 平均 MAE：`{avg('model_raw_mae_um'):.2f} um`。",
        f"- 模型 post 平均 MAE：`{avg('model_post_mae_um'):.2f} um`。",
        f"- 模型 + 原方案完整后处理平均 MAE：`{avg('model_full_post_mae_um'):.2f} um`。",
        f"- full-post 保守融合平均 MAE：`{avg('fused_full_post_mae_um'):.2f} um`。",
        "",
        "各样本最佳方法如下：",
        "",
        "| 样本 | 最佳方法 | 风险面积/% | 原算法post | GADFF+完整后处理 | 模型+完整后处理 | full-post融合 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sim_rows:
        lines.append(
            f"| {row['sample']} | {row['best_by_mae']} | {float(row['risk_area_percent']):.2f} | "
            f"{float(row['original_post_mae_um']):.2f} | {float(row['gadff_full_post_mae_um']):.2f} | "
            f"{float(row['model_full_post_mae_um']):.2f} | {float(row['fused_full_post_mae_um']):.2f} |"
        )
    lines += [
        "",
        "## 4. 真实样本无真值验证",
        "",
        "真实样本只做相对高度和无真值诊断。评价项包括风险区域高度跳变、低置信区域尖峰数量、边缘保留度、融合权重和融合改变量。",
        "",
        "| 样本 | 风险面积/% | 低置信面积/% | 原算法跳变 | 融合跳变 | 原算法尖峰 | 融合尖峰 | 判断 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in real_rows:
        lines.append(
            f"| {row['sample']} | {float(row['risk_area_percent']):.2f} | {float(row['low_confidence_percent']):.2f} | "
            f"{float(row['original_high_risk_jump']):.4f} | {float(row['fused_high_risk_jump']):.4f} | "
            f"{int(row['original_low_conf_spikes'])} | {int(row['fused_low_conf_spikes'])} | {row['judgment']} |"
        )
    lines += [
        "",
        "## 5. 最终汇报口径",
        "",
        "建议向导师说明：本项目中期算法不是被废弃，而是作为真实样本稳定重建基线保留。后续眩光先验模型的动机来自真实样本中高亮区域聚焦曲线不稳定，以及仿真样本中可定量验证的高风险区域误差。本次真实迁移还暴露出模型输出方向/标尺需要校准的问题，因此更不能声称真实深度精度已经提升。",
        "",
        "推荐最终表述：`原算法提供稳定的工程化 DFF 后处理，眩光先验模型提供面向高反光和低置信区域的学习型校正，真实样本采用保守融合输出；真实精度提升仍需后续标准样块或轮廓仪真值验证。`",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: FocusResUNet | None = None
    model_loaded = False
    invert_model_output = False
    if MODEL_PATH.exists():
        model = load_focus_resunet(device)
        model_loaded = True
        invert_model_output = calibrate_model_direction(model, device)
    real_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        row = evaluate_real_sample(sample, model, device, invert_model_output=invert_model_output)
        if row is not None:
            real_rows.append(row)
    if real_rows:
        with (OUT / "real_sample_no_gt_metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(real_rows[0].keys()))
            writer.writeheader()
            writer.writerows(real_rows)
        (OUT / "real_sample_no_gt_metrics_summary.json").write_text(json.dumps(real_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    sim_rows = evaluate_simulation_baselines(model, device, invert_model_output=invert_model_output)
    write_report(real_rows, sim_rows, model_loaded)
    (OUT / "run_manifest.json").write_text(
        json.dumps(
            {
                "model_path": str(MODEL_PATH),
                "model_loaded": model_loaded,
                "invert_model_output_by_validation": invert_model_output,
                "device": device,
                "real_sample_count": len(real_rows),
                "simulation_test_count": len(sim_rows),
                "output_dir": str(OUT),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote real/simulation fusion validation to: {OUT}")


if __name__ == "__main__":
    main()
