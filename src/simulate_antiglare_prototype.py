from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import font_manager

from dff_depth_direction import focus_index_to_relative_height, focus_positions_norm


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结题交付包" / "05_图表与结果" / "仿真抗眩光原型"
REAL_STACK_ROOT = ROOT / "DFFcode" / "ALL_IMAGES"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
]
for font_path in FONT_CANDIDATES:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class SimConfig:
    height: int = 64
    width: int = 64
    stack_layers: int = 9
    train_count: int = 192
    val_count: int = 48
    epochs: int = 10
    batch_size: int = 16
    seed: int = 20260518


class TinyDepthNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 24, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        b = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.out(d1))


def normalize01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def imwrite_float(path: Path, arr: np.ndarray, cmap: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(normalize01(arr) * 255, 0, 255).astype(np.uint8)
    if cmap is not None:
        u8 = cv2.applyColorMap(u8, cmap)
    ok, buf = cv2.imencode(path.suffix or ".png", u8)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    buf.tofile(str(path))


def gaussian_field(rng: np.random.Generator, h: int, w: int, sigma: float) -> np.ndarray:
    field = rng.normal(0, 1, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(field, (0, 0), sigma)


def make_depth_truth(rng: np.random.Generator, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    h, w = cfg.height, cfg.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx / max(w - 1, 1)) * 2 - 1
    y = (yy / max(h - 1, 1)) * 2 - 1

    angle = rng.uniform(0, math.pi)
    coord = x * math.cos(angle) + y * math.sin(angle)
    tool_period = rng.uniform(0.09, 0.18)
    tool_marks = 0.030 * np.sin(2 * math.pi * coord / tool_period + rng.uniform(0, 2 * math.pi))
    fine_chatter = 0.012 * np.sin(2 * math.pi * coord / rng.uniform(0.035, 0.060))

    depth = 0.50 + rng.uniform(-0.12, 0.12) * x + rng.uniform(-0.10, 0.10) * y
    depth += tool_marks + fine_chatter
    depth += 0.055 * normalize01(gaussian_field(rng, h, w, rng.uniform(5, 10))) - 0.025

    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.uniform(-0.85, 0.85, 2)
        sx, sy = rng.uniform(0.05, 0.18, 2)
        amp = rng.uniform(0.04, 0.15) * (-1 if rng.random() < 0.78 else 1)
        pit = np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        depth += amp * pit

    for _ in range(rng.integers(2, 5)):
        theta = rng.uniform(0, math.pi)
        dist = x * math.cos(theta) + y * math.sin(theta) - rng.uniform(-0.65, 0.65)
        along = -x * math.sin(theta) + y * math.cos(theta)
        width = rng.uniform(0.010, 0.035)
        length_gate = np.exp(-(along / rng.uniform(0.45, 1.2)) ** 8)
        depth -= rng.uniform(0.025, 0.075) * np.exp(-(dist / width) ** 2) * length_gate

    depth = normalize01(depth)
    depth = 0.06 + 0.88 * depth
    depth_gt_limited = np.round(depth * 64.0) / 64.0
    return depth.astype(np.float32), depth_gt_limited.astype(np.float32)


def surface_normals(depth: np.ndarray) -> np.ndarray:
    dzdy, dzdx = np.gradient(depth.astype(np.float32))
    scale = 5.0
    normals = np.dstack((-dzdx * scale, -dzdy * scale, np.ones_like(depth)))
    norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6
    return normals / norm


def render_metal_reflectance(rng: np.random.Generator, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    normals = surface_normals(depth)
    roughness = 0.20 + 0.42 * normalize01(gaussian_field(rng, h, w, rng.uniform(3, 7)))
    roughness += 0.05 * normalize01(cv2.Laplacian(depth, cv2.CV_32F))
    roughness = np.clip(roughness, 0.12, 0.75)

    light = np.array([rng.uniform(-0.65, 0.65), rng.uniform(-0.55, 0.35), rng.uniform(0.65, 1.0)], dtype=np.float32)
    light /= np.linalg.norm(light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = light + view
    half_vec /= np.linalg.norm(half_vec)

    ndotl = np.clip(np.sum(normals * light, axis=2), 0, 1)
    ndoth = np.clip(np.sum(normals * half_vec, axis=2), 0, 1)
    vdh = float(np.clip(np.dot(view, half_vec), 0, 1))
    f0 = rng.uniform(0.62, 0.86)
    fresnel = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
    shininess = 20 + 210 * (1 - roughness) ** 2
    spec = fresnel * np.power(ndoth, shininess) * (0.25 + 0.95 * ndotl)

    albedo = 0.32 + 0.24 * normalize01(gaussian_field(rng, h, w, 9))
    diffuse = albedo * (0.35 + 0.65 * ndotl)
    machining_texture = 0.10 * normalize01(cv2.Laplacian(depth, cv2.CV_32F))
    radiance = 0.10 + 0.42 * diffuse + 1.9 * spec + machining_texture
    radiance = normalize01(radiance)

    spec_norm = normalize01(spec)
    glare_seed = np.where(spec_norm > np.percentile(spec_norm, 97.7), spec_norm, 0)
    glare_bloom = cv2.GaussianBlur(glare_seed, (0, 0), rng.uniform(1.2, 2.4))
    glare_bloom = normalize01(glare_seed + 1.8 * glare_bloom)
    return radiance.astype(np.float32), glare_bloom.astype(np.float32), roughness.astype(np.float32)


def real_style_stats() -> dict[str, float]:
    candidates = [
        REAL_STACK_ROOT / "钥匙纹路100um",
        REAL_STACK_ROOT / "磕碰孔5um",
        REAL_STACK_ROOT / "3D表面",
    ]
    values: list[np.ndarray] = []
    for folder in candidates:
        if not folder.exists():
            continue
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
        for path in files[:4]:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                values.append(img.astype(np.float32) / 255.0)
    if not values:
        return {"mean": 0.48, "std": 0.18, "p995": 0.94}
    arr = np.concatenate([v.reshape(-1) for v in values])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p995": float(np.percentile(arr, 99.5)),
    }


def synthesize_focus_stack(
    rng: np.random.Generator,
    depth: np.ndarray,
    radiance: np.ndarray,
    glare: np.ndarray,
    cfg: SimConfig,
    style: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    focus_positions = focus_positions_norm(cfg.stack_layers, low=0.06, high=0.94)

    sharp = np.clip(radiance + 0.09 * normalize01(cv2.Laplacian(depth, cv2.CV_32F)), 0, 1)
    blur_soft = cv2.GaussianBlur(sharp, (0, 0), 1.25)
    blur_heavy = cv2.GaussianBlur(sharp, (0, 0), 3.0)

    prnu = 1.0 + rng.normal(0, 0.018, (h, w)).astype(np.float32)
    dsnu = rng.normal(0, 0.012, (h, 1)).astype(np.float32) + rng.normal(0, 0.010, (1, w)).astype(np.float32)
    stack = []
    risk_layers = []

    glare_phase = rng.uniform(0, 2 * math.pi)
    glare_jitter = rng.uniform(0.65, 1.35, cfg.stack_layers)
    for i, focus_z in enumerate(focus_positions):
        dist = np.abs(depth - focus_z)
        focus_weight = np.exp(-0.5 * (dist / rng.uniform(0.075, 0.105)) ** 2).astype(np.float32)
        mid_weight = np.exp(-0.5 * (dist / 0.20) ** 2).astype(np.float32)
        image = focus_weight * sharp + (1 - focus_weight) * (mid_weight * blur_soft + (1 - mid_weight) * blur_heavy)

        moving_glare = glare * glare_jitter[i] * (0.82 + 0.26 * math.sin(2 * math.pi * i / cfg.stack_layers + glare_phase))
        bloom = cv2.GaussianBlur(moving_glare, (0, 0), 1.0 + 1.2 * (1 - float(np.mean(focus_weight))))
        image = image + 0.55 * moving_glare + 0.38 * bloom

        image = (image - np.mean(image)) / (np.std(image) + 1e-6) * max(style["std"], 0.12) + style["mean"]
        image = image * prnu + dsnu
        image = np.clip(image, 0, 1)

        photons = np.maximum(image * rng.uniform(180, 340), 0)
        shot = rng.poisson(photons).astype(np.float32) / max(float(np.max(photons)), 1.0)
        read = rng.normal(0, rng.uniform(0.006, 0.014), (h, w)).astype(np.float32)
        quantized = np.clip(0.62 * image + 0.38 * shot + read, 0, 1)
        quantized = np.round(quantized * 255) / 255.0

        local_med = cv2.medianBlur((quantized * 255).astype(np.uint8), 15).astype(np.float32) / 255.0
        local_excess = quantized - local_med
        risk = ((quantized > 0.94) | ((quantized > max(0.72, style["p995"] * 0.80)) & (local_excess > 0.08))).astype(np.float32)
        risk = cv2.GaussianBlur(risk, (0, 0), 1.2)

        stack.append(quantized.astype(np.float32))
        risk_layers.append(np.clip(risk, 0, 1).astype(np.float32))

    return np.stack(stack, axis=0), np.stack(risk_layers, axis=0)


def focus_maps_from_stack(stack: np.ndarray) -> np.ndarray:
    maps = []
    for layer in stack:
        u8 = np.clip(layer * 255, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(u8, (3, 3), 0)
        lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
        ten_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        ten_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        ten = ten_x * ten_x + ten_y * ten_y
        fm = cv2.boxFilter(lap, -1, (5, 5), normalize=True) + 0.002 * cv2.boxFilter(ten, -1, (5, 5), normalize=True)
        maps.append(fm.astype(np.float32))
    return np.stack(maps, axis=0)


def dff_depth_and_confidence(stack: np.ndarray, risk_layers: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    focus = focus_maps_from_stack(stack)
    if risk_layers is not None:
        focus = focus * np.clip(1.0 - 0.72 * risk_layers, 0.18, 1.0)
    idx = np.argmax(focus, axis=0)
    depth = focus_index_to_relative_height(idx, focus.shape[0])

    sorted_focus = np.sort(focus, axis=0)
    peak = sorted_focus[-1]
    second = sorted_focus[-2] if focus.shape[0] > 1 else np.zeros_like(peak)
    confidence = (peak - second) / (peak + 1e-6)
    confidence = np.clip(confidence / (np.percentile(confidence, 98) + 1e-6), 0, 1)
    return depth.astype(np.float32), confidence.astype(np.float32)


def make_sample(rng: np.random.Generator, cfg: SimConfig, style: dict[str, float]) -> dict[str, np.ndarray]:
    depth_true, depth_limited = make_depth_truth(rng, cfg)
    radiance, glare, roughness = render_metal_reflectance(rng, depth_true)
    stack, risk_layers = synthesize_focus_stack(rng, depth_true, radiance, glare, cfg, style)
    dff, conf = dff_depth_and_confidence(stack)
    gadff, ga_conf = dff_depth_and_confidence(stack, risk_layers)
    risk_mean = np.clip(np.mean(risk_layers, axis=0), 0, 1).astype(np.float32)

    features = np.concatenate(
        [
            stack,
            risk_mean[None, :, :],
            dff[None, :, :],
            conf[None, :, :],
            gadff[None, :, :],
            ga_conf[None, :, :],
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "features": features,
        "target": depth_limited[None, :, :],
        "truth": depth_true,
        "stack": stack,
        "risk": risk_mean,
        "dff": dff,
        "gadff": gadff,
        "confidence": conf,
        "radiance": radiance,
        "glare": glare,
        "roughness": roughness,
    }


def build_dataset(cfg: SimConfig, style: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    samples = [make_sample(rng, cfg, style) for _ in range(cfg.train_count + cfg.val_count)]
    train_samples = samples[: cfg.train_count]
    val_samples = samples[cfg.train_count :]

    def pack(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            "x": np.stack([it["features"] for it in items]).astype(np.float32),
            "y": np.stack([it["target"] for it in items]).astype(np.float32),
            "truth": np.stack([it["truth"] for it in items]).astype(np.float32),
            "dff": np.stack([it["dff"] for it in items]).astype(np.float32),
            "gadff": np.stack([it["gadff"] for it in items]).astype(np.float32),
            "risk": np.stack([it["risk"] for it in items]).astype(np.float32),
            "stack": np.stack([it["stack"] for it in items]).astype(np.float32),
            "glare": np.stack([it["glare"] for it in items]).astype(np.float32),
        }

    return pack(train_samples), pack(val_samples), val_samples[0]


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def train_model(cfg: SimConfig, train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> tuple[TinyDepthNet, list[dict[str, float]], str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = TinyDepthNet(train["x"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    x_train = torch.from_numpy(train["x"])
    y_train = torch.from_numpy(train["y"])
    x_val = torch.from_numpy(val["x"]).to(device)
    y_val = torch.from_numpy(val["y"]).to(device)
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0])
        total = 0.0
        for start in range(0, x_train.shape[0], cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb) + 0.10 * edge_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_mae = torch.mean(torch.abs(val_pred - y_val)).item()
            val_rmse = torch.sqrt(torch.mean((val_pred - y_val) ** 2)).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / x_train.shape[0],
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
    return model, history, device


def metrics_for_prediction(pred: np.ndarray, target: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - target)
    high_risk = risk > max(float(np.percentile(risk, 82)), 0.10)
    low_risk = ~high_risk
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "p90_abs_error": float(np.percentile(err, 90)),
        "high_risk_mae": float(np.mean(err[high_risk])) if np.any(high_risk) else float("nan"),
        "low_risk_mae": float(np.mean(err[low_risk])) if np.any(low_risk) else float("nan"),
    }


def evaluate_model(model: TinyDepthNet, device: str, val: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(val["x"]).to(device)).detach().cpu().numpy()[:, 0]
    target = val["y"][:, 0]
    metrics = {
        "DFF_baseline": metrics_for_prediction(val["dff"], target, val["risk"]),
        "GA_DFF_heuristic": metrics_for_prediction(val["gadff"], target, val["risk"]),
        "TinyCNN_synthetic": metrics_for_prediction(pred, target, val["risk"]),
    }
    return metrics, pred


def save_training_curve(history: list[dict[str, float]]) -> None:
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(7.2, 4.2), dpi=160)
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    plt.plot(epochs, [h["val_mae"] for h in history], marker="s", label="val MAE")
    plt.plot(epochs, [h["val_rmse"] for h in history], marker="^", label="val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized depth error")
    plt.title("轻量抗眩光深度恢复网络训练曲线")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "training_curve.png")
    plt.close()


def save_comparison_panel(val: dict[str, np.ndarray], pred: np.ndarray, cfg: SimConfig) -> int:
    target = val["y"][:, 0]
    dff_err = np.mean(np.abs(val["dff"] - target), axis=(1, 2))
    cnn_err = np.mean(np.abs(pred - target), axis=(1, 2))
    risk_score = np.mean(val["risk"], axis=(1, 2))
    score = dff_err - cnn_err + 0.3 * risk_score
    idx = int(np.argmax(score))

    truth = target[idx]
    dff = val["dff"][idx]
    gadff = val["gadff"][idx]
    cnn = pred[idx]
    risk = val["risk"][idx]
    stack = val["stack"][idx]
    glare = val["glare"][idx]
    mid = cfg.stack_layers // 2

    panels = [
        ("金属深度真值", truth, "viridis"),
        ("代表成像帧", stack[mid], "gray"),
        ("眩光风险图", risk, "magma"),
        ("DFF 基线", dff, "viridis"),
        ("眩光降权 DFF", gadff, "viridis"),
        ("CNN 恢复", cnn, "viridis"),
        ("DFF 误差", np.abs(dff - truth), "inferno"),
        ("CNN 误差", np.abs(cnn - truth), "inferno"),
        ("镜面高光真值", glare, "magma"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.0), dpi=170)
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("仿真金属表面抗眩光 DFF 深度恢复闭环验证", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_vs_model_panel.png")
    plt.close(fig)

    imwrite_float(OUT / "sample_truth_depth.png", truth, cv2.COLORMAP_VIRIDIS)
    imwrite_float(OUT / "sample_glare_risk.png", risk, cv2.COLORMAP_MAGMA)
    imwrite_float(OUT / "sample_dff_error.png", np.abs(dff - truth), cv2.COLORMAP_INFERNO)
    imwrite_float(OUT / "sample_cnn_error.png", np.abs(cnn - truth), cv2.COLORMAP_INFERNO)
    return idx


def write_metrics(metrics: dict[str, dict[str, float]], history: list[dict[str, float]], elapsed: float, device: str, cfg: SimConfig, style: dict[str, float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "simulation_antiglare_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mae", "rmse", "p90_abs_error", "high_risk_mae", "low_risk_mae"])
        for method, vals in metrics.items():
            writer.writerow([method, vals["mae"], vals["rmse"], vals["p90_abs_error"], vals["high_risk_mae"], vals["low_risk_mae"]])
    payload = {
        "config": cfg.__dict__,
        "device": device,
        "elapsed_seconds": elapsed,
        "real_style_stats": style,
        "metrics": metrics,
        "history": history,
    }
    (OUT / "simulation_antiglare_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(metrics: dict[str, dict[str, float]], elapsed: float, device: str, cfg: SimConfig) -> None:
    dff = metrics["DFF_baseline"]["mae"]
    gadff = metrics["GA_DFF_heuristic"]["mae"]
    cnn = metrics["TinyCNN_synthetic"]["mae"]
    gain_vs_dff = (dff - cnn) / max(dff, 1e-6) * 100
    gain_vs_gadff = (gadff - cnn) / max(gadff, 1e-6) * 100

    lines = [
        "# 仿真抗眩光原型执行结果",
        "",
        "该结果来自轻量 Python 近似物理仿真，不是严格光线追踪。它用于验证 5 小时内能否跑通“深度真值生成-金属反射/眩光成像-DFF 基线-抗眩光网络训练-误差评估”的概念闭环。",
        "",
        f"- 训练设备：`{device}`",
        f"- 总耗时：`{elapsed:.1f} s`",
        f"- 训练样本：`{cfg.train_count}`，验证样本：`{cfg.val_count}`",
        f"- 输入：`{cfg.stack_layers}` 层模拟焦距堆栈 + 眩光风险图 + DFF/置信度先验",
        f"- 输出：归一化相对深度图，训练真值按 1/64 深度范围量化，避免给网络不现实的无限精度真值。",
        "",
        "| 方法 | MAE | RMSE | P90误差 | 高眩光区MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, vals in metrics.items():
        lines.append(
            f"| {name} | {vals['mae']:.4f} | {vals['rmse']:.4f} | {vals['p90_abs_error']:.4f} | {vals['high_risk_mae']:.4f} |"
        )
    lines += [
        "",
        f"相对 DFF 基线，TinyCNN 在该合成验证集上的 MAE 改善约 `{gain_vs_dff:.1f}%`；相对眩光降权启发式 DFF 改善约 `{gain_vs_gadff:.1f}%`。",
        "",
        "结论：5 小时内可以完成轻量闭环和可视化证据，但不能声称完成真实工业泛化模型。严格版本仍需要 BlenderProc/Mitsuba 等物理渲染器、标定相机参数、真实样品域适配和独立测试集。",
    ]
    (OUT / "仿真抗眩光原型执行结果.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    cfg = SimConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    style = real_style_stats()
    train, val, _ = build_dataset(cfg, style)
    model, history, device = train_model(cfg, train, val)
    metrics, pred = evaluate_model(model, device, val)
    elapsed = time.time() - start
    save_training_curve(history)
    save_comparison_panel(val, pred, cfg)
    write_metrics(metrics, history, elapsed, device, cfg, style)
    write_summary(metrics, elapsed, device, cfg)
    torch.save(model.state_dict(), OUT / "tiny_antiglare_depth_net.pt")
    print(f"Wrote simulation anti-glare prototype outputs to: {OUT}")


if __name__ == "__main__":
    main()
