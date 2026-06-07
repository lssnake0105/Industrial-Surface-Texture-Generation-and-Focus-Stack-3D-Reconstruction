from __future__ import annotations

import csv
import json
import math
import os
import time
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

from final_dataset_training import build_dataset
from simulate_antiglare_highres_samples import (
    DEFAULT_STACK_LAYERS,
    Scenario,
    generate_sample_arrays,
    metrics,
    save_3d_surface_preview,
    save_sample_panel,
    tile_positions,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结题交付包" / "05_图表与结果" / "模型与损失函数升级实验"
MODEL_DIR = OUT / "model"
STACK_LAYERS = DEFAULT_STACK_LAYERS
SEED = 20260519
PATCH_SIZE = 128
BATCH_SIZE = 6
TRAIN_PATCHES_PER_EPOCH = 384
VAL_PATCHES_PER_EPOCH = 96
EPOCHS = 12


def augment_features(features: np.ndarray) -> np.ndarray:
    stack = features[:STACK_LAYERS]
    priors = features[STACK_LAYERS:]
    diffs = np.diff(stack, axis=0)
    return np.concatenate([stack, diffs, priors], axis=0).astype(np.float32)


def upgraded_channel_count() -> int:
    return STACK_LAYERS + (STACK_LAYERS - 1) + 5


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.silu(out + self.skip(x))


class FocusResUNet(nn.Module):
    """Lightweight focus-stack model with explicit focal-difference channels."""

    def __init__(self, stack_layers: int = STACK_LAYERS, base: int = 24) -> None:
        super().__init__()
        self.stack_layers = stack_layers
        self.diff_layers = stack_layers - 1
        self.stack_stem = nn.Sequential(nn.Conv2d(stack_layers, base, 3, padding=1), nn.GroupNorm(6, base), nn.SiLU())
        self.diff_stem = nn.Sequential(nn.Conv2d(self.diff_layers, base, 3, padding=1), nn.GroupNorm(6, base), nn.SiLU())
        self.prior_stem = nn.Sequential(nn.Conv2d(5, base // 2, 3, padding=1), nn.GroupNorm(6, base // 2), nn.SiLU())
        self.fuse = ResidualBlock(base * 2 + base // 2, base)

        self.enc1 = ResidualBlock(base, base)
        self.enc2 = ResidualBlock(base, base * 2)
        self.enc3 = ResidualBlock(base * 2, base * 3)
        self.bottleneck = ResidualBlock(base * 3, base * 4)
        self.dec3 = ResidualBlock(base * 4 + base * 3, base * 3)
        self.dec2 = ResidualBlock(base * 3 + base * 2, base * 2)
        self.dec1 = ResidualBlock(base * 2 + base, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack = x[:, : self.stack_layers]
        diffs = x[:, self.stack_layers : self.stack_layers + self.diff_layers]
        priors = x[:, self.stack_layers + self.diff_layers :]
        x0 = self.fuse(torch.cat([self.stack_stem(stack), self.diff_stem(diffs), self.prior_stem(priors)], dim=1))
        e1 = self.enc1(x0)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def laplacian(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def normal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pdx, pdy = grad_xy(pred)
    tdx, tdy = grad_xy(target)
    h = min(pdx.shape[2], pdy.shape[2], tdx.shape[2], tdy.shape[2])
    w = min(pdx.shape[3], pdy.shape[3], tdx.shape[3], tdy.shape[3])
    pdx, pdy = pdx[:, :, :h, :w], pdy[:, :, :h, :w]
    tdx, tdy = tdx[:, :, :h, :w], tdy[:, :, :h, :w]
    pn = torch.cat([-pdx, -pdy, torch.ones_like(pdx)], dim=1)
    tn = torch.cat([-tdx, -tdy, torch.ones_like(tdx)], dim=1)
    pn = F.normalize(pn, dim=1)
    tn = F.normalize(tn, dim=1)
    return torch.mean(1.0 - torch.sum(pn * tn, dim=1, keepdim=True))


class HybridDFFLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        prior_offset = STACK_LAYERS + (STACK_LAYERS - 1)
        risk = features[:, prior_offset + 0 : prior_offset + 1]
        dff = features[:, prior_offset + 1 : prior_offset + 2]
        conf = features[:, prior_offset + 2 : prior_offset + 3]
        gadff = features[:, prior_offset + 3 : prior_offset + 4]
        ga_conf = features[:, prior_offset + 4 : prior_offset + 5]

        glare_weight = 1.0 + 0.80 * torch.clamp(risk, 0, 1)
        data = torch.mean(glare_weight * charbonnier(pred - target))

        pdx, pdy = grad_xy(pred)
        tdx, tdy = grad_xy(target)
        grad = torch.mean(charbonnier(pdx - tdx)) + torch.mean(charbonnier(pdy - tdy))
        curv = torch.mean(charbonnier(laplacian(pred) - laplacian(target)))
        nrm = normal_loss(pred, target)

        prior_weight = torch.clamp((0.65 * conf + 0.35 * ga_conf) * (1.0 - risk).pow(1.5), 0, 1)
        prior_target = 0.45 * dff + 0.55 * gadff
        prior = torch.sum(prior_weight * charbonnier(pred - prior_target)) / torch.clamp(torch.sum(prior_weight), min=1.0)

        loss = data + 0.22 * grad + 0.055 * curv + 0.035 * nrm + 0.045 * prior
        parts = {
            "data": float(data.detach().cpu()),
            "gradient": float(grad.detach().cpu()),
            "curvature": float(curv.detach().cpu()),
            "normal": float(nrm.detach().cpu()),
            "focus_prior": float(prior.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
        return loss, parts


def prepare_samples(items: list[tuple[str, Scenario]]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for category, scenario in items:
        arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
        base = arrays["features"]
        assert isinstance(base, np.ndarray)
        arrays["model_features"] = augment_features(base)
        arrays["category"] = category
        samples.append(arrays)
    return samples


def random_patch_batch(
    samples: list[dict[str, object]],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _ in range(batch_size):
        item = samples[int(rng.integers(0, len(samples)))]
        features = item["model_features"]
        truth = item["truth"]
        assert isinstance(features, np.ndarray)
        assert isinstance(truth, np.ndarray)
        _, h, w = features.shape
        y0 = int(rng.integers(0, h - PATCH_SIZE + 1))
        x0 = int(rng.integers(0, w - PATCH_SIZE + 1))
        x = features[:, y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
        y = truth[None, y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
        if rng.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        if rng.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def train_model(train_samples: list[dict[str, object]], val_samples: list[dict[str, object]], device: str) -> tuple[FocusResUNet, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = FocusResUNet().to(device)
    criterion = HybridDFFLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1.2e-4)
    train_steps = math.ceil(TRAIN_PATCHES_PER_EPOCH / BATCH_SIZE)
    val_steps = math.ceil(VAL_PATCHES_PER_EPOCH / BATCH_SIZE)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_parts: dict[str, float] = {"data": 0.0, "gradient": 0.0, "curvature": 0.0, "normal": 0.0, "focus_prior": 0.0}
        for _ in range(train_steps):
            xb, yb = random_patch_batch(train_samples, rng, BATCH_SIZE)
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss, parts = criterion(pred, yb, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += parts["total"]
            for key in train_parts:
                train_parts[key] += parts[key]
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = random_patch_batch(val_samples, rng, BATCH_SIZE)
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss, parts = criterion(pred, yb, xb)
                val_loss += parts["total"]
                val_mae += float(torch.mean(torch.abs(pred - yb)).detach().cpu())
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / train_steps,
            "val_loss": val_loss / val_steps,
            "val_mae_norm": val_mae / val_steps,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        for key, value in train_parts.items():
            row[f"train_{key}"] = value / train_steps
        history.append(row)
        print(
            f"epoch {epoch:02d}/{EPOCHS}: train={row['train_loss']:.5f}, "
            f"val={row['val_loss']:.5f}, val_mae={row['val_mae_norm']:.5f}",
            flush=True,
        )
        if row["val_mae_norm"] < best_val:
            best_val = row["val_mae_norm"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_tiled_upgraded(model: nn.Module, features: np.ndarray, device: str, tile: int = 256, overlap: int = 80) -> np.ndarray:
    c, h, w = features.shape
    pred_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    win_1d = np.hanning(tile).astype(np.float32)
    win_1d = np.maximum(win_1d, 0.12)
    window = np.outer(win_1d, win_1d).astype(np.float32)
    for y in tile_positions(h, tile, overlap):
        for x in tile_positions(w, tile, overlap):
            tile_arr = features[:, y : y + tile, x : x + tile]
            pad_h = tile - tile_arr.shape[1]
            pad_w = tile - tile_arr.shape[2]
            if pad_h or pad_w:
                tile_arr = np.pad(tile_arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            with torch.no_grad():
                out = model(torch.from_numpy(tile_arr[None]).to(device)).detach().cpu().numpy()[0, 0]
            out = out[: min(tile, h - y), : min(tile, w - x)]
            ww = window[: out.shape[0], : out.shape[1]]
            pred_sum[y : y + out.shape[0], x : x + out.shape[1]] += out * ww
            weight_sum[y : y + out.shape[0], x : x + out.shape[1]] += ww
    return np.clip(pred_sum / np.maximum(weight_sum, 1e-6), 0, 1).astype(np.float32)


def write_history(history: list[dict[str, float]]) -> None:
    with (OUT / "focus_resunet_training_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plt.figure(figsize=(7.6, 4.6), dpi=150)
    epochs = [int(h["epoch"]) for h in history]
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train hybrid loss")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation hybrid loss")
    plt.plot(epochs, [h["val_mae_norm"] for h in history], marker="^", label="validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / normalized MAE")
    plt.title("Focus-ResUNet hybrid-loss training curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "focus_resunet_training_curve.png")
    plt.close()


def load_tiny_baseline() -> dict[str, dict[str, float]]:
    path = ROOT / "结题交付包" / "05_图表与结果" / "最终仿真数据集训练验证" / "final_metrics.csv"
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["sample"]] = {
                "tiny_model_mae_um": float(row["model_mae_um"]),
                "tiny_gain_percent": float(row["model_vs_dff_gain_percent"]),
            }
    return rows


def evaluate_split(
    split: str,
    items: list[tuple[str, Scenario]],
    model: nn.Module,
    device: str,
    tiny_baseline: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, scenario in items:
        print(f"evaluate {split}: {scenario.name}", flush=True)
        sample_dir = OUT / split / scenario.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
        base_features = arrays["features"]
        truth = arrays["truth"]
        stack = arrays["stack"]
        risk = arrays["risk"]
        dff = arrays["dff"]
        gadff = arrays["gadff"]
        camera = arrays["camera"]
        assert isinstance(base_features, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(stack, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(dff, np.ndarray)
        assert isinstance(gadff, np.ndarray)
        pred = predict_tiled_upgraded(model, augment_features(base_features), device)
        save_sample_panel(sample_dir, scenario, camera, truth, stack, risk, dff, gadff, None, pred)
        save_3d_surface_preview(sample_dir, scenario, camera, truth, pred)
        dff_m = metrics(dff, truth, risk, scenario.depth_range_um)
        ga_m = metrics(gadff, truth, risk, scenario.depth_range_um)
        up_m = metrics(pred, truth, risk, scenario.depth_range_um)
        tiny = tiny_baseline.get(scenario.name, {"tiny_model_mae_um": float("nan"), "tiny_gain_percent": float("nan")})
        rows.append(
            {
                "split": split,
                "category": category,
                "sample": scenario.name,
                "resolution": f"{scenario.width}x{scenario.height}",
                "depth_range_um": scenario.depth_range_um,
                "z_step_um": scenario.depth_range_um / max(STACK_LAYERS - 1, 1),
                "dff_mae_um": dff_m["mae_um"],
                "ga_dff_mae_um": ga_m["mae_um"],
                "tiny_model_mae_um": tiny["tiny_model_mae_um"],
                "focus_resunet_mae_um": up_m["mae_um"],
                "focus_resunet_edge_mae_um": up_m["edge_mae_um"],
                "focus_resunet_high_risk_mae_um": up_m["high_risk_mae_um"],
                "focus_resunet_vs_dff_gain_percent": (dff_m["mae_um"] - up_m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100,
                "focus_resunet_vs_tiny_gain_percent": (tiny["tiny_model_mae_um"] - up_m["mae_um"]) / max(tiny["tiny_model_mae_um"], 1e-6) * 100,
                "panel": str(sample_dir / "00_highres_comparison_panel.png"),
                "surface_3d": str(sample_dir / "11_3d_surface_preview.png"),
            }
        )
    return rows


def write_metric_plots(rows: list[dict[str, object]]) -> None:
    test = [r for r in rows if r["split"] == "test"]
    labels = [str(r["category"]).replace("-", "\n") for r in test]
    x = np.arange(len(test))
    width = 0.22
    plt.figure(figsize=(12.4, 5.4), dpi=150)
    plt.bar(x - 1.5 * width, [float(r["dff_mae_um"]) for r in test], width, label="DFF")
    plt.bar(x - 0.5 * width, [float(r["ga_dff_mae_um"]) for r in test], width, label="Glare-aware DFF")
    plt.bar(x + 0.5 * width, [float(r["tiny_model_mae_um"]) for r in test], width, label="TinyDepthNet")
    plt.bar(x + 1.5 * width, [float(r["focus_resunet_mae_um"]) for r in test], width, label="Focus-ResUNet")
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("MAE / um")
    plt.title("Model and loss upgrade: test-set MAE")
    plt.grid(axis="y", alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "focus_resunet_test_metrics_bar.png")
    plt.close()

    plt.figure(figsize=(10.8, 4.8), dpi=150)
    gains = [float(r["focus_resunet_vs_tiny_gain_percent"]) for r in test]
    colors = ["#2a9d8f" if g >= 0 else "#e76f51" for g in gains]
    plt.bar(x, gains, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("Gain vs TinyDepthNet / %")
    plt.title("Focus-ResUNet relative gain over previous model")
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(OUT / "focus_resunet_vs_tiny_gain.png")
    plt.close()


def split_mean(rows: list[dict[str, object]], split: str, key: str) -> float:
    vals = [float(r[key]) for r in rows if r["split"] == split and not math.isnan(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def write_report(rows: list[dict[str, object]], history: list[dict[str, float]], elapsed_s: float) -> None:
    test = [r for r in rows if r["split"] == "test"]
    p10 = next(r for r in test if "P10" in str(r["sample"]))
    lines = [
        "# 模型与损失函数升级实验报告",
        "",
        "## 调研结论",
        "",
        "- DDFFNet 说明 DFF 可以端到端从焦栈中学习深度，而不是只依赖手工清晰度算子。",
        "- CVPR 2022 的 Differential Focus Volume 思路强调沿焦平面维度的一阶差分特征，有助于捕捉焦点变化和上下文信息。",
        "- 近期 DualFocus 将 DFF 约束拆成空间约束和焦平面约束，说明 DFF 模型应同时保持空间边缘与焦栈物理一致性。",
        "",
        "## 模型设计",
        "",
        "本轮将 TinyDepthNet 替换为 Focus-ResUNet。输入由三部分组成：17 层焦栈、16 层相邻焦平面差分、5 个 DFF/眩光先验通道。网络采用轻量残差 U-Net 结构，分别对原始焦栈、焦平面差分和先验通道进行 stem 编码，再通过多尺度残差编码器-解码器恢复高度图。",
        "",
        "## 损失函数设计",
        "",
        "总损失为：L = L_data + 0.22 L_grad + 0.055 L_lap + 0.035 L_normal + 0.045 L_prior。",
        "",
        "- L_data：眩光风险加权 Charbonnier 深度误差，强眩光区域权重更高。",
        "- L_grad：一阶梯度一致性，减少高度边缘被过度平滑。",
        "- L_lap：Laplacian 曲率一致性，约束细微凹坑、刃脊和周期纹理的局部形状。",
        "- L_normal：表面法向一致性，鼓励三维预览中的坡面方向更稳定。",
        "- L_prior：高置信、低眩光区域与 DFF/GADFF 先验保持一致，避免模型在可靠区域无意义偏移。",
        "",
        "## 测试集结果",
        "",
        "| 样品 | DFF MAE/um | TinyDepthNet MAE/um | Focus-ResUNet MAE/um | 相对 Tiny 改善/% |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in test:
        lines.append(
            f"| {r['category']} | {float(r['dff_mae_um']):.2f} | {float(r['tiny_model_mae_um']):.2f} | "
            f"{float(r['focus_resunet_mae_um']):.2f} | {float(r['focus_resunet_vs_tiny_gain_percent']):.2f} |"
        )
    lines += [
        "",
        "## 平均指标",
        "",
        f"- 测试集 DFF 平均 MAE：{split_mean(rows, 'test', 'dff_mae_um'):.2f} um。",
        f"- 测试集 TinyDepthNet 平均 MAE：{split_mean(rows, 'test', 'tiny_model_mae_um'):.2f} um。",
        f"- 测试集 Focus-ResUNet 平均 MAE：{split_mean(rows, 'test', 'focus_resunet_mae_um'):.2f} um。",
        f"- Focus-ResUNet 相对 TinyDepthNet 平均改善：{split_mean(rows, 'test', 'focus_resunet_vs_tiny_gain_percent'):.2f}%。",
        "",
        "## P10 结果",
        "",
        f"- P10 上 DFF MAE：{float(p10['dff_mae_um']):.2f} um。",
        f"- P10 上 TinyDepthNet MAE：{float(p10['tiny_model_mae_um']):.2f} um。",
        f"- P10 上 Focus-ResUNet MAE：{float(p10['focus_resunet_mae_um']):.2f} um。",
        f"- P10 相对 TinyDepthNet 改善：{float(p10['focus_resunet_vs_tiny_gain_percent']):.2f}%。",
        "",
        "## 结论",
        "",
        "若 Focus-ResUNet 的测试集平均指标优于 TinyDepthNet，可在结题材料中将其作为“后续算法探索的升级版本”；若只有部分样品改善，则应表述为“损失函数和焦栈差分特征对复杂纹理有效，但仍需扩大训练集和真实标定数据”。",
        "",
        f"训练耗时约 {elapsed_s / 60:.1f} min；最佳模型按验证 MAE 选取；最后一轮验证 MAE 为 {history[-1]['val_mae_norm']:.4f}。",
        "",
        "## 参考来源",
        "",
        "- Hazirbas et al., Deep Depth From Focus, arXiv:1704.01085, https://arxiv.org/abs/1704.01085",
        "- Yang et al., Deep Depth From Focus With Differential Focus Volume, CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Deep_Depth_From_Focus_With_Differential_Focus_Volume_CVPR_2022_paper.html",
        "- Woo and Lee, DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints, arXiv:2509.21992, https://arxiv.org/abs/2509.21992",
    ]
    (OUT / "模型与损失函数升级实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    print("Preparing samples...", flush=True)
    train_samples = prepare_samples(dataset["train"])
    val_samples = prepare_samples(dataset["validation"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Focus-ResUNet on {device}, channels={upgraded_channel_count()}", flush=True)
    model, history = train_model(train_samples, val_samples, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stack_layers": STACK_LAYERS,
            "input_channels": upgraded_channel_count(),
            "history": history,
            "loss": "weighted Charbonnier + gradient + laplacian + normal + focus-prior consistency",
            "seed": SEED,
        },
        MODEL_DIR / "focus_resunet_hybrid_loss.pt",
    )
    write_history(history)
    tiny_baseline = load_tiny_baseline()
    rows: list[dict[str, object]] = []
    for split, items in dataset.items():
        rows.extend(evaluate_split(split, items, model, device, tiny_baseline))
    with (OUT / "focus_resunet_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "focus_resunet_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metric_plots(rows)
    write_report(rows, history, time.time() - start)
    print(f"Wrote upgraded model experiment to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
