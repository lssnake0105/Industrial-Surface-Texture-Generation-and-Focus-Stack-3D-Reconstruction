from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
from train_focus_resunet_loss_experiment import (
    HybridDFFLoss,
    ResidualBlock,
    augment_features,
    load_tiny_baseline,
    normal_loss,
    upgraded_channel_count,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结题交付包" / "05_图表与结果" / "模型与损失函数升级实验_残差保护版"
MODEL_DIR = OUT / "model"
STACK_LAYERS = DEFAULT_STACK_LAYERS
SEED = 20260519 + 17
PATCH_SIZE = 128
BATCH_SIZE = 6
TRAIN_PATCHES_PER_EPOCH = 384
VAL_PATCHES_PER_EPOCH = 96
EPOCHS = 10


class ResidualFocusResUNet(nn.Module):
    """Predicts a bounded residual over a DFF/GADFF prior instead of absolute depth."""

    def __init__(self, stack_layers: int = STACK_LAYERS, base: int = 24, residual_scale: float = 0.18) -> None:
        super().__init__()
        self.stack_layers = stack_layers
        self.diff_layers = stack_layers - 1
        self.residual_scale = residual_scale
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
        self.residual_head = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.SiLU(), nn.Conv2d(base, 1, 1))
        self.gate_head = nn.Sequential(nn.Conv2d(base, base // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(base // 2, 1, 1), nn.Sigmoid())

    def prior_from_features(self, x: torch.Tensor) -> torch.Tensor:
        prior_offset = self.stack_layers + self.diff_layers
        risk = x[:, prior_offset + 0 : prior_offset + 1]
        dff = x[:, prior_offset + 1 : prior_offset + 2]
        conf = x[:, prior_offset + 2 : prior_offset + 3]
        gadff = x[:, prior_offset + 3 : prior_offset + 4]
        ga_conf = x[:, prior_offset + 4 : prior_offset + 5]
        ga_weight = torch.clamp(ga_conf * (1.0 - risk), 0.0, 1.0)
        dff_weight = torch.clamp(conf * (1.0 - 0.35 * risk), 0.0, 1.0)
        total = dff_weight + ga_weight + 1e-4
        return torch.clamp((dff * dff_weight + gadff * ga_weight) / total, 0.0, 1.0)

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
        prior = self.prior_from_features(x)
        residual = self.residual_scale * torch.tanh(self.residual_head(d1))
        gate = self.gate_head(d1)
        return torch.clamp(prior + gate * residual, 0.0, 1.0)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def random_patch_batch(samples: list[dict[str, object]], rng: np.random.Generator, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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


class ResidualHybridLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hybrid = HybridDFFLoss()

    def forward(self, model: ResidualFocusResUNet, pred: torch.Tensor, target: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        hybrid, parts = self.hybrid(pred, target, features)
        prior = model.prior_from_features(features)
        prior_offset = STACK_LAYERS + (STACK_LAYERS - 1)
        risk = features[:, prior_offset + 0 : prior_offset + 1]
        conf = features[:, prior_offset + 2 : prior_offset + 3]
        ga_conf = features[:, prior_offset + 4 : prior_offset + 5]
        safe = torch.clamp((0.5 * conf + 0.5 * ga_conf) * (1.0 - risk).pow(1.3), 0.0, 1.0)
        residual_guard = torch.sum(safe * charbonnier(pred - prior)) / torch.clamp(torch.sum(safe), min=1.0)
        nrm = normal_loss(pred, target)
        loss = hybrid + 0.10 * residual_guard + 0.02 * nrm
        parts["residual_guard"] = float(residual_guard.detach().cpu())
        parts["normal_extra"] = float(nrm.detach().cpu())
        parts["total"] = float(loss.detach().cpu())
        return loss, parts


def train_model(train_samples: list[dict[str, object]], val_samples: list[dict[str, object]], device: str) -> tuple[ResidualFocusResUNet, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = ResidualFocusResUNet().to(device)
    criterion = ResidualHybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1.2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1.2e-4)
    train_steps = math.ceil(TRAIN_PATCHES_PER_EPOCH / BATCH_SIZE)
    val_steps = math.ceil(VAL_PATCHES_PER_EPOCH / BATCH_SIZE)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for _ in range(train_steps):
            xb, yb = random_patch_batch(train_samples, rng, BATCH_SIZE)
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss, parts = criterion(model, pred, yb, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += parts["total"]
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
                loss, parts = criterion(model, pred, yb, xb)
                val_loss += parts["total"]
                val_mae += float(torch.mean(torch.abs(pred - yb)).detach().cpu())
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / train_steps,
            "val_loss": val_loss / val_steps,
            "val_mae_norm": val_mae / val_steps,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(f"epoch {epoch:02d}/{EPOCHS}: train={row['train_loss']:.5f}, val={row['val_loss']:.5f}, val_mae={row['val_mae_norm']:.5f}", flush=True)
        if row["val_mae_norm"] < best_val:
            best_val = row["val_mae_norm"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_tiled(model: nn.Module, features: np.ndarray, device: str, tile: int = 256, overlap: int = 80) -> np.ndarray:
    _, h, w = features.shape
    pred_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    win_1d = np.maximum(np.hanning(tile).astype(np.float32), 0.12)
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
    with (OUT / "residual_focus_resunet_training_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plt.figure(figsize=(7.4, 4.4), dpi=150)
    epochs = [int(h["epoch"]) for h in history]
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation")
    plt.plot(epochs, [h["val_mae_norm"] for h in history], marker="^", label="validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.title("Residual Focus-ResUNet training curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "residual_focus_resunet_training_curve.png")
    plt.close()


def evaluate_split(split: str, items: list[tuple[str, Scenario]], model: nn.Module, device: str, tiny_baseline: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, scenario in items:
        print(f"evaluate {split}: {scenario.name}", flush=True)
        sample_dir = OUT / split / scenario.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
        base = arrays["features"]
        truth = arrays["truth"]
        stack = arrays["stack"]
        risk = arrays["risk"]
        dff = arrays["dff"]
        gadff = arrays["gadff"]
        camera = arrays["camera"]
        assert isinstance(base, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(stack, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(dff, np.ndarray)
        assert isinstance(gadff, np.ndarray)
        pred = predict_tiled(model, augment_features(base), device)
        save_sample_panel(sample_dir, scenario, camera, truth, stack, risk, dff, gadff, None, pred)
        save_3d_surface_preview(sample_dir, scenario, camera, truth, pred)
        dff_m = metrics(dff, truth, risk, scenario.depth_range_um)
        ga_m = metrics(gadff, truth, risk, scenario.depth_range_um)
        m = metrics(pred, truth, risk, scenario.depth_range_um)
        tiny = tiny_baseline.get(scenario.name, {"tiny_model_mae_um": float("nan")})
        rows.append(
            {
                "split": split,
                "category": category,
                "sample": scenario.name,
                "resolution": f"{scenario.width}x{scenario.height}",
                "dff_mae_um": dff_m["mae_um"],
                "ga_dff_mae_um": ga_m["mae_um"],
                "tiny_model_mae_um": tiny["tiny_model_mae_um"],
                "residual_focus_resunet_mae_um": m["mae_um"],
                "residual_focus_resunet_edge_mae_um": m["edge_mae_um"],
                "residual_focus_resunet_high_risk_mae_um": m["high_risk_mae_um"],
                "residual_vs_dff_gain_percent": (dff_m["mae_um"] - m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100,
                "residual_vs_tiny_gain_percent": (tiny["tiny_model_mae_um"] - m["mae_um"]) / max(tiny["tiny_model_mae_um"], 1e-6) * 100,
                "panel": str(sample_dir / "00_highres_comparison_panel.png"),
                "surface_3d": str(sample_dir / "11_3d_surface_preview.png"),
            }
        )
    return rows


def split_mean(rows: list[dict[str, object]], split: str, key: str) -> float:
    vals = [float(r[key]) for r in rows if r["split"] == split and not math.isnan(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def write_plots(rows: list[dict[str, object]]) -> None:
    test = [r for r in rows if r["split"] == "test"]
    labels = [str(r["category"]).replace("-", "\n") for r in test]
    x = np.arange(len(test))
    width = 0.24
    plt.figure(figsize=(12.4, 5.4), dpi=150)
    plt.bar(x - width, [float(r["dff_mae_um"]) for r in test], width, label="DFF")
    plt.bar(x, [float(r["tiny_model_mae_um"]) for r in test], width, label="TinyDepthNet")
    plt.bar(x + width, [float(r["residual_focus_resunet_mae_um"]) for r in test], width, label="Residual Focus-ResUNet")
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("MAE / um")
    plt.title("Residual protected model: test-set MAE")
    plt.grid(axis="y", alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "residual_focus_resunet_test_metrics_bar.png")
    plt.close()

    gains = [float(r["residual_vs_tiny_gain_percent"]) for r in test]
    colors = ["#2a9d8f" if g >= 0 else "#e76f51" for g in gains]
    plt.figure(figsize=(10.8, 4.8), dpi=150)
    plt.bar(x, gains, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("Gain vs TinyDepthNet / %")
    plt.title("Residual model relative gain over TinyDepthNet")
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(OUT / "residual_focus_resunet_vs_tiny_gain.png")
    plt.close()


def write_report(rows: list[dict[str, object]], history: list[dict[str, float]], elapsed_s: float) -> None:
    test = [r for r in rows if r["split"] == "test"]
    p10 = next(r for r in test if "P10" in str(r["sample"]))
    lines = [
        "# 残差保护版 Focus-ResUNet 实验报告",
        "",
        "## 设计动机",
        "",
        "第一版 Focus-ResUNet 在 P10、A 型刃脊、山脊和周期纹理上明显优于 TinyDepthNet，但在复合凹坑和阶跃样品上出现过度校正。第二版将网络改为残差式结构：先由 DFF/GADFF 先验得到基础深度，再由模型预测有界残差和门控系数。这样可以让模型主要修正眩光与复杂纹理导致的错误，同时在 DFF 已经可靠的区域保持保守。",
        "",
        "## 损失函数调整",
        "",
        "保留眩光加权 Charbonnier、梯度、Laplacian、法向和焦栈先验一致性损失，同时新增 residual guard：在高置信、低眩光区域惩罚模型偏离 DFF/GADFF 先验，避免复合凹坑和阶跃边缘被强行拉向训练集中更常见的纹理模式。",
        "",
        "## 测试集结果",
        "",
        "| 样品 | DFF MAE/um | TinyDepthNet MAE/um | 残差 Focus-ResUNet MAE/um | 相对 Tiny 改善/% |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in test:
        lines.append(
            f"| {r['category']} | {float(r['dff_mae_um']):.2f} | {float(r['tiny_model_mae_um']):.2f} | "
            f"{float(r['residual_focus_resunet_mae_um']):.2f} | {float(r['residual_vs_tiny_gain_percent']):.2f} |"
        )
    lines += [
        "",
        "## 平均指标",
        "",
        f"- 测试集 DFF 平均 MAE：{split_mean(rows, 'test', 'dff_mae_um'):.2f} um。",
        f"- 测试集 TinyDepthNet 平均 MAE：{split_mean(rows, 'test', 'tiny_model_mae_um'):.2f} um。",
        f"- 测试集残差 Focus-ResUNet 平均 MAE：{split_mean(rows, 'test', 'residual_focus_resunet_mae_um'):.2f} um。",
        f"- 残差 Focus-ResUNet 相对 TinyDepthNet 平均改善：{split_mean(rows, 'test', 'residual_vs_tiny_gain_percent'):.2f}%。",
        "",
        "## P10 结果",
        "",
        f"- P10 上 DFF MAE：{float(p10['dff_mae_um']):.2f} um。",
        f"- P10 上 TinyDepthNet MAE：{float(p10['tiny_model_mae_um']):.2f} um。",
        f"- P10 上残差 Focus-ResUNet MAE：{float(p10['residual_focus_resunet_mae_um']):.2f} um。",
        f"- P10 相对 TinyDepthNet 改善：{float(p10['residual_vs_tiny_gain_percent']):.2f}%。",
        "",
        "## 结论",
        "",
        "残差保护版模型更适合作为结题后的算法升级路线：它把深度学习定位为对 DFF 的有界校正，而不是完全替代 DFF。若平均指标优于 TinyDepthNet，可作为新版推荐模型；若仍存在个别样品退化，应在报告中保留“需要真实标定和更丰富凹坑样本”的边界说明。",
        "",
        f"训练耗时约 {elapsed_s / 60:.1f} min；最佳模型按验证 MAE 选取；最后一轮验证 MAE 为 {history[-1]['val_mae_norm']:.4f}。",
    ]
    (OUT / "残差保护版模型实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    print("Preparing samples...", flush=True)
    train_samples = prepare_samples(dataset["train"])
    val_samples = prepare_samples(dataset["validation"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training residual Focus-ResUNet on {device}, channels={upgraded_channel_count()}", flush=True)
    model, history = train_model(train_samples, val_samples, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stack_layers": STACK_LAYERS,
            "input_channels": upgraded_channel_count(),
            "history": history,
            "loss": "hybrid DFF loss + residual guard",
            "seed": SEED,
        },
        MODEL_DIR / "residual_focus_resunet_hybrid_loss.pt",
    )
    write_history(history)
    tiny_baseline = load_tiny_baseline()
    rows: list[dict[str, object]] = []
    for split, items in dataset.items():
        rows.extend(evaluate_split(split, items, model, device, tiny_baseline))
    with (OUT / "residual_focus_resunet_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "residual_focus_resunet_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plots(rows)
    write_report(rows, history, time.time() - start)
    print(f"Wrote residual model experiment to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
