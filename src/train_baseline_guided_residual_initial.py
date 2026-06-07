from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
from collections import OrderedDict
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
from real_sample_glare_prior_fusion_validation import (
    edge_mae,
    edge_retention,
    focus_maps_from_stack,
    laplace_raw_from_stack,
    original_full_postprocess,
    postprocess_depth,
)
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays, metrics, save_float_image, tile_positions
from train_focus_resunet_loss_experiment import ResidualBlock, augment_features, charbonnier, grad_xy


ROOT = Path(__file__).resolve().parent
RUN_LABEL = os.environ.get("BGR_RUN_LABEL", "原算法引导残差校正初训")
RUN_SLUG = os.environ.get("BGR_RUN_SLUG", "baseline_guided_residual_initial")
OUT = ROOT / "结题交付包" / "05_图表与结果" / os.environ.get("BGR_OUT_NAME", "原算法引导残差校正初训")
MODEL_DIR = OUT / "model"

STACK_LAYERS = DEFAULT_STACK_LAYERS
DIFF_LAYERS = STACK_LAYERS - 1
PRIOR_OFFSET = STACK_LAYERS + DIFF_LAYERS
RISK_CH = PRIOR_OFFSET
CONF_CH = PRIOR_OFFSET + 2
ORIGINAL_RAW_CH = PRIOR_OFFSET + 5
BASELINE_MODE = os.environ.get("BGR_BASELINE_MODE", "full")
ORIGINAL_POST_FULL_CH = PRIOR_OFFSET + 6
if BASELINE_MODE == "mid_dual":
    GADFF_FULL_POST_CH = PRIOR_OFFSET + 7
    ORIGINAL_POST_CH = PRIOR_OFFSET + 9
    ORIGINAL_POST_MID_CH = ORIGINAL_POST_CH
    ORIGINAL_POST_FULL_CH_FOR_DIFF = ORIGINAL_POST_FULL_CH
    INPUT_CHANNELS = PRIOR_OFFSET + 11
else:
    ORIGINAL_POST_CH = ORIGINAL_POST_FULL_CH
    ORIGINAL_POST_MID_CH = ORIGINAL_POST_CH
    GADFF_FULL_POST_CH = PRIOR_OFFSET + 7
    ORIGINAL_POST_FULL_CH_FOR_DIFF = ORIGINAL_POST_FULL_CH
    INPUT_CHANNELS = PRIOR_OFFSET + 9

SEED = 20260520
PATCH_SIZE = 128
BATCH_SIZE = int(os.environ.get("BGR_BATCH_SIZE", "6"))
TRAIN_PATCHES_PER_EPOCH = int(os.environ.get("BGR_TRAIN_PATCHES_PER_EPOCH", "512"))
VAL_PATCHES_PER_EPOCH = int(os.environ.get("BGR_VAL_PATCHES_PER_EPOCH", "128"))
EPOCHS = int(os.environ.get("BGR_EPOCHS", "12"))
TRAIN_CACHE_SIZE = int(os.environ.get("BGR_TRAIN_CACHE_SIZE", "8"))
VAL_CACHE_SIZE = int(os.environ.get("BGR_VAL_CACHE_SIZE", "4"))
RESUME_TRAINING = os.environ.get("BGR_RESUME", "0") == "1"
RESUME_SOURCE = os.environ.get("BGR_RESUME_SOURCE", "latest").lower()


def prepare_sample(category: str, scenario) -> dict[str, object]:
    arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
    stack = arrays["stack"]
    truth = arrays["truth"]
    base = arrays["features"]
    gadff = arrays["gadff"]
    risk = arrays["risk"]
    assert isinstance(stack, np.ndarray)
    assert isinstance(truth, np.ndarray)
    assert isinstance(base, np.ndarray)
    assert isinstance(gadff, np.ndarray)
    assert isinstance(risk, np.ndarray)

    original_raw = laplace_raw_from_stack(stack, truth)
    original_post_full = original_full_postprocess(original_raw)
    original_post_mid = postprocess_depth(original_raw, median_kernel=5, gaussian_kernel=9, morph_kernel=9, order="median_gaussian_morph")
    original_post = original_post_mid if BASELINE_MODE == "mid_dual" else original_post_full
    gadff_full_post = original_full_postprocess(gadff)
    if BASELINE_MODE == "mid_dual":
        extra_features = [
            original_raw[None],
            original_post_full[None],
            gadff_full_post[None],
            (original_post_full - gadff_full_post)[None],
            original_post_mid[None],
            (original_post_full - original_post_mid)[None],
        ]
    else:
        extra_features = [
            original_raw[None],
            original_post_full[None],
            gadff_full_post[None],
            (original_post_full - gadff_full_post)[None],
        ]
    model_features = np.concatenate([augment_features(base), *extra_features], axis=0).astype(np.float32)
    baseline_error = np.abs(original_post - truth).astype(np.float32)
    arrays["model_features"] = model_features
    arrays["original_raw"] = original_raw
    arrays["original_post"] = original_post
    arrays["original_post_full"] = original_post_full
    arrays["original_post_mid"] = original_post_mid
    arrays["gadff_full_post"] = gadff_full_post
    arrays["baseline_error"] = baseline_error
    arrays["category"] = category
    return arrays


def prepare_samples(items: list[tuple[str, object]]) -> list[dict[str, object]]:
    return [prepare_sample(category, scenario) for category, scenario in items]


class PreparedSamplePool:
    def __init__(self, items: list[tuple[str, object]], cache_size: int) -> None:
        self.items = items
        self.cache_size = max(1, cache_size)
        self.cache: OrderedDict[int, dict[str, object]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        category, scenario = self.items[index]
        sample = prepare_sample(category, scenario)
        self.cache[index] = sample
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        gc.collect()
        return sample


def choose_patch_origin(score: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    h, w = score.shape
    if h <= PATCH_SIZE or w <= PATCH_SIZE:
        return 0, 0
    work = score.copy().astype(np.float32)
    margin = PATCH_SIZE // 2
    work[:margin, :] = 0
    work[-margin:, :] = 0
    work[:, :margin] = 0
    work[:, -margin:] = 0
    if float(np.max(work)) <= 1e-6:
        return int(rng.integers(0, h - PATCH_SIZE + 1)), int(rng.integers(0, w - PATCH_SIZE + 1))
    flat = work.ravel()
    threshold = np.percentile(flat[flat > 0], 75) if np.any(flat > 0) else 0.0
    ys, xs = np.where(work >= threshold)
    if len(ys) == 0:
        y = int(np.argmax(work) // w)
        x = int(np.argmax(work) % w)
    else:
        idx = int(rng.integers(0, len(ys)))
        y, x = int(ys[idx]), int(xs[idx])
    return int(np.clip(y - PATCH_SIZE // 2, 0, h - PATCH_SIZE)), int(np.clip(x - PATCH_SIZE // 2, 0, w - PATCH_SIZE))


def random_patch_batch(samples, rng: np.random.Generator, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _ in range(batch_size):
        item = samples[int(rng.integers(0, len(samples)))]
        features = item["model_features"]
        truth = item["truth"]
        risk = item["risk"]
        baseline_error = item["baseline_error"]
        assert isinstance(features, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(baseline_error, np.ndarray)
        _, h, w = features.shape
        mode = float(rng.random())
        if mode < 0.50:
            y0 = int(rng.integers(0, h - PATCH_SIZE + 1))
            x0 = int(rng.integers(0, w - PATCH_SIZE + 1))
        elif mode < 0.75:
            y0, x0 = choose_patch_origin(risk, rng)
        else:
            y0, x0 = choose_patch_origin(baseline_error, rng)
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


class BaselineGuidedResidualNet(nn.Module):
    def __init__(self, in_channels: int = INPUT_CHANNELS, base: int = 24, residual_scale: float = 0.16) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.stem = nn.Sequential(nn.Conv2d(in_channels, base, 3, padding=1), nn.GroupNorm(6, base), nn.SiLU())
        self.enc1 = ResidualBlock(base, base)
        self.enc2 = ResidualBlock(base, base * 2)
        self.enc3 = ResidualBlock(base * 2, base * 3)
        self.bottleneck = ResidualBlock(base * 3, base * 4)
        self.dec3 = ResidualBlock(base * 4 + base * 3, base * 3)
        self.dec2 = ResidualBlock(base * 3 + base * 2, base * 2)
        self.dec1 = ResidualBlock(base * 2 + base, base)
        self.head = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.SiLU(), nn.Conv2d(base, 3, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        original_post = x[:, ORIGINAL_POST_CH : ORIGINAL_POST_CH + 1]
        x0 = self.stem(x)
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
        raw = self.head(d1)
        delta = self.residual_scale * torch.tanh(raw[:, 0:1])
        mask = torch.sigmoid(raw[:, 1:2])
        corr_conf = torch.sigmoid(raw[:, 2:3])
        correction = mask * corr_conf * delta
        pred = torch.clamp(original_post + correction, 0.0, 1.0)
        return {"pred": pred, "delta": delta, "mask": mask, "corr_conf": corr_conf, "correction": correction}


class BaselineGuidedLoss(nn.Module):
    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred = outputs["pred"]
        delta = outputs["delta"]
        mask = outputs["mask"]
        corr_conf = outputs["corr_conf"]
        correction = outputs["correction"]
        risk = torch.clamp(features[:, RISK_CH : RISK_CH + 1], 0, 1)
        conf = torch.clamp(features[:, CONF_CH : CONF_CH + 1], 0, 1)
        original_post = torch.clamp(features[:, ORIGINAL_POST_CH : ORIGINAL_POST_CH + 1], 0, 1)
        target_delta = torch.clamp(target - original_post, -0.16, 0.16)

        depth = torch.mean(charbonnier(pred - target))
        delta_loss = torch.mean(charbonnier(correction - target_delta))
        pdx, pdy = grad_xy(pred)
        tdx, tdy = grad_xy(target)
        edge = torch.mean(charbonnier(pdx - tdx)) + torch.mean(charbonnier(pdy - tdy))

        baseline_err = torch.abs(original_post - target).detach()
        safe = torch.clamp((1.0 - risk).pow(1.5) * conf * (baseline_err < 0.035).float(), 0.0, 1.0)
        guard = torch.sum(safe * charbonnier(pred - original_post)) / torch.clamp(torch.sum(safe), min=1.0)

        high_weight = torch.clamp(risk + (1.0 - conf) + (baseline_err > 0.055).float(), 0.0, 2.0)
        highrisk = torch.sum(high_weight * charbonnier(pred - target)) / torch.clamp(torch.sum(high_weight), min=1.0)
        sparse = torch.mean(mask * corr_conf)

        loss = depth + 0.20 * delta_loss + 0.12 * edge + 0.18 * guard + 0.10 * highrisk + 0.04 * sparse
        parts = {
            "depth": float(depth.detach().cpu()),
            "delta": float(delta_loss.detach().cpu()),
            "edge": float(edge.detach().cpu()),
            "guard": float(guard.detach().cpu()),
            "highrisk": float(highrisk.detach().cpu()),
            "mask_sparse": float(sparse.detach().cpu()),
            "total": float(loss.detach().cpu()),
            "mean_mask": float(torch.mean(mask).detach().cpu()),
            "mean_conf": float(torch.mean(corr_conf).detach().cpu()),
            "mean_abs_delta": float(torch.mean(torch.abs(delta)).detach().cpu()),
        }
        return loss, parts


def train_model(train_samples, val_samples, device: str) -> tuple[BaselineGuidedResidualNet, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = BaselineGuidedResidualNet().to(device)
    criterion = BaselineGuidedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1.5e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=8e-5)
    train_steps = math.ceil(TRAIN_PATCHES_PER_EPOCH / BATCH_SIZE)
    val_steps = math.ceil(VAL_PATCHES_PER_EPOCH / BATCH_SIZE)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    start_epoch = 1
    latest_path = MODEL_DIR / f"{RUN_SLUG}_latest_checkpoint.pt"
    best_path = MODEL_DIR / f"{RUN_SLUG}_best_checkpoint.pt"
    if RESUME_TRAINING and RESUME_SOURCE == "best" and best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        history = list(ckpt.get("history", []))
        best_val = float(ckpt.get("best_val_mae_norm", best_val))
        start_epoch = int(ckpt.get("epoch", len(history))) + 1
        print(f"Resumed best checkpoint weights from epoch {start_epoch - 1}, best_val={best_val:.5f}", flush=True)
    elif RESUME_TRAINING and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = list(ckpt.get("history", []))
        best_val = float(ckpt.get("best_val_mae_norm", best_val))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed checkpoint from epoch {start_epoch - 1}, best_val={best_val:.5f}", flush=True)
    elif RESUME_TRAINING and best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        history = list(ckpt.get("history", []))
        best_val = float(ckpt.get("best_val_mae_norm", best_val))
        start_epoch = int(ckpt.get("epoch", len(history))) + 1
        print(f"Resumed best checkpoint weights from epoch {start_epoch - 1}, best_val={best_val:.5f}", flush=True)
    if history:
        write_history(history)
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for _ in range(train_steps):
            xb, yb = random_patch_batch(train_samples, rng, BATCH_SIZE)
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss, parts = criterion(outputs, yb, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += parts["total"]
        scheduler.step()
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mask = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = random_patch_batch(val_samples, rng, BATCH_SIZE)
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss, parts = criterion(outputs, yb, xb)
                val_loss += parts["total"]
                val_mae += float(torch.mean(torch.abs(outputs["pred"] - yb)).detach().cpu())
                val_mask += parts["mean_mask"]
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / train_steps,
            "val_loss": val_loss / val_steps,
            "val_mae_norm": val_mae / val_steps,
            "val_mean_mask": val_mask / val_steps,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        write_history(history)
        print(f"epoch {epoch:02d}/{EPOCHS}: train={row['train_loss']:.5f}, val={row['val_loss']:.5f}, val_mae={row['val_mae_norm']:.5f}, mask={row['val_mean_mask']:.3f}", flush=True)
        latest_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": latest_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "input_channels": INPUT_CHANNELS,
                "stack_layers": STACK_LAYERS,
                "history": history,
                "seed": SEED,
                "formula": "pred = original_post + sigmoid(mask) * sigmoid(conf) * 0.16*tanh(delta)",
                "run_label": RUN_LABEL,
                "best_val_mae_norm": best_val,
                "epoch": epoch,
                "baseline_mode": BASELINE_MODE,
            },
            latest_path,
        )
        if row["val_mae_norm"] < best_val:
            best_val = row["val_mae_norm"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_channels": INPUT_CHANNELS,
                    "stack_layers": STACK_LAYERS,
                    "history": history,
                    "seed": SEED,
            "formula": "pred = original_post + sigmoid(mask) * sigmoid(conf) * 0.16*tanh(delta)",
            "run_label": RUN_LABEL,
            "best_val_mae_norm": best_val,
            "epoch": epoch,
            "baseline_mode": BASELINE_MODE,
                    },
                best_path,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_tiled(model: nn.Module, features: np.ndarray, device: str, tile: int = 256, overlap: int = 80) -> dict[str, np.ndarray]:
    _, h, w = features.shape
    sums = {k: np.zeros((h, w), dtype=np.float32) for k in ["pred", "delta", "mask", "corr_conf", "correction"]}
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
                out = model(torch.from_numpy(tile_arr[None]).to(device))
                out_np = {k: v.detach().cpu().numpy()[0, 0] for k, v in out.items()}
            for key in sums:
                arr = out_np[key][: min(tile, h - y), : min(tile, w - x)]
                ww = window[: arr.shape[0], : arr.shape[1]]
                sums[key][y : y + arr.shape[0], x : x + arr.shape[1]] += arr * ww
            weight_sum[y : y + min(tile, h - y), x : x + min(tile, w - x)] += window[: min(tile, h - y), : min(tile, w - x)]
    return {k: np.clip(v / np.maximum(weight_sum, 1e-6), 0.0, 1.0).astype(np.float32) if k in {"pred", "mask", "corr_conf"} else (v / np.maximum(weight_sum, 1e-6)).astype(np.float32) for k, v in sums.items()}


def save_height_panel(sample_name: str, truth: np.ndarray, original_post: np.ndarray, pred: np.ndarray, pred_full_post: np.ndarray, outputs: dict[str, np.ndarray], out: Path) -> None:
    panels = [
        ("truth", truth, "viridis"),
        ("original_post", original_post, "viridis"),
        ("pred", pred, "viridis"),
        ("pred_full_post", pred_full_post, "viridis"),
        ("abs error pred", np.abs(pred - truth), "inferno"),
        ("abs error original", np.abs(original_post - truth), "inferno"),
        ("delta", outputs["delta"], "coolwarm"),
        ("mask", outputs["mask"], "magma"),
        ("confidence", outputs["corr_conf"], "magma"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 10.2), dpi=150)
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(f"{sample_name}: baseline-guided residual initial", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_3d_comparison(sample_name: str, truth: np.ndarray, original_post: np.ndarray, pred: np.ndarray, out: Path) -> None:
    def small(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr.astype(np.float32), (220, 180), interpolation=cv2.INTER_AREA)

    fig = plt.figure(figsize=(15.0, 5.1), dpi=150)
    for i, (title, arr) in enumerate([("truth", truth), ("original_post", original_post), ("pred", pred)], start=1):
        s = small(arr)
        yy, xx = np.mgrid[0 : s.shape[0], 0 : s.shape[1]]
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(xx, yy, s, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_zlim(0, 1)
        ax.view_init(elev=38, azim=-58)
    fig.suptitle(f"{sample_name}: 3D comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def sample_metrics(pred: np.ndarray, truth: np.ndarray, risk: np.ndarray, depth_range_um: float, original_post: np.ndarray) -> dict[str, float]:
    m = metrics(pred, truth, risk, depth_range_um)
    err_pred = np.abs(pred - truth)
    err_base = np.abs(original_post - truth)
    return {
        "mae_um": float(m["mae_um"]),
        "high_risk_mae_um": float(m["high_risk_mae_um"]),
        "edge_mae_um": edge_mae(pred, truth, depth_range_um),
        "p90_error_um": float(m["p90_norm"] * depth_range_um),
        "worse_than_original_area_percent": float(np.mean(err_pred > err_base + 1e-6) * 100),
        "edge_retention": edge_retention(pred, truth),
    }


def evaluate_split(split: str, items: list[tuple[str, object]], model: nn.Module, device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, scenario in items:
        print(f"evaluate {split}: {scenario.name}", flush=True)
        sample_dir = OUT / split / scenario.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        item = prepare_sample(category, scenario)
        features = item["model_features"]
        truth = item["truth"]
        risk = item["risk"]
        original_post = item["original_post"]
        gadff_full_post = item["gadff_full_post"]
        assert isinstance(features, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(original_post, np.ndarray)
        assert isinstance(gadff_full_post, np.ndarray)
        outputs = predict_tiled(model, features, device)
        pred = np.clip(outputs["pred"], 0, 1)
        pred_full_post = original_full_postprocess(pred)
        for name, arr in [
            ("truth", truth),
            ("original_post", original_post),
            ("gadff_full_post", gadff_full_post),
            ("pred", pred),
            ("pred_full_post", pred_full_post),
            ("delta", outputs["delta"]),
            ("mask", outputs["mask"]),
            ("confidence", outputs["corr_conf"]),
        ]:
            save_float_image(sample_dir / f"{name}.png", arr, cv2.COLORMAP_VIRIDIS if name not in {"mask", "confidence"} else cv2.COLORMAP_MAGMA)
        save_height_panel(scenario.name, truth, original_post, pred, pred_full_post, outputs, sample_dir / "error_panel.png")
        save_3d_comparison(scenario.name, truth, original_post, pred, sample_dir / "3d_comparison.png")

        original_m = sample_metrics(original_post, truth, risk, scenario.depth_range_um, original_post)
        gadff_m = sample_metrics(gadff_full_post, truth, risk, scenario.depth_range_um, original_post)
        pred_m = sample_metrics(pred, truth, risk, scenario.depth_range_um, original_post)
        pred_post_m = sample_metrics(pred_full_post, truth, risk, scenario.depth_range_um, original_post)
        row: dict[str, object] = {
            "split": split,
            "category": category,
            "sample": scenario.name,
            "depth_range_um": scenario.depth_range_um,
            "risk_area_percent": float(np.mean(risk > 0.08) * 100),
            "original_post_mae_um": original_m["mae_um"],
            "gadff_full_post_mae_um": gadff_m["mae_um"],
            "pred_mae_um": pred_m["mae_um"],
            "pred_full_post_mae_um": pred_post_m["mae_um"],
            "original_high_risk_mae_um": original_m["high_risk_mae_um"],
            "pred_high_risk_mae_um": pred_m["high_risk_mae_um"],
            "pred_full_post_high_risk_mae_um": pred_post_m["high_risk_mae_um"],
            "original_edge_mae_um": original_m["edge_mae_um"],
            "pred_edge_mae_um": pred_m["edge_mae_um"],
            "pred_full_post_edge_mae_um": pred_post_m["edge_mae_um"],
            "pred_p90_error_um": pred_m["p90_error_um"],
            "pred_worse_than_original_area_percent": pred_m["worse_than_original_area_percent"],
            "pred_full_post_worse_than_original_area_percent": pred_post_m["worse_than_original_area_percent"],
            "mean_mask_value": float(np.mean(outputs["mask"])),
            "mean_confidence_value": float(np.mean(outputs["corr_conf"])),
            "mean_abs_delta": float(np.mean(np.abs(outputs["delta"]))),
            "panel": str(sample_dir / "error_panel.png"),
            "surface_3d": str(sample_dir / "3d_comparison.png"),
        }
        rows.append(row)
    return rows


def write_history(history: list[dict[str, float]]) -> None:
    with (OUT / f"{RUN_SLUG}_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plt.figure(figsize=(7.4, 4.4), dpi=150)
    epochs = [int(h["epoch"]) for h in history]
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation")
    plt.plot(epochs, [h["val_mae_norm"] for h in history], marker="^", label="val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / normalized MAE")
    plt.title(RUN_LABEL)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"{RUN_SLUG}_training_curve.png")
    plt.close()


def split_mean(rows: list[dict[str, object]], split: str, key: str) -> float:
    vals = [float(r[key]) for r in rows if r["split"] == split]
    return float(np.mean(vals)) if vals else float("nan")


def write_report(rows: list[dict[str, object]], elapsed: float) -> None:
    test = [r for r in rows if r["split"] == "test"]
    original_mean = split_mean(rows, "test", "original_post_mae_um")
    pred_mean = split_mean(rows, "test", "pred_mae_um")
    pred_post_mean = split_mean(rows, "test", "pred_full_post_mae_um")
    original_hr = split_mean(rows, "test", "original_high_risk_mae_um")
    pred_hr = split_mean(rows, "test", "pred_high_risk_mae_um")
    original_edge = split_mean(rows, "test", "original_edge_mae_um")
    pred_edge = split_mean(rows, "test", "pred_edge_mae_um")
    best_pred = min(pred_mean, pred_post_mean)
    if best_pred < original_mean:
        conclusion = "成功：残差校正平均 MAE 低于原算法 post。"
    elif best_pred <= original_mean * 1.05 and pred_hr <= original_hr * 0.92 and pred_edge <= original_edge * 1.10:
        conclusion = "部分成功：整体接近原算法，并在高风险区域有改善。"
    else:
        conclusion = "失败：当前模型未超过原算法 post，需补充失败样本或调整模型。"
    lines = [
        f"# {RUN_LABEL}报告",
        "",
        "## 1. 结论",
        "",
        conclusion,
        "",
        f"- 原算法 post 测试集平均 MAE：`{original_mean:.2f} um`。",
        f"- 残差模型 pred 测试集平均 MAE：`{pred_mean:.2f} um`。",
        f"- 残差模型 pred + full post 测试集平均 MAE：`{pred_post_mean:.2f} um`。",
        f"- 原算法 high-risk MAE：`{original_hr:.2f} um`。",
        f"- 残差模型 high-risk MAE：`{pred_hr:.2f} um`。",
        f"- 训练耗时：`{elapsed / 60:.1f} min`。",
        "",
        "## 2. 测试集逐样本结果",
        "",
        "| 样本 | original_post | pred | pred_full_post | high-risk pred | worse area/% | mean mask |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in test:
        lines.append(
            f"| {r['sample']} | {float(r['original_post_mae_um']):.2f} | {float(r['pred_mae_um']):.2f} | "
            f"{float(r['pred_full_post_mae_um']):.2f} | {float(r['pred_high_risk_mae_um']):.2f} | "
            f"{float(r['pred_worse_than_original_area_percent']):.2f} | {float(r['mean_mask_value']):.3f} |"
        )
    lines += [
        "",
        "## 3. 解释",
        "",
        "本模型不直接重建高度，而是在 `original_post` 上预测有界残差、修正 mask 和校正置信度。本轮训练若优于初训，说明补充反射域样本有助于模型识别眩光、低纹理与杂散光条件下的原算法失败区域；若退化，则说明样本难度或分布需要重新平衡。",
        "",
        "## 4. 运行说明与边界",
        "",
        f"- 本次训练使用 `BATCH_SIZE={BATCH_SIZE}`，`EPOCHS={EPOCHS}`，每轮训练 patch 目标数为 `{TRAIN_PATCHES_PER_EPOCH}`，验证 patch 目标数为 `{VAL_PATCHES_PER_EPOCH}`。",
        f"- 残差基准模式：`{BASELINE_MODE}`。`full` 表示以 current full post 为基准；`mid_dual` 表示以 median(5)+Gaussian(9)+open(9) 的 `original_post_mid` 为基准，并保留 full-mid 差异通道。",
        "- 由于当前机器 CUDA 显存较紧，脚本支持用环境变量 `BGR_BATCH_SIZE` 降低 batch；这只影响单步显存与训练步数，不改变输入通道、模型输出、损失函数和评价口径。",
        "- 该结论仍只对现有仿真 test split 成立；真实样本没有 ground truth，后续需要继续使用无真值诊断指标和可视化结果谨慎验证。",
    ]
    (OUT / f"{RUN_LABEL}报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("BGR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset()
    print("Preparing lazy train/validation sample pools...", flush=True)
    train_samples = PreparedSamplePool(dataset["train"], TRAIN_CACHE_SIZE)
    val_samples = PreparedSamplePool(dataset["validation"], VAL_CACHE_SIZE)
    print(f"Training baseline-guided residual model on {device}, channels={INPUT_CHANNELS}", flush=True)
    model, history = train_model(train_samples, val_samples, device)
    write_history(history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": INPUT_CHANNELS,
            "stack_layers": STACK_LAYERS,
            "history": history,
            "seed": SEED,
            "formula": "pred = original_post + sigmoid(mask) * sigmoid(conf) * 0.16*tanh(delta)",
            "run_label": RUN_LABEL,
            "baseline_mode": BASELINE_MODE,
        },
        MODEL_DIR / f"{RUN_SLUG}.pt",
    )
    rows: list[dict[str, object]] = []
    for split, items in dataset.items():
        rows.extend(evaluate_split(split, items, model, device))
    with (OUT / f"{RUN_SLUG}_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / f"{RUN_SLUG}_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(rows, time.time() - start)
    print(f"Wrote baseline-guided residual initial experiment to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
