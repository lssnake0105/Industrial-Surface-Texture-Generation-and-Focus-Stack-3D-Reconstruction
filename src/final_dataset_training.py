from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from simulate_antiglare_highres_samples import (
    CameraConfig,
    DEFAULT_STACK_LAYERS,
    Scenario,
    edge_loss,
    feature_channel_count,
    generate_sample_arrays,
    metrics,
    predict_tiled,
    save_3d_surface_preview,
    save_sample_panel,
)
from simulate_antiglare_prototype import TinyDepthNet
from surface_sample_generator import SurfaceConfig


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结题交付包" / "05_图表与结果" / "最终仿真数据集训练验证"
MODEL_DIR = OUT / "model"
STACK_LAYERS = DEFAULT_STACK_LAYERS


TRAIN_PATCHES_PER_EPOCH = 448
VAL_PATCHES_PER_EPOCH = 96
EPOCHS = 10
BATCH_SIZE = 8
PATCH_SIZE = 128
SEED = 20260518


def surface_scenario(
    name: str,
    width: int,
    height: int,
    seed: int,
    depth_range_um: float,
    baseline_type: str,
    noise_type: str,
    *,
    tilt_x_um: float = 0.0,
    tilt_y_um: float = 0.0,
    feature_amp_um: float = 600.0,
    noise_amp_um: float = 45.0,
    stray_level: float = 0.16,
    roughness_base: float = 0.36,
    f0: float = 0.84,
    period_count: float = 8.0,
    ridge_width: float = 0.11,
    step_position: float = 0.0,
    perlin_octaves: int = 6,
    perlin_grid: int = 0,
    perlin_persistence: float = 0.5,
    valley_width: float = 0.36,
    valley_floor: float = 0.08,
    valley_sharpness: float = 1.0,
    orientation_deg: float = 0.0,
) -> Scenario:
    cfg = SurfaceConfig(
        name=name,
        width=width,
        height=height,
        depth_range_um=depth_range_um,
        baseline_type=baseline_type,
        noise_type=noise_type,
        seed=seed,
        tilt_x_um=tilt_x_um,
        tilt_y_um=tilt_y_um,
        feature_amp_um=feature_amp_um,
        noise_amp_um=noise_amp_um,
        period_count=period_count,
        ridge_width=ridge_width,
        step_position=step_position,
        perlin_octaves=perlin_octaves,
        perlin_grid=perlin_grid,
        perlin_persistence=perlin_persistence,
        valley_width=valley_width,
        valley_floor=valley_floor,
        valley_sharpness=valley_sharpness,
        orientation_deg=orientation_deg,
    )
    return Scenario(
        name=name,
        width=width,
        height=height,
        seed=seed,
        depth_range_um=depth_range_um,
        tilt_x_um=tilt_x_um,
        tilt_y_um=tilt_y_um,
        micro_amp_um=0.0,
        pit_style="none",
        pit_count=0,
        scratch_count=0,
        stray_level=stray_level,
        roughness_base=roughness_base,
        f0=f0,
        surface_config=cfg,
    )


def pit_scenario(
    name: str,
    width: int,
    height: int,
    seed: int,
    depth_range_um: float,
    tilt_x_um: float,
    tilt_y_um: float,
    micro_amp_um: float,
    pit_style: str,
    pit_count: int,
    scratch_count: int,
    stray_level: float,
    roughness_base: float,
    f0: float,
) -> Scenario:
    return Scenario(
        name,
        width,
        height,
        seed,
        depth_range_um,
        tilt_x_um,
        tilt_y_um,
        micro_amp_um,
        pit_style,
        pit_count,
        scratch_count,
        stray_level,
        roughness_base,
        f0,
    )


def add_reflectance_augmented_samples(
    train: list[tuple[str, Scenario]],
    validation: list[tuple[str, Scenario]],
) -> None:
    """Add same-geometry / different-reflectance cases for glare robustness."""
    train.extend(
        [
            (
                "反射增强-山峰-镜面强眩光",
                surface_scenario("train_reflect_山峰_镜面强眩光_13", 640, 360, 801, 1080, "mountain", "perlin", tilt_x_um=-115, tilt_y_um=70, feature_amp_um=690, noise_amp_um=48, perlin_octaves=6, perlin_grid=120, perlin_persistence=0.56, stray_level=0.30, roughness_base=0.20, f0=0.96),
            ),
            (
                "反射增强-山峰-粗糙漫反射",
                surface_scenario("train_reflect_山峰_粗糙漫反射_14", 640, 360, 802, 1080, "mountain", "perlin", tilt_x_um=-115, tilt_y_um=70, feature_amp_um=690, noise_amp_um=72, perlin_octaves=7, perlin_grid=155, perlin_persistence=0.66, stray_level=0.10, roughness_base=0.58, f0=0.58),
            ),
            (
                "反射增强-山脊-窄高光带",
                surface_scenario("train_reflect_山脊_窄高光带_15", 640, 360, 803, 1040, "ridge", "perlin", tilt_x_um=115, tilt_y_um=-105, feature_amp_um=660, noise_amp_um=54, ridge_width=0.090, perlin_grid=110, perlin_persistence=0.55, orientation_deg=12, stray_level=0.27, roughness_base=0.18, f0=0.97),
            ),
            (
                "反射增强-山脊-低纹理粗糙",
                surface_scenario("train_reflect_山脊_低纹理粗糙_16", 640, 360, 804, 1040, "ridge", "smooth", tilt_x_um=115, tilt_y_um=-105, feature_amp_um=660, noise_amp_um=24, ridge_width=0.110, orientation_deg=12, stray_level=0.12, roughness_base=0.62, f0=0.55),
            ),
            (
                "反射增强-A型刃脊-金属强眩光",
                surface_scenario("train_reflect_A型刃脊_金属强眩光_17", 640, 360, 805, 1260, "a_ridge", "perlin", tilt_x_um=150, tilt_y_um=-95, feature_amp_um=740, noise_amp_um=66, ridge_width=0.095, perlin_octaves=7, perlin_grid=115, perlin_persistence=0.58, stray_level=0.31, roughness_base=0.19, f0=0.98),
            ),
            (
                "反射增强-A型刃脊-雾面散射",
                surface_scenario("train_reflect_A型刃脊_雾面散射_18", 640, 360, 806, 1260, "a_ridge", "perlin", tilt_x_um=150, tilt_y_um=-95, feature_amp_um=740, noise_amp_um=82, ridge_width=0.095, perlin_octaves=7, perlin_grid=150, perlin_persistence=0.66, stray_level=0.18, roughness_base=0.64, f0=0.60),
            ),
            (
                "反射增强-阶跃-高杂散光",
                surface_scenario("train_reflect_阶跃_高杂散光_19", 640, 360, 807, 1040, "step", "perlin", tilt_x_um=105, tilt_y_um=95, feature_amp_um=670, noise_amp_um=54, step_position=-0.05, perlin_grid=125, perlin_persistence=0.58, stray_level=0.32, roughness_base=0.24, f0=0.92),
            ),
            (
                "反射增强-阶跃-低反射低纹理",
                surface_scenario("train_reflect_阶跃_低反射低纹理_20", 640, 360, 808, 1040, "step", "smooth", tilt_x_um=105, tilt_y_um=95, feature_amp_um=670, noise_amp_um=20, step_position=-0.05, stray_level=0.09, roughness_base=0.54, f0=0.50),
            ),
            (
                "反射增强-周期-条纹强高光",
                surface_scenario("train_reflect_周期_条纹强高光_21", 640, 360, 809, 920, "periodic", "stripe", tilt_x_um=-90, tilt_y_um=115, feature_amp_um=460, noise_amp_um=40, period_count=8.8, stray_level=0.28, roughness_base=0.21, f0=0.96),
            ),
            (
                "反射增强-周期-粗糙漫反射",
                surface_scenario("train_reflect_周期_粗糙漫反射_22", 640, 360, 810, 920, "periodic", "perlin", tilt_x_um=-90, tilt_y_um=115, feature_amp_um=460, noise_amp_um=74, period_count=8.8, perlin_grid=145, perlin_persistence=0.64, stray_level=0.11, roughness_base=0.60, f0=0.56),
            ),
            (
                "反射增强-P10V谷-镜面强眩光",
                surface_scenario("train_reflect_P10V谷_镜面强眩光_23", 640, 360, 811, 1200, "v_valley", "perlin", tilt_x_um=80, tilt_y_um=-45, feature_amp_um=820, noise_amp_um=72, perlin_octaves=6, perlin_grid=160, perlin_persistence=0.60, valley_width=0.54, valley_floor=0.160, valley_sharpness=0.70, orientation_deg=-15, stray_level=0.33, roughness_base=0.21, f0=0.98),
            ),
            (
                "反射增强-P10V谷-雾面强粗糙",
                surface_scenario("train_reflect_P10V谷_雾面强粗糙_24", 640, 360, 812, 1200, "v_valley", "perlin", tilt_x_um=80, tilt_y_um=-45, feature_amp_um=820, noise_amp_um=92, perlin_octaves=7, perlin_grid=175, perlin_persistence=0.68, valley_width=0.54, valley_floor=0.160, valley_sharpness=0.70, orientation_deg=-15, stray_level=0.16, roughness_base=0.66, f0=0.62),
            ),
            (
                "反射增强-腐蚀凹坑-湿润高反射",
                pit_scenario("train_reflect_腐蚀凹坑_湿润高反射_25", 640, 360, 813, 1280, -320, 230, 11.0, "mixed_deep", 36, 10, 0.31, 0.22, 0.96),
            ),
            (
                "反射增强-划痕凹槽-强杂散光",
                pit_scenario("train_reflect_划痕凹槽_强杂散光_26", 640, 360, 814, 960, -280, -185, 8.4, "elongated_dents", 32, 28, 0.30, 0.28, 0.93),
            ),
            (
                "反射增强-微坑阵列-低反射粗糙",
                pit_scenario("train_reflect_微坑阵列_低反射粗糙_27", 640, 360, 815, 980, 180, -250, 6.5, "micro_pitting", 72, 16, 0.12, 0.62, 0.58),
            ),
        ]
    )

    validation.extend(
        [
            (
                "反射验证-山峰-镜面留出",
                surface_scenario("val_reflect_山峰_镜面留出", 640, 360, 901, 1080, "mountain", "perlin", tilt_x_um=105, tilt_y_um=-95, feature_amp_um=660, noise_amp_um=58, perlin_grid=135, perlin_persistence=0.56, stray_level=0.29, roughness_base=0.22, f0=0.96),
            ),
            (
                "反射验证-A型刃脊-粗糙留出",
                surface_scenario("val_reflect_A型刃脊_粗糙留出", 640, 360, 902, 1260, "a_ridge", "perlin", tilt_x_um=140, tilt_y_um=-100, feature_amp_um=720, noise_amp_um=86, ridge_width=0.10, perlin_octaves=7, perlin_grid=155, perlin_persistence=0.66, stray_level=0.17, roughness_base=0.65, f0=0.60),
            ),
            (
                "反射验证-P10V谷-强杂散留出",
                surface_scenario("val_reflect_P10V谷_强杂散留出", 640, 360, 903, 1200, "v_valley", "perlin", tilt_x_um=80, tilt_y_um=-45, feature_amp_um=820, noise_amp_um=80, perlin_octaves=6, perlin_grid=160, perlin_persistence=0.60, valley_width=0.54, valley_floor=0.160, valley_sharpness=0.70, orientation_deg=-15, stray_level=0.34, roughness_base=0.25, f0=0.94),
            ),
            (
                "反射验证-阶跃-低纹理留出",
                surface_scenario("val_reflect_阶跃_低纹理留出", 640, 360, 904, 1040, "step", "smooth", tilt_x_um=120, tilt_y_um=90, feature_amp_um=650, noise_amp_um=18, step_position=-0.05, stray_level=0.10, roughness_base=0.56, f0=0.52),
            ),
            (
                "反射验证-腐蚀凹坑-湿润留出",
                pit_scenario("val_reflect_腐蚀凹坑_湿润留出", 640, 360, 905, 1320, -340, 240, 11.4, "mixed_deep", 38, 11, 0.32, 0.24, 0.96),
            ),
        ]
    )


def build_dataset() -> dict[str, list[tuple[str, Scenario]]]:
    train = [
        (
            "山峰-分形粗糙",
            surface_scenario("train_山峰_分形粗糙_01", 640, 360, 501, 1040, "mountain", "fractal", tilt_x_um=90, tilt_y_um=-70, feature_amp_um=640, noise_amp_um=40, stray_level=0.14),
        ),
        (
            "山峰-柏林粗糙",
            surface_scenario("train_山峰_柏林粗糙_02", 640, 360, 502, 1120, "mountain", "perlin", tilt_x_um=-120, tilt_y_um=80, feature_amp_um=700, noise_amp_um=65, perlin_octaves=6, perlin_grid=120, perlin_persistence=0.58, stray_level=0.17),
        ),
        (
            "山脊-柏林粗糙",
            surface_scenario("train_山脊_柏林粗糙_03", 640, 360, 503, 980, "ridge", "perlin", tilt_x_um=110, tilt_y_um=-90, feature_amp_um=620, noise_amp_um=55, ridge_width=0.105, perlin_grid=105, orientation_deg=14, stray_level=0.16),
        ),
        (
            "A型刃脊-柏林粗糙",
            surface_scenario("train_A型刃脊_柏林粗糙_04", 640, 360, 504, 1260, "a_ridge", "perlin", tilt_x_um=150, tilt_y_um=-95, feature_amp_um=740, noise_amp_um=70, ridge_width=0.095, perlin_octaves=7, perlin_grid=115, perlin_persistence=0.58, stray_level=0.18),
        ),
        (
            "阶跃-平滑粗糙",
            surface_scenario("train_阶跃_平滑粗糙_05", 640, 360, 505, 900, "step", "smooth", tilt_x_um=80, tilt_y_um=120, feature_amp_um=570, noise_amp_um=32, step_position=-0.18, stray_level=0.13),
        ),
        (
            "阶跃-柏林粗糙",
            surface_scenario("train_阶跃_柏林粗糙_06", 640, 360, 506, 1080, "step", "perlin", tilt_x_um=-120, tilt_y_um=95, feature_amp_um=680, noise_amp_um=64, step_position=0.12, perlin_grid=125, perlin_persistence=0.62, stray_level=0.17),
        ),
        (
            "周期-条纹粗糙",
            surface_scenario("train_周期_条纹粗糙_07", 640, 360, 507, 860, "periodic", "stripe", tilt_x_um=70, tilt_y_um=-100, feature_amp_um=430, noise_amp_um=38, period_count=9.0, stray_level=0.15),
        ),
        (
            "周期-柏林粗糙",
            surface_scenario("train_周期_柏林粗糙_08", 640, 360, 508, 940, "periodic", "perlin", tilt_x_um=-70, tilt_y_um=110, feature_amp_um=480, noise_amp_um=62, period_count=6.5, perlin_grid=100, perlin_persistence=0.56, stray_level=0.17),
        ),
        (
            "V谷-尖锐中噪",
            surface_scenario("train_V谷_尖锐中噪_09", 640, 360, 509, 1180, "v_valley", "perlin", tilt_x_um=95, tilt_y_um=-70, feature_amp_um=790, noise_amp_um=56, perlin_octaves=7, perlin_grid=110, valley_width=0.28, valley_floor=0.045, valley_sharpness=1.35, orientation_deg=8, stray_level=0.18),
        ),
        (
            "V谷-宽谷强噪",
            surface_scenario("train_V谷_宽谷强噪_10", 640, 360, 510, 1220, "v_valley", "perlin", tilt_x_um=-85, tilt_y_um=65, feature_amp_um=810, noise_amp_um=86, perlin_octaves=7, perlin_grid=145, perlin_persistence=0.66, valley_width=0.47, valley_floor=0.12, valley_sharpness=0.82, orientation_deg=-18, stray_level=0.20),
        ),
        (
            "腐蚀凹坑-多尺度",
            pit_scenario("train_多尺度腐蚀凹坑_11", 640, 360, 511, 1120, -230, 240, 10.5, "multi_scale", 38, 9, 0.18, 0.40, 0.84),
        ),
        (
            "划痕凹槽-混合",
            pit_scenario("train_斜面细槽混合划痕_12", 640, 360, 512, 930, -280, -185, 8.2, "elongated_dents", 30, 24, 0.19, 0.36, 0.81),
        ),
    ]

    validation = [
        (
            "山峰-柏林粗糙",
            surface_scenario("val_山峰_柏林粗糙", 640, 360, 601, 1080, "mountain", "perlin", tilt_x_um=105, tilt_y_um=-95, feature_amp_um=660, noise_amp_um=60, perlin_grid=135, perlin_persistence=0.56, stray_level=0.16),
        ),
        (
            "山脊-分形粗糙",
            surface_scenario("val_山脊_分形粗糙", 640, 360, 602, 1010, "ridge", "fractal", tilt_x_um=-80, tilt_y_um=120, feature_amp_um=640, noise_amp_um=48, ridge_width=0.095, stray_level=0.16),
        ),
        (
            "阶跃-条纹粗糙",
            surface_scenario("val_阶跃_条纹粗糙", 640, 360, 603, 980, "step", "stripe", tilt_x_um=130, tilt_y_um=85, feature_amp_um=620, noise_amp_um=42, step_position=0.08, period_count=8.0, stray_level=0.15),
        ),
        (
            "周期-分形粗糙",
            surface_scenario("val_周期_分形粗糙", 640, 360, 604, 900, "periodic", "fractal", tilt_x_um=-110, tilt_y_um=70, feature_amp_um=450, noise_amp_um=45, period_count=7.4, stray_level=0.15),
        ),
        (
            "腐蚀凹坑-复合",
            pit_scenario("val_复合凹坑结构", 640, 360, 605, 1260, -320, 210, 11.0, "mixed_deep", 34, 10, 0.20, 0.42, 0.85),
        ),
    ]

    test = [
        (
            "P10 V谷-宽谷粗糙平底",
            surface_scenario(
                "test_V谷_P10_宽谷粗糙平底",
                960,
                540,
                410,
                1200,
                "v_valley",
                "perlin",
                tilt_x_um=80,
                tilt_y_um=-45,
                feature_amp_um=820,
                noise_amp_um=75,
                perlin_octaves=6,
                perlin_grid=160,
                perlin_persistence=0.60,
                valley_width=0.54,
                valley_floor=0.160,
                valley_sharpness=0.70,
                orientation_deg=-15,
                stray_level=0.20,
                roughness_base=0.39,
                f0=0.85,
            ),
        ),
        (
            "A型刃脊-柏林粗糙",
            surface_scenario("test_A型突起刃脊_柏林粗糙", 960, 540, 701, 1280, "a_ridge", "perlin", tilt_x_um=110, tilt_y_um=-90, feature_amp_um=720, noise_amp_um=58, ridge_width=0.095, perlin_octaves=6, perlin_grid=120, perlin_persistence=0.56, stray_level=0.17),
        ),
        (
            "山峰-分形粗糙",
            surface_scenario("test_山峰_分形粗糙", 640, 360, 702, 1100, "mountain", "fractal", tilt_x_um=-95, tilt_y_um=75, feature_amp_um=680, noise_amp_um=48, stray_level=0.16),
        ),
        (
            "山脊-柏林粗糙",
            surface_scenario("test_山脊_柏林粗糙", 640, 360, 703, 1020, "ridge", "perlin", tilt_x_um=115, tilt_y_um=-105, feature_amp_um=650, noise_amp_um=62, ridge_width=0.10, perlin_grid=115, perlin_persistence=0.58, stray_level=0.18),
        ),
        (
            "阶跃-柏林粗糙",
            surface_scenario("test_阶跃_柏林粗糙", 640, 360, 704, 1040, "step", "perlin", tilt_x_um=105, tilt_y_um=95, feature_amp_um=670, noise_amp_um=66, step_position=-0.05, perlin_grid=130, perlin_persistence=0.62, stray_level=0.17),
        ),
        (
            "周期-条纹粗糙",
            surface_scenario("test_周期_条纹粗糙", 640, 360, 705, 920, "periodic", "stripe", tilt_x_um=-90, tilt_y_um=115, feature_amp_um=460, noise_amp_um=43, period_count=8.8, stray_level=0.16),
        ),
        (
            "腐蚀凹坑-复合",
            pit_scenario("test_复合腐蚀凹坑", 960, 540, 706, 1420, -360, 270, 11.2, "mixed_deep", 42, 12, 0.20, 0.43, 0.84),
        ),
    ]
    add_reflectance_augmented_samples(train, validation)
    return {"train": train, "validation": validation, "test": test}


def random_patch_batch(
    samples: list[dict[str, object]],
    rng: np.random.Generator,
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _ in range(batch_size):
        item = samples[int(rng.integers(0, len(samples)))]
        features = item["features"]
        truth = item["truth"]
        assert isinstance(features, np.ndarray)
        assert isinstance(truth, np.ndarray)
        _, h, w = features.shape
        y0 = int(rng.integers(0, h - patch_size + 1))
        x0 = int(rng.integers(0, w - patch_size + 1))
        patch_x = features[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_y = truth[None, y0 : y0 + patch_size, x0 : x0 + patch_size]
        if rng.random() < 0.5:
            patch_x = patch_x[:, :, ::-1].copy()
            patch_y = patch_y[:, :, ::-1].copy()
        if rng.random() < 0.5:
            patch_x = patch_x[:, ::-1, :].copy()
            patch_y = patch_y[:, ::-1, :].copy()
        xs.append(patch_x.astype(np.float32))
        ys.append(patch_y.astype(np.float32))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def train_model(
    train_samples: list[dict[str, object]],
    val_samples: list[dict[str, object]],
    device: str,
    channels: int,
) -> tuple[TinyDepthNet, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = TinyDepthNet(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=1e-4)
    train_steps = math.ceil(TRAIN_PATCHES_PER_EPOCH / BATCH_SIZE)
    val_steps = math.ceil(VAL_PATCHES_PER_EPOCH / BATCH_SIZE)
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for _ in range(train_steps):
            xb, yb = random_patch_batch(train_samples, rng, BATCH_SIZE, PATCH_SIZE)
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(pred, yb) + 0.08 * edge_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = random_patch_batch(val_samples, rng, BATCH_SIZE, PATCH_SIZE)
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = torch.nn.functional.smooth_l1_loss(pred, yb) + 0.08 * edge_loss(pred, yb)
                val_loss += float(loss.detach().cpu())
                val_mae += float(torch.mean(torch.abs(pred - yb)).detach().cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss / train_steps,
                "val_loss": val_loss / val_steps,
                "val_mae_norm": val_mae / val_steps,
            }
        )
        print(
            f"epoch {epoch:02d}/{EPOCHS}: train_loss={history[-1]['train_loss']:.5f}, "
            f"val_loss={history[-1]['val_loss']:.5f}, val_mae_norm={history[-1]['val_mae_norm']:.5f}",
            flush=True,
        )
    return model, history


def write_dataset_split(dataset: dict[str, list[tuple[str, Scenario]]]) -> None:
    rows: list[dict[str, object]] = []
    for split, items in dataset.items():
        for category, scenario in items:
            rows.append(
                {
                    "split": split,
                    "sample": scenario.name,
                    "category": category,
                    "resolution": f"{scenario.width}x{scenario.height}",
                    "depth_range_um": scenario.depth_range_um,
                    "stack_layers": STACK_LAYERS,
                    "z_step_um": scenario.depth_range_um / max(STACK_LAYERS - 1, 1),
                    "stray_level": scenario.stray_level,
                    "surface_baseline": scenario.surface_config.baseline_type if scenario.surface_config else "pit/groove",
                    "surface_noise": scenario.surface_config.noise_type if scenario.surface_config else "procedural",
                }
            )
    with (OUT / "dataset_split.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "dataset_split.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_training_curve(history: list[dict[str, float]]) -> None:
    if not history:
        return
    epochs = [int(h["epoch"]) for h in history]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.3), dpi=150)
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SmoothL1 + edge loss")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_mae_norm"] for h in history], color="#d95f02", marker="^", label="validation MAE")
    ax2.set_ylabel("Validation MAE (normalized height)")
    ax2.legend(loc="center right")
    fig.suptitle("最终多类型样品训练曲线")
    fig.tight_layout()
    fig.savefig(OUT / "final_training_curve.png")
    plt.close(fig)


def evaluate_one(
    split: str,
    category: str,
    scenario: Scenario,
    model: TinyDepthNet,
    device: str,
) -> dict[str, object]:
    sample_dir = OUT / split / scenario.name
    sample_dir.mkdir(parents=True, exist_ok=True)
    arrays = generate_sample_arrays(scenario, stack_layers=STACK_LAYERS)
    camera = arrays["camera"]
    truth = arrays["truth"]
    stack = arrays["stack"]
    risk = arrays["risk"]
    dff = arrays["dff"]
    gadff = arrays["gadff"]
    features = arrays["features"]
    focus_positions_norm = arrays["focus_positions_norm"]
    assert isinstance(camera, CameraConfig)
    assert isinstance(truth, np.ndarray)
    assert isinstance(stack, np.ndarray)
    assert isinstance(risk, np.ndarray)
    assert isinstance(dff, np.ndarray)
    assert isinstance(gadff, np.ndarray)
    assert isinstance(features, np.ndarray)
    assert isinstance(focus_positions_norm, np.ndarray)

    model_pred = predict_tiled(model, features, device, tile=256, overlap=80)
    save_sample_panel(sample_dir, scenario, camera, truth, stack, risk, dff, gadff, None, model_pred)
    save_3d_surface_preview(sample_dir, scenario, camera, truth, model_pred)

    config_payload = {
        "split": split,
        "category": category,
        "camera": {
            **asdict(camera),
            "fov_width_mm": camera.fov_width_mm,
            "fov_height_mm": camera.fov_height_mm,
            "focal_length_mm": camera.focal_length_mm,
            "object_pixel_um": camera.object_pixel_um,
            "focus_positions_norm": focus_positions_norm.tolist(),
            "focus_positions_um": (focus_positions_norm * scenario.depth_range_um).tolist(),
        },
        "scenario": asdict(scenario),
    }
    (sample_dir / "camera_and_scene_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dff_m = metrics(dff, truth, risk, scenario.depth_range_um)
    ga_m = metrics(gadff, truth, risk, scenario.depth_range_um)
    model_m = metrics(model_pred, truth, risk, scenario.depth_range_um)
    row: dict[str, object] = {
        "split": split,
        "category": category,
        "sample": scenario.name,
        "resolution": f"{scenario.width}x{scenario.height}",
        "stack_layers": camera.stack_layers,
        "z_step_um": scenario.depth_range_um / max(camera.stack_layers - 1, 1),
        "depth_range_um": scenario.depth_range_um,
        "object_pixel_um": camera.object_pixel_um,
        "stray_level": scenario.stray_level,
        "risk_area_percent": float(np.mean(risk > 0.08) * 100),
        "dff_mae_um": dff_m["mae_um"],
        "ga_dff_mae_um": ga_m["mae_um"],
        "model_mae_um": model_m["mae_um"],
        "dff_high_risk_mae_um": dff_m["high_risk_mae_um"],
        "ga_dff_high_risk_mae_um": ga_m["high_risk_mae_um"],
        "model_high_risk_mae_um": model_m["high_risk_mae_um"],
        "dff_edge_mae_um": dff_m["edge_mae_um"],
        "ga_dff_edge_mae_um": ga_m["edge_mae_um"],
        "model_edge_mae_um": model_m["edge_mae_um"],
        "model_vs_dff_gain_percent": (dff_m["mae_um"] - model_m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100,
        "ga_vs_dff_gain_percent": (dff_m["mae_um"] - ga_m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100,
        "comparison_panel": str(sample_dir / "00_highres_comparison_panel.png"),
        "surface_3d_preview": str(sample_dir / "11_3d_surface_preview.png"),
    }
    return row


def write_metrics(rows: list[dict[str, object]]) -> None:
    with (OUT / "final_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "final_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def split_average(rows: list[dict[str, object]], split: str, key: str) -> float:
    values = [float(r[key]) for r in rows if r["split"] == split and not math.isnan(float(r[key]))]
    return float(np.mean(values)) if values else float("nan")


def write_metric_plots(rows: list[dict[str, object]]) -> None:
    test_rows = [r for r in rows if r["split"] == "test"]
    labels = [str(r["category"]).replace("-", "\n") for r in test_rows]
    x = np.arange(len(test_rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.2, 5.2), dpi=150)
    ax.bar(x - width, [float(r["dff_mae_um"]) for r in test_rows], width, label="DFF baseline")
    ax.bar(x, [float(r["ga_dff_mae_um"]) for r in test_rows], width, label="Glare-aware DFF")
    ax.bar(x + width, [float(r["model_mae_um"]) for r in test_rows], width, label="Trained model")
    ax.set_ylabel("MAE / um")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("最终测试集误差对比")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "final_test_metrics_bar.png")
    plt.close(fig)

    gains = [float(r["model_vs_dff_gain_percent"]) for r in test_rows]
    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=150)
    colors = ["#2a9d8f" if g >= 0 else "#e76f51" for g in gains]
    ax.bar(x, gains, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MAE gain vs DFF / %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("模型相对传统 DFF 的改善率")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "final_test_gain_bar.png")
    plt.close(fig)


def fmt(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_summary_report(rows: list[dict[str, object]], history: list[dict[str, float]], elapsed_s: float) -> None:
    test_rows = [r for r in rows if r["split"] == "test"]
    p10 = next(r for r in test_rows if "P10" in str(r["sample"]))
    lines = [
        "# 最终仿真数据集训练验证结果报告",
        "",
        "本轮将用户选定效果最好的 P10 V 谷样本固定为正式测试样本，并补充山峰、山脊、阶跃、周期、A 型刃脊、复合凹坑/划痕等不同结构类型，形成 17 层焦栈条件下的多类型仿真数据集。",
        "",
        "## 数据集划分",
        "",
        f"- 训练集：{sum(1 for r in rows if r['split'] == 'train')} 个样品类型，用于 patch 级监督训练。",
        f"- 验证集：{sum(1 for r in rows if r['split'] == 'validation')} 个样品类型，用于训练过程中的泛化误差监控。",
        f"- 测试集：{sum(1 for r in rows if r['split'] == 'test')} 个样品类型，只在训练完成后评估，其中包含 `test_V谷_P10_宽谷粗糙平底`。",
        f"- 焦栈层数：{STACK_LAYERS} 层；模型输入包含焦栈、眩光风险、DFF/眩光降权 DFF 及置信度通道。",
        "",
        "## 测试集核心结果",
        "",
        "| 样品类别 | 分辨率 | DFF MAE/um | 眩光降权 DFF MAE/um | 模型 MAE/um | 模型改善率/% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in test_rows:
        lines.append(
            f"| {r['category']} | {r['resolution']} | {fmt(float(r['dff_mae_um']))} | "
            f"{fmt(float(r['ga_dff_mae_um']))} | {fmt(float(r['model_mae_um']))} | "
            f"{fmt(float(r['model_vs_dff_gain_percent']))} |"
        )
    lines += [
        "",
        "## 平均指标",
        "",
        f"- 测试集传统 DFF 平均 MAE：{fmt(split_average(rows, 'test', 'dff_mae_um'))} um。",
        f"- 测试集眩光降权 DFF 平均 MAE：{fmt(split_average(rows, 'test', 'ga_dff_mae_um'))} um。",
        f"- 测试集训练模型平均 MAE：{fmt(split_average(rows, 'test', 'model_mae_um'))} um。",
        f"- 测试集模型相对传统 DFF 的平均改善率：{fmt(split_average(rows, 'test', 'model_vs_dff_gain_percent'))}%。",
        "",
        "## P10 样品结论",
        "",
        f"- P10 样品分辨率为 {p10['resolution']}，深度范围为 {fmt(float(p10['depth_range_um']), 0)} um，17 层焦栈对应 Z 向采样步长约 {fmt(float(p10['z_step_um']))} um。",
        f"- P10 上传统 DFF MAE 为 {fmt(float(p10['dff_mae_um']))} um，模型 MAE 为 {fmt(float(p10['model_mae_um']))} um，相对改善 {fmt(float(p10['model_vs_dff_gain_percent']))}%。",
        "- P10 的宽谷、粗糙平底和强柏林噪声使其适合作为最终答辩中的代表性困难样品。",
        "",
        "## 口径说明",
        "",
        "- 本结果属于仿真域监督验证，证明“眩光/杂散光机理建模 + DFF 先验 + 小型 CNN 校正”这一路线具备可行性。",
        "- 真实样品仍以现有 DFF 重建、眩光风险分析和可视化验证为主，不声称已经完成真实工业数据的大规模深度学习训练。",
        "- 若后续继续推进，应优先补充真实标定高度或轮廓仪对照，再将仿真训练迁移到真实焦栈数据。",
        "",
        "## 生成文件",
        "",
        "- `dataset_split.csv/json`：训练集、验证集、测试集划分。",
        "- `final_metrics.csv/json`：全部样品的 DFF、眩光降权 DFF、模型指标。",
        "- `final_training_curve.png`：训练/验证曲线。",
        "- `final_test_metrics_bar.png` 与 `final_test_gain_bar.png`：测试集结果汇总图。",
        "- 各样品子目录：对比面板、误差图、三维预览图、相机与场景配置。",
        "",
        f"运行耗时约 {elapsed_s / 60:.1f} min；最后一轮验证 MAE 归一化误差为 {history[-1]['val_mae_norm']:.4f}。",
    ]
    (OUT / "最终仿真训练验证结果报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    write_dataset_split(dataset)

    print("Generating training/validation arrays...", flush=True)
    train_samples = [generate_sample_arrays(s, stack_layers=STACK_LAYERS) for _, s in dataset["train"]]
    val_samples = [generate_sample_arrays(s, stack_layers=STACK_LAYERS) for _, s in dataset["validation"]]
    channels = feature_channel_count(STACK_LAYERS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}, channels={channels}, stack_layers={STACK_LAYERS}", flush=True)
    model, history = train_model(train_samples, val_samples, device, channels)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stack_layers": STACK_LAYERS,
            "channels": channels,
            "history": history,
            "seed": SEED,
            "dataset_split": {
                split: [{"category": c, "scenario": asdict(s)} for c, s in items]
                for split, items in dataset.items()
            },
        },
        MODEL_DIR / "final_antiglare_depth_net.pt",
    )
    write_training_curve(history)

    print("Evaluating train/validation/test samples...", flush=True)
    rows: list[dict[str, object]] = []
    for split, items in dataset.items():
        for category, scenario in items:
            print(f"  {split}: {scenario.name}", flush=True)
            rows.append(evaluate_one(split, category, scenario, model, device))
    write_metrics(rows)
    write_metric_plots(rows)
    elapsed_s = time.time() - start
    write_summary_report(rows, history, elapsed_s)
    print(f"Wrote final dataset outputs to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
