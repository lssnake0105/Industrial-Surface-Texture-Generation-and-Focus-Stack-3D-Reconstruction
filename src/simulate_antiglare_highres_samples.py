from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from simulate_antiglare_prototype import TinyDepthNet
from surface_sample_generator import SurfaceConfig, generate_surface
from dff_depth_direction import focus_index_to_relative_height, focus_positions_um


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结题交付包" / "05_图表与结果" / "仿真抗眩光高分辨率样本"
LOWRES_MODEL = ROOT / "结题交付包" / "05_图表与结果" / "仿真抗眩光原型" / "tiny_antiglare_depth_net.pt"
DEFAULT_STACK_LAYERS = 17

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
class CameraConfig:
    width: int
    height: int
    working_distance_mm: float = 25.0
    horizontal_fov_deg: float = 8.0
    sensor_width_mm: float = 6.4
    f_number: float = 5.6
    stack_layers: int = DEFAULT_STACK_LAYERS

    @property
    def fov_width_mm(self) -> float:
        return 2 * self.working_distance_mm * math.tan(math.radians(self.horizontal_fov_deg) / 2)

    @property
    def fov_height_mm(self) -> float:
        return self.fov_width_mm * self.height / self.width

    @property
    def focal_length_mm(self) -> float:
        return self.sensor_width_mm / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2))

    @property
    def object_pixel_um(self) -> float:
        return self.fov_width_mm * 1000 / self.width


@dataclass
class Scenario:
    name: str
    width: int
    height: int
    seed: int
    depth_range_um: float
    tilt_x_um: float
    tilt_y_um: float
    micro_amp_um: float
    pit_style: str
    pit_count: int
    scratch_count: int
    stray_level: float
    roughness_base: float
    f0: float
    surface_config: SurfaceConfig | None = None


@dataclass
class StackSensitivityCase:
    name: str
    scenario: Scenario
    stack_layers: int


@dataclass
class PatchTrainConfig:
    patch_size: int = 128
    train_patches: int = 384
    epochs: int = 8
    batch_size: int = 8
    seed: int = 20260518


SCENARIOS = [
    Scenario("360p_倾斜平面_细微刀纹", 640, 360, 101, 760, 230, -160, 6.0, "shallow_dense", 42, 12, 0.11, 0.32, 0.78),
    Scenario("360p_多尺度腐蚀凹坑", 640, 360, 102, 940, -120, 210, 9.0, "multi_scale", 34, 8, 0.15, 0.38, 0.82),
    Scenario("360p_深凹坑_抬边结构", 640, 360, 103, 1180, 290, 110, 4.0, "rimmed_crater", 18, 5, 0.13, 0.30, 0.86),
    Scenario("360p_斜面细槽_混合划痕", 640, 360, 104, 830, -260, -180, 7.5, "elongated_dents", 26, 22, 0.17, 0.34, 0.80),
    Scenario("540p_倾斜平面_微结构阵列", 960, 540, 201, 980, 330, -260, 5.5, "micro_pitting", 64, 16, 0.12, 0.36, 0.81),
    Scenario("540p_大深度范围_复合凹坑", 960, 540, 202, 1450, -380, 270, 11.0, "mixed_deep", 40, 10, 0.18, 0.42, 0.84),
    Scenario(
        "540p_A型突起刃脊_柏林噪声",
        960,
        540,
        203,
        1280,
        110,
        -90,
        0.0,
        "none",
        0,
        0,
        0.16,
        0.34,
        0.83,
        SurfaceConfig(
            name="A型突起刃脊_柏林噪声",
            width=960,
            height=540,
            depth_range_um=1280,
            baseline_type="a_ridge",
            noise_type="perlin",
            seed=203,
            tilt_x_um=110,
            tilt_y_um=-90,
            feature_amp_um=720,
            noise_amp_um=55,
            ridge_width=0.095,
        ),
    ),
]

PATCH_TRAIN_SCENARIOS = [
    Scenario("train_斜面浅坑_01", 640, 360, 301, 840, 260, -210, 7.0, "shallow_dense", 44, 14, 0.13, 0.34, 0.80),
    Scenario("train_多尺度坑_02", 640, 360, 302, 1080, -240, 230, 10.5, "multi_scale", 36, 10, 0.17, 0.39, 0.83),
    Scenario("train_抬边坑_03", 640, 360, 303, 1320, 310, 160, 5.0, "rimmed_crater", 22, 7, 0.14, 0.31, 0.86),
    Scenario("train_细槽划痕_04", 640, 360, 304, 900, -300, -190, 8.0, "elongated_dents", 30, 24, 0.19, 0.36, 0.81),
    Scenario("train_微坑阵列_05", 640, 360, 305, 980, 180, -250, 6.5, "micro_pitting", 70, 16, 0.15, 0.35, 0.82),
    Scenario("train_深复合坑_06", 640, 360, 306, 1500, -360, 280, 11.5, "mixed_deep", 42, 11, 0.20, 0.43, 0.84),
    Scenario(
        "train_A型刃脊_柏林_07",
        640,
        360,
        307,
        1260,
        140,
        -100,
        0.0,
        "none",
        0,
        0,
        0.17,
        0.35,
        0.84,
        SurfaceConfig(
            name="train_A型刃脊_柏林_07",
            width=640,
            height=360,
            depth_range_um=1260,
            baseline_type="a_ridge",
            noise_type="perlin",
            seed=307,
            tilt_x_um=140,
            tilt_y_um=-100,
            feature_amp_um=700,
            noise_amp_um=60,
            ridge_width=0.10,
        ),
    ),
]

STACK_SENSITIVITY_CASES = [
    StackSensitivityCase("9层_深凹坑", SCENARIOS[2], 9),
    StackSensitivityCase("17层_深凹坑", SCENARIOS[2], 17),
    StackSensitivityCase("25层_深凹坑", SCENARIOS[2], 25),
    StackSensitivityCase("9层_540p复合凹坑", SCENARIOS[5], 9),
    StackSensitivityCase("17层_540p复合凹坑", SCENARIOS[5], 17),
    StackSensitivityCase("25层_540p复合凹坑", SCENARIOS[5], 25),
]


def normalize01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def encode_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def save_float_image(path: Path, arr: np.ndarray, cmap: int | None = None) -> None:
    u8 = np.clip(normalize01(arr) * 255, 0, 255).astype(np.uint8)
    if cmap is not None:
        u8 = cv2.applyColorMap(u8, cmap)
    encode_image(path, u8)


def gaussian_field(rng: np.random.Generator, shape: tuple[int, int], sigma: float, scale: float = 1.0) -> np.ndarray:
    field = rng.normal(0, 1, shape).astype(np.float32)
    return cv2.GaussianBlur(field, (0, 0), sigma) * scale


def coordinate_grids(camera: CameraConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_um = np.linspace(-camera.fov_width_mm * 500, camera.fov_width_mm * 500, camera.width, dtype=np.float32)
    y_um = np.linspace(-camera.fov_height_mm * 500, camera.fov_height_mm * 500, camera.height, dtype=np.float32)
    xx_um, yy_um = np.meshgrid(x_um, y_um)
    xx = xx_um / max(float(np.max(np.abs(xx_um))), 1.0)
    yy = yy_um / max(float(np.max(np.abs(yy_um))), 1.0)
    return xx_um, yy_um, xx, yy


def add_linear_grooves(rng: np.random.Generator, z: np.ndarray, xx_um: np.ndarray, yy_um: np.ndarray, count: int, depth_scale: float) -> np.ndarray:
    out = z.copy()
    max_extent = max(float(np.max(np.abs(xx_um))), float(np.max(np.abs(yy_um))))
    for _ in range(count):
        theta = rng.uniform(0, math.pi)
        dist = xx_um * math.cos(theta) + yy_um * math.sin(theta) - rng.uniform(-0.72, 0.72) * max_extent
        along = -xx_um * math.sin(theta) + yy_um * math.cos(theta)
        width = rng.uniform(5, 22)
        length = rng.uniform(max_extent * 0.25, max_extent * 0.90)
        gate = np.exp(-(along / length) ** 8)
        out -= rng.uniform(0.4, 1.0) * depth_scale * np.exp(-(dist / width) ** 2) * gate
    return out


def add_pits(rng: np.random.Generator, z: np.ndarray, xx_um: np.ndarray, yy_um: np.ndarray, scenario: Scenario) -> np.ndarray:
    out = z.copy()
    xlim = float(np.max(np.abs(xx_um)))
    ylim = float(np.max(np.abs(yy_um)))
    for i in range(scenario.pit_count):
        cx = rng.uniform(-0.86, 0.86) * xlim
        cy = rng.uniform(-0.84, 0.84) * ylim
        if scenario.pit_style == "shallow_dense":
            rx, ry = rng.uniform(18, 55), rng.uniform(18, 55)
            depth = rng.uniform(10, 55)
            exponent = 1.25
        elif scenario.pit_style == "multi_scale":
            rx, ry = rng.uniform(18, 130), rng.uniform(18, 110)
            depth = rng.uniform(18, 130)
            exponent = rng.uniform(1.0, 2.2)
        elif scenario.pit_style == "rimmed_crater":
            rx, ry = rng.uniform(60, 190), rng.uniform(45, 170)
            depth = rng.uniform(80, 260)
            exponent = 1.45
        elif scenario.pit_style == "elongated_dents":
            rx, ry = rng.uniform(30, 210), rng.uniform(12, 55)
            if rng.random() < 0.5:
                rx, ry = ry, rx
            depth = rng.uniform(25, 130)
            exponent = 1.25
        elif scenario.pit_style == "micro_pitting":
            rx, ry = rng.uniform(10, 42), rng.uniform(10, 42)
            depth = rng.uniform(8, 65)
            exponent = 1.1
        else:
            rx, ry = rng.uniform(32, 180), rng.uniform(24, 160)
            depth = rng.uniform(35, 240)
            exponent = rng.uniform(1.0, 2.0)

        dx = (xx_um - cx) / rx
        dy = (yy_um - cy) / ry
        r2 = dx * dx + dy * dy
        bowl = np.exp(-(r2**exponent))
        out -= depth * bowl

        if scenario.pit_style in {"rimmed_crater", "mixed_deep"} or (scenario.pit_style == "multi_scale" and i % 4 == 0):
            rim = np.exp(-((np.sqrt(r2) - 1.0) / rng.uniform(0.11, 0.22)) ** 2)
            out += rng.uniform(0.10, 0.28) * depth * rim

        if scenario.pit_style in {"mixed_deep", "elongated_dents"} and i % 5 == 0:
            inner = np.exp(-(r2 / rng.uniform(0.12, 0.28)))
            out -= rng.uniform(0.20, 0.45) * depth * inner
    return out


def generate_depth_um(scenario: Scenario, camera: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    if scenario.surface_config is not None:
        cfg = scenario.surface_config
        cfg = SurfaceConfig(
            **{
                **asdict(cfg),
                "width": camera.width,
                "height": camera.height,
                "depth_range_um": scenario.depth_range_um,
                "seed": scenario.seed,
            }
        )
        return generate_surface(cfg)

    rng = np.random.default_rng(scenario.seed)
    xx_um, yy_um, xx, yy = coordinate_grids(camera)
    h, w = camera.height, camera.width

    z = scenario.depth_range_um * 0.50 + scenario.tilt_x_um * xx + scenario.tilt_y_um * yy
    z += gaussian_field(rng, (h, w), sigma=max(h, w) / 18, scale=scenario.depth_range_um * 0.030)
    z += gaussian_field(rng, (h, w), sigma=max(h, w) / 42, scale=scenario.depth_range_um * 0.012)

    angle = rng.uniform(0, math.pi)
    coord = xx_um * math.cos(angle) + yy_um * math.sin(angle)
    period_um = rng.uniform(42, 115)
    z += scenario.micro_amp_um * np.sin(2 * math.pi * coord / period_um + rng.uniform(0, 2 * math.pi))
    z += 0.42 * scenario.micro_amp_um * np.sin(2 * math.pi * coord / rng.uniform(15, 35))

    z = add_pits(rng, z, xx_um, yy_um, scenario)
    z = add_linear_grooves(rng, z, xx_um, yy_um, scenario.scratch_count, depth_scale=max(scenario.micro_amp_um * 4.2, 18))

    z -= float(np.min(z))
    current_span = float(np.max(z) - np.min(z))
    if current_span > 1e-6:
        z *= scenario.depth_range_um / current_span
    z_norm = (z / max(scenario.depth_range_um, 1e-6)).astype(np.float32)
    return z.astype(np.float32), z_norm


def surface_normals_from_um(z_um: np.ndarray, camera: CameraConfig) -> np.ndarray:
    dy_um, dx_um = np.gradient(z_um.astype(np.float32), camera.object_pixel_um, camera.object_pixel_um)
    normals = np.dstack((-dx_um, -dy_um, np.ones_like(z_um, dtype=np.float32)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6
    return normals.astype(np.float32)


def render_coaxial_metal(
    rng: np.random.Generator,
    z_um: np.ndarray,
    camera: CameraConfig,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = z_um.shape
    normals = surface_normals_from_um(z_um, camera)
    nz = np.clip(normals[:, :, 2], 0, 1)

    roughness = scenario.roughness_base + 0.22 * normalize01(gaussian_field(rng, (h, w), sigma=max(h, w) / 34))
    roughness += 0.08 * normalize01(np.abs(cv2.Laplacian(z_um, cv2.CV_32F)))
    roughness = np.clip(roughness, 0.12, 0.78).astype(np.float32)

    albedo = 0.30 + 0.20 * normalize01(gaussian_field(rng, (h, w), sigma=max(h, w) / 24))
    diffuse = albedo * (0.36 + 0.64 * nz)
    shininess = 38 + 240 * (1 - roughness) ** 2
    specular = scenario.f0 * np.power(nz, shininess) * (0.55 + 0.45 * nz)
    edge_boost = 0.08 * normalize01(np.abs(cv2.Laplacian(z_um, cv2.CV_32F)))
    radiance = normalize01(0.10 + 0.52 * diffuse + 2.2 * specular + edge_boost)

    hard_seed = np.where(specular > np.percentile(specular, 97.5), specular, 0)
    bloom = cv2.GaussianBlur(hard_seed, (0, 0), max(1.2, 0.004 * max(h, w)))
    glare = normalize01(hard_seed + 1.8 * bloom)
    return radiance.astype(np.float32), glare.astype(np.float32), roughness


def make_stray_light(rng: np.random.Generator, shape: tuple[int, int], glare: np.ndarray, level: float) -> np.ndarray:
    h, w = shape
    low = normalize01(gaussian_field(rng, shape, sigma=max(h, w) / 5))
    bloom = cv2.GaussianBlur(glare, (0, 0), max(3.0, 0.018 * max(h, w)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ghost = np.zeros(shape, dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        cx = rng.uniform(0.10, 0.90) * w
        cy = rng.uniform(0.08, 0.92) * h
        sx = rng.uniform(0.06, 0.20) * w
        sy = rng.uniform(0.02, 0.10) * h
        ghost += rng.uniform(0.10, 0.34) * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    return np.clip(level * (0.35 + 0.55 * low + 0.95 * normalize01(bloom) + 0.55 * normalize01(ghost)), 0, 1).astype(np.float32)


def bloom_like(image: np.ndarray) -> np.ndarray:
    return normalize01(cv2.GaussianBlur(image.astype(np.float32), (0, 0), max(1.5, 0.006 * max(image.shape))))


def synthesize_focus_stack(
    rng: np.random.Generator,
    z_um: np.ndarray,
    z_norm: np.ndarray,
    radiance: np.ndarray,
    glare: np.ndarray,
    camera: CameraConfig,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = z_um.shape
    focus_positions = focus_positions_um(scenario.depth_range_um, camera.stack_layers)
    sharp = np.clip(radiance + 0.10 * normalize01(np.abs(cv2.Laplacian(z_um, cv2.CV_32F))), 0, 1)
    blur_soft = cv2.GaussianBlur(sharp, (0, 0), max(0.8, 0.0025 * max(h, w)))
    blur_mid = cv2.GaussianBlur(sharp, (0, 0), max(1.5, 0.0060 * max(h, w)))
    blur_heavy = cv2.GaussianBlur(sharp, (0, 0), max(2.8, 0.0130 * max(h, w)))
    stray = make_stray_light(rng, (h, w), glare, scenario.stray_level)

    prnu = 1.0 + rng.normal(0, 0.014, (h, w)).astype(np.float32)
    row_bias = rng.normal(0, 0.010, (h, 1)).astype(np.float32)
    col_bias = rng.normal(0, 0.006, (1, w)).astype(np.float32)

    stack = []
    risk_layers = []
    dof_um = max(45.0, scenario.depth_range_um / 14.0)
    for i, focus_z in enumerate(focus_positions):
        dist = np.abs(z_um - focus_z)
        focus_weight = np.exp(-0.5 * (dist / dof_um) ** 2).astype(np.float32)
        mid_weight = np.exp(-0.5 * (dist / (dof_um * 2.4)) ** 2).astype(np.float32)
        base = focus_weight * sharp + (1 - focus_weight) * (mid_weight * blur_soft + (1 - mid_weight) * blur_mid)
        far = dist > dof_um * 2.8
        base[far] = 0.65 * base[far] + 0.35 * blur_heavy[far]

        layer_glare = glare * (0.65 + 0.30 * focus_weight) * rng.uniform(0.86, 1.18)
        layer_stray = stray * rng.uniform(0.86, 1.16)
        image = np.clip(base + 0.62 * layer_glare + layer_stray, 0, 1)
        image = image * prnu + row_bias + col_bias
        image = np.clip(image, 0, 1)

        photons = image * rng.uniform(220, 520)
        shot = rng.poisson(np.maximum(photons, 0)).astype(np.float32) / max(float(np.max(photons)), 1.0)
        read = rng.normal(0, rng.uniform(0.004, 0.010), (h, w)).astype(np.float32)
        image = np.clip(0.70 * image + 0.30 * shot + read, 0, 1)
        image = np.round(image * 255) / 255.0

        local_med = cv2.medianBlur((image * 255).astype(np.uint8), 21).astype(np.float32) / 255.0
        local_excess = image - local_med
        stray_risk = layer_stray > max(float(np.percentile(layer_stray, 82)), 0.035)
        glare_risk = (layer_glare > max(float(np.percentile(layer_glare, 92)), 0.045)) | (bloom_like(layer_glare) > 0.08)
        risk = ((image > 0.94) | ((image > 0.78) & (local_excess > 0.075)) | stray_risk | glare_risk).astype(np.float32)
        risk = cv2.GaussianBlur(risk, (0, 0), max(1.0, 0.0035 * max(h, w)))
        stack.append(image.astype(np.float32))
        risk_layers.append(np.clip(risk, 0, 1).astype(np.float32))

    return np.stack(stack, axis=0), np.stack(risk_layers, axis=0), focus_positions / max(scenario.depth_range_um, 1e-6)


def focus_maps_from_stack(stack: np.ndarray) -> np.ndarray:
    maps = []
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


def dff_depth(stack: np.ndarray, risk_layers: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    focus = focus_maps_from_stack(stack)
    if risk_layers is not None:
        focus = focus * np.clip(1.0 - 0.70 * risk_layers, 0.20, 1.0)
    idx = np.argmax(focus, axis=0)
    depth = focus_index_to_relative_height(idx, stack.shape[0])
    sorted_focus = np.sort(focus, axis=0)
    peak = sorted_focus[-1]
    second = sorted_focus[-2] if focus.shape[0] > 1 else np.zeros_like(peak)
    confidence = (peak - second) / (peak + 1e-6)
    confidence = np.clip(confidence / (np.percentile(confidence, 98.5) + 1e-6), 0, 1)
    return depth.astype(np.float32), confidence.astype(np.float32)


def features_for_model(stack: np.ndarray, risk: np.ndarray, dff: np.ndarray, conf: np.ndarray, gadff: np.ndarray, ga_conf: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [stack, risk[None, :, :], dff[None, :, :], conf[None, :, :], gadff[None, :, :], ga_conf[None, :, :]],
        axis=0,
    ).astype(np.float32)


def feature_channel_count(stack_layers: int = DEFAULT_STACK_LAYERS) -> int:
    return stack_layers + 5


def tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    pos = list(range(0, max(length - tile, 0) + 1, step))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def load_model(device: str, channels: int) -> TinyDepthNet | None:
    if not LOWRES_MODEL.exists():
        return None
    model = TinyDepthNet(channels).to(device)
    state = torch.load(LOWRES_MODEL, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        return None
    model.eval()
    return model


def generate_sample_arrays(scenario: Scenario, stack_layers: int | None = None) -> dict[str, np.ndarray | CameraConfig | Scenario]:
    camera = CameraConfig(width=scenario.width, height=scenario.height, stack_layers=stack_layers or DEFAULT_STACK_LAYERS)
    rng = np.random.default_rng(scenario.seed)
    z_um, truth = generate_depth_um(scenario, camera)
    radiance, glare, roughness = render_coaxial_metal(rng, z_um, camera, scenario)
    stack, risk_layers, focus_positions_norm = synthesize_focus_stack(rng, z_um, truth, radiance, glare, camera, scenario)
    risk = np.clip(np.mean(risk_layers, axis=0), 0, 1).astype(np.float32)
    dff, conf = dff_depth(stack)
    gadff, ga_conf = dff_depth(stack, risk_layers)
    features = features_for_model(stack, risk, dff, conf, gadff, ga_conf)
    return {
        "camera": camera,
        "scenario": scenario,
        "z_um": z_um,
        "truth": truth,
        "stack": stack,
        "risk_layers": risk_layers,
        "risk": risk,
        "dff": dff,
        "confidence": conf,
        "gadff": gadff,
        "ga_confidence": ga_conf,
        "features": features,
        "focus_positions_norm": focus_positions_norm,
        "roughness": roughness,
    }


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return torch.nn.functional.l1_loss(pred_dx, target_dx) + torch.nn.functional.l1_loss(pred_dy, target_dy)


def train_patch_model(device: str, channels: int) -> tuple[TinyDepthNet, list[dict[str, float]]]:
    cfg = PatchTrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    samples = [generate_sample_arrays(s) for s in PATCH_TRAIN_SCENARIOS]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _ in range(cfg.train_patches):
        item = samples[int(rng.integers(0, len(samples)))]
        features = item["features"]
        truth = item["truth"]
        assert isinstance(features, np.ndarray)
        assert isinstance(truth, np.ndarray)
        _, h, w = features.shape
        y0 = int(rng.integers(0, h - cfg.patch_size + 1))
        x0 = int(rng.integers(0, w - cfg.patch_size + 1))
        patch_x = features[:, y0 : y0 + cfg.patch_size, x0 : x0 + cfg.patch_size]
        patch_y = truth[None, y0 : y0 + cfg.patch_size, x0 : x0 + cfg.patch_size]
        if rng.random() < 0.5:
            patch_x = patch_x[:, :, ::-1].copy()
            patch_y = patch_y[:, :, ::-1].copy()
        if rng.random() < 0.5:
            patch_x = patch_x[:, ::-1, :].copy()
            patch_y = patch_y[:, ::-1, :].copy()
        xs.append(patch_x.astype(np.float32))
        ys.append(patch_y.astype(np.float32))

    x_train = torch.from_numpy(np.stack(xs))
    y_train = torch.from_numpy(np.stack(ys))
    model = TinyDepthNet(channels).to(device)
    if LOWRES_MODEL.exists():
        try:
            model.load_state_dict(torch.load(LOWRES_MODEL, map_location=device))
        except RuntimeError:
            pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
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
            loss = torch.nn.functional.smooth_l1_loss(pred, yb) + 0.08 * edge_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
        history.append({"epoch": epoch, "train_loss": total / x_train.shape[0]})
    model.eval()
    return model, history


def predict_tiled(model: TinyDepthNet, features: np.ndarray, device: str, tile: int = 256, overlap: int = 64) -> np.ndarray:
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
            tensor = torch.from_numpy(tile_arr[None]).to(device)
            with torch.no_grad():
                out = model(tensor).detach().cpu().numpy()[0, 0]
            out = out[: min(tile, h - y), : min(tile, w - x)]
            ww = window[: out.shape[0], : out.shape[1]]
            pred_sum[y : y + out.shape[0], x : x + out.shape[1]] += out * ww
            weight_sum[y : y + out.shape[0], x : x + out.shape[1]] += ww
    return np.clip(pred_sum / np.maximum(weight_sum, 1e-6), 0, 1).astype(np.float32)


def metrics(pred: np.ndarray, truth: np.ndarray, risk: np.ndarray, depth_range_um: float) -> dict[str, float]:
    err = np.abs(pred - truth)
    high_risk = risk > max(float(np.percentile(risk, 84)), 0.08)
    grad_x = cv2.Sobel(truth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(truth, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge_mask = edge > np.percentile(edge, 88)
    return {
        "mae_norm": float(np.mean(err)),
        "rmse_norm": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "p90_norm": float(np.percentile(err, 90)),
        "mae_um": float(np.mean(err) * depth_range_um),
        "high_risk_mae_um": float(np.mean(err[high_risk]) * depth_range_um) if np.any(high_risk) else float("nan"),
        "edge_mae_um": float(np.mean(err[edge_mask]) * depth_range_um) if np.any(edge_mask) else float("nan"),
    }


def save_sample_panel(
    sample_dir: Path,
    scenario: Scenario,
    camera: CameraConfig,
    truth: np.ndarray,
    stack: np.ndarray,
    risk: np.ndarray,
    dff: np.ndarray,
    gadff: np.ndarray,
    cnn: np.ndarray | None,
    cnn_patch: np.ndarray | None,
) -> None:
    mid = stack.shape[0] // 2
    cnn_patch_img = cnn_patch if cnn_patch is not None else np.zeros_like(truth)
    dff_err = np.abs(dff - truth)
    patch_err = np.abs(cnn_patch_img - truth)
    improvement = dff_err - patch_err
    panels = [
        ("深度真值", truth, "viridis"),
        ("代表成像帧", stack[mid], "gray"),
        ("眩光/杂散光风险", risk, "magma"),
        ("DFF基线", dff, "viridis"),
        ("眩光降权DFF", gadff, "viridis"),
        ("高分辨率patch微调", cnn_patch_img, "viridis"),
        ("DFF误差", dff_err, "inferno"),
        ("patch微调误差", patch_err, "inferno"),
        ("误差改善(DFF-patch)", improvement, "coolwarm"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.2), dpi=145)
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{scenario.name} | {camera.width}x{camera.height} | 物方像素 {camera.object_pixel_um:.2f} um/px", fontsize=13)
    fig.tight_layout()
    fig.savefig(sample_dir / "00_highres_comparison_panel.png")
    plt.close(fig)

    save_float_image(sample_dir / "01_truth_depth.png", truth, cv2.COLORMAP_VIRIDIS)
    save_float_image(sample_dir / "02_representative_frame.png", stack[mid], None)
    save_float_image(sample_dir / "03_glare_stray_risk.png", risk, cv2.COLORMAP_MAGMA)
    save_float_image(sample_dir / "04_dff_baseline.png", dff, cv2.COLORMAP_VIRIDIS)
    save_float_image(sample_dir / "05_glare_aware_dff.png", gadff, cv2.COLORMAP_VIRIDIS)
    if cnn is not None:
        save_float_image(sample_dir / "06_tinycnn_transfer.png", cnn, cv2.COLORMAP_VIRIDIS)
        save_float_image(sample_dir / "07_tinycnn_error.png", np.abs(cnn - truth), cv2.COLORMAP_INFERNO)
    if cnn_patch is not None:
        save_float_image(sample_dir / "08_patch_finetuned_cnn.png", cnn_patch, cv2.COLORMAP_VIRIDIS)
        save_float_image(sample_dir / "09_patch_finetuned_error.png", np.abs(cnn_patch - truth), cv2.COLORMAP_INFERNO)
    save_float_image(sample_dir / "10_dff_error.png", np.abs(dff - truth), cv2.COLORMAP_INFERNO)


def process_scenario(
    scenario: Scenario,
    model: TinyDepthNet | None,
    patch_model: TinyDepthNet | None,
    device: str,
) -> dict[str, float | str | int]:
    sample_dir = OUT / scenario.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    arrays = generate_sample_arrays(scenario)
    camera = arrays["camera"]
    truth = arrays["truth"]
    stack = arrays["stack"]
    risk_layers = arrays["risk_layers"]
    risk = arrays["risk"]
    dff = arrays["dff"]
    gadff = arrays["gadff"]
    features = arrays["features"]
    focus_positions_norm = arrays["focus_positions_norm"]
    assert isinstance(camera, CameraConfig)
    assert isinstance(truth, np.ndarray)
    assert isinstance(stack, np.ndarray)
    assert isinstance(risk_layers, np.ndarray)
    assert isinstance(risk, np.ndarray)
    assert isinstance(dff, np.ndarray)
    assert isinstance(gadff, np.ndarray)
    assert isinstance(features, np.ndarray)
    assert isinstance(focus_positions_norm, np.ndarray)

    cnn = None
    if model is not None:
        cnn = predict_tiled(model, features, device)
    patch_cnn = None
    if patch_model is not None:
        patch_cnn = predict_tiled(patch_model, features, device)

    save_sample_panel(sample_dir, scenario, camera, truth, stack, risk, dff, gadff, cnn, patch_cnn)
    save_3d_surface_preview(sample_dir, scenario, camera, truth, patch_cnn)
    camera_payload = {
        **asdict(camera),
        "fov_width_mm": camera.fov_width_mm,
        "fov_height_mm": camera.fov_height_mm,
        "focal_length_mm": camera.focal_length_mm,
        "object_pixel_um": camera.object_pixel_um,
        "focus_positions_norm": focus_positions_norm.tolist(),
        "focus_positions_um": (focus_positions_norm * scenario.depth_range_um).tolist(),
    }
    (sample_dir / "camera_and_scene_config.json").write_text(
        json.dumps({"camera": camera_payload, "scenario": asdict(scenario)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dff_m = metrics(dff, truth, risk, scenario.depth_range_um)
    ga_m = metrics(gadff, truth, risk, scenario.depth_range_um)
    cnn_m = metrics(cnn, truth, risk, scenario.depth_range_um) if cnn is not None else None
    patch_m = metrics(patch_cnn, truth, risk, scenario.depth_range_um) if patch_cnn is not None else None

    row: dict[str, float | str | int] = {
        "sample": scenario.name,
        "resolution": f"{scenario.width}x{scenario.height}",
        "stack_layers": camera.stack_layers,
        "z_step_um": scenario.depth_range_um / max(camera.stack_layers - 1, 1),
        "depth_range_um": scenario.depth_range_um,
        "object_pixel_um": camera.object_pixel_um,
        "working_distance_mm": camera.working_distance_mm,
        "horizontal_fov_deg": camera.horizontal_fov_deg,
        "focal_length_mm": camera.focal_length_mm,
        "stray_level": scenario.stray_level,
        "risk_area_percent": float(np.mean(risk > 0.08) * 100),
        "dff_mae_um": dff_m["mae_um"],
        "ga_dff_mae_um": ga_m["mae_um"],
        "cnn_mae_um": cnn_m["mae_um"] if cnn_m else float("nan"),
        "patch_cnn_mae_um": patch_m["mae_um"] if patch_m else float("nan"),
        "dff_high_risk_mae_um": dff_m["high_risk_mae_um"],
        "cnn_high_risk_mae_um": cnn_m["high_risk_mae_um"] if cnn_m else float("nan"),
        "patch_cnn_high_risk_mae_um": patch_m["high_risk_mae_um"] if patch_m else float("nan"),
        "dff_edge_mae_um": dff_m["edge_mae_um"],
        "cnn_edge_mae_um": cnn_m["edge_mae_um"] if cnn_m else float("nan"),
        "patch_cnn_edge_mae_um": patch_m["edge_mae_um"] if patch_m else float("nan"),
        "cnn_vs_dff_mae_gain_percent": (dff_m["mae_um"] - cnn_m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100 if cnn_m else float("nan"),
        "patch_cnn_vs_dff_mae_gain_percent": (dff_m["mae_um"] - patch_m["mae_um"]) / max(dff_m["mae_um"], 1e-6) * 100 if patch_m else float("nan"),
    }
    return row


def write_training_curve(history: list[dict[str, float]]) -> None:
    if not history:
        return
    plt.figure(figsize=(6.8, 4.0), dpi=150)
    plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Patch train loss")
    plt.title("高分辨率 patch 微调训练曲线")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "patch_finetune_training_curve.png")
    plt.close()


def save_3d_surface_preview(sample_dir: Path, scenario: Scenario, camera: CameraConfig, truth: np.ndarray, pred: np.ndarray | None) -> None:
    h, w = truth.shape
    stride = max(1, max(h, w) // 170)
    x = np.linspace(-camera.fov_width_mm / 2, camera.fov_width_mm / 2, w, dtype=np.float32)
    y = np.linspace(-camera.fov_height_mm / 2, camera.fov_height_mm / 2, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    truth_um = truth * scenario.depth_range_um

    fig = plt.figure(figsize=(13.2, 6.2), dpi=145)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        truth_um[::stride, ::stride],
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )
    ax1.set_title("三维真值表面")
    ax1.set_xlabel("X/mm")
    ax1.set_ylabel("Y/mm")
    ax1.set_zlabel("Height/um")
    ax1.view_init(elev=32, azim=-132)

    ax2 = fig.add_subplot(122, projection="3d")
    pred_um = (pred if pred is not None else truth) * scenario.depth_range_um
    ax2.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        pred_um[::stride, ::stride],
        cmap="magma",
        linewidth=0,
        antialiased=True,
    )
    ax2.set_title("模型恢复三维表面" if pred is not None else "三维真值表面")
    ax2.set_xlabel("X/mm")
    ax2.set_ylabel("Y/mm")
    ax2.set_zlabel("Height/um")
    ax2.view_init(elev=32, azim=-132)
    fig.suptitle(f"{scenario.name} 三维预览", fontsize=13)
    fig.tight_layout()
    fig.savefig(sample_dir / "11_3d_surface_preview.png")
    plt.close(fig)


def run_stack_sensitivity() -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    panel_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for case in STACK_SENSITIVITY_CASES:
        arrays = generate_sample_arrays(case.scenario, stack_layers=case.stack_layers)
        camera = arrays["camera"]
        truth = arrays["truth"]
        risk = arrays["risk"]
        dff = arrays["dff"]
        gadff = arrays["gadff"]
        stack = arrays["stack"]
        assert isinstance(camera, CameraConfig)
        assert isinstance(truth, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(dff, np.ndarray)
        assert isinstance(gadff, np.ndarray)
        assert isinstance(stack, np.ndarray)
        dff_m = metrics(dff, truth, risk, case.scenario.depth_range_um)
        ga_m = metrics(gadff, truth, risk, case.scenario.depth_range_um)
        z_step_um = case.scenario.depth_range_um / max(case.stack_layers - 1, 1)
        rows.append(
            {
                "case": case.name,
                "sample": case.scenario.name,
                "resolution": f"{case.scenario.width}x{case.scenario.height}",
                "stack_layers": case.stack_layers,
                "z_step_um": z_step_um,
                "depth_range_um": case.scenario.depth_range_um,
                "risk_area_percent": float(np.mean(risk > 0.08) * 100),
                "dff_mae_um": dff_m["mae_um"],
                "ga_dff_mae_um": ga_m["mae_um"],
                "dff_edge_mae_um": dff_m["edge_mae_um"],
                "ga_dff_edge_mae_um": ga_m["edge_mae_um"],
            }
        )
        if "复合凹坑" in case.name:
            panel_rows.append((case.name, truth, dff, np.abs(dff - truth)))

    with (OUT / "z_step_sensitivity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "z_step_sensitivity_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if panel_rows:
        fig, axes = plt.subplots(len(panel_rows), 3, figsize=(11.5, 3.4 * len(panel_rows)), dpi=145)
        if len(panel_rows) == 1:
            axes = np.expand_dims(axes, axis=0)
        for ax_row, (title, truth, dff, err) in zip(axes, panel_rows):
            for ax, img, sub, cmap in [
                (ax_row[0], truth, "真值", "viridis"),
                (ax_row[1], dff, "DFF深度", "viridis"),
                (ax_row[2], err, "DFF误差", "inferno"),
            ]:
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(f"{title} {sub}", fontsize=10)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle("Z向焦距采样步长敏感性：540p复合凹坑", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / "z_step_sensitivity_panel.png")
        plt.close(fig)
    return rows


def write_summary(
    rows: list[dict[str, float | str | int]],
    elapsed: float,
    device: str,
    patch_history: list[dict[str, float]],
    z_rows: list[dict[str, float | str | int]],
) -> None:
    csv_path = OUT / "highres_antiglare_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "highres_antiglare_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_patch_gain = float(np.nanmean([float(r["patch_cnn_vs_dff_mae_gain_percent"]) for r in rows]))
    avg_dff = float(np.nanmean([float(r["dff_mae_um"]) for r in rows]))
    avg_patch = float(np.nanmean([float(r["patch_cnn_mae_um"]) for r in rows]))
    final_loss = patch_history[-1]["train_loss"] if patch_history else float("nan")
    lines = [
        "# 高分辨率仿真抗眩光样本扩展报告",
        "",
        "本次扩展在原 64×64 合成闭环之外，新增 360p 与 540p 两档高分辨率样本，用于验证更复杂几何、较大深度范围、同轴光照和杂散光条件下的抗眩光原型表现。",
        "",
        "## 执行设置",
        "",
        f"- 推理设备：`{device}`",
        f"- 总耗时：`{elapsed:.1f} s`",
        "- 分辨率：`640×360` 与 `960×540`",
        f"- 焦距堆栈：{DEFAULT_STACK_LAYERS} 层",
        "- 光照假设：同轴光照，观察方向与主光照方向近似一致",
        "- 新增退化：杂散光 veiling glare、低频照明漂移、高光 bloom、椭圆 ghost flare、PRNU/DSNU、Poisson-Gaussian 噪声",
        "- 相机假设：工作距离 25 mm，水平视场角 8°，传感器宽度 6.4 mm，F 数 5.6",
        "- 推算结果：物方视场宽约 3.50 mm；360p 物方像素约 5.46 um/px，540p 物方像素约 3.64 um/px；等效焦距约 45.7 mm",
        f"- 额外训练：基于 7 组独立 360p 高分辨率仿真场景抽取 128×128 patch，微调 8 轮；最终训练损失约 `{final_loss:.4f}`",
        "",
        "## 样本设计",
        "",
        "| 样本 | 分辨率 | 层数 | Z步长/um | 深度范围/um | 特征 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample']} | {row['resolution']} | {int(row['stack_layers'])} | {float(row['z_step_um']):.2f} | "
            f"{float(row['depth_range_um']):.0f} | 倾斜平面/生成器样品 + {row['stray_level']:.2f} 杂散光等级 |"
        )
    lines += [
        "",
        "## 指标结果",
        "",
        "| 样本 | DFF MAE/um | GA-DFF MAE/um | 17层patch模型 MAE/um | patch相对DFF改善/% |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample']} | {float(row['dff_mae_um']):.2f} | {float(row['ga_dff_mae_um']):.2f} | "
            f"{float(row['patch_cnn_mae_um']):.2f} | {float(row['patch_cnn_vs_dff_mae_gain_percent']):.1f} |"
        )
    lines += [
        "",
        "## 综合判断",
        "",
        f"- 七组高分辨率样本平均 DFF MAE 约 `{avg_dff:.2f} um`，17层 patch 模型平均 MAE 约 `{avg_patch:.2f} um`，相对 DFF 平均改善约 `{avg_patch_gain:.1f}%`。",
        "- 旧低分辨率 TinyCNN 因输入通道数不同不再直接比较；本轮模型针对 17 层焦距堆栈重新训练。",
        "- 17层 patch 模型在大深度凹坑、复合凹坑和 A 型突起刃脊上改善明显，但在细微刀纹/细槽场景仍可能略差，说明后续还需要多尺度网络和边缘保持损失。",
        "- 同轴光照下，平缓区域更容易出现镜面强反射；杂散光使整幅图像产生低频抬升和局部 ghost flare，会降低传统 DFF 焦点曲线的峰值可信度。",
        "- 本轮主流程已统一采用 17 层焦距堆栈；对大深度范围样本比原 9 层更公平，但仍建议加入亚层峰值拟合和峰值置信度约束。",
        "",
        "## Z 向步长敏感性",
        "",
        "为回应“传统 DFF 基线步长可能过大”的问题，额外比较了 9 层、17 层、25 层焦距堆栈。深度范围不变，层数增加意味着 Z 向采样步长减小。",
        "",
        "| 样本 | 层数 | Z步长/um | DFF MAE/um | GA-DFF MAE/um |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in z_rows:
        lines.append(
            f"| {row['sample']} | {int(row['stack_layers'])} | {float(row['z_step_um']):.2f} | "
            f"{float(row['dff_mae_um']):.2f} | {float(row['ga_dff_mae_um']):.2f} |"
        )
    lines += [
        "",
        "结论：9 层堆栈确实会让大深度范围样本的 DFF 基线偏粗；对 540p 大深度复合凹坑，层数从 9 增至 17/25 后误差持续下降。但对 360p 深凹坑，17 层改善而 25 层反弹，说明在眩光/杂散光和伪峰存在时，单纯减小步长并不必然单调提升，还需要峰值置信度、亚层拟合和眩光抑制共同配合。",
        "",
        "## 后续建议",
        "",
        "- 若继续走仿真监督路线，下一步应把训练分辨率提升到 256×256 patch，并加入多尺度输入、边缘保持损失和真实样品域适配。",
        "- 若继续走物理建模路线，建议用 BlenderProc 或 Mitsuba 替换当前近似成像模型，尤其是更准确模拟同轴照明、镜头杂散光和景深。",
        "- 若继续走真实验证路线，应优先补标准台阶或已知深度凹坑样品，否则所有真实样品上的改善仍只能称为无真值诊断。",
    ]
    (OUT / "高分辨率仿真抗眩光样本扩展报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = feature_channel_count(DEFAULT_STACK_LAYERS)
    model = load_model(device, channels=channels)
    patch_model, patch_history = train_patch_model(device, channels=channels)
    write_training_curve(patch_history)
    torch.save(patch_model.state_dict(), OUT / "patch_finetuned_highres_depth_net.pt")
    rows = [process_scenario(s, model, patch_model, device) for s in SCENARIOS]
    z_rows = run_stack_sensitivity()
    elapsed = time.time() - start
    write_summary(rows, elapsed, device, patch_history, z_rows)
    print(f"Wrote high-resolution anti-glare samples to: {OUT}")


if __name__ == "__main__":
    main()
