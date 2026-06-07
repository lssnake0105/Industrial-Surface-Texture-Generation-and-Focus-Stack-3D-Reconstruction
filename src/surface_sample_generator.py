from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib
    import numpy as np
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"缺少依赖 `{missing}`。\n"
        "你当前使用的 Python 环境没有安装本脚本需要的科学计算/绘图库。\n"
        "推荐在项目环境中执行：\n"
        f"  \"{sys.executable}\" -m pip install numpy matplotlib\n"
        "如果你要运行完整抗眩光仿真脚本，还需要安装 opencv-python 和 torch。"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "结题交付包" / "05_图表与结果" / "三维样品生成器"

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
class SurfaceConfig:
    name: str = "generated_surface"
    width: int = 640
    height: int = 360
    depth_range_um: float = 1000.0
    baseline_type: str = "ridge"
    noise_type: str = "fractal"
    seed: int = 42
    tilt_x_um: float = 0.0
    tilt_y_um: float = 0.0
    feature_amp_um: float = 420.0
    noise_amp_um: float = 35.0
    period_count: float = 8.0
    ridge_width: float = 0.13
    step_position: float = 0.0
    perlin_octaves: int = 6
    perlin_grid: int = 0
    perlin_persistence: float = 0.5
    valley_width: float = 0.36
    valley_floor: float = 0.08
    valley_sharpness: float = 1.0
    orientation_deg: float = 0.0


def normalize01(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def coordinate_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = x / max(width - 1, 1) * 2 - 1
    y = y / max(height - 1, 1) * 2 - 1
    return x, y


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    return gaussian_blur_np(rng.normal(0, 1, shape).astype(np.float32), sigma)


def resize_bilinear(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = arr.shape
    x = np.linspace(0, src_w - 1, width)
    y = np.linspace(0, src_h - 1, height)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    top = (1 - wx)[None, :] * arr[y0[:, None], x0[None, :]] + wx[None, :] * arr[y0[:, None], x1[None, :]]
    bottom = (1 - wx)[None, :] * arr[y1[:, None], x0[None, :]] + wx[None, :] * arr[y1[:, None], x1[None, :]]
    return ((1 - wy)[:, None] * top + wy[:, None] * bottom).astype(np.float32)


def gaussian_blur_np(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.25:
        return arr.astype(np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= np.sum(kernel)
    padded = np.pad(arr, ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, padded)
    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)
    return out.astype(np.float32)


def value_noise(rng: np.random.Generator, shape: tuple[int, int], grid: int) -> np.ndarray:
    h, w = shape
    coarse = rng.random((max(2, h // grid + 2), max(2, w // grid + 2))).astype(np.float32)
    resized = resize_bilinear(coarse, w, h)
    return resized * 2 - 1


def perlin_like_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    octaves: int = 5,
    base_grid: int | None = None,
    persistence: float = 0.5,
) -> np.ndarray:
    h, w = shape
    total = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    grid = base_grid if base_grid and base_grid > 0 else max(min(h, w) // 5, 8)
    for _ in range(octaves):
        total += amplitude * value_noise(rng, shape, max(4, grid))
        total_amp += amplitude
        amplitude *= persistence
        grid = max(4, grid // 2)
    return total / max(total_amp, 1e-6)


def baseline_surface(config: SurfaceConfig) -> np.ndarray:
    x, y = coordinate_grid(config.width, config.height)
    z = config.tilt_x_um * x + config.tilt_y_um * y
    t = config.baseline_type.lower()

    theta = math.radians(config.orientation_deg)
    xr = x * math.cos(theta) + y * math.sin(theta)
    yr = -x * math.sin(theta) + y * math.cos(theta)

    if t in {"mountain", "山峰"}:
        z += config.feature_amp_um * np.exp(-((x / 0.45) ** 2 + (y / 0.36) ** 2))
        z += 0.38 * config.feature_amp_um * np.exp(-(((x + 0.45) / 0.25) ** 2 + ((y - 0.25) / 0.20) ** 2))
    elif t in {"ridge", "山脊"}:
        theta = math.radians(18)
        dist = x * math.cos(theta) + y * math.sin(theta)
        z += config.feature_amp_um * np.exp(-(dist / max(config.ridge_width, 0.02)) ** 2)
    elif t in {"a_ridge", "a-blade", "a型刃脊", "blade"}:
        theta = math.radians(-8)
        dist = x * math.cos(theta) + y * math.sin(theta)
        along = -x * math.sin(theta) + y * math.cos(theta)
        triangular = np.maximum(0.0, 1.0 - np.abs(dist) / max(config.ridge_width, 0.02))
        taper = 0.58 + 0.42 * np.exp(-(along / 0.70) ** 4)
        shoulder = 0.16 * np.exp(-((np.abs(dist) - config.ridge_width * 1.20) / (config.ridge_width * 0.36 + 1e-6)) ** 2)
        z += config.feature_amp_um * triangular * taper
        z += config.feature_amp_um * shoulder
    elif t in {"step", "阶跃"}:
        transition = 0.035
        z += config.feature_amp_um * (0.5 + 0.5 * np.tanh((x - config.step_position) / transition))
    elif t in {"periodic", "周期"}:
        z += config.feature_amp_um * 0.50 * (np.sin(config.period_count * math.pi * x) + 1.0)
        z += config.feature_amp_um * 0.18 * np.sin((config.period_count * 0.55) * math.pi * (x + 0.4 * y))
    elif t in {"v_valley", "v-valley", "v谷", "v_groove", "v槽"}:
        half_width = max(config.valley_width, 0.03)
        v_profile = np.clip(np.abs(xr) / half_width, config.valley_floor, 1.0)
        v_profile = np.power(v_profile, max(config.valley_sharpness, 0.2))
        length_taper = 0.86 + 0.14 * np.exp(-(yr / 0.92) ** 8)
        shoulder = 0.10 * np.exp(-((np.abs(xr) - half_width) / max(half_width * 0.18, 1e-6)) ** 2)
        z += config.feature_amp_um * (v_profile * length_taper + shoulder)
    else:
        raise ValueError(f"Unsupported baseline_type: {config.baseline_type}")
    return z.astype(np.float32)


def noise_surface(config: SurfaceConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    shape = (config.height, config.width)
    t = config.noise_type.lower()
    if t in {"none", "无"}:
        noise = np.zeros(shape, dtype=np.float32)
    elif t in {"smooth", "gaussian", "高斯"}:
        noise = smooth_noise(rng, shape, max(config.height, config.width) / 48)
    elif t in {"fractal", "分形"}:
        noise = np.zeros(shape, dtype=np.float32)
        amp = 1.0
        sigma = max(config.height, config.width) / 12
        for _ in range(5):
            noise += amp * smooth_noise(rng, shape, sigma)
            amp *= 0.55
            sigma = max(sigma / 2.0, 1.2)
    elif t in {"perlin", "柏林"}:
        noise = perlin_like_noise(
            rng,
            shape,
            octaves=config.perlin_octaves,
            base_grid=config.perlin_grid if config.perlin_grid > 0 else None,
            persistence=config.perlin_persistence,
        )
    elif t in {"stripe", "条纹"}:
        x, y = coordinate_grid(config.width, config.height)
        theta = rng.uniform(0, math.pi)
        coord = x * math.cos(theta) + y * math.sin(theta)
        noise = np.sin(config.period_count * math.pi * coord + rng.uniform(0, 2 * math.pi))
        noise += 0.35 * smooth_noise(rng, shape, max(config.height, config.width) / 80)
    else:
        raise ValueError(f"Unsupported noise_type: {config.noise_type}")
    noise = normalize01(noise) * 2 - 1
    return (config.noise_amp_um * noise).astype(np.float32)


def generate_surface(config: SurfaceConfig) -> tuple[np.ndarray, np.ndarray]:
    z = baseline_surface(config) + noise_surface(config)
    z -= float(np.min(z))
    span = float(np.max(z) - np.min(z))
    if span > 1e-6:
        z = z / span * config.depth_range_um
    return z.astype(np.float32), (z / max(config.depth_range_um, 1e-6)).astype(np.float32)


def save_surface_outputs(config: SurfaceConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    z_um, z_norm = generate_surface(config)
    np.save(out_dir / f"{config.name}_depth_um.npy", z_um)
    np.save(out_dir / f"{config.name}_depth_norm.npy", z_norm)
    (out_dir / f"{config.name}_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(8.8, 5.2), dpi=150)
    plt.imshow(z_um, cmap="viridis")
    plt.colorbar(label="Height/um")
    plt.title(f"{config.name}: depth preview")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{config.name}_depth_preview.png")
    plt.close()

    y = np.linspace(-1, 1, config.height)
    x = np.linspace(-1, 1, config.width)
    xx, yy = np.meshgrid(x, y)
    stride = max(1, max(config.width, config.height) // 180)
    fig = plt.figure(figsize=(9.2, 6.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        z_um[::stride, ::stride],
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=180,
        ccount=180,
    )
    ax.set_title(f"{config.name}: {config.baseline_type} + {config.noise_type}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height/um")
    ax.view_init(elev=32, azim=-132)
    fig.tight_layout()
    fig.savefig(out_dir / f"{config.name}_3d_preview.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 3D surface samples.")
    parser.add_argument("--name", default="generated_surface")
    parser.add_argument("--baseline", default="ridge", choices=["mountain", "ridge", "a_ridge", "v_valley", "step", "periodic", "山峰", "山脊", "a型刃脊", "v谷", "v槽", "阶跃", "周期"])
    parser.add_argument("--noise", default="fractal", choices=["none", "smooth", "fractal", "perlin", "stripe", "无", "高斯", "分形", "柏林", "条纹"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--depth-range-um", type=float, default=1000.0)
    parser.add_argument("--feature-amp-um", type=float, default=420.0)
    parser.add_argument("--noise-amp-um", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perlin-octaves", type=int, default=6)
    parser.add_argument("--perlin-grid", type=int, default=0)
    parser.add_argument("--perlin-persistence", type=float, default=0.5)
    parser.add_argument("--valley-width", type=float, default=0.36)
    parser.add_argument("--valley-floor", type=float, default=0.08)
    parser.add_argument("--valley-sharpness", type=float, default=1.0)
    parser.add_argument("--orientation-deg", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SurfaceConfig(
        name=args.name,
        width=args.width,
        height=args.height,
        depth_range_um=args.depth_range_um,
        baseline_type=args.baseline,
        noise_type=args.noise,
        seed=args.seed,
        feature_amp_um=args.feature_amp_um,
        noise_amp_um=args.noise_amp_um,
        perlin_octaves=args.perlin_octaves,
        perlin_grid=args.perlin_grid,
        perlin_persistence=args.perlin_persistence,
        valley_width=args.valley_width,
        valley_floor=args.valley_floor,
        valley_sharpness=args.valley_sharpness,
        orientation_deg=args.orientation_deg,
    )
    save_surface_outputs(config, args.out / args.name)
    print(f"Wrote generated surface to: {args.out / args.name}")


if __name__ == "__main__":
    main()
