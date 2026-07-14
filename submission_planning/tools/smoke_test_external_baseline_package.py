"""Dataloader-only smoke test for exported external baseline packages.

The check validates shapes, frame count, metadata consistency, and normalized
value ranges. It does not import external baseline repositories or run models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SAMPLE_DIR = Path("tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底")


def read_frames(stack_dir: Path) -> np.ndarray:
    frame_paths = sorted(stack_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No PNG frames found under {stack_dir}")
    frames = []
    for path in frame_paths:
        image = Image.open(path).convert("L")
        frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)


def assert_range(name: str, array: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> None:
    arr_min = float(np.nanmin(array))
    arr_max = float(np.nanmax(array))
    if arr_min < lo - 1e-6 or arr_max > hi + 1e-6:
        raise ValueError(f"{name} range [{arr_min}, {arr_max}] is outside [{lo}, {hi}]")


def run_check(sample_dir: Path) -> dict[str, object]:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    frames = read_frames(sample_dir / "stack")
    height = np.load(sample_dir / "height_gt.npy")
    risk = np.load(sample_dir / "masks" / "high_risk_mask.npy")
    risk_layers = np.load(sample_dir / "masks" / "risk_layers.npy")
    dff = np.load(sample_dir / "priors" / "dff_depth.npy")
    gadff = np.load(sample_dir / "priors" / "gadff_depth.npy")
    conf = np.load(sample_dir / "priors" / "focus_confidence.npy")
    ga_conf = np.load(sample_dir / "priors" / "gadff_confidence.npy")
    focus_positions = np.load(sample_dir / "focus_positions_norm.npy")

    stack_layers = int(meta["stack_layers"])
    width, height_px = meta["resolution"]
    expected_hw = (int(height_px), int(width))
    if frames.shape != (stack_layers, *expected_hw):
        raise ValueError(f"Frame shape mismatch: {frames.shape} vs {(stack_layers, *expected_hw)}")
    for name, array in {
        "height_gt": height,
        "high_risk_mask": risk,
        "dff_depth": dff,
        "gadff_depth": gadff,
        "focus_confidence": conf,
        "gadff_confidence": ga_conf,
    }.items():
        if array.shape != expected_hw:
            raise ValueError(f"{name} shape mismatch: {array.shape} vs {expected_hw}")
    if risk_layers.shape != (stack_layers, *expected_hw):
        raise ValueError(f"risk_layers shape mismatch: {risk_layers.shape}")
    if focus_positions.shape != (stack_layers,):
        raise ValueError(f"focus_positions shape mismatch: {focus_positions.shape}")

    assert_range("frames", frames)
    assert_range("height_gt", height)
    assert_range("high_risk_mask", risk)
    assert_range("risk_layers", risk_layers)
    assert_range("dff_depth", dff)
    assert_range("gadff_depth", gadff)
    assert_range("focus_confidence", conf)
    assert_range("gadff_confidence", ga_conf)
    assert_range("focus_positions_norm", focus_positions)

    # Shape commonly expected by PyTorch focal-stack baselines.
    torch_like_gray = frames[None, :, None, :, :]
    torch_like_rgb = np.repeat(torch_like_gray, 3, axis=2)

    return {
        "sample_id": meta["sample_id"],
        "sample_dir": str(sample_dir),
        "status": "pass",
        "frames_shape": list(frames.shape),
        "height_shape": list(height.shape),
        "risk_layers_shape": list(risk_layers.shape),
        "focus_positions": focus_positions.tolist(),
        "torch_like_gray_shape": list(torch_like_gray.shape),
        "torch_like_rgb_shape": list(torch_like_rgb.shape),
        "frame_min": float(frames.min()),
        "frame_max": float(frames.max()),
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "dff_min": float(dff.min()),
        "dff_max": float(dff.max()),
        "notes": "Dataloader-only check passed. No external model was executed.",
    }


def write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# External Baseline Package Smoke Test",
        "",
        f"- Sample: `{payload['sample_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Source package: `{payload['sample_dir']}`",
        "",
        "## Shapes",
        "",
        f"- Frames: `{payload['frames_shape']}`",
        f"- Height GT: `{payload['height_shape']}`",
        f"- Risk layers: `{payload['risk_layers_shape']}`",
        f"- PyTorch-like grayscale batch: `{payload['torch_like_gray_shape']}`",
        f"- PyTorch-like RGB batch: `{payload['torch_like_rgb_shape']}`",
        "",
        "## Range Checks",
        "",
        f"- Frame range: `{payload['frame_min']:.6f}` to `{payload['frame_max']:.6f}`",
        f"- Height range: `{payload['height_min']:.6f}` to `{payload['height_max']:.6f}`",
        f"- DFF prior range: `{payload['dff_min']:.6f}` to `{payload['dff_max']:.6f}`",
        "",
        "## Interpretation",
        "",
        "The exported P10 package is readable as a focal-stack sample for a future external baseline dataloader. This check does not provide DFV/DDFFNet results and does not support a SOTA numerical claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/external_baseline_data/smoke_reports"))
    args = parser.parse_args()

    payload = run_check(args.sample_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "p10_dataloader_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir / "p10_dataloader_smoke.md", payload)
    print("Dataloader smoke test passed")
    print(args.out_dir / "p10_dataloader_smoke.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
