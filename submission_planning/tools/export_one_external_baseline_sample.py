"""Export synthetic sample packages for external DFF baseline smoke tests.

The exporter regenerates arrays from the existing simulator and writes them
under tmp/external_baseline_data by default. The default exports only one
sample. Use --split or --all explicitly for larger temporary exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUT = Path("tmp/external_baseline_data")
DEFAULT_SAMPLE = "test_V谷_P10_宽谷粗糙平底"


def add_src_to_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_dataset(repo_root: Path) -> dict[str, list[tuple[str, Any]]]:
    add_src_to_path(repo_root)
    from final_dataset_training import build_dataset  # noqa: PLC0415

    return build_dataset()


def find_scenario(dataset: dict[str, list[tuple[str, Any]]], sample_id: str) -> tuple[str, str, Any]:
    for split, items in dataset.items():
        for category, scenario in items:
            if scenario.name == sample_id:
                return split, category, scenario
    available = [scenario.name for items in dataset.values() for _, scenario in items]
    raise SystemExit(f"Sample not found: {sample_id}\nAvailable samples: {available}")


def save_png(path: Path, array: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required to export PNG frames.") from exc
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def normalize_for_png(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def export_sample(repo_root: Path, sample_id: str, out_root: Path, overwrite: bool) -> dict[str, object]:
    add_src_to_path(repo_root)
    from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: PLC0415

    dataset = load_dataset(repo_root)
    split, category, scenario = find_scenario(dataset, sample_id)
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)

    sample_dir = out_root / "samples" / scenario.name
    if sample_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Pass --overwrite to replace: {sample_dir}")
    for subdir in ("stack", "masks", "priors", "previews"):
        (sample_dir / subdir).mkdir(parents=True, exist_ok=True)

    stack = np.asarray(arrays["stack"], dtype=np.float32)
    truth = np.asarray(arrays["truth"], dtype=np.float32)
    risk = np.asarray(arrays["risk"], dtype=np.float32)
    risk_layers = np.asarray(arrays["risk_layers"], dtype=np.float32)
    dff = np.asarray(arrays["dff"], dtype=np.float32)
    gadff = np.asarray(arrays["gadff"], dtype=np.float32)
    confidence = np.asarray(arrays["confidence"], dtype=np.float32)
    ga_confidence = np.asarray(arrays["ga_confidence"], dtype=np.float32)
    focus_positions = np.asarray(arrays["focus_positions_norm"], dtype=np.float32)

    np.save(sample_dir / "height_gt.npy", truth)
    np.save(sample_dir / "focus_positions_norm.npy", focus_positions)
    np.save(sample_dir / "masks" / "high_risk_mask.npy", risk)
    np.save(sample_dir / "masks" / "risk_layers.npy", risk_layers)
    np.save(sample_dir / "priors" / "dff_depth.npy", dff)
    np.save(sample_dir / "priors" / "gadff_depth.npy", gadff)
    np.save(sample_dir / "priors" / "focus_confidence.npy", confidence)
    np.save(sample_dir / "priors" / "gadff_confidence.npy", ga_confidence)

    for idx, frame in enumerate(stack):
        save_png(sample_dir / "stack" / f"{idx:03d}.png", frame)
    save_png(sample_dir / "masks" / "high_risk_mask.png", risk)
    save_png(sample_dir / "previews" / "height_gt_norm.png", normalize_for_png(truth))
    save_png(sample_dir / "previews" / "dff_depth_norm.png", normalize_for_png(dff))
    save_png(sample_dir / "previews" / "gadff_depth_norm.png", normalize_for_png(gadff))

    meta = {
        "sample_id": scenario.name,
        "split": split,
        "category": category,
        "resolution": [int(scenario.width), int(scenario.height)],
        "stack_layers": int(stack.shape[0]),
        "focus_positions_norm": focus_positions.tolist(),
        "z_step_um": float(scenario.depth_range_um / max(stack.shape[0] - 1, 1)),
        "depth_range_um": float(scenario.depth_range_um),
        "height_unit": "normalized_0_to_1",
        "height_scale_note": "Multiply normalized height by depth_range_um for micrometer-scale errors.",
        "surface_baseline": scenario.surface_config.baseline_type if scenario.surface_config else "procedural",
        "surface_noise": scenario.surface_config.noise_type if scenario.surface_config else "procedural",
        "stray_level": float(scenario.stray_level),
        "has_height_gt": True,
        "has_focus_stack_frames": True,
        "source": "src/simulate_antiglare_highres_samples.py::generate_sample_arrays",
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sample_id": scenario.name,
        "split": split,
        "category": category,
        "width": scenario.width,
        "height": scenario.height,
        "stack_layers": int(stack.shape[0]),
        "z_step_um": meta["z_step_um"],
        "depth_range_um": scenario.depth_range_um,
        "height_unit": meta["height_unit"],
        "stack_path": str(sample_dir / "stack"),
        "gt_path": str(sample_dir / "height_gt.npy"),
        "sample_dir": str(sample_dir),
    }


def select_sample_ids(repo_root: Path, sample_id: str | None, split: str | None, export_all: bool) -> list[str]:
    dataset = load_dataset(repo_root)
    if export_all:
        return [scenario.name for items in dataset.values() for _, scenario in items]
    if split:
        if split not in dataset:
            raise SystemExit(f"Unknown split: {split}. Available: {sorted(dataset)}")
        return [scenario.name for _, scenario in dataset[split]]
    return [sample_id or DEFAULT_SAMPLE]


def write_manifest(out_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = out_root / "manifest.csv"
    fieldnames = [
        "sample_id",
        "split",
        "category",
        "width",
        "height",
        "stack_layers",
        "z_step_um",
        "depth_range_um",
        "height_unit",
        "stack_path",
        "gt_path",
        "sample_dir",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(out_root: Path, rows: list[dict[str, object]]) -> None:
    splits = sorted({str(row["split"]) for row in rows})
    run_config = {
        "purpose": "dataloader_smoke_test",
        "eligible_for_main_table": False,
        "reason": "external baselines have not been run yet",
        "exported_samples": len(rows),
        "splits": splits,
        "recommended_methods": ["DFV", "DDFFNet"],
        "scale_alignment": "height_gt is normalized; convert prediction consistently before MAE",
    }
    (out_root / "run_config_template.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--all", action="store_true", help="Export all train/validation/test samples.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sample_ids = select_sample_ids(repo_root, args.sample_id, args.split, args.all)
    rows = []
    for sample_id in sample_ids:
        row = export_sample(repo_root, sample_id, args.out_dir, args.overwrite)
        rows.append(row)
        print(f"Exported sample to {row['sample_dir']}")
    write_manifest(args.out_dir, rows)
    write_run_config(args.out_dir, rows)
    print(f"Wrote manifest for {len(rows)} sample(s) to {args.out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
