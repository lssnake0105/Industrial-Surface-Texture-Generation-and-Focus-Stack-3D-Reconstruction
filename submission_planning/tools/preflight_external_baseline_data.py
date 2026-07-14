"""Preflight checks for external DFF baseline data readiness.

This script reads the existing project metadata and writes a small report under
tmp/external_baseline_preflight by default. It does not export image stacks,
modify source data, download repositories, or run any model.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SPLIT = Path("论文与PPT制作项目包/03_Data/synthetic_training/dataset_split.csv")
DEFAULT_SAMPLE_ROOT = Path("论文与PPT制作项目包/06_Samples/simulated_generated_samples")
DEFAULT_OUT = Path("tmp/external_baseline_preflight")


@dataclass(frozen=True)
class SampleCheck:
    sample_id: str
    split: str
    category: str
    width: int
    height: int
    stack_layers: int
    z_step_um: float
    depth_range_um: float
    surface_baseline: str
    surface_noise: str
    stray_level: float
    sample_dir: str
    has_depth_um: bool
    has_depth_norm: bool
    png_count: int
    candidate_stack_frames: int
    has_complete_stack: bool
    readiness: str
    notes: str


def parse_resolution(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_sample_dir(sample_root: Path, sample_id: str) -> Path | None:
    target = sample_id
    if "_" in sample_id:
        # Stored generated-sample folders often omit train/test prefixes.
        parts = sample_id.split("_", 1)
        target = parts[1]

    candidates: list[Path] = []
    for path in sample_root.rglob("*"):
        if path.is_dir() and (path.name == sample_id or path.name == target):
            candidates.append(path)
    if not candidates:
        # Some generated folders use a shorter scenario name, for example
        # test_V... -> V... . Keep this as a conservative fallback.
        tail = target.replace("test_", "").replace("train_", "").replace("val_", "")
        for path in sample_root.rglob("*"):
            if path.is_dir() and tail and tail in path.name:
                candidates.append(path)
    return sorted(candidates, key=lambda p: len(str(p)))[0] if candidates else None


def check_sample(sample_root: Path, row: dict[str, str]) -> SampleCheck:
    sample_id = row["sample"]
    width, height = parse_resolution(row["resolution"])
    sample_dir = find_sample_dir(sample_root, sample_id)

    files = list(sample_dir.iterdir()) if sample_dir and sample_dir.exists() else []
    png_files = [p for p in files if p.suffix.lower() == ".png"]
    depth_um = [p for p in files if p.name.endswith("_depth_um.npy")]
    depth_norm = [p for p in files if p.name.endswith("_depth_norm.npy")]

    preview_tokens = ("preview", "panel", "overview", "3d", "depth")
    candidate_stack = [
        p
        for p in png_files
        if not any(token in p.stem.lower() for token in preview_tokens)
        and any(ch.isdigit() for ch in p.stem)
    ]
    stack_layers = int(row["stack_layers"])
    has_complete_stack = len(candidate_stack) >= stack_layers

    if has_complete_stack and depth_um:
        readiness = "ready_for_stack_baseline"
        notes = "Full frame stack and height GT appear available."
    elif depth_um:
        readiness = "gt_only"
        notes = "Height GT is available, but focus-stack frames were not found in this folder."
    elif sample_dir:
        readiness = "metadata_only"
        notes = "Sample folder exists, but height GT and complete focus stack were not found."
    else:
        readiness = "missing_sample_folder"
        notes = "No matching sample folder was found under the generated-sample root."

    return SampleCheck(
        sample_id=sample_id,
        split=row["split"],
        category=row["category"],
        width=width,
        height=height,
        stack_layers=stack_layers,
        z_step_um=float(row["z_step_um"]),
        depth_range_um=float(row["depth_range_um"]),
        surface_baseline=row["surface_baseline"],
        surface_noise=row["surface_noise"],
        stray_level=float(row["stray_level"]),
        sample_dir=str(sample_dir) if sample_dir else "",
        has_depth_um=bool(depth_um),
        has_depth_norm=bool(depth_norm),
        png_count=len(png_files),
        candidate_stack_frames=len(candidate_stack),
        has_complete_stack=has_complete_stack,
        readiness=readiness,
        notes=notes,
    )


def write_csv(path: Path, rows: list[SampleCheck]) -> None:
    fieldnames = list(SampleCheck.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_summary(path: Path, rows: list[SampleCheck], split_path: Path, sample_root: Path) -> None:
    by_readiness: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_readiness[row.readiness] = by_readiness.get(row.readiness, 0) + 1
        by_split[row.split] = by_split.get(row.split, 0) + 1

    test_rows = [row for row in rows if row.split == "test"]
    lines = [
        "# External Baseline Data Preflight",
        "",
        "Purpose: check whether the current local sample files are ready for DFV/DDFFNet-style external focus-stack baselines without modifying source data.",
        "",
        "## Inputs",
        "",
        f"- Split metadata: `{split_path}`",
        f"- Generated sample root: `{sample_root}`",
        "",
        "## Summary",
        "",
        f"- Total split rows checked: {len(rows)}",
        "- Rows by split: "
        + ", ".join(f"{key}={value}" for key, value in sorted(by_split.items())),
        "- Rows by readiness: "
        + ", ".join(f"{key}={value}" for key, value in sorted(by_readiness.items())),
        "",
        "## Test Split Readiness",
        "",
        "| Sample | Readiness | Depth GT | Candidate stack frames | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for row in test_rows:
        lines.append(
            f"| `{row.sample_id}` | {row.readiness} | {str(row.has_depth_um).lower()} | "
            f"{row.candidate_stack_frames}/{row.stack_layers} | {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `ready_for_stack_baseline`: enough local evidence for a stack-based external baseline smoke test.",
            "- `gt_only`: synthetic height GT exists, but focus-stack frames need to be regenerated or located before DFV/DDFFNet can run.",
            "- `metadata_only`: metadata exists, but both GT and complete stack assets are missing from the checked folder.",
            "- `missing_sample_folder`: no matching generated-sample folder was found.",
            "",
            "## Next Action",
            "",
            "Use `src/simulate_antiglare_highres_samples.py::generate_sample_arrays` as the safest source for regenerating stack, GT, risk maps, and priors into `tmp/external_baseline_data/` for a one-sample smoke test. Keep generated data under `tmp/` until the export format is reviewed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true", help="Also write JSON output.")
    args = parser.parse_args()

    rows = [check_sample(args.sample_root, row) for row in read_split(args.split_csv)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "preflight_manifest.csv", rows)
    write_summary(args.out_dir / "README.md", rows, args.split_csv, args.sample_root)
    if args.json:
        payload = [{name: getattr(row, name) for name in SampleCheck.__dataclass_fields__} for row in rows]
        (args.out_dir / "preflight_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Wrote preflight report to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
