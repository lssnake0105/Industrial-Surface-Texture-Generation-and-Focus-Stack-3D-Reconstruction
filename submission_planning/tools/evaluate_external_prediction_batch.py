"""Batch-evaluate exported external baseline predictions.

The script reads an exported-sample manifest and a prediction manifest, evaluates
all matching .npy predictions with the same metric implementation as
evaluate_external_prediction.py, and writes per-sample plus method-mean tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_external_prediction import evaluate  # noqa: E402


DEFAULT_EXPORT_MANIFEST = Path("tmp/external_baseline_data/manifest.csv")
DEFAULT_OUT_DIR = Path("tmp/external_baseline_results/batch_evaluation")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_namespace(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def find_sample_dir(export_rows: list[dict[str, str]], sample_id: str) -> Path:
    for row in export_rows:
        if row.get("sample_id") == sample_id:
            if row.get("sample_dir"):
                return Path(row["sample_dir"])
            gt_path = Path(row["gt_path"])
            return gt_path.parent
    raise ValueError(f"Sample {sample_id} not found in export manifest.")


def mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["training_setting"]), str(row["scale_mode"]))].append(row)
    summary = []
    metric_keys = ["mae_um", "rmse_um", "p90_um", "edge_mae_um", "high_risk_mae_um"]
    for (method, training_setting, scale_mode), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "method": method,
            "training_setting": training_setting,
            "scale_mode": scale_mode,
            "sample_count": len(items),
        }
        for key in metric_keys:
            out[f"mean_{key}"] = mean_or_nan([float(item[key]) for item in items])
        summary.append(out)
    return summary


def write_readme(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Batch External Prediction Evaluation",
        "",
        f"- Per-sample rows: {len(rows)}",
        f"- Method summaries: {len(summary)}",
        "",
        "## Method Mean Metrics",
        "",
        "| Method | Training | Scale | Samples | Mean MAE | Mean Edge MAE | Mean High-risk MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['training_setting']} | {row['scale_mode']} | "
            f"{row['sample_count']} | {row['mean_mae_um']:.4f} | "
            f"{row['mean_edge_mae_um']:.4f} | {row['mean_high_risk_mae_um']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a metric aggregation utility. It does not run external models. Results are eligible for manuscript comparison only when predictions cover the fixed evaluation split with documented training settings and scale alignment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--edge-percentile", type=float, default=88.0)
    parser.add_argument("--high-risk-percentile", type=float, default=84.0)
    parser.add_argument("--high-risk-floor", type=float, default=0.08)
    args = parser.parse_args()

    export_rows = read_csv(args.export_manifest)
    prediction_rows = read_csv(args.prediction_manifest)
    out_rows: list[dict[str, Any]] = []
    for pred_row in prediction_rows:
        sample_id = pred_row["sample_id"]
        sample_dir = find_sample_dir(export_rows, sample_id)
        scale_mode = pred_row.get("scale_mode") or "raw_norm"
        method = pred_row["method"]
        training_setting = pred_row.get("training_setting") or "unknown"
        payload = evaluate(
            as_namespace(
                sample_dir=sample_dir,
                prediction=Path(pred_row["prediction_path"]),
                method=method,
                scale_mode=scale_mode,
                edge_percentile=args.edge_percentile,
                high_risk_percentile=args.high_risk_percentile,
                high_risk_floor=args.high_risk_floor,
                clip=str(pred_row.get("clip", "")).lower() in {"1", "true", "yes"},
            )
        )
        metrics = payload["metrics"]
        out_rows.append(
            {
                "method": method,
                "training_setting": training_setting,
                "sample_id": sample_id,
                "prediction_path": pred_row["prediction_path"],
                "scale_mode": scale_mode,
                "alignment_note": payload["alignment_note"],
                **metrics,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(out_rows)
    write_csv(args.out_dir / "per_sample_metrics.csv", out_rows)
    write_csv(args.out_dir / "method_summary_metrics.csv", summary)
    (args.out_dir / "per_sample_metrics.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "method_summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(args.out_dir / "README.md", out_rows, summary)
    print(f"Wrote batch evaluation to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
