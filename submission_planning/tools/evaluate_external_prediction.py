"""Evaluate an external baseline prediction against an exported synthetic sample.

The evaluator is designed for future DFV/DDFFNet outputs. It accepts a .npy
prediction map, aligns it to the exported synthetic GT, and reports MAE,
edge MAE, high-risk MAE, RMSE, and P90 error. It does not train or run models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SAMPLE_DIR = Path("tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底")
DEFAULT_OUT_DIR = Path("tmp/external_baseline_results/evaluation_smoke")


def load_meta(sample_dir: Path) -> dict[str, object]:
    path = sample_dir / "meta.json"
    if not path.exists():
        raise ValueError(f"Missing meta.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError("Only .npy prediction files are supported by this lightweight evaluator.")
    pred = np.load(path)
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3 and 1 in pred.shape:
        pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Prediction must be 2D after squeeze, got shape {pred.shape}")
    return pred


def scale_prediction(pred: np.ndarray, gt: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if mode == "raw_norm":
        return pred, "prediction used as normalized height"
    if mode == "scale_to_um":
        return pred, "prediction interpreted as micrometer height; GT converted to micrometer outside"
    if mode == "minmax":
        lo = float(np.nanmin(pred))
        hi = float(np.nanmax(pred))
        if hi <= lo:
            return np.zeros_like(pred), "constant prediction minmax-aligned to zero"
        return (pred - lo) / (hi - lo), "prediction min-max aligned to normalized GT"
    if mode == "affine":
        x = pred.reshape(-1).astype(np.float64)
        y = gt.reshape(-1).astype(np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 2:
            raise ValueError("Not enough finite pixels for affine alignment.")
        design = np.stack([x[good], np.ones(good.sum())], axis=1)
        slope, intercept = np.linalg.lstsq(design, y[good], rcond=None)[0]
        return (pred * float(slope) + float(intercept)).astype(np.float32), f"affine aligned: y={slope:.6g}*x+{intercept:.6g}"
    raise ValueError(f"Unknown scale mode: {mode}")


def edge_mask_from_gt(gt: np.ndarray, percentile: float) -> np.ndarray:
    gt32 = gt.astype(np.float32)
    grad_x = cv2.Sobel(gt32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt32, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    threshold = float(np.percentile(edge, percentile))
    return edge > threshold


def high_risk_mask(risk: np.ndarray, percentile: float, floor: float) -> np.ndarray:
    threshold = max(float(np.percentile(risk, percentile)), floor)
    return risk > threshold


def metric_summary(
    pred: np.ndarray,
    gt: np.ndarray,
    high_risk: np.ndarray,
    depth_range_um: float,
    edge_percentile: float,
    high_risk_percentile: float,
    high_risk_floor: float,
) -> dict[str, float]:
    # Keep these masks synchronized with src/simulate_antiglare_highres_samples.py::metrics.
    err_norm = np.abs(pred - gt)
    err_um = err_norm * depth_range_um
    edge = edge_mask_from_gt(gt, edge_percentile)
    risk = high_risk_mask(high_risk, high_risk_percentile, high_risk_floor)
    result = {
        "mae_um": float(np.mean(err_um)),
        "rmse_um": float(np.sqrt(np.mean(err_um * err_um))),
        "p90_um": float(np.percentile(err_um, 90)),
        "edge_mae_um": float(np.mean(err_um[edge])) if np.any(edge) else float("nan"),
        "high_risk_mae_um": float(np.mean(err_um[risk])) if np.any(risk) else float("nan"),
        "edge_pixel_fraction": float(np.mean(edge)),
        "high_risk_pixel_fraction": float(np.mean(risk)),
        "edge_percentile": float(edge_percentile),
        "high_risk_percentile": float(high_risk_percentile),
        "high_risk_floor": float(high_risk_floor),
    }
    return result


def write_outputs(out_dir: Path, payload: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics.csv"
    flat = {
        "method": payload["method"],
        "sample_id": payload["sample_id"],
        "prediction_path": payload["prediction_path"],
        "scale_mode": payload["scale_mode"],
        "alignment_note": payload["alignment_note"],
        **payload["metrics"],
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    metrics = payload["metrics"]
    lines = [
        "# External Prediction Evaluation",
        "",
        f"- Method: `{payload['method']}`",
        f"- Sample: `{payload['sample_id']}`",
        f"- Prediction: `{payload['prediction_path']}`",
        f"- Scale mode: `{payload['scale_mode']}`",
        f"- Alignment: {payload['alignment_note']}",
        "",
        "## Metrics",
        "",
        f"- MAE: `{metrics['mae_um']:.4f} um`",
        f"- RMSE: `{metrics['rmse_um']:.4f} um`",
        f"- P90: `{metrics['p90_um']:.4f} um`",
        f"- Edge MAE: `{metrics['edge_mae_um']:.4f} um`",
        f"- High-risk MAE: `{metrics['high_risk_mae_um']:.4f} um`",
        "",
        "## Boundary",
        "",
        "This evaluates one exported synthetic sample only. It is useful for validating the metric pipeline, but it is not a full external SOTA comparison.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    sample_dir = args.sample_dir
    meta = load_meta(sample_dir)
    gt = np.load(sample_dir / "height_gt.npy").astype(np.float32)
    risk = np.load(sample_dir / "masks" / "high_risk_mask.npy").astype(np.float32)
    pred = load_prediction(args.prediction)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt.shape}")
    aligned_pred, note = scale_prediction(pred, gt, args.scale_mode)
    if args.clip:
        aligned_pred = np.clip(aligned_pred, 0.0, 1.0)
        note += "; clipped to [0, 1]"

    metrics = metric_summary(
        aligned_pred,
        gt,
        risk,
        float(meta["depth_range_um"]),
        args.edge_percentile,
        args.high_risk_percentile,
        args.high_risk_floor,
    )
    return {
        "method": args.method,
        "sample_id": meta["sample_id"],
        "sample_dir": str(sample_dir),
        "prediction_path": str(args.prediction),
        "scale_mode": args.scale_mode,
        "alignment_note": note,
        "edge_percentile": args.edge_percentile,
        "high_risk_percentile": args.high_risk_percentile,
        "high_risk_floor": args.high_risk_floor,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--method", default="external_prediction")
    parser.add_argument(
        "--scale-mode",
        choices=["raw_norm", "scale_to_um", "minmax", "affine"],
        default="raw_norm",
    )
    parser.add_argument("--edge-percentile", type=float, default=88.0)
    parser.add_argument("--high-risk-percentile", type=float, default=84.0)
    parser.add_argument("--high-risk-floor", type=float, default=0.08)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    payload = evaluate(args)
    write_outputs(args.out_dir, payload)
    metrics = payload["metrics"]
    print(
        f"{payload['method']}: MAE={metrics['mae_um']:.4f} um, "
        f"edge={metrics['edge_mae_um']:.4f} um, high-risk={metrics['high_risk_mae_um']:.4f} um"
    )
    print(args.out_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
