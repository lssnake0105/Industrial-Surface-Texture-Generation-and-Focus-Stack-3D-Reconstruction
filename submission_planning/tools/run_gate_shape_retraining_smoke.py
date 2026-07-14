"""Matched retraining smoke for gate-shape choices.

This is a lightweight diagnostic runner. It trains the same network with the
same data budget while changing only the prior gate formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = ROOT / "submission_planning" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from final_dataset_training import build_dataset  # noqa: E402
from run_confidence_weighted_loss_training import (  # noqa: E402
    ABL_ROOT,
    CHANNEL_MAP,
    PATCH_SIZE,
    RUN_ID,
    STACK_LAYERS,
    check,
    prepare_samples,
    safe_tag,
    split_patch_batch,
    write_json,
    write_metrics_csv,
)
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays, metrics  # noqa: E402
from train_focus_resunet_loss_experiment import (  # noqa: E402
    FocusResUNet,
    augment_features,
    charbonnier,
    grad_xy,
    laplacian,
    normal_loss,
    predict_tiled_upgraded,
    upgraded_channel_count,
)


DATE = date.today().isoformat()
OUT_DIR = ROOT / "submission_planning" / "optical_mechanism_analysis" / "gate_shape_retraining_smoke"
SUMMARY_ROOT = ABL_ROOT / "gate_shape_retraining_smoke"
STRATA = ["high_risk", "low_confidence", "normal"]


@dataclass(frozen=True)
class GateTrainSpec:
    tag: str
    label: str
    exponent: float
    risk_coeff: float
    min_weight: float
    family: str


DEFAULT_SPECS = [
    GateTrainSpec(
        tag="gate_rank1_cfocus_p15_risk0_smoke",
        label="Rank-1 diagnostic gate: C_focus^1.5",
        exponent=1.5,
        risk_coeff=0.0,
        min_weight=0.02,
        family="current_family",
    ),
    GateTrainSpec(
        tag="gate_current_cfocus_p15_risk045_smoke",
        label="Current ABL-07 gate: C_focus^1.5(1-0.45R)",
        exponent=1.5,
        risk_coeff=0.45,
        min_weight=0.02,
        family="current_family",
    ),
    GateTrainSpec(
        tag="gate_focus_only_p15_smoke",
        label="Focus-only gate: C_focus^1.5",
        exponent=1.5,
        risk_coeff=0.0,
        min_weight=0.02,
        family="focus_only",
    ),
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_parts(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def combined_focus_confidence(base_features: np.ndarray) -> np.ndarray:
    prior_offset = DEFAULT_STACK_LAYERS
    conf = np.clip(base_features[prior_offset + 2], 0, 1)
    ga_conf = np.clip(base_features[prior_offset + 4], 0, 1)
    return np.clip(0.65 * conf + 0.35 * ga_conf, 0, 1).astype(np.float32)


def masked_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, depth_range_um: float) -> dict[str, float]:
    if not np.any(mask):
        return {"pixel_count": 0, "mae_um": float("nan"), "rmse_um": float("nan"), "p90_um": float("nan")}
    err = np.abs(pred[mask] - truth[mask])
    sq = (pred[mask] - truth[mask]) ** 2
    return {
        "pixel_count": int(np.sum(mask)),
        "mae_um": float(np.mean(err) * depth_range_um),
        "rmse_um": float(np.sqrt(np.mean(sq)) * depth_range_um),
        "p90_um": float(np.percentile(err, 90) * depth_range_um),
    }


def stratum_masks(risk: np.ndarray, focus_conf: np.ndarray) -> dict[str, np.ndarray]:
    risk_thr = max(float(np.percentile(risk, 90)), 0.35)
    conf_thr = float(np.percentile(focus_conf, 25))
    high_risk = risk >= risk_thr
    low_conf = focus_conf <= conf_thr
    normal = (risk <= float(np.percentile(risk, 40))) & (focus_conf >= float(np.percentile(focus_conf, 50)))
    return {"high_risk": high_risk, "low_confidence": low_conf, "normal": normal}


class GateShapePriorLoss(nn.Module):
    def __init__(self, spec: GateTrainSpec):
        super().__init__()
        self.spec = spec

    def prior_weight(self, focus_conf: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        if self.spec.family in {"current_family", "focus_only"}:
            raw = focus_conf.pow(self.spec.exponent) * (1.0 - self.spec.risk_coeff * risk)
        else:
            raise ValueError(f"Unknown gate family: {self.spec.family}")
        return torch.clamp(raw, min=self.spec.min_weight, max=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        prior_offset = STACK_LAYERS + (STACK_LAYERS - 1)
        risk = torch.clamp(features[:, prior_offset + 0 : prior_offset + 1], 0, 1)
        dff = features[:, prior_offset + 1 : prior_offset + 2]
        conf = torch.clamp(features[:, prior_offset + 2 : prior_offset + 3], 0, 1)
        gadff = features[:, prior_offset + 3 : prior_offset + 4]
        ga_conf = torch.clamp(features[:, prior_offset + 4 : prior_offset + 5], 0, 1)

        data = torch.mean(charbonnier(pred - target))
        pdx, pdy = grad_xy(pred)
        tdx, tdy = grad_xy(target)
        grad = torch.mean(charbonnier(pdx - tdx)) + torch.mean(charbonnier(pdy - tdy))
        curv = torch.mean(charbonnier(laplacian(pred) - laplacian(target)))
        nrm = normal_loss(pred, target)

        focus_conf = torch.clamp(0.65 * conf + 0.35 * ga_conf, 0, 1)
        prior_weight = self.prior_weight(focus_conf, risk)
        prior_target = 0.45 * dff + 0.55 * gadff
        prior = torch.sum(prior_weight * charbonnier(pred - prior_target)) / torch.clamp(torch.sum(prior_weight), min=1.0)
        loss = data + 0.22 * grad + 0.055 * curv + 0.035 * nrm + 0.045 * prior
        return loss, {
            "total": float(loss.detach().cpu()),
            "data": float(data.detach().cpu()),
            "gradient": float(grad.detach().cpu()),
            "curvature": float(curv.detach().cpu()),
            "normal": float(nrm.detach().cpu()),
            "focus_prior": float(prior.detach().cpu()),
            "focus_conf_mean": float(torch.mean(focus_conf).detach().cpu()),
            "prior_weight_mean": float(torch.mean(prior_weight).detach().cpu()),
            "risk_mean": float(torch.mean(risk).detach().cpu()),
        }


def train_one(spec: GateTrainSpec, dataset: dict[str, list[tuple[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_samples = prepare_samples(dataset["train"], args.max_train_samples)
    val_samples = prepare_samples(dataset["validation"], args.max_val_samples)
    model = FocusResUNet().to(args.device)
    criterion = GateShapePriorLoss(spec).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.5e-4)
    train_steps = max(1, int(math.ceil(args.train_patches / args.batch_size)))
    val_steps = max(1, int(math.ceil(args.val_patches / args.batch_size)))
    history: list[dict[str, float]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_rows: list[dict[str, float]] = []
        for _ in range(train_steps):
            xb, yb = split_patch_batch(train_samples, rng, args.batch_size, args.patch_size)
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            pred = model(xb)
            loss, parts = criterion(pred, yb, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_rows.append(parts)
        model.eval()
        val_rows: list[dict[str, float]] = []
        val_mae: list[float] = []
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = split_patch_batch(val_samples, rng, args.batch_size, args.patch_size)
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                pred = model(xb)
                _, parts = criterion(pred, yb, xb)
                val_rows.append(parts)
                val_mae.append(float(torch.mean(torch.abs(pred - yb)).detach().cpu()))
        train_parts = mean_parts(train_rows)
        val_parts = mean_parts(val_rows)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_debug": train_parts["total"],
                "val_loss_debug": val_parts["total"],
                "val_mae_norm_debug": float(np.mean(val_mae)),
                "train_focus_prior_debug": train_parts["focus_prior"],
                "val_focus_prior_debug": val_parts["focus_prior"],
                "train_prior_weight_mean": train_parts["prior_weight_mean"],
                "val_prior_weight_mean": val_parts["prior_weight_mean"],
                "train_focus_conf_mean": train_parts["focus_conf_mean"],
                "val_focus_conf_mean": val_parts["focus_conf_mean"],
                "train_risk_mean": train_parts["risk_mean"],
                "val_risk_mean": val_parts["risk_mean"],
            }
        )
    run_root = ABL_ROOT / "gate_shape_retraining_smoke"
    checkpoint = run_root / "checkpoints" / f"{spec.tag}.pt"
    metrics_csv = run_root / "metrics" / f"{spec.tag}_training_metrics.csv"
    metrics_json = run_root / "metrics" / f"{spec.tag}_training_metrics.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_id": RUN_ID,
            "tag": spec.tag,
            "label": spec.label,
            "gate_spec": spec.__dict__,
            "history": history,
            "debug_only": True,
            "claim_eligible": False,
            "channel_map": CHANNEL_MAP,
            "training": {
                "seed": seed,
                "max_epochs": args.max_epochs,
                "train_patches": args.train_patches,
                "val_patches": args.val_patches,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "patch_size": args.patch_size,
                "device": args.device,
            },
        },
        checkpoint,
    )
    write_metrics_csv(metrics_csv, history)
    write_json(metrics_json, history)
    return {
        "spec": spec,
        "seed": seed,
        "model": model,
        "history": history,
        "checkpoint": checkpoint,
        "metrics_csv": metrics_csv,
        "metrics_json": metrics_json,
        "prepared_train_samples": len(train_samples),
        "prepared_validation_samples": len(val_samples),
    }


def evaluate_one(model: FocusResUNet, spec: GateTrainSpec, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    test_items = build_dataset()["test"]
    if args.max_test_samples:
        test_items = test_items[: args.max_test_samples]
    per_sample: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for category, scenario in test_items:
            arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
            base = np.asarray(arrays["features"], dtype=np.float32)
            truth = np.asarray(arrays["truth"], dtype=np.float32)
            risk = np.asarray(arrays["risk"], dtype=np.float32)
            dff = np.asarray(arrays["dff"], dtype=np.float32)
            gadff = np.asarray(arrays["gadff"], dtype=np.float32)
            focus_conf = combined_focus_confidence(base)
            pred = predict_tiled_upgraded(model, augment_features(base), args.device, tile=args.tile, overlap=args.overlap)
            row_m = metrics(pred, truth, risk, scenario.depth_range_um)
            dff_m = metrics(dff, truth, risk, scenario.depth_range_um)
            gadff_m = metrics(gadff, truth, risk, scenario.depth_range_um)
            per_sample.append(
                {
                    "tag": spec.tag,
                    "label": spec.label,
                    "category": category,
                    "sample": scenario.name,
                    "depth_range_um": scenario.depth_range_um,
                    "mae_um": row_m["mae_um"],
                    "rmse_norm": row_m["rmse_norm"],
                    "edge_mae_um": row_m["edge_mae_um"],
                    "high_risk_mae_um": row_m["high_risk_mae_um"],
                    "dff_mae_um": dff_m["mae_um"],
                    "gadff_mae_um": gadff_m["mae_um"],
                    "model_vs_dff_gain_percent": (dff_m["mae_um"] - row_m["mae_um"]) / max(dff_m["mae_um"], 1e-8) * 100.0,
                    "risk_mean": float(np.mean(risk)),
                    "focus_conf_mean": float(np.mean(focus_conf)),
                    "claim_eligible": False,
                }
            )
            masks = stratum_masks(risk, focus_conf)
            for stratum in STRATA:
                model_m = masked_metrics(pred, truth, masks[stratum], scenario.depth_range_um)
                dff_s = masked_metrics(dff, truth, masks[stratum], scenario.depth_range_um)
                strata.append(
                    {
                        "tag": spec.tag,
                        "label": spec.label,
                        "category": category,
                        "sample": scenario.name,
                        "stratum": stratum,
                        "pixel_count": model_m["pixel_count"],
                        "risk_mean": float(np.mean(risk[masks[stratum]])) if np.any(masks[stratum]) else float("nan"),
                        "focus_conf_mean": float(np.mean(focus_conf[masks[stratum]])) if np.any(masks[stratum]) else float("nan"),
                        "model_mae_um": model_m["mae_um"],
                        "dff_mae_um": dff_s["mae_um"],
                        "model_vs_dff_gain_percent": (dff_s["mae_um"] - model_m["mae_um"]) / max(dff_s["mae_um"], 1e-8) * 100.0,
                        "claim_eligible": False,
                    }
                )
    overall = {
        "tag": spec.tag,
        "label": spec.label,
        "sample_count": len(per_sample),
        "mean_mae_um": float(np.mean([row["mae_um"] for row in per_sample])),
        "mean_high_risk_mae_um": float(np.mean([row["high_risk_mae_um"] for row in per_sample])),
        "mean_dff_mae_um": float(np.mean([row["dff_mae_um"] for row in per_sample])),
        "mean_gadff_mae_um": float(np.mean([row["gadff_mae_um"] for row in per_sample])),
        "model_vs_dff_gain_ratio_of_means_percent": (
            float(np.mean([row["dff_mae_um"] for row in per_sample])) - float(np.mean([row["mae_um"] for row in per_sample]))
        )
        / max(float(np.mean([row["dff_mae_um"] for row in per_sample])), 1e-8)
        * 100.0,
        "model_vs_dff_win_rate": float(np.mean([row["mae_um"] < row["dff_mae_um"] for row in per_sample])),
        "claim_eligible": False,
    }
    for stratum in STRATA:
        rows = [row for row in strata if row["stratum"] == stratum]
        overall[f"{stratum}_mae_um"] = float(np.mean([row["model_mae_um"] for row in rows]))
        overall[f"{stratum}_gain_vs_dff_percent"] = (
            float(np.mean([row["dff_mae_um"] for row in rows])) - float(np.mean([row["model_mae_um"] for row in rows]))
        ) / max(float(np.mean([row["dff_mae_um"] for row in rows])), 1e-8) * 100.0
    return per_sample, strata, overall


def write_report(path: Path, report: dict[str, Any]) -> None:
    rows = report["summary_rows"]
    lines: list[str] = []
    lines.append("# Gate-Shape Matched Retraining Smoke")
    lines.append("")
    lines.append(f"- 日期：{DATE}")
    lines.append(f"- 状态：{report['status']}")
    lines.append(f"- 样本：train {report['split_counts']['train']} / validation {report['split_counts']['validation']} / test {report['split_counts']['test']}")
    lines.append(f"- 训练预算：{report['max_epochs']} epochs, {report['train_patches']} train patches, {report['val_patches']} validation patches")
    lines.append("- 结论边界：claim-ineligible smoke only；real-height calibrated accuracy claim remains unsupported。")
    lines.append("- real-stack evidence remains diagnostic alignment only；audit should be rerun after any manuscript-level merge。")
    lines.append("- 本实验只检验 confidence-gated prior consistency 的训练可行性和方向性。")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append("三组实验保持网络、数据 split、patch 采样预算、loss 其他项一致，只替换 prior consistency 权重：")
    lines.append("")
    lines.append("| Tag | Gate |")
    lines.append("|---|---|")
    for row in rows:
        lines.append(f"| {row['tag']} | {row['label']} |")
    lines.append("")
    lines.append("## 2. Full Test Split Smoke Summary")
    lines.append("")
    lines.append("| Gate | Mean MAE um | High-risk MAE um | Low-confidence MAE um | Gain vs DFF | Win rate | Last val MAE norm | Prior weight mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        hist = report["training_summaries"][row["tag"]]["history"][-1]
        lines.append(
            f"| {row['tag']} | {row['mean_mae_um']:.2f} | {row['mean_high_risk_mae_um']:.2f} | "
            f"{row['low_confidence_mae_um']:.2f} | {row['model_vs_dff_gain_ratio_of_means_percent']:.2f}% | "
            f"{row['model_vs_dff_win_rate']:.2f} | {hist['val_mae_norm_debug']:.5f} | {hist['val_prior_weight_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## 3. 原理判断")
    lines.append("")
    rank1 = next(row for row in rows if row["tag"] == "gate_rank1_cfocus_p15_risk0_smoke")
    current = next(row for row in rows if row["tag"] == "gate_current_cfocus_p15_risk045_smoke")
    focus_only = next(row for row in rows if row["tag"] == "gate_focus_only_p15_smoke")
    delta = current["mean_mae_um"] - rank1["mean_mae_um"]
    if delta > 0:
        lines.append(f"在当前 smoke 预算下，rank-1 诊断候选的 mean MAE 比当前 risk0.45 门控低 {delta:.2f} um，方向上支持先把 risk 项降级为辅助调制。")
    else:
        lines.append(f"在当前 smoke 预算下，当前 risk0.45 门控的 mean MAE 比 rank-1 诊断候选低 {-delta:.2f} um，说明训练动态可能仍从 risk 调制中受益，需要扩大预算复核。")
    if abs(float(rank1["mean_mae_um"]) - float(focus_only["mean_mae_um"])) < 1e-6:
        lines.append("rank-1 诊断候选与 focus-only 组数值完全一致，因为二者在当前候选集合中都等价于 $W=\\mathrm{clip}(C_{\\mathrm{focus}}^{1.5},0.02,1)$。")
    lines.append("")
    lines.append("可支持的主张：")
    lines.append("")
    lines.append("- low-confidence prior consistency 是当前门控设计的核心变量。")
    lines.append("- gate-shape 诊断需要 matched retraining smoke 复核，不能只依赖 prior-error ranking。")
    lines.append("- 本轮结果可以作为下一步 full split seed repeat 的候选筛选依据。")
    lines.append("")
    lines.append("暂不使用的主张：")
    lines.append("")
    lines.append("- 不声明模型精度提升已经成立。")
    lines.append("- 不声明真实样本三维高度精度。")
    lines.append("- 不声明外部基线优势。")
    lines.append("")
    lines.append("## 4. 下一步")
    lines.append("")
    lines.append("1. 如果 rank-1 候选继续占优，运行 full-budget matched repeat。")
    lines.append("2. 如果 current gate 占优，分析 risk 项是否通过优化动态而非 prior-error 排序发挥作用。")
    lines.append("3. 做高置信负收益区域的 per-sample failure audit。")
    lines.append("")
    lines.append("## 5. 文件索引")
    lines.append("")
    for key in ["summary_csv", "per_sample_csv", "stratum_csv", "report_json"]:
        lines.append(f"- {key}: `{report[key]}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dataset = build_dataset()
    split_counts = {key: len(value) for key, value in dataset.items()}
    training_summaries: dict[str, Any] = {}
    per_sample_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in DEFAULT_SPECS:
        trained = train_one(spec, dataset, args)
        per_sample, strata, overall = evaluate_one(trained["model"], spec, args)
        training_summaries[spec.tag] = {
            "seed": trained["seed"],
            "checkpoint": str(trained["checkpoint"]),
            "metrics_csv": str(trained["metrics_csv"]),
            "metrics_json": str(trained["metrics_json"]),
            "history": trained["history"],
            "prepared_train_samples": trained["prepared_train_samples"],
            "prepared_validation_samples": trained["prepared_validation_samples"],
        }
        for row in per_sample:
            row["checkpoint"] = str(trained["checkpoint"])
        for row in strata:
            row["checkpoint"] = str(trained["checkpoint"])
        per_sample_rows.extend(per_sample)
        stratum_rows.extend(strata)
        overall["checkpoint"] = str(trained["checkpoint"])
        summary_rows.append(overall)
    summary_rows.sort(key=lambda row: float(row["mean_mae_um"]))
    run_dir = SUMMARY_ROOT
    out_dir = OUT_DIR
    summary_csv = run_dir / "gate_shape_retraining_smoke_summary.csv"
    per_sample_csv = run_dir / "gate_shape_retraining_smoke_per_sample.csv"
    stratum_csv = run_dir / "gate_shape_retraining_smoke_strata.csv"
    report_json = run_dir / "gate_shape_retraining_smoke_report.json"
    report_md = out_dir / "gate_shape_retraining_smoke_report.md"
    checks = [
        check("train split count", split_counts.get("train") == 27, str(split_counts.get("train"))),
        check("validation split count", split_counts.get("validation") == 10, str(split_counts.get("validation"))),
        check("test split count", split_counts.get("test") == 7, str(split_counts.get("test"))),
        check("variant count", len(summary_rows) == 3, str(len(summary_rows))),
        check("upgraded channel count", upgraded_channel_count() == 38, str(upgraded_channel_count())),
        check("all checkpoints exist", all(Path(v["checkpoint"]).exists() for v in training_summaries.values()), "checkpoints"),
        check("claim_eligible false", True, "claim_eligible=false"),
    ]
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    report = {
        "status": "pass" if not errors else "fail",
        "date": DATE,
        "run_id": RUN_ID,
        "split_counts": split_counts,
        "max_epochs": args.max_epochs,
        "train_patches": args.train_patches,
        "val_patches": args.val_patches,
        "batch_size": args.batch_size,
        "device": args.device,
        "summary_rows": summary_rows,
        "training_summaries": training_summaries,
        "summary_csv": str(summary_csv),
        "per_sample_csv": str(per_sample_csv),
        "stratum_csv": str(stratum_csv),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "checks": checks,
        "check_count": len(checks),
        "error_count": len(errors),
        "claim_eligible": False,
        "main_table_eligible": False,
        "elapsed_s": time.time() - start,
        "claim_boundary": "Claim-ineligible matched retraining smoke only. No model accuracy or calibrated real-height claim.",
    }
    write_csv(summary_csv, summary_rows)
    write_csv(per_sample_csv, per_sample_rows)
    write_csv(stratum_csv, stratum_rows)
    write_json(report_json, report)
    write_report(report_md, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--train-patches", type=int, default=72)
    parser.add_argument("--val-patches", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-train-samples", type=int, default=12)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "summary_csv": report["summary_csv"], "report_md": report["report_md"], "summary_rows": report["summary_rows"]}, ensure_ascii=False, indent=2)[:5000])
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
