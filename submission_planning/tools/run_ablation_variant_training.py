"""Protected scaffold for future ablation training runs.

The default mode is intentionally non-training: it validates the run plan,
feature masks, patch sampling, model/loss interface, and output locations.
Explicit training modes are debug-only and write only under tmp/ablation_results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_dataset_training import build_dataset, surface_scenario  # noqa: E402
from simulate_antiglare_highres_samples import DEFAULT_STACK_LAYERS, generate_sample_arrays  # noqa: E402
from train_focus_resunet_loss_experiment import (  # noqa: E402
    FocusResUNet,
    HybridDFFLoss,
    augment_features,
    upgraded_channel_count,
)


ABL_ROOT = Path("tmp/ablation_results")
DEFAULT_RUN_IDS = ["ABL-00", "ABL-03"]
PATCH_SIZE = 64
SEED = 20260619
ALLOWED_CONFIG_STATUSES = {
    "scaffold_only_no_training_run",
    "small_training_debug_run",
    "controlled_pilot_debug_run",
    "matched_training_smoke_run",
    "matched_training_full_candidate_run",
    "matched_training_longer_repeat_run",
}

TRAINING_KINDS = {
    "small_debug": {
        "default_tag": "2026-06-19_small_training_debug",
        "summary_dir": "training_runner_small_train",
        "summary_stem": "ablation_training_runner_small_train_summary",
        "console_name": "Ablation small-training debug",
        "report_status": "small_training_debug_completed",
        "config_status": "small_training_debug_run",
        "config_key": "debug_training",
        "log_title": "Small Training Debug Run",
        "summary_title": "Ablation Small-Training Debug Summary",
        "interpretation": "Small-scale debug training only. Results are not manuscript evidence and claim_eligible remains false.",
        "summary_interpretation": "Explicit small-scale debug training summary. These results are not manuscript evidence.",
    },
    "controlled_pilot": {
        "default_tag": "2026-06-19_controlled_pilot",
        "summary_dir": "training_runner_controlled_pilot",
        "summary_stem": "2026-06-19_controlled_pilot_summary",
        "console_name": "Ablation controlled pilot",
        "report_status": "controlled_pilot_debug_completed",
        "config_status": "controlled_pilot_debug_run",
        "config_key": "pilot_training",
        "log_title": "Controlled Pilot Training Run",
        "summary_title": "Ablation Controlled Pilot Summary",
        "interpretation": "Controlled pilot debug training only. Results are not manuscript evidence and claim_eligible remains false.",
        "summary_interpretation": "Controlled ABL pilot summary. These results are not manuscript evidence.",
    },
    "matched_smoke": {
        "default_tag": "2026-06-19_matched_training_smoke",
        "summary_dir": "training_runner_matched_smoke",
        "summary_stem": "2026-06-19_matched_training_smoke_summary",
        "console_name": "Ablation matched training smoke",
        "report_status": "matched_training_smoke_completed",
        "config_status": "matched_training_smoke_run",
        "config_key": "matched_smoke_training",
        "log_title": "Matched Training Smoke Run",
        "summary_title": "Ablation Matched Training Smoke Summary",
        "metrics_stem": "2026-06-19_matched_training_history",
        "interpretation": "Matched split smoke training only. Results are not manuscript evidence and claim_eligible remains false.",
        "summary_interpretation": "Matched split smoke training summary. These results verify runner continuity only and are not manuscript evidence.",
    },
    "matched_full_candidate": {
        "default_tag": "2026-06-19_matched_training_full_candidate",
        "summary_dir": "training_runner_matched_full_candidate",
        "summary_stem": "2026-06-19_matched_training_full_candidate_summary",
        "console_name": "Ablation matched full candidate",
        "report_status": "matched_training_full_candidate_completed",
        "config_status": "matched_training_full_candidate_run",
        "config_key": "matched_full_candidate_training",
        "log_title": "Matched Full Candidate Training Run",
        "summary_title": "Ablation Matched Full Candidate Summary",
        "metrics_stem": "2026-06-19_matched_training_full_candidate_history",
        "interpretation": "Matched full candidate training under tmp only. Results remain claim-ineligible until full-split evaluation and eligibility audit pass.",
        "summary_interpretation": "Matched full candidate training summary. These results remain outside manuscript tables until full-split metrics and a separate eligibility audit pass.",
    },
    "matched_longer_repeat": {
        "default_tag": "2026-06-19_matched_training_longer_repeat",
        "summary_dir": "training_runner_matched_longer_repeat",
        "summary_stem": "2026-06-19_matched_training_longer_repeat_summary",
        "console_name": "Ablation matched longer repeat",
        "report_status": "matched_training_longer_repeat_completed",
        "config_status": "matched_training_longer_repeat_run",
        "config_key": "matched_longer_repeat_training",
        "log_title": "Matched Longer Repeat Training Run",
        "summary_title": "Ablation Matched Longer Repeat Summary",
        "metrics_stem": "2026-06-19_matched_training_longer_repeat_history",
        "interpretation": "Matched longer-budget repeat under tmp only. Results remain supervisor-review evidence until repeat evaluation and audit pass.",
        "summary_interpretation": "Matched longer-budget repeat summary. These results test whether the previous full-candidate result was budget-limited.",
    },
}

RUN_SPECS = {
    "ABL-00": {
        "variant": "Full S2R-FocusNet",
        "runner_mode": "focus_resunet_upgraded",
        "zero_channels": [],
        "status": "dry_run_supported",
    },
    "ABL-01": {
        "variant": "Direct image-to-depth",
        "runner_mode": "lower_prior_focus_stack_only",
        "zero_channels": [],
        "status": "blocked_by_architecture_decision",
    },
    "ABL-02": {
        "variant": "w/o DFF/GADFF prior",
        "runner_mode": "focus_resunet_upgraded_masked",
        "zero_channels": [34, 35, 36, 37],
        "status": "dry_run_supported",
    },
    "ABL-03": {
        "variant": "w/o focal difference",
        "runner_mode": "focus_resunet_upgraded_masked",
        "zero_channels": list(range(17, 33)),
        "status": "dry_run_supported",
    },
    "ABL-04": {
        "variant": "w/o glare cue",
        "runner_mode": "focus_resunet_upgraded_masked",
        "zero_channels": [33],
        "status": "dry_run_supported",
    },
}


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def build_p10_scenario():
    return surface_scenario(
        "test_V谷_P10_宽谷粗糙平底",
        960,
        540,
        611,
        1200,
        "v_valley",
        "perlin",
        tilt_x_um=80,
        tilt_y_um=-45,
        feature_amp_um=820,
        noise_amp_um=72,
        perlin_octaves=6,
        perlin_grid=160,
        perlin_persistence=0.60,
        valley_width=0.54,
        valley_floor=0.160,
        valley_sharpness=0.70,
        orientation_deg=-15,
        stray_level=0.20,
        roughness_base=0.40,
        f0=0.76,
    )


def center_patch(features: np.ndarray, truth: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = features.shape
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return features[:, y0 : y0 + size, x0 : x0 + size], truth[y0 : y0 + size, x0 : x0 + size]


def apply_zero_channels(features: np.ndarray, zero_channels: list[int]) -> np.ndarray:
    masked = features.copy()
    if zero_channels:
        masked[zero_channels, :, :] = 0.0
    return masked


def risky_artifacts(run_id: str) -> list[str]:
    run_root = ABL_ROOT / run_id
    risky_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".png", ".jpg", ".jpeg"}
    if not run_root.exists():
        return []
    return [str(path) for path in sorted(run_root.rglob("*")) if path.is_file() and path.suffix.lower() in risky_suffixes]


def run_config(run_id: str) -> dict[str, Any] | None:
    path = ABL_ROOT / run_id / "run_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_tag(value: str) -> str:
    tag = value.strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not tag or any(char not in allowed for char in tag):
        raise ValueError("Tag must contain only letters, numbers, underscores, and hyphens.")
    return tag


def dry_run_forward(features: np.ndarray, truth: np.ndarray, device: str) -> dict[str, Any]:
    model = FocusResUNet().to(device)
    model.eval()
    criterion = HybridDFFLoss()
    x = torch.from_numpy(features[None].astype(np.float32)).to(device)
    y = torch.from_numpy(truth[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x)
        loss, parts = criterion(out, y, x)
    grad_files = [name for name, param in model.named_parameters() if param.grad is not None]
    return {
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "output_shape": list(out.shape),
        "diagnostic_loss": float(loss.detach().cpu()),
        "diagnostic_parts": parts,
        "params_with_grad": grad_files,
    }


def patch_batch(
    features: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    _, height, width = features.shape
    for _ in range(batch_size):
        y0 = int(rng.integers(0, height - patch_size + 1))
        x0 = int(rng.integers(0, width - patch_size + 1))
        x = features[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        y = truth[None, y0 : y0 + patch_size, x0 : x0 + patch_size]
        if rng.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        if rng.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def split_patch_batch(
    samples: list[dict[str, Any]],
    rng: np.random.Generator,
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _ in range(batch_size):
        item = samples[int(rng.integers(0, len(samples)))]
        features = np.asarray(item["model_features"], dtype=np.float32)
        truth = np.asarray(item["truth"], dtype=np.float32)
        _, height, width = features.shape
        y0 = int(rng.integers(0, height - patch_size + 1))
        x0 = int(rng.integers(0, width - patch_size + 1))
        x = features[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        y = truth[None, y0 : y0 + patch_size, x0 : x0 + patch_size]
        if rng.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        if rng.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def prepared_features(run_id: str) -> tuple[str, np.ndarray, np.ndarray, list[int]]:
    spec = RUN_SPECS[run_id]
    scenario = build_p10_scenario()
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
    base = np.asarray(arrays["features"], dtype=np.float32)
    truth = np.asarray(arrays["truth"], dtype=np.float32)
    upgraded = augment_features(base)
    masked = apply_zero_channels(upgraded, list(spec["zero_channels"]))
    return scenario.name, masked, truth, list(spec["zero_channels"])


def prepare_matched_samples(
    split_items: list[tuple[str, Any]],
    run_id: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    spec = RUN_SPECS[run_id]
    prepared: list[dict[str, Any]] = []
    selected = split_items[: max_samples or None]
    for category, scenario in selected:
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        base = np.asarray(arrays["features"], dtype=np.float32)
        arrays["model_features"] = apply_zero_channels(augment_features(base), list(spec["zero_channels"]))
        arrays["category"] = category
        arrays["sample_id"] = scenario.name
        prepared.append(arrays)
    return prepared


def build_run_report(run_id: str, device: str, dry_run: bool) -> dict[str, Any]:
    if run_id not in RUN_SPECS:
        raise ValueError(f"Unsupported run id: {run_id}")
    spec = RUN_SPECS[run_id]
    checks: list[dict[str, Any]] = []
    cfg = run_config(run_id)
    checks.append(check(f"{run_id} run_config exists", cfg is not None, str(ABL_ROOT / run_id / "run_config.json")))
    if cfg is not None:
        checks.append(check(f"{run_id} claim_eligible false", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
        checks.append(check(f"{run_id} safe config status", cfg.get("status") in ALLOWED_CONFIG_STATUSES, str(cfg.get("status"))))
    artifacts = risky_artifacts(run_id)
    checks.append(check(f"{run_id} no existing risky artifacts", not artifacts, f"artifact_files={artifacts}", "warning"))

    if spec["status"] != "dry_run_supported":
        checks.append(check(f"{run_id} architecture gate", True, spec["status"]))
        return {
            "run_id": run_id,
            "variant": spec["variant"],
            "runner_mode": spec["runner_mode"],
            "dry_run": dry_run,
            "status": "skipped_by_design_gate",
            "checks": checks,
            "interpretation": "No training or forward pass was run for this variant.",
        }

    scenario = build_p10_scenario()
    arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
    base = np.asarray(arrays["features"], dtype=np.float32)
    truth = np.asarray(arrays["truth"], dtype=np.float32)
    upgraded = augment_features(base)
    patch, truth_patch = center_patch(upgraded, truth, PATCH_SIZE)
    masked = apply_zero_channels(patch, list(spec["zero_channels"]))
    target = masked[spec["zero_channels"], :, :] if spec["zero_channels"] else np.empty((0,), dtype=np.float32)
    checks.append(check(f"{run_id} upgraded feature count", upgraded.shape[0] == upgraded_channel_count(), str(upgraded.shape)))
    checks.append(check(f"{run_id} patch shape", list(masked.shape) == [38, PATCH_SIZE, PATCH_SIZE], str(masked.shape)))
    checks.append(check(f"{run_id} target channels zeroed", bool(target.size == 0 or np.max(np.abs(target)) == 0.0), f"zero_channels={spec['zero_channels']}"))
    forward = dry_run_forward(masked, truth_patch, device)
    checks.append(check(f"{run_id} forward output shape", forward["output_shape"] == [1, 1, PATCH_SIZE, PATCH_SIZE], str(forward["output_shape"])))
    checks.append(check(f"{run_id} diagnostic loss finite", np.isfinite(forward["diagnostic_loss"]), str(forward["diagnostic_loss"])))
    checks.append(check(f"{run_id} no gradients accumulated", not forward["params_with_grad"], str(forward["params_with_grad"])))
    return {
        "run_id": run_id,
        "variant": spec["variant"],
        "runner_mode": spec["runner_mode"],
        "dry_run": dry_run,
        "status": "dry_run_passed",
        "sample_id": scenario.name,
        "patch_size": PATCH_SIZE,
        "zero_channels": spec["zero_channels"],
        "forward": forward,
        "checks": checks,
        "interpretation": "Dry-run only. No optimizer, backward pass, checkpoint, prediction file, figure, metric result, or claim update was produced.",
    }


def build_training_report(
    run_id: str,
    device: str,
    max_epochs: int,
    train_patches: int,
    val_patches: int,
    batch_size: int,
    learning_rate: float,
    run_kind: str,
    tag: str,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> dict[str, Any]:
    if run_id not in RUN_SPECS:
        raise ValueError(f"Unsupported run id: {run_id}")
    kind_cfg = TRAINING_KINDS[run_kind]
    spec = RUN_SPECS[run_id]
    checks: list[dict[str, Any]] = []
    cfg = run_config(run_id)
    checks.append(check(f"{run_id} run_config exists", cfg is not None, str(ABL_ROOT / run_id / "run_config.json")))
    if cfg is not None:
        checks.append(check(f"{run_id} claim_eligible false before training", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
        checks.append(check(f"{run_id} safe config status before training", cfg.get("status") in ALLOWED_CONFIG_STATUSES, str(cfg.get("status"))))
    if spec["status"] != "dry_run_supported":
        checks.append(check(f"{run_id} training gate", False, spec["status"]))
        return {
            "run_id": run_id,
            "variant": spec["variant"],
            "runner_mode": spec["runner_mode"],
            "status": "training_skipped_by_design_gate",
            "checks": checks,
            "interpretation": "No training was run for this variant.",
        }

    start = time.time()
    rng = np.random.default_rng(SEED + int(run_id.split("-")[1]))
    torch.manual_seed(SEED + int(run_id.split("-")[1]))
    split_counts: dict[str, int] | None = None
    train_samples: list[dict[str, Any]] | None = None
    val_samples: list[dict[str, Any]] | None = None
    features: np.ndarray | None = None
    truth: np.ndarray | None = None
    zero_channels = list(spec["zero_channels"])
    if run_kind in {"matched_smoke", "matched_full_candidate", "matched_longer_repeat"}:
        dataset = build_dataset()
        split_counts = {name: len(items) for name, items in dataset.items()}
        train_samples = prepare_matched_samples(dataset["train"], run_id, max_train_samples)
        val_samples = prepare_matched_samples(dataset["validation"], run_id, max_val_samples)
        sample_id = "matched_train_validation_split"
        checks.append(check(f"{run_id} matched train split count", split_counts.get("train") == 27, str(split_counts.get("train"))))
        checks.append(check(f"{run_id} matched validation split count", split_counts.get("validation") == 10, str(split_counts.get("validation"))))
        checks.append(check(f"{run_id} matched test split count", split_counts.get("test") == 7, str(split_counts.get("test"))))
        checks.append(check(f"{run_id} prepared matched train samples", bool(train_samples), f"prepared={len(train_samples)}"))
        checks.append(check(f"{run_id} prepared matched validation samples", bool(val_samples), f"prepared={len(val_samples)}"))
    else:
        sample_id, features, truth, zero_channels = prepared_features(run_id)
    model = FocusResUNet().to(device)
    criterion = HybridDFFLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.5e-4)
    history: list[dict[str, float]] = []

    train_steps = max(1, int(np.ceil(train_patches / batch_size)))
    val_steps = max(1, int(np.ceil(val_patches / batch_size)))
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for _ in range(train_steps):
            if train_samples is not None:
                xb, yb = split_patch_batch(train_samples, rng, batch_size, PATCH_SIZE)
            else:
                assert features is not None and truth is not None
                xb, yb = patch_batch(features, truth, rng, batch_size, PATCH_SIZE)
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss, parts = criterion(pred, yb, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += parts["total"]

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                if val_samples is not None:
                    xb, yb = split_patch_batch(val_samples, rng, batch_size, PATCH_SIZE)
                else:
                    assert features is not None and truth is not None
                    xb, yb = patch_batch(features, truth, rng, batch_size, PATCH_SIZE)
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss, parts = criterion(pred, yb, xb)
                val_loss += parts["total"]
                val_mae += float(torch.mean(torch.abs(pred - yb)).detach().cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_debug": train_loss / train_steps,
                "val_loss_debug": val_loss / val_steps,
                "val_mae_norm_debug": val_mae / val_steps,
            }
        )

    elapsed_s = time.time() - start
    run_root = ABL_ROOT / run_id
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{tag}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_id": run_id,
            "variant": spec["variant"],
            "runner_mode": spec["runner_mode"],
            "debug_only": True,
            "claim_eligible": False,
            "run_kind": run_kind,
            "tag": tag,
            "sample_id": sample_id,
            "zero_channels": zero_channels,
            "history": history,
            "training": {
                "max_epochs": max_epochs,
                "train_patches": train_patches,
                "val_patches": val_patches,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "patch_size": PATCH_SIZE,
                "device": device,
                "split_counts": split_counts,
                "prepared_train_samples": len(train_samples) if train_samples is not None else None,
                "prepared_validation_samples": len(val_samples) if val_samples is not None else None,
            },
        },
        checkpoint_path,
    )
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if kind_cfg.get("metrics_stem"):
        metrics_stem = str(kind_cfg["metrics_stem"])
    elif run_kind == "small_debug" and tag == TRAINING_KINDS["small_debug"]["default_tag"]:
        metrics_stem = "small_training_debug_metrics"
    else:
        metrics_stem = f"{tag}_metrics"
    metrics_csv = metrics_dir / f"{metrics_stem}.csv"
    metrics_json = metrics_dir / f"{metrics_stem}.json"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metrics_json.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg is not None:
        cfg["status"] = kind_cfg["config_status"]
        cfg["claim_eligible"] = False
        cfg["main_table_eligible"] = False
        cfg[kind_cfg["config_key"]] = {
            "date": "2026-06-19",
            "debug_only": True,
            "run_kind": run_kind,
            "tag": tag,
            "sample_id": sample_id,
            "checkpoint": str(checkpoint_path),
            "metrics_csv": str(metrics_csv),
            "max_epochs": max_epochs,
            "train_patches": train_patches,
            "val_patches": val_patches,
            "batch_size": batch_size,
            "split_counts": split_counts,
            "prepared_train_samples": len(train_samples) if train_samples is not None else None,
            "prepared_validation_samples": len(val_samples) if val_samples is not None else None,
            "interpretation": kind_cfg["interpretation"],
        }
        (run_root / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    checks.append(check(f"{run_id} checkpoint under tmp", str(checkpoint_path).startswith(str(run_root)), str(checkpoint_path)))
    checks.append(check(f"{run_id} debug metrics CSV exists", metrics_csv.exists(), str(metrics_csv)))
    checks.append(check(f"{run_id} debug metrics JSON exists", metrics_json.exists(), str(metrics_json)))
    checks.append(check(f"{run_id} history rows", len(history) == max_epochs, str(len(history))))
    checks.append(check(f"{run_id} claim_eligible remains false", run_config(run_id).get("claim_eligible") is False if run_config(run_id) else False, str(run_config(run_id).get("claim_eligible") if run_config(run_id) else None)))
    return {
        "run_id": run_id,
        "variant": spec["variant"],
        "runner_mode": spec["runner_mode"],
        "status": kind_cfg["report_status"],
        "run_kind": run_kind,
        "tag": tag,
        "sample_id": sample_id,
        "split_counts": split_counts,
        "prepared_train_samples": len(train_samples) if train_samples is not None else None,
        "prepared_validation_samples": len(val_samples) if val_samples is not None else None,
        "zero_channels": zero_channels,
        "checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "history": history,
        "elapsed_s": elapsed_s,
        "checks": checks,
        "interpretation": kind_cfg["interpretation"],
    }


def write_run_log(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {report['run_id']} Training Runner Dry Run",
        "",
        f"- Status: {report['status']}",
        f"- Variant: {report['variant']}",
        f"- Runner mode: {report['runner_mode']}",
        f"- Dry run: {report['dry_run']}",
        "",
        report["interpretation"],
        "",
    ]
    forward = report.get("forward")
    if forward:
        lines.extend(
            [
                "## Forward Diagnostic",
                "",
                f"- Input shape: `{forward['input_shape']}`",
                f"- Output shape: `{forward['output_shape']}`",
                f"- Diagnostic loss: `{forward['diagnostic_loss']:.8f}`",
                "",
            ]
        )
    lines.extend(["## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_training_log(path: Path, report: dict[str, Any]) -> None:
    kind_cfg = TRAINING_KINDS.get(report.get("run_kind", "small_debug"), TRAINING_KINDS["small_debug"])
    lines = [
        f"# {report['run_id']} {kind_cfg['log_title']}",
        "",
        f"- Status: {report['status']}",
        f"- Run kind: {report.get('run_kind', '')}",
        f"- Tag: {report.get('tag', '')}",
        f"- Variant: {report['variant']}",
        f"- Runner mode: {report['runner_mode']}",
        f"- Checkpoint: `{report.get('checkpoint', '')}`",
        f"- Metrics CSV: `{report.get('metrics_csv', '')}`",
        "",
        report["interpretation"],
        "",
        "## Debug History",
        "",
        "| Epoch | Train Loss | Val Loss | Val MAE Norm |",
        "|---:|---:|---:|---:|",
    ]
    for row in report.get("history", []):
        lines.append(
            f"| {int(row['epoch'])} | {row['train_loss_debug']:.8f} | "
            f"{row['val_loss_debug']:.8f} | {row['val_mae_norm_debug']:.8f} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    reports: list[dict[str, Any]],
    device: str,
    run_kind: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    checks = [check for report in reports for check in report["checks"]]
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    split_counts = next((report.get("split_counts") for report in reports if report.get("split_counts")), None)
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "device": device,
        "run_kind": run_kind,
        "tag": tag,
        "run_ids": [report["run_id"] for report in reports],
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "interpretation": "Training runner dry-run summary. No real training or result artifact was produced.",
        "split_counts": split_counts,
        "reports": reports,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Ablation Training Runner Dry-Run Summary",
        "",
        f"- Status: {summary['status']}",
        f"- Date: {summary['date']}",
        f"- Device: {summary['device']}",
        f"- Runs: {summary['run_ids']}",
        f"- Errors: {summary['error_count']}",
        f"- Warnings: {summary['warning_count']}",
        "",
        summary["interpretation"],
        "",
        "| Run | Status | Runner Mode |",
        "|---|---|---|",
    ]
    for report in summary["reports"]:
        lines.append(f"| {report['run_id']} | {report['status']} | {report['runner_mode']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_training_summary(path: Path, summary: dict[str, Any]) -> None:
    kind_cfg = TRAINING_KINDS.get(summary.get("run_kind", "small_debug"), TRAINING_KINDS["small_debug"])
    lines = [
        f"# {kind_cfg['summary_title']}",
        "",
        f"- Status: {summary['status']}",
        f"- Date: {summary['date']}",
        f"- Device: {summary['device']}",
        f"- Run kind: {summary.get('run_kind')}",
        f"- Tag: {summary.get('tag')}",
        f"- Runs: {summary['run_ids']}",
        f"- Errors: {summary['error_count']}",
        f"- Warnings: {summary['warning_count']}",
        "",
        summary["interpretation"],
        "",
    ]
    if summary.get("split_counts"):
        lines.extend([f"- Split counts: `{summary['split_counts']}`", ""])
    lines.extend(["| Run | Status | Last Val MAE Norm | Checkpoint |", "|---|---|---:|---|"])
    for report in summary["reports"]:
        history = report.get("history") or [{}]
        last = history[-1]
        val_mae = last.get("val_mae_norm_debug")
        val_text = f"{val_mae:.8f}" if isinstance(val_mae, float) else "n/a"
        lines.append(f"| {report['run_id']} | {report['status']} | {val_text} | `{report.get('checkpoint', '')}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=sorted(RUN_SPECS), help="Run id to dry-run. May be repeated.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--execute-training", action="store_true", help="Run explicitly guarded small-scale debug training.")
    parser.add_argument("--run-kind", default="small_debug", choices=sorted(TRAINING_KINDS), help="Training output and config mode.")
    parser.add_argument("--tag", help="Safe artifact tag for logs, metrics, checkpoints, and summaries.")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-patches", type=int, default=8)
    parser.add_argument("--val-patches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-train-samples", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.manual_seed(SEED)
    run_ids = args.run_id or DEFAULT_RUN_IDS
    if args.execute_training:
        tag = safe_tag(args.tag or TRAINING_KINDS[args.run_kind]["default_tag"])
        reports = [
            build_training_report(
                run_id,
                args.device,
                args.max_epochs,
                args.train_patches,
                args.val_patches,
                args.batch_size,
                args.learning_rate,
                args.run_kind,
                tag,
                args.max_train_samples,
                args.max_val_samples,
            )
            for run_id in run_ids
        ]
        for report in reports:
            log_dir = ABL_ROOT / report["run_id"] / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            json_path = log_dir / f"{tag}.json"
            md_path = log_dir / f"{tag}.md"
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            write_training_log(md_path, report)
        summary = build_summary(reports, args.device, args.run_kind, tag)
        kind_cfg = TRAINING_KINDS[args.run_kind]
        summary["interpretation"] = kind_cfg["summary_interpretation"]
        summary_dir = ABL_ROOT / kind_cfg["summary_dir"]
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_json = summary_dir / f"{kind_cfg['summary_stem']}.json"
        summary_md = summary_dir / f"{kind_cfg['summary_stem']}.md"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_training_summary(summary_md, summary)
        print(f"{kind_cfg['console_name']}: {summary['status']}")
        print(f"Runs: {', '.join(summary['run_ids'])}")
        print(f"Checks: {summary['check_count']}, errors: {summary['error_count']}, warnings: {summary['warning_count']}")
        print(f"Wrote {summary_json}")
        print(f"Wrote {summary_md}")
        return 0 if summary["status"] == "pass" else 1

    reports = [build_run_report(run_id, args.device, dry_run=True) for run_id in run_ids]
    for report in reports:
        run_root = ABL_ROOT / report["run_id"]
        log_dir = run_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        json_path = log_dir / "2026-06-19_training_runner_dry_run.json"
        md_path = log_dir / "2026-06-19_training_runner_dry_run.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_run_log(md_path, report)
    summary = build_summary(reports, args.device)
    summary_dir = ABL_ROOT / "training_runner_dry_run"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summary_dir / "ablation_training_runner_dry_run_summary.json"
    summary_md = summary_dir / "ablation_training_runner_dry_run_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(summary_md, summary)
    print(f"Ablation training runner dry run: {summary['status']}")
    print(f"Runs: {', '.join(summary['run_ids'])}")
    print(f"Checks: {summary['check_count']}, errors: {summary['error_count']}, warnings: {summary['warning_count']}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
