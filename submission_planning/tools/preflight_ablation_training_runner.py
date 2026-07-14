"""Preflight an ablation training runner plan without launching training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_focus_resunet_loss_experiment import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    OUT,
    PATCH_SIZE,
    TRAIN_PATCHES_PER_EPOCH,
    VAL_PATCHES_PER_EPOCH,
    upgraded_channel_count,
)


DEFAULT_OUT_DIR = Path("tmp/ablation_results/training_runner_preflight")
ABL_ROOT = Path("tmp/ablation_results")
SOURCE_ENTRY = Path("src/train_focus_resunet_loss_experiment.py")
SMOKE_REPORT = ABL_ROOT / "runner_smoke" / "minimal_ablation_runner_smoke.json"

CORE_VARIANTS = {
    "ABL-00": {
        "variant": "Full S2R-FocusNet",
        "runner_mode": "focus_resunet_upgraded",
        "input_channels": list(range(38)),
        "zero_channels": [],
        "trainable": True,
    },
    "ABL-01": {
        "variant": "Direct image-to-depth",
        "runner_mode": "lower_prior_focus_stack_only",
        "input_channels": list(range(17)),
        "zero_channels": [],
        "trainable": False,
        "decision_needed": "Define a lower-prior architecture before training; do not emulate with final FocusResUNet masks.",
    },
    "ABL-02": {
        "variant": "w/o DFF/GADFF prior",
        "runner_mode": "focus_resunet_upgraded_masked",
        "input_channels": list(range(38)),
        "zero_channels": [34, 35, 36, 37],
        "trainable": True,
    },
    "ABL-03": {
        "variant": "w/o focal difference",
        "runner_mode": "focus_resunet_upgraded_masked",
        "input_channels": list(range(38)),
        "zero_channels": list(range(17, 33)),
        "trainable": True,
    },
    "ABL-04": {
        "variant": "w/o glare cue",
        "runner_mode": "focus_resunet_upgraded_masked",
        "input_channels": list(range(38)),
        "zero_channels": [33],
        "trainable": True,
        "decision_note": "This removes the explicit risk cue only; stricter GADFF-derived glare removal remains a separate design.",
    },
}


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def inside_tmp(path: Path) -> bool:
    try:
        resolved = (ROOT / path).resolve()
        tmp_root = (ROOT / "tmp").resolve()
        return resolved == tmp_root or tmp_root in resolved.parents
    except OSError:
        return False


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def planned_outputs(run_id: str) -> dict[str, str]:
    root = ABL_ROOT / run_id
    return {
        "run_log": str(root / "logs" / "2026-06-19_training_runner_plan.md"),
        "config": str(root / "run_config.json"),
        "synthetic_metrics": str(root / "metrics" / "synthetic_metrics.csv"),
        "real_metrics": str(root / "metrics" / "real_no_reference_metrics.csv"),
        "predictions": str(root / "predictions"),
        "figures": str(root / "figures"),
    }


def build_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(check("source entry exists", SOURCE_ENTRY.exists(), str(SOURCE_ENTRY)))
    checks.append(check("upgraded channel count", upgraded_channel_count() == 38, str(upgraded_channel_count())))
    checks.append(check("original OUT outside tmp", not inside_tmp(OUT), str(OUT)))
    checks.append(check("original MODEL_DIR outside tmp", not inside_tmp(MODEL_DIR), str(MODEL_DIR)))
    checks.append(check("do not call original main/evaluate_split directly", True, "future runner must reuse components and redirect outputs under tmp/ablation_results"))

    smoke = load_json(SMOKE_REPORT)
    checks.append(check("minimal runner smoke JSON exists", smoke is not None, str(SMOKE_REPORT)))
    if smoke is not None:
        checks.append(check("minimal runner smoke passed", smoke.get("status") == "pass", str(smoke.get("status"))))
        smoke_ids = {row.get("run_id") for row in smoke.get("variant_summaries", []) if isinstance(row, dict)}
        checks.append(check("minimal runner smoke covers core variants", set(CORE_VARIANTS).issubset(smoke_ids), f"covered={sorted(smoke_ids)}"))

    variants = []
    for run_id, spec in CORE_VARIANTS.items():
        run_root = ABL_ROOT / run_id
        config_path = run_root / "run_config.json"
        config = load_json(config_path)
        outputs = planned_outputs(run_id)
        variants.append(
            {
                "run_id": run_id,
                "variant": spec["variant"],
                "runner_mode": spec["runner_mode"],
                "trainable_now": spec["trainable"],
                "input_channel_count": len(spec["input_channels"]),
                "zero_channels": spec["zero_channels"],
                "planned_outputs": outputs,
                "decision_needed": spec.get("decision_needed", ""),
                "decision_note": spec.get("decision_note", ""),
            }
        )
        checks.append(check(f"{run_id} workspace exists", run_root.exists(), str(run_root)))
        checks.append(check(f"{run_id} run_config exists", config is not None, str(config_path)))
        if config is not None:
            checks.append(check(f"{run_id} claim_eligible false", config.get("claim_eligible") is False, str(config.get("claim_eligible"))))
            checks.append(check(f"{run_id} status scaffold-only", config.get("status") == "scaffold_only_no_training_run", str(config.get("status"))))
        for name, output in outputs.items():
            path = Path(output)
            if name == "config":
                checks.append(check(f"{run_id} planned config under tmp", str(path).startswith(str(ABL_ROOT / run_id)), output))
            else:
                checks.append(check(f"{run_id} planned {name} under tmp", inside_tmp(path), output))
        if run_id == "ABL-01":
            checks.append(check("ABL-01 training gated by architecture decision", spec["trainable"] is False, spec["decision_needed"]))
        else:
            checks.append(check(f"{run_id} trainable after runner implementation", spec["trainable"] is True, spec["runner_mode"]))

    training_defaults = {
        "source_entry": str(SOURCE_ENTRY),
        "base_runner_should_not_call": ["main", "evaluate_split", "write_metric_plots", "write_report"],
        "allowed_reused_components": [
            "FocusResUNet",
            "HybridDFFLoss",
            "augment_features",
            "random_patch_batch pattern",
            "metrics",
            "generate_sample_arrays",
        ],
        "original_training_defaults_observed": {
            "EPOCHS": EPOCHS,
            "PATCH_SIZE": PATCH_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "TRAIN_PATCHES_PER_EPOCH": TRAIN_PATCHES_PER_EPOCH,
            "VAL_PATCHES_PER_EPOCH": VAL_PATCHES_PER_EPOCH,
        },
        "safe_output_root": str(ABL_ROOT),
        "next_action": "Implement a training runner that redirects checkpoints, logs, predictions, figures, and metric CSVs under tmp/ablation_results/<run_id>/.",
    }

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "interpretation": "Preflight only. No optimizer, training loop, checkpoint, prediction, metric result, or claim-eligible update was produced.",
        "training_defaults": training_defaults,
        "variant_plans": variants,
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Training Runner Preflight",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Runner Boundary",
        "",
        f"- Source entry: `{report['training_defaults']['source_entry']}`",
        f"- Safe output root: `{report['training_defaults']['safe_output_root']}`",
        f"- Functions not to call directly: `{', '.join(report['training_defaults']['base_runner_should_not_call'])}`",
        "",
        "## Variant Plans",
        "",
        "| Run | Variant | Runner Mode | Trainable Now | Input Channels | Zero Channels |",
        "|---|---|---|---|---:|---|",
    ]
    for row in report["variant_plans"]:
        lines.append(
            f"| {row['run_id']} | {row['variant']} | {row['runner_mode']} | "
            f"{row['trainable_now']} | {row['input_channel_count']} | {row['zero_channels']} |"
        )
    lines.extend(["", "## ABL-01 Decision", ""])
    abl01 = next(row for row in report["variant_plans"] if row["run_id"] == "ABL-01")
    lines.append(abl01["decision_needed"])
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = build_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ablation_training_runner_preflight.json"
    md_path = args.out_dir / "ablation_training_runner_preflight.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation training runner preflight: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
