"""Preflight the full matched ablation training configuration.

This tool does not train models or evaluate checkpoints. It records the
candidate full-run budget, required outputs, evaluator boundary, and eligibility
gates before any full matched ablation run is started.
"""

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

from final_dataset_training import build_dataset  # noqa: E402

from run_ablation_variant_training import ABL_ROOT, RUN_SPECS, TRAINING_KINDS, safe_tag  # noqa: E402


DEFAULT_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]
DEFAULT_TAG = "2026-06-19_matched_training_full_candidate"
DEFAULT_OUT_DIR = ABL_ROOT / "matched_training_full_config"
SEED = 20260619
PATCH_SIZE = 64


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_config(run_id: str) -> dict[str, Any] | None:
    return read_json(ABL_ROOT / run_id / "run_config.json")


def build_variant_plan(run_id: str, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = RUN_SPECS[run_id]
    run_root = ABL_ROOT / run_id
    cfg = run_config(run_id)
    checks: list[dict[str, Any]] = []
    checks.append(check(f"{run_id} run_config exists", cfg is not None, str(run_root / "run_config.json")))
    if cfg:
        checks.append(check(f"{run_id} claim_eligible false", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
        checks.append(check(f"{run_id} main_table_eligible false", cfg.get("main_table_eligible") is False, str(cfg.get("main_table_eligible"))))
        checks.append(check(f"{run_id} matched smoke exists", isinstance(cfg.get("matched_smoke_training"), dict), str(bool(cfg.get("matched_smoke_training")))))
    checks.append(check(f"{run_id} runner mode supported", spec["status"] == "dry_run_supported", str(spec["status"])))
    return {
        "run_id": run_id,
        "variant": spec["variant"],
        "runner_mode": spec["runner_mode"],
        "zero_channels": spec["zero_channels"],
        "trainable_in_current_runner": spec["status"] == "dry_run_supported",
        "planned_command": (
            "python -B -X utf8 submission_planning/tools/run_ablation_variant_training.py "
            f"--execute-training --run-kind matched_full_candidate --tag {tag} --run-id {run_id} "
            f"--max-epochs {args.epochs} --train-patches {args.train_patches} --val-patches {args.val_patches} "
            f"--batch-size {args.batch_size} --learning-rate {args.learning_rate} "
            f"--max-train-samples {args.max_train_samples} --max-val-samples {args.max_val_samples}"
        ),
        "planned_checkpoint": str(run_root / "checkpoints" / f"{tag}.pt"),
        "planned_history_csv": str(run_root / "metrics" / f"{tag}_history.csv"),
        "planned_log_md": str(run_root / "logs" / f"{tag}.md"),
        "claim_eligible_after_training": False,
        "main_table_eligible_after_training": False,
        "requires_full_split_evaluation": True,
        "checks": checks,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tag = safe_tag(args.tag)
    dataset = build_dataset()
    split_counts = {name: len(items) for name, items in dataset.items()}
    checks: list[dict[str, Any]] = [
        check("tag is safe", tag == args.tag, tag),
        check("runner supports matched_full_candidate", "matched_full_candidate" in TRAINING_KINDS, str(sorted(TRAINING_KINDS))),
        check("train split count", split_counts.get("train") == 27, str(split_counts.get("train"))),
        check("validation split count", split_counts.get("validation") == 10, str(split_counts.get("validation"))),
        check("test split count", split_counts.get("test") == 7, str(split_counts.get("test"))),
        check("uses all train samples", args.max_train_samples == 0 or args.max_train_samples == split_counts.get("train"), str(args.max_train_samples)),
        check("uses all validation samples", args.max_val_samples == 0 or args.max_val_samples == split_counts.get("validation"), str(args.max_val_samples)),
        check("candidate epochs positive", args.epochs >= 2, str(args.epochs)),
        check("candidate train patches sufficient for first full run", args.train_patches >= 64, str(args.train_patches)),
        check("candidate validation patches sufficient for first full run", args.val_patches >= 16, str(args.val_patches)),
        check("batch size supported", args.batch_size == 1, str(args.batch_size)),
    ]
    variant_plans = [build_variant_plan(run_id, tag, args) for run_id in (args.run_id or DEFAULT_RUN_IDS)]
    for plan in variant_plans:
        checks.extend(plan["checks"])

    evaluator_plan = {
        "current_evaluator": "submission_planning/tools/evaluate_ablation_full_split_metrics.py",
        "current_limitation": "It is currently tied to controlled-pilot checkpoint lookup and training_scope labels.",
        "required_next_tool": "evaluate_ablation_matched_full_split_metrics.py or an updated evaluator with --checkpoint-tag and --training-scope",
        "required_metrics": ["mae_um", "edge_mae_um", "high_risk_mae_um", "p90_norm"],
        "required_test_samples": 7,
        "planned_evaluation_scope": "matched_full_candidate_test_split_eval",
    }
    eligibility_plan = {
        "required_audit": "audit_ablation_matched_training_eligibility.py",
        "claim_eligible_before_audit": False,
        "main_table_eligible_before_audit": False,
        "minimum_requirements": [
            "ABL-00/02/03/04 full candidate checkpoints exist under tmp/ablation_results/<run_id>/checkpoints/",
            "each run has full candidate history CSV/JSON and Markdown/JSON log",
            "training config records train=27, validation=10, test=7",
            "full-split evaluator produces 28 per-sample metric rows and 4 method summary rows",
            "run_config.json for every run keeps claim_eligible=false until audit passes",
            "audit explicitly sets eligibility level before manuscript tables are updated",
        ],
    }
    full_config = {
        "tag": tag,
        "date": "2026-06-19",
        "seed": SEED,
        "patch_size": PATCH_SIZE,
        "epochs": args.epochs,
        "train_patches_per_epoch": args.train_patches,
        "validation_patches_per_epoch": args.val_patches,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "split_counts": split_counts,
        "run_kind": "matched_full_candidate",
        "output_scope": "tmp/ablation_results/<run_id>/",
        "claim_eligible_after_training": False,
        "main_table_eligible_after_training": False,
    }
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-19",
        "tag": tag,
        "full_matched_training_config": full_config,
        "variant_plans": variant_plans,
        "evaluator_plan": evaluator_plan,
        "eligibility_plan": eligibility_plan,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": "Configuration preflight only. No full matched training, checkpoint, prediction, or manuscript-eligible result was produced.",
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    cfg = report["full_matched_training_config"]
    lines = [
        "# Ablation Full Matched Training Configuration Preflight",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Tag: {report['tag']}",
        f"- Claim eligible: {str(report['claim_eligible']).lower()}",
        f"- Main table eligible: {str(report['main_table_eligible']).lower()}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Candidate Training Configuration",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in cfg.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Variant Plans",
            "",
            "| Run | Variant | Trainable | Planned Checkpoint |",
            "|---|---|---|---|",
        ]
    )
    for row in report["variant_plans"]:
        lines.append(
            f"| {row['run_id']} | {row['variant']} | {row['trainable_in_current_runner']} | "
            f"`{row['planned_checkpoint']}` |"
        )
    lines.extend(["", "## Evaluator Plan", "", "| Field | Value |", "|---|---|"])
    for key, value in report["evaluator_plan"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Eligibility Plan", "", "| Field | Value |", "|---|---|"])
    for key, value in report["eligibility_plan"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=sorted(RUN_SPECS), help="Run id to configure. May be repeated.")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-patches", type=int, default=128)
    parser.add_argument("--val-patches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{report['tag']}_config_preflight.json"
    md_path = args.out_dir / f"{report['tag']}_config_preflight.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation full matched configuration preflight: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
