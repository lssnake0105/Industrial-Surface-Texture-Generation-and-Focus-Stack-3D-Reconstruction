"""Audit matched full-candidate ablation training for current evidence eligibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ABL_ROOT = Path("tmp/ablation_results")
DEFAULT_TRAINING_SUMMARY = ABL_ROOT / "training_runner_matched_full_candidate" / "2026-06-19_matched_training_full_candidate_summary.json"
DEFAULT_EVAL_SUMMARY = ABL_ROOT / "matched_full_split_eval" / "2026-06-19_matched_full_candidate_eval" / "2026-06-19_matched_full_candidate_eval_summary.json"
DEFAULT_OUT_DIR = ABL_ROOT / "eligibility_audits"
REQUIRED_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]
EXPECTED_SPLIT_COUNTS = {"train": 27, "validation": 10, "test": 7}
EXPECTED_TRAINING_TAG = "2026-06-19_matched_training_full_candidate"
EXPECTED_EVAL_TAG = "2026-06-19_matched_full_candidate_eval"


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_columns(rows: list[dict[str, str]], columns: list[str]) -> bool:
    if not rows:
        return False
    for row in rows:
        for column in columns:
            try:
                if not math.isfinite(float(row.get(column, ""))):
                    return False
            except ValueError:
                return False
    return True


def finite_history(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    rows = read_csv_rows(path)
    if len(rows) != expected_rows:
        return False
    return finite_columns(rows, ["train_loss_debug", "val_loss_debug", "val_mae_norm_debug"])


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    training_summary = load_json(args.training_summary)
    eval_summary = load_json(args.eval_summary)
    checks.append(check("matched full candidate training summary exists", training_summary is not None, str(args.training_summary)))
    checks.append(check("matched full candidate eval summary exists", eval_summary is not None, str(args.eval_summary)))

    training_reports: list[dict[str, Any]] = []
    if training_summary:
        training_reports = [row for row in training_summary.get("reports", []) if isinstance(row, dict)]
        run_ids = set(training_summary.get("run_ids", []))
        checks.append(check("training status pass", training_summary.get("status") == "pass", str(training_summary.get("status"))))
        checks.append(check("training run_kind matched_full_candidate", training_summary.get("run_kind") == "matched_full_candidate", str(training_summary.get("run_kind"))))
        checks.append(check("training tag", training_summary.get("tag") == EXPECTED_TRAINING_TAG, str(training_summary.get("tag"))))
        checks.append(check("training required run ids", set(REQUIRED_RUN_IDS).issubset(run_ids), f"run_ids={sorted(run_ids)}"))
        checks.append(check("training has no errors", training_summary.get("error_count") == 0, str(training_summary.get("error_count"))))
        checks.append(check("training split counts", training_summary.get("split_counts") == EXPECTED_SPLIT_COUNTS, str(training_summary.get("split_counts"))))

    per_sample_rows: list[dict[str, str]] = []
    method_summary_rows: list[dict[str, str]] = []
    if eval_summary:
        per_sample_csv = Path(str(eval_summary.get("per_sample_metrics_csv", "")))
        method_summary_csv = Path(str(eval_summary.get("method_summary_metrics_csv", "")))
        eval_run_ids = set(eval_summary.get("run_ids", []))
        checks.append(check("evaluation status pass", eval_summary.get("status") == "pass", str(eval_summary.get("status"))))
        checks.append(check("evaluation tag", eval_summary.get("tag") == EXPECTED_EVAL_TAG, str(eval_summary.get("tag"))))
        checks.append(check("evaluation checkpoint tag", eval_summary.get("checkpoint_tag") == EXPECTED_TRAINING_TAG, str(eval_summary.get("checkpoint_tag"))))
        checks.append(check("evaluation sample count 7", eval_summary.get("sample_count") == 7, str(eval_summary.get("sample_count"))))
        checks.append(check("evaluation required run ids", set(REQUIRED_RUN_IDS).issubset(eval_run_ids), f"run_ids={sorted(eval_run_ids)}"))
        checks.append(check("evaluation is not smoke", eval_summary.get("smoke_evaluation") is False, str(eval_summary.get("smoke_evaluation"))))
        checks.append(check("evaluation training scope", eval_summary.get("training_scope") == "matched_full_candidate_train_validation_split", str(eval_summary.get("training_scope"))))
        checks.append(check("evaluation scope", eval_summary.get("evaluation_scope") == "matched_full_candidate_test_split_eval", str(eval_summary.get("evaluation_scope"))))
        checks.append(check("evaluation claim false before audit", eval_summary.get("claim_eligible") is False, str(eval_summary.get("claim_eligible"))))
        checks.append(check("evaluation main table false before audit", eval_summary.get("main_table_eligible") is False, str(eval_summary.get("main_table_eligible"))))
        checks.append(check("per-sample metrics CSV exists", per_sample_csv.exists(), str(per_sample_csv)))
        checks.append(check("method summary CSV exists", method_summary_csv.exists(), str(method_summary_csv)))
        if per_sample_csv.exists():
            per_sample_rows = read_csv_rows(per_sample_csv)
        if method_summary_csv.exists():
            method_summary_rows = read_csv_rows(method_summary_csv)

    checks.append(check("per-sample rows count", len(per_sample_rows) == 28, f"rows={len(per_sample_rows)}"))
    checks.append(check("method summary rows count", len(method_summary_rows) == 4, f"rows={len(method_summary_rows)}"))
    checks.append(check("per-sample metrics finite", finite_columns(per_sample_rows, ["mae_um", "edge_mae_um", "high_risk_mae_um", "p90_norm"]), "mae/edge/high-risk/p90"))
    checks.append(check("method summary metrics finite", finite_columns(method_summary_rows, ["mean_mae_um", "mean_edge_mae_um", "mean_high_risk_mae_um", "mean_p90_norm"]), "mean metrics"))

    report_by_id = {row.get("run_id"): row for row in training_reports}
    for run_id in REQUIRED_RUN_IDS:
        run_root = ABL_ROOT / run_id
        config_path = run_root / "run_config.json"
        config = load_json(config_path)
        training_report = report_by_id.get(run_id, {})
        checkpoint = Path(str(training_report.get("checkpoint", "")))
        metrics_csv = Path(str(training_report.get("metrics_csv", "")))
        log_md = run_root / "logs" / f"{EXPECTED_TRAINING_TAG}.md"
        log_json = run_root / "logs" / f"{EXPECTED_TRAINING_TAG}.json"
        run_eval_rows = [row for row in per_sample_rows if row.get("run_id") == run_id]

        checks.append(check(f"{run_id} run_config exists", config is not None, str(config_path)))
        checks.append(check(f"{run_id} training report completed", training_report.get("status") == "matched_training_full_candidate_completed", str(training_report.get("status"))))
        checks.append(check(f"{run_id} checkpoint exists", checkpoint.exists(), str(checkpoint)))
        checks.append(check(f"{run_id} checkpoint under tmp", str(checkpoint).startswith(str(run_root / "checkpoints")), str(checkpoint)))
        checks.append(check(f"{run_id} history CSV exists", metrics_csv.exists(), str(metrics_csv)))
        checks.append(check(f"{run_id} history finite with 4 epochs", finite_history(metrics_csv, expected_rows=4), str(metrics_csv)))
        checks.append(check(f"{run_id} training markdown log exists", log_md.exists(), str(log_md)))
        checks.append(check(f"{run_id} training JSON log exists", log_json.exists(), str(log_json)))
        checks.append(check(f"{run_id} has 7 per-sample eval rows", len(run_eval_rows) == 7, f"rows={len(run_eval_rows)}"))
        checks.append(check(f"{run_id} report split counts", training_report.get("split_counts") == EXPECTED_SPLIT_COUNTS, str(training_report.get("split_counts"))))
        checks.append(check(f"{run_id} prepared train samples all 27", training_report.get("prepared_train_samples") == 27, str(training_report.get("prepared_train_samples"))))
        checks.append(check(f"{run_id} prepared validation samples all 10", training_report.get("prepared_validation_samples") == 10, str(training_report.get("prepared_validation_samples"))))
        if config:
            checks.append(check(f"{run_id} status full candidate", config.get("status") == "matched_training_full_candidate_run", str(config.get("status"))))
            checks.append(check(f"{run_id} claim_eligible false before report use", config.get("claim_eligible") is False, str(config.get("claim_eligible"))))
            checks.append(check(f"{run_id} main_table_eligible false before report use", config.get("main_table_eligible") is False, str(config.get("main_table_eligible"))))
            training_entry = config.get("matched_full_candidate_training", {})
            checks.append(check(f"{run_id} matched_full_candidate_training exists", isinstance(training_entry, dict) and bool(training_entry), str(bool(training_entry))))
            checks.append(check(f"{run_id} full candidate tag in config", training_entry.get("tag") == EXPECTED_TRAINING_TAG, str(training_entry.get("tag"))))
            checks.append(check(f"{run_id} full candidate split counts in config", training_entry.get("split_counts") == EXPECTED_SPLIT_COUNTS, str(training_entry.get("split_counts"))))
            evaluations = config.get("matched_full_split_evaluations", {})
            eval_entry = evaluations.get(EXPECTED_EVAL_TAG, {}) if isinstance(evaluations, dict) else {}
            checks.append(check(f"{run_id} matched full eval entry exists", isinstance(eval_entry, dict) and bool(eval_entry), str(bool(eval_entry))))
            checks.append(check(f"{run_id} matched full eval sample count", eval_entry.get("sample_count") == 7, str(eval_entry.get("sample_count"))))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    passed = not errors
    return {
        "status": "pass" if passed else "fail",
        "eligibility_level": "Current-stage matched ablation evidence; manuscript table candidate pending supervisor review and external-baseline context" if passed else "Not eligible",
        "date": "2026-06-19",
        "training_summary": str(args.training_summary),
        "eval_summary": str(args.eval_summary),
        "required_run_ids": REQUIRED_RUN_IDS,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_eligible": bool(passed),
        "main_table_eligible": bool(passed),
        "interpretation": (
            "The matched full-candidate run covers ABL-00/02/03/04 under the declared 27/10/7 split, "
            "has checkpoints and 4-epoch histories for every run, and has 7-sample test metrics. The results "
            "are usable as current-stage ablation evidence for supervisor discussion. Before manuscript use, "
            "the unexpectedly stronger w/o focal-difference and w/o glare-cue variants should be interpreted "
            "as model-design feedback and checked with a stronger/longer final-method training schedule."
        ),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Matched Training Eligibility Audit",
        "",
        f"- Status: {report['status']}",
        f"- Eligibility: {report['eligibility_level']}",
        f"- Date: {report['date']}",
        f"- Required runs: {report['required_run_ids']}",
        f"- Expected split counts: {report['expected_split_counts']}",
        f"- Claim eligible: {str(report['claim_eligible']).lower()}",
        f"- Main table eligible: {str(report['main_table_eligible']).lower()}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---|---|---|",
    ]
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--eval-summary", type=Path, default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ABL_matched_training_eligibility.json"
    md_path = args.out_dir / "ABL_matched_training_eligibility.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation matched training eligibility audit: {report['status']}")
    print(f"Eligibility: {report['eligibility_level']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
