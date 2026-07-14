"""Audit matched ablation smoke training for claim eligibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ABL_ROOT = Path("tmp/ablation_results")
DEFAULT_SUMMARY = ABL_ROOT / "training_runner_matched_smoke" / "2026-06-19_matched_training_smoke_summary.json"
DEFAULT_OUT_DIR = ABL_ROOT / "eligibility_audits"
REQUIRED_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]
EXPECTED_SPLIT_COUNTS = {"train": 27, "validation": 10, "test": 7}


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_history(path: Path) -> bool:
    if not path.exists():
        return False
    rows = read_csv_rows(path)
    if not rows:
        return False
    keys = ["train_loss_debug", "val_loss_debug", "val_mae_norm_debug"]
    for row in rows:
        for key in keys:
            try:
                if not math.isfinite(float(row.get(key, ""))):
                    return False
            except ValueError:
                return False
    return True


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    summary = load_json(args.summary)
    checks.append(check("matched smoke summary exists", summary is not None, str(args.summary)))
    reports: list[dict[str, Any]] = []
    if summary:
        reports = [row for row in summary.get("reports", []) if isinstance(row, dict)]
        run_ids = set(summary.get("run_ids", []))
        checks.append(check("matched smoke status pass", summary.get("status") == "pass", str(summary.get("status"))))
        checks.append(check("matched smoke run_kind", summary.get("run_kind") == "matched_smoke", str(summary.get("run_kind"))))
        checks.append(check("matched smoke required run ids", set(REQUIRED_RUN_IDS).issubset(run_ids), f"run_ids={sorted(run_ids)}"))
        checks.append(check("matched smoke has no errors", summary.get("error_count") == 0, str(summary.get("error_count"))))
        checks.append(check("matched smoke split counts", summary.get("split_counts") == EXPECTED_SPLIT_COUNTS, str(summary.get("split_counts"))))
    report_by_id = {row.get("run_id"): row for row in reports}

    for run_id in REQUIRED_RUN_IDS:
        run_root = ABL_ROOT / run_id
        config_path = run_root / "run_config.json"
        config = load_json(config_path)
        report = report_by_id.get(run_id, {})
        checkpoint = Path(str(report.get("checkpoint", "")))
        metrics_csv = Path(str(report.get("metrics_csv", "")))
        log_md = run_root / "logs" / "2026-06-19_matched_training_smoke.md"
        log_json = run_root / "logs" / "2026-06-19_matched_training_smoke.json"

        checks.append(check(f"{run_id} run_config exists", config is not None, str(config_path)))
        checks.append(check(f"{run_id} report status completed", report.get("status") == "matched_training_smoke_completed", str(report.get("status"))))
        checks.append(check(f"{run_id} checkpoint exists", checkpoint.exists(), str(checkpoint)))
        checks.append(check(f"{run_id} checkpoint under tmp", str(checkpoint).startswith(str(run_root / "checkpoints")), str(checkpoint)))
        checks.append(check(f"{run_id} history CSV exists", metrics_csv.exists(), str(metrics_csv)))
        checks.append(check(f"{run_id} history under tmp", str(metrics_csv).startswith(str(run_root / "metrics")), str(metrics_csv)))
        checks.append(check(f"{run_id} history finite", finite_history(metrics_csv), str(metrics_csv)))
        checks.append(check(f"{run_id} matched smoke markdown log exists", log_md.exists(), str(log_md)))
        checks.append(check(f"{run_id} matched smoke JSON log exists", log_json.exists(), str(log_json)))
        checks.append(check(f"{run_id} report split counts", report.get("split_counts") == EXPECTED_SPLIT_COUNTS, str(report.get("split_counts"))))
        checks.append(check(f"{run_id} prepared train samples recorded", report.get("prepared_train_samples") == 2, str(report.get("prepared_train_samples"))))
        checks.append(check(f"{run_id} prepared validation samples recorded", report.get("prepared_validation_samples") == 1, str(report.get("prepared_validation_samples"))))
        if config:
            checks.append(check(f"{run_id} status matched smoke", config.get("status") == "matched_training_smoke_run", str(config.get("status"))))
            checks.append(check(f"{run_id} claim_eligible false", config.get("claim_eligible") is False, str(config.get("claim_eligible"))))
            checks.append(check(f"{run_id} main_table_eligible false", config.get("main_table_eligible") is False, str(config.get("main_table_eligible"))))
            smoke = config.get("matched_smoke_training", {})
            checks.append(check(f"{run_id} matched_smoke_training exists", isinstance(smoke, dict) and bool(smoke), str(bool(smoke))))
            checks.append(check(f"{run_id} matched smoke debug_only", smoke.get("debug_only") is True, str(smoke.get("debug_only"))))
            checks.append(check(f"{run_id} matched smoke split counts in config", smoke.get("split_counts") == EXPECTED_SPLIT_COUNTS, str(smoke.get("split_counts"))))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    passed = not errors
    return {
        "status": "pass" if passed else "fail",
        "eligibility_level": "Matched smoke only; not manuscript ablation evidence" if passed else "Not eligible",
        "date": "2026-06-19",
        "summary": str(args.summary),
        "required_run_ids": REQUIRED_RUN_IDS,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": (
            "The matched smoke run proves ABL-00/02/03/04 can train through the protected runner under the same "
            "declared train/validation/test split boundary. It uses only a tiny sample/patch budget and cannot "
            "support manuscript ablation claims."
        ),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Matched Smoke Eligibility Audit",
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
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ABL_matched_smoke_eligibility.json"
    md_path = args.out_dir / "ABL_matched_smoke_eligibility.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation matched smoke eligibility audit: {report['status']}")
    print(f"Eligibility: {report['eligibility_level']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
