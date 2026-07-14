"""Audit whether the controlled ablation pilot can support manuscript claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ABL_ROOT = Path("tmp/ablation_results")
DEFAULT_SUMMARY = ABL_ROOT / "training_runner_controlled_pilot" / "2026-06-19_controlled_pilot_summary.json"
DEFAULT_OUT_DIR = ABL_ROOT / "eligibility_audits"
REQUIRED_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_metric_rows(path: Path) -> bool:
    if not path.exists():
        return False
    rows = read_csv_rows(path)
    if not rows:
        return False
    metric_keys = ["train_loss_debug", "val_loss_debug", "val_mae_norm_debug"]
    for row in rows:
        for key in metric_keys:
            try:
                if not math.isfinite(float(row.get(key, ""))):
                    return False
            except ValueError:
                return False
    return True


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    summary = load_json(args.summary)
    checks.append(check("controlled pilot summary exists", summary is not None, str(args.summary)))
    reports = []
    if summary:
        reports = [row for row in summary.get("reports", []) if isinstance(row, dict)]
        run_ids = set(summary.get("run_ids", []))
        checks.append(check("controlled pilot status pass", summary.get("status") == "pass", str(summary.get("status"))))
        checks.append(check("controlled pilot run_kind", summary.get("run_kind") == "controlled_pilot", str(summary.get("run_kind"))))
        checks.append(check("controlled pilot required run ids", set(REQUIRED_RUN_IDS).issubset(run_ids), f"run_ids={sorted(run_ids)}"))
        checks.append(check("controlled pilot has no errors", summary.get("error_count") == 0, str(summary.get("error_count"))))
    report_by_id = {row.get("run_id"): row for row in reports}

    for run_id in REQUIRED_RUN_IDS:
        run_root = ABL_ROOT / run_id
        config_path = run_root / "run_config.json"
        config = load_json(config_path)
        checks.append(check(f"{run_id} run_config exists", config is not None, str(config_path)))
        if config:
            checks.append(check(f"{run_id} status controlled pilot", config.get("status") == "controlled_pilot_debug_run", str(config.get("status"))))
            checks.append(check(f"{run_id} claim_eligible false", config.get("claim_eligible") is False, str(config.get("claim_eligible"))))
            checks.append(check(f"{run_id} main_table_eligible false", config.get("main_table_eligible") is False, str(config.get("main_table_eligible"))))
            pilot = config.get("pilot_training", {})
            checks.append(check(f"{run_id} pilot_training exists", isinstance(pilot, dict) and bool(pilot), str(bool(pilot))))
            checks.append(check(f"{run_id} pilot marked debug_only", pilot.get("debug_only") is True, str(pilot.get("debug_only"))))
        report = report_by_id.get(run_id, {})
        checkpoint = Path(str(report.get("checkpoint", "")))
        metrics_csv = Path(str(report.get("metrics_csv", "")))
        checks.append(check(f"{run_id} report status completed", report.get("status") == "controlled_pilot_debug_completed", str(report.get("status"))))
        checks.append(check(f"{run_id} checkpoint exists", checkpoint.exists(), str(checkpoint)))
        checks.append(check(f"{run_id} checkpoint under tmp", str(checkpoint).startswith(str(run_root / "checkpoints")), str(checkpoint)))
        checks.append(check(f"{run_id} metrics CSV exists", metrics_csv.exists(), str(metrics_csv)))
        checks.append(check(f"{run_id} metrics under tmp", str(metrics_csv).startswith(str(run_root / "metrics")), str(metrics_csv)))
        checks.append(check(f"{run_id} metrics finite", finite_metric_rows(metrics_csv), str(metrics_csv)))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    pilot_passed = not errors
    return {
        "status": "pass" if pilot_passed else "fail",
        "eligibility_level": "Debug-only; not manuscript main-table evidence" if pilot_passed else "Not eligible",
        "date": "2026-06-19",
        "required_run_ids": REQUIRED_RUN_IDS,
        "summary": str(args.summary),
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": (
            "The controlled pilot proves the protected runner can train ABL-00/02/03/04 under one tiny P10 "
            "debug setup. It is insufficient for manuscript ablation claims until a full split, repeated seeds, "
            "per-sample metrics, and a separate claim eligibility update exist."
        ),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Pilot Eligibility Audit",
        "",
        f"- Status: {report['status']}",
        f"- Eligibility: {report['eligibility_level']}",
        f"- Date: {report['date']}",
        f"- Required runs: {report['required_run_ids']}",
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
    json_path = args.out_dir / "ABL_pilot_eligibility.json"
    md_path = args.out_dir / "ABL_pilot_eligibility.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation pilot eligibility audit: {report['status']}")
    print(f"Eligibility: {report['eligibility_level']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
