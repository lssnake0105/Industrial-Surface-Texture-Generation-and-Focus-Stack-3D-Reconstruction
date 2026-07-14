"""Audit full-split ablation diagnostic metrics for claim eligibility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ABL_ROOT = Path("tmp/ablation_results")
DEFAULT_SUMMARY = ABL_ROOT / "full_split_debug_eval" / "2026-06-19_full_split_debug_eval_summary.json"
DEFAULT_OUT_DIR = ABL_ROOT / "eligibility_audits"
REQUIRED_RUN_IDS = ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
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


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    summary = load_json(args.summary)
    checks.append(check("full-split debug summary exists", summary is not None, str(args.summary)))
    per_sample_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    if summary:
        per_sample_csv = Path(str(summary.get("per_sample_metrics_csv", "")))
        method_summary_csv = Path(str(summary.get("method_summary_metrics_csv", "")))
        checks.append(check("full-split debug status pass", summary.get("status") == "pass", str(summary.get("status"))))
        checks.append(check("full-split sample count is 7", summary.get("sample_count") == 7, str(summary.get("sample_count"))))
        checks.append(check("full-split required run ids", set(REQUIRED_RUN_IDS).issubset(set(summary.get("run_ids", []))), str(summary.get("run_ids"))))
        checks.append(check("full-split claim_eligible false", summary.get("claim_eligible") is False, str(summary.get("claim_eligible"))))
        checks.append(check("full-split main_table_eligible false", summary.get("main_table_eligible") is False, str(summary.get("main_table_eligible"))))
        checks.append(check("per-sample metrics CSV exists", per_sample_csv.exists(), str(per_sample_csv)))
        checks.append(check("method summary CSV exists", method_summary_csv.exists(), str(method_summary_csv)))
        if per_sample_csv.exists():
            per_sample_rows = read_csv(per_sample_csv)
        if method_summary_csv.exists():
            summary_rows = read_csv(method_summary_csv)
    checks.append(check("per-sample rows count", len(per_sample_rows) == 28, f"rows={len(per_sample_rows)}"))
    checks.append(check("summary rows count", len(summary_rows) == 4, f"rows={len(summary_rows)}"))
    checks.append(check("per-sample metrics finite", finite_columns(per_sample_rows, ["mae_um", "edge_mae_um", "high_risk_mae_um"]), "mae/edge/high-risk"))
    checks.append(check("summary metrics finite", finite_columns(summary_rows, ["mean_mae_um", "mean_edge_mae_um", "mean_high_risk_mae_um"]), "mean metrics"))
    for run_id in REQUIRED_RUN_IDS:
        cfg_path = ABL_ROOT / run_id / "run_config.json"
        cfg = load_json(cfg_path)
        checks.append(check(f"{run_id} run_config exists", cfg is not None, str(cfg_path)))
        run_rows = [row for row in per_sample_rows if row.get("run_id") == run_id]
        checks.append(check(f"{run_id} has 7 per-sample rows", len(run_rows) == 7, f"rows={len(run_rows)}"))
        if cfg:
            checks.append(check(f"{run_id} claim_eligible false", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
            checks.append(check(f"{run_id} main_table_eligible false", cfg.get("main_table_eligible") is False, str(cfg.get("main_table_eligible"))))
            eval_payload = cfg.get("full_split_debug_evaluation", {})
            checks.append(check(f"{run_id} full_split_debug_evaluation exists", isinstance(eval_payload, dict) and bool(eval_payload), str(bool(eval_payload))))
            checks.append(check(f"{run_id} full_split debug_only true", eval_payload.get("debug_only") is True, str(eval_payload.get("debug_only"))))
            checks.append(check(f"{run_id} training scope remains pilot", eval_payload.get("training_scope") == "controlled_pilot_p10_debug", str(eval_payload.get("training_scope"))))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    passed = not errors
    return {
        "status": "pass" if passed else "fail",
        "eligibility_level": "Diagnostic full-split metrics only; not manuscript ablation evidence" if passed else "Not eligible",
        "date": "2026-06-19",
        "summary": str(args.summary),
        "required_run_ids": REQUIRED_RUN_IDS,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_eligible": False,
        "main_table_eligible": False,
        "interpretation": (
            "The diagnostic evaluation covers the full 7-sample synthetic test split, but the checkpoints were "
            "trained only by a tiny P10 controlled pilot. These metrics can guide debugging and full experiment "
            "planning, but they cannot support manuscript ablation claims."
        ),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Full-Split Eligibility Audit",
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
    json_path = args.out_dir / "ABL_full_split_eligibility.json"
    md_path = args.out_dir / "ABL_full_split_eligibility.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation full-split eligibility audit: {report['status']}")
    print(f"Eligibility: {report['eligibility_level']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
