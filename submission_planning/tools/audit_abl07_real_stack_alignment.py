"""Audit ABL-07 real focus-stack no-reference alignment diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_ROOT = Path("submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment")
DEFAULT_OUT_DIR = Path("tmp/ablation_results/eligibility_audits")
EXPECTED_CHECKPOINTS = {
    "2026-06-22_confidence_gated_prior_full_candidate",
    "2026-06-22_confidence_gated_prior_seed_repeat",
}
EXPECTED_STACKS = {"1124", "3D层纹", "3D表面", "圆孔50um", "磕碰孔5um", "钥匙尖头50um", "钥匙纹路100um"}
MIN_REAL_STACKS = 7


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: str | float | int | None) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def finite_values(rows: list[dict[str, str]], columns: list[str]) -> tuple[bool, str]:
    missing: list[str] = []
    for idx, row in enumerate(rows, start=1):
        for column in columns:
            if not math.isfinite(finite_float(row.get(column))):
                missing.append(f"row {idx}: {column}")
    return not missing, "; ".join(missing[:10])


def image_summary(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(root.glob("*.png")):
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            rows.append({"name": path.name, "width": width, "height": height, "bytes": path.stat().st_size})
        except Exception as exc:  # pragma: no cover - audit should report file-level failures
            errors.append(f"{path.name}: {exc}")
    return rows, errors


def build_report(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stack_csv = root / "abl07_real_stack_alignment_stack_metrics.csv"
    class_csv = root / "abl07_real_stack_alignment_class_metrics.csv"
    aggregate_csv = root / "abl07_real_stack_alignment_aggregate.csv"
    summary_json = root / "abl07_real_stack_alignment_summary.json"
    report_md = root / "abl07_real_stack_alignment_report.md"

    for path in [stack_csv, class_csv, aggregate_csv, summary_json, report_md]:
        checks.append(check(f"artifact exists: {path.name}", path.exists(), str(path)))

    stack_rows = read_csv(stack_csv) if stack_csv.exists() else []
    class_rows = read_csv(class_csv) if class_csv.exists() else []
    aggregate_rows = read_csv(aggregate_csv) if aggregate_csv.exists() else []
    summary = load_json(summary_json)
    report_text = report_md.read_text(encoding="utf-8") if report_md.exists() else ""

    stacks = {row.get("stack", "") for row in stack_rows}
    checkpoints = {row.get("checkpoint_tag", "") for row in stack_rows}
    checks.append(check("expected real stacks covered", stacks == EXPECTED_STACKS, ", ".join(sorted(stacks))))
    checks.append(check("at least seven real stacks", len(stacks) >= MIN_REAL_STACKS, str(len(stacks))))
    checks.append(check("expected checkpoints covered", checkpoints == EXPECTED_CHECKPOINTS, ", ".join(sorted(checkpoints))))
    checks.append(check("stack rows equal stacks times checkpoints", len(stack_rows) == len(stacks) * len(checkpoints), str(len(stack_rows))))
    checks.append(check("aggregate rows equal checkpoints", len(aggregate_rows) == len(EXPECTED_CHECKPOINTS), str(len(aggregate_rows))))
    checks.append(check("class rows populated", len(class_rows) >= len(stack_rows), str(len(class_rows))))

    if isinstance(summary, dict):
        checks.append(check("summary status pass", summary.get("status") == "pass", str(summary.get("status"))))
        checks.append(check("summary stack count seven", summary.get("stack_count") == MIN_REAL_STACKS, str(summary.get("stack_count"))))
        boundary = str(summary.get("claim_boundary", "")).lower()
        checks.append(check("summary blocks absolute real-height claim", "no calibrated real height ground truth" in boundary, boundary))
    else:
        checks.append(check("summary JSON parsed", False, str(summary_json)))

    stack_finite_cols = [
        "dff_dev_mean",
        "model_dev_mean",
        "low_confidence_model_dev_reduction_percent",
        "spike_top10_model_dev_reduction_percent",
        "quality_top10_model_dev_reduction_percent",
        "confident_pearson_model_dff",
        "confident_model_std_over_dff_std",
    ]
    ok, detail = finite_values(stack_rows, stack_finite_cols)
    checks.append(check("stack metrics finite", ok, detail or "all finite"))
    saturated_issues = []
    for idx, row in enumerate(stack_rows, start=1):
        saturated_fraction = finite_float(row.get("saturated_fraction"))
        saturated_reduction = finite_float(row.get("saturated_model_dev_reduction_percent"))
        if saturated_fraction > 0 and not math.isfinite(saturated_reduction):
            saturated_issues.append(f"row {idx}: saturated reduction")
    checks.append(
        check(
            "saturated metrics finite when saturated mask exists",
            not saturated_issues,
            "; ".join(saturated_issues[:10]) or "empty saturated masks allowed",
        )
    )

    aggregate_finite_cols = [
        "mean_low_confidence_model_dev_reduction_percent",
        "mean_spike_top10_model_dev_reduction_percent",
        "mean_saturated_model_dev_reduction_percent",
        "mean_quality_top10_model_dev_reduction_percent",
        "mean_confident_pearson_model_dff",
        "mean_confident_model_std_over_dff_std",
    ]
    ok, detail = finite_values(aggregate_rows, aggregate_finite_cols)
    checks.append(check("aggregate metrics finite", ok, detail or "all finite"))

    for row in aggregate_rows:
        tag = row.get("checkpoint_tag", "unknown")
        checks.append(
            check(
                f"{tag} mean low-confidence reduction >=80%",
                finite_float(row.get("mean_low_confidence_model_dev_reduction_percent")) >= 80.0,
                str(row.get("mean_low_confidence_model_dev_reduction_percent")),
            )
        )
        checks.append(
            check(
                f"{tag} mean spike-top10 reduction >=85%",
                finite_float(row.get("mean_spike_top10_model_dev_reduction_percent")) >= 85.0,
                str(row.get("mean_spike_top10_model_dev_reduction_percent")),
            )
        )
        checks.append(
            check(
                f"{tag} mean saturated reduction >=70%",
                finite_float(row.get("mean_saturated_model_dev_reduction_percent")) >= 70.0,
                str(row.get("mean_saturated_model_dev_reduction_percent")),
            )
        )
        corr = finite_float(row.get("mean_confident_pearson_model_dff"))
        checks.append(check(f"{tag} confident correlation in valid range", -1.0 <= corr <= 1.0, f"{corr:.4f}"))

    min_low = min((finite_float(row.get("low_confidence_model_dev_reduction_percent")) for row in stack_rows), default=float("nan"))
    min_spike = min((finite_float(row.get("spike_top10_model_dev_reduction_percent")) for row in stack_rows), default=float("nan"))
    saturated_values = [
        finite_float(row.get("saturated_model_dev_reduction_percent"))
        for row in stack_rows
        if math.isfinite(finite_float(row.get("saturated_model_dev_reduction_percent")))
    ]
    min_saturated = min(saturated_values, default=float("nan"))
    checks.append(check("all stack low-confidence reductions >=80%", min_low >= 80.0, f"min={min_low:.2f}%"))
    checks.append(check("all stack spike-top10 reductions >=85%", min_spike >= 85.0, f"min={min_spike:.2f}%"))
    checks.append(check("all stack saturated reductions >=70%", min_saturated >= 70.0, f"min={min_saturated:.2f}%"))

    images, image_errors = image_summary(root)
    expected_image_count = len(EXPECTED_STACKS) * len(EXPECTED_CHECKPOINTS) * 2
    checks.append(check("expected figure count", len(images) == expected_image_count, f"{len(images)} / {expected_image_count}"))
    checks.append(check("all figures readable", not image_errors, "; ".join(image_errors[:5]) or "all readable"))
    checks.append(check("all figures non-empty", all(row["width"] > 0 and row["height"] > 0 and row["bytes"] > 0 for row in images), str(len(images))))

    lowered_report = report_text.lower()
    required_phrases = [
        "no-reference real-stack diagnostic",
        "no real calibrated height ground truth",
        "does not prove absolute real-height accuracy",
    ]
    for phrase in required_phrases:
        checks.append(check(f"report contains boundary phrase: {phrase}", phrase in lowered_report, phrase))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    summary_payload = {
        "stack_count": len(stacks),
        "checkpoint_count": len(checkpoints),
        "stack_rows": len(stack_rows),
        "class_rows": len(class_rows),
        "figure_count": len(images),
        "min_low_confidence_reduction_percent": min_low,
        "min_spike_top10_reduction_percent": min_spike,
        "min_saturated_reduction_percent": min_saturated,
    }
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-22",
        "scope": "ABL-07 real focus-stack no-reference alignment diagnostics",
        "claim_eligible_for_real_height_accuracy": False,
        "claim_eligible_for_real_stack_alignment": not errors,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "summary": summary_payload,
        "interpretation": (
            "The audit validates artifact completeness and diagnostic consistency for real-stack alignment. "
            "It does not validate calibrated real-height accuracy because no real height ground truth is available."
        ),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ABL-07 Real-Stack Alignment Audit",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Scope: {report['scope']}",
        f"- Real-stack alignment claim eligible: {str(report['claim_eligible_for_real_stack_alignment']).lower()}",
        f"- Real-height accuracy claim eligible: {str(report['claim_eligible_for_real_height_accuracy']).lower()}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Checks", "", "| Check | Status | Severity | Detail |", "|---|---|---|---|"])
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {detail} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = build_report(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ABL07_real_stack_alignment_audit.json"
    md_path = args.out_dir / "ABL07_real_stack_alignment_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"ABL-07 real-stack alignment audit: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
