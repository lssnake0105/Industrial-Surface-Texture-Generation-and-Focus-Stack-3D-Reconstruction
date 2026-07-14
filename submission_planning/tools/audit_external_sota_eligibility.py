"""Audit whether external SOTA predictions are eligible for a main table.

This script is a read-only gate for future DFV/DDFFNet results. It checks
manifests, prediction files, batch-evaluation outputs, and run logs, then writes
a small JSON/Markdown audit report under tmp/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_EXPORT_MANIFEST = Path("tmp/external_baseline_data/manifest.csv")
DEFAULT_OUT_DIR = Path("tmp/external_baseline_results/eligibility_audits")
VALID_SCALE_MODES = {"raw_norm", "minmax", "affine", "scale_to_um"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_read_csv(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.exists():
        return [], f"missing file: {path}"
    try:
        return read_csv(path), None
    except Exception as exc:  # pragma: no cover - defensive CLI report
        return [], f"failed to read {path}: {exc}"


def normalized_key(method: str, sample_id: str) -> tuple[str, str]:
    return method.strip(), sample_id.strip()


def finite_float(value: str | float | int | None) -> bool:
    if value in {None, ""}:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def infer_method(args: argparse.Namespace, pred_rows: list[dict[str, str]]) -> str:
    if args.method:
        return args.method
    methods = sorted({row.get("method", "").strip() for row in pred_rows if row.get("method", "").strip()})
    if len(methods) == 1:
        return methods[0]
    return "external_method"


def find_logs(method: str, explicit_log_dir: Path | None) -> list[Path]:
    log_dir = explicit_log_dir or Path("tmp/external_baseline_results") / method / "logs"
    if not log_dir.exists():
        return []
    return sorted(path for path in log_dir.glob("*") if path.is_file())


def gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail}


def determine_level(gates: list[dict[str, Any]], pred_rows: list[dict[str, str]], full_coverage: bool, enough_samples: bool) -> str:
    if all(item["passed"] for item in gates) and full_coverage and enough_samples:
        return "Eligible-main-table"
    if pred_rows and any(item["gate"] == "prediction files" and item["passed"] for item in gates):
        return "Eligible-auxiliary"
    return "Not eligible"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# External SOTA Eligibility Audit",
        "",
        f"- Method: {report['method']}",
        f"- Eligibility: {report['eligibility_level']}",
        f"- Required samples: {report['required_sample_count']}",
        f"- Prediction rows: {report['prediction_row_count']}",
        f"- Per-sample metric rows: {report['per_sample_row_count']}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for item in report["gates"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"| {item['gate']} | {status} | {item['detail']} |")
    lines.extend(["", "## Missing Samples", ""])
    if report["missing_samples"]:
        for sample_id in report["missing_samples"]:
            lines.append(f"- {sample_id}")
    else:
        lines.append("- none")
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    export_rows, export_error = safe_read_csv(args.export_manifest)
    pred_rows, pred_error = safe_read_csv(args.prediction_manifest)
    per_sample_path = args.batch_result_dir / "per_sample_metrics.csv"
    summary_path = args.batch_result_dir / "method_summary_metrics.csv"
    per_rows, per_error = safe_read_csv(per_sample_path)
    summary_rows, summary_error = safe_read_csv(summary_path)

    method = infer_method(args, pred_rows)
    if method and method != "external_method":
        pred_rows = [row for row in pred_rows if row.get("method", "").strip() == method]
        per_rows = [row for row in per_rows if row.get("method", "").strip() == method]
        summary_rows = [row for row in summary_rows if row.get("method", "").strip() == method]
    required_samples = sorted(
        {
            row.get("sample_id", "").strip()
            for row in export_rows
            if row.get("sample_id", "").strip()
            and (not args.required_split or row.get("split", "").strip() == args.required_split)
        }
    )
    pred_samples = sorted({row.get("sample_id", "").strip() for row in pred_rows if row.get("sample_id", "").strip()})
    missing_samples = sorted(set(required_samples) - set(pred_samples))
    unexpected_samples = sorted(set(pred_samples) - set(required_samples)) if required_samples else []

    pred_keys = {normalized_key(row.get("method", ""), row.get("sample_id", "")) for row in pred_rows}
    per_keys = {normalized_key(row.get("method", ""), row.get("sample_id", "")) for row in per_rows}
    missing_metric_rows = sorted(pred_keys - per_keys)

    existing_prediction_files = []
    missing_prediction_files = []
    invalid_scale_rows = []
    unknown_training_rows = []
    for row in pred_rows:
        prediction_path = Path(row.get("prediction_path", ""))
        if prediction_path.exists():
            existing_prediction_files.append(str(prediction_path))
        else:
            missing_prediction_files.append(str(prediction_path))
        scale_mode = row.get("scale_mode", "").strip()
        if scale_mode not in VALID_SCALE_MODES:
            invalid_scale_rows.append({"sample_id": row.get("sample_id", ""), "scale_mode": scale_mode})
        training = row.get("training_setting", "").strip().lower()
        if training in {"", "unknown", "todo", "tbd"}:
            unknown_training_rows.append(row.get("sample_id", ""))

    metric_keys = ["mae_um", "rmse_um", "p90_um", "edge_mae_um", "high_risk_mae_um"]
    invalid_metric_rows = [
        row.get("sample_id", "")
        for row in per_rows
        if any(not finite_float(row.get(key)) for key in metric_keys)
    ]

    log_files = find_logs(method, args.log_dir)
    full_coverage = bool(required_samples) and not missing_samples and len(pred_samples) >= len(required_samples)
    one_sample_only = len(pred_samples) == 1
    enough_samples = len(pred_samples) >= args.min_samples_for_main_table
    summary_methods = {row.get("method", "").strip() for row in summary_rows}

    gates = [
        gate("export manifest", export_error is None and bool(export_rows), export_error or f"{len(export_rows)} rows"),
        gate("prediction manifest", pred_error is None and bool(pred_rows), pred_error or f"{len(pred_rows)} rows"),
        gate("prediction coverage", full_coverage, f"missing={len(missing_samples)}, unexpected={len(unexpected_samples)}"),
        gate("prediction files", bool(pred_rows) and not missing_prediction_files, f"missing files={len(missing_prediction_files)}"),
        gate("result files", per_error is None and summary_error is None and bool(per_rows) and bool(summary_rows), per_error or summary_error or "per-sample and summary files found"),
        gate("metric row consistency", not missing_metric_rows and bool(per_rows), f"missing metric rows={len(missing_metric_rows)}"),
        gate("metric finite values", not invalid_metric_rows and bool(per_rows), f"invalid metric rows={len(invalid_metric_rows)}"),
        gate("training setting", not unknown_training_rows and bool(pred_rows), f"unknown rows={len(unknown_training_rows)}"),
        gate("scale mode", not invalid_scale_rows and bool(pred_rows), f"invalid rows={len(invalid_scale_rows)}"),
        gate("method summary", method in summary_methods, f"summary methods={sorted(summary_methods)}"),
        gate("minimum sample count", enough_samples, f"prediction samples={len(pred_samples)}, required minimum={args.min_samples_for_main_table}"),
        gate("not single-sample smoke", not one_sample_only, f"prediction samples={len(pred_samples)}"),
        gate("run log", bool(log_files), f"log files={len(log_files)}"),
    ]

    notes = []
    if one_sample_only:
        notes.append("One-sample results should remain smoke-test or auxiliary evidence.")
    if any(row.get("scale_mode", "").strip() == "affine" for row in pred_rows):
        notes.append("Affine scale alignment should be labeled clearly and may be auxiliary if fitted per sample.")
    if export_error or pred_error or per_error or summary_error:
        notes.append("At least one required table is missing or unreadable.")
    if not pred_rows:
        notes.append("No external prediction rows exist; the method cannot enter the main table.")

    return {
        "method": method,
        "eligibility_level": determine_level(gates, pred_rows, full_coverage, enough_samples),
        "required_sample_count": len(required_samples),
        "prediction_row_count": len(pred_rows),
        "per_sample_row_count": len(per_rows),
        "summary_row_count": len(summary_rows),
        "missing_samples": missing_samples,
        "unexpected_samples": unexpected_samples,
        "missing_prediction_files": missing_prediction_files,
        "existing_prediction_files": existing_prediction_files,
        "missing_metric_rows": [list(item) for item in missing_metric_rows],
        "invalid_scale_rows": invalid_scale_rows,
        "unknown_training_rows": unknown_training_rows,
        "invalid_metric_rows": invalid_metric_rows,
        "log_files": [str(path) for path in log_files],
        "gates": gates,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--batch-result-dir", type=Path, required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--required-split", default="")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-samples-for-main-table", type=int, default=2)
    args = parser.parse_args()

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = report["method"].replace(" ", "_").replace("/", "_")
    json_path = args.out_dir / f"{slug}_eligibility_audit.json"
    md_path = args.out_dir / f"{slug}_eligibility_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"{report['method']}: {report['eligibility_level']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["eligibility_level"] == "Eligible-main-table" else 2


if __name__ == "__main__":
    raise SystemExit(main())
