"""Audit ABL-07 confidence-gated prior evidence and claim boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ABL_ROOT = Path("tmp/ablation_results")
RUN_ROOT = ABL_ROOT / "ABL-07"
OUT_DIR = ABL_ROOT / "eligibility_audits"
RUN_ID = "ABL-07"
EXPECTED_SPLIT = {"train": 27, "validation": 10, "test": 7}
FULL_TAG = "2026-06-22_confidence_gated_prior_full_candidate"
FULL_EVAL_TAG = "2026-06-22_confidence_gated_prior_full_candidate_eval"
REPEAT_TAG = "2026-06-22_confidence_gated_prior_seed_repeat"
REPEAT_EVAL_TAG = "2026-06-22_confidence_gated_prior_seed_repeat_eval"
OLD_BEST_ABL04_MAE_UM = 75.45722307477679


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def load_json(path: Path) -> Any | None:
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


def history_ok(path: Path, expected_rows: int = 12) -> bool:
    if not path.exists():
        return False
    rows = read_csv(path)
    if len(rows) != expected_rows:
        return False
    return finite_columns(rows, ["train_loss_debug", "val_loss_debug", "val_mae_norm_debug"])


def eval_dir(tag: str) -> Path:
    return ABL_ROOT / "confidence_gated_full_split_eval" / tag


def load_eval_summary(tag: str) -> dict[str, Any] | None:
    return load_json(eval_dir(tag) / f"{tag}_summary.json")


def load_method_row(tag: str) -> dict[str, str] | None:
    path = eval_dir(tag) / f"{tag}_method_summary_metrics.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    return rows[0] if rows else None


def load_stratum_row(tag: str, stratum: str) -> dict[str, str] | None:
    path = eval_dir(tag) / f"{tag}_stratum_summary_metrics.csv"
    if not path.exists():
        return None
    for row in read_csv(path):
        if row.get("stratum") == stratum:
            return row
    return None


def numeric(row: dict[str, str] | None, key: str) -> float:
    if not row:
        return float("nan")
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


def audit_run_artifacts(tag: str, expected_seed: int | None, checks: list[dict[str, Any]]) -> None:
    checkpoint = RUN_ROOT / "checkpoints" / f"{tag}.pt"
    metrics_csv = RUN_ROOT / "metrics" / f"{tag}_metrics.csv"
    metrics_json = RUN_ROOT / "metrics" / f"{tag}_metrics.json"
    log_md = RUN_ROOT / "logs" / f"{tag}.md"
    log_json = RUN_ROOT / "logs" / f"{tag}.json"
    log_payload = load_json(log_json)
    checks.append(check(f"{tag} checkpoint exists", checkpoint.exists(), str(checkpoint)))
    checks.append(check(f"{tag} metrics CSV exists", metrics_csv.exists(), str(metrics_csv)))
    checks.append(check(f"{tag} metrics JSON exists", metrics_json.exists(), str(metrics_json)))
    checks.append(check(f"{tag} markdown log exists", log_md.exists(), str(log_md)))
    checks.append(check(f"{tag} JSON log exists", log_json.exists(), str(log_json)))
    checks.append(check(f"{tag} history has 12 finite rows", history_ok(metrics_csv), str(metrics_csv)))
    if isinstance(log_payload, dict):
        checks.append(check(f"{tag} prepared train samples 27", log_payload.get("prepared_train_samples") == 27, str(log_payload.get("prepared_train_samples"))))
        checks.append(check(f"{tag} prepared validation samples 10", log_payload.get("prepared_validation_samples") == 10, str(log_payload.get("prepared_validation_samples"))))
        checks.append(check(f"{tag} status completed", log_payload.get("status") == "confidence_gated_prior_smoke_completed", str(log_payload.get("status"))))
        checks.append(check(f"{tag} error count zero", log_payload.get("error_count") == 0, str(log_payload.get("error_count"))))
        if expected_seed is not None:
            checks.append(check(f"{tag} seed recorded", log_payload.get("seed") == expected_seed, str(log_payload.get("seed"))))


def audit_eval(tag: str, checkpoint_tag: str, checks: list[dict[str, Any]]) -> None:
    summary = load_eval_summary(tag)
    checks.append(check(f"{tag} eval summary exists", summary is not None, str(eval_dir(tag) / f"{tag}_summary.json")))
    if not isinstance(summary, dict):
        return
    checks.append(check(f"{tag} eval status pass", summary.get("status") == "pass", str(summary.get("status"))))
    checks.append(check(f"{tag} checkpoint tag", summary.get("checkpoint_tag") == checkpoint_tag, str(summary.get("checkpoint_tag"))))
    checks.append(check(f"{tag} sample count 7", summary.get("sample_count") == 7, str(summary.get("sample_count"))))
    checks.append(check(f"{tag} not smoke", summary.get("smoke_evaluation") is False, str(summary.get("smoke_evaluation"))))
    checks.append(check(f"{tag} errors zero", summary.get("error_count") == 0, str(summary.get("error_count"))))
    checks.append(check(f"{tag} claim false", summary.get("claim_eligible") is False, str(summary.get("claim_eligible"))))
    checks.append(check(f"{tag} main table false", summary.get("main_table_eligible") is False, str(summary.get("main_table_eligible"))))
    for stem in ("per_sample_metrics", "method_summary_metrics", "stratum_metrics", "stratum_summary_metrics"):
        csv_path = eval_dir(tag) / f"{tag}_{stem}.csv"
        json_path = eval_dir(tag) / f"{tag}_{stem}.json"
        checks.append(check(f"{tag} {stem} CSV exists", csv_path.exists(), str(csv_path)))
        checks.append(check(f"{tag} {stem} JSON exists", json_path.exists(), str(json_path)))


def build_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    cfg_path = RUN_ROOT / "run_config.json"
    cfg = load_json(cfg_path)
    checks.append(check("ABL-07 run_config exists", isinstance(cfg, dict), str(cfg_path)))
    if isinstance(cfg, dict):
        checks.append(check("run id ABL-07", cfg.get("run_id") == RUN_ID, str(cfg.get("run_id"))))
        checks.append(check("claim_eligible false in run_config", cfg.get("claim_eligible") is False, str(cfg.get("claim_eligible"))))
        checks.append(check("main_table_eligible false in run_config", cfg.get("main_table_eligible") is False, str(cfg.get("main_table_eligible"))))
        checks.append(check("confidence-gated loss enabled", cfg.get("loss_switches", {}).get("confidence_gated_prior_consistency") is True, str(cfg.get("loss_switches"))))
        checks.append(check("direct glare data upweight disabled", cfg.get("loss_switches", {}).get("direct_glare_data_upweight") is False, str(cfg.get("loss_switches"))))
        evaluations = cfg.get("confidence_gated_full_split_evaluations", {})
        checks.append(check("full candidate eval recorded in config", isinstance(evaluations, dict) and FULL_EVAL_TAG in evaluations, str(bool(isinstance(evaluations, dict) and FULL_EVAL_TAG in evaluations))))
        checks.append(check("seed repeat eval recorded in config", isinstance(evaluations, dict) and REPEAT_EVAL_TAG in evaluations, str(bool(isinstance(evaluations, dict) and REPEAT_EVAL_TAG in evaluations))))

    audit_run_artifacts(FULL_TAG, None, checks)
    audit_run_artifacts(REPEAT_TAG, 20260623, checks)
    audit_eval(FULL_EVAL_TAG, FULL_TAG, checks)
    audit_eval(REPEAT_EVAL_TAG, REPEAT_TAG, checks)

    full_row = load_method_row(FULL_EVAL_TAG)
    repeat_row = load_method_row(REPEAT_EVAL_TAG)
    full_mae = numeric(full_row, "mean_mae_um")
    repeat_mae = numeric(repeat_row, "mean_mae_um")
    full_gain = numeric(full_row, "model_vs_dff_gain_ratio_of_means_percent")
    repeat_gain = numeric(repeat_row, "model_vs_dff_gain_ratio_of_means_percent")
    full_win = numeric(full_row, "model_vs_dff_win_rate")
    repeat_win = numeric(repeat_row, "model_vs_dff_win_rate")
    checks.append(check("full candidate mean MAE below old ABL-04", full_mae < OLD_BEST_ABL04_MAE_UM, f"{full_mae:.4f} < {OLD_BEST_ABL04_MAE_UM:.4f}"))
    checks.append(check("seed repeat mean MAE below old ABL-04", repeat_mae < OLD_BEST_ABL04_MAE_UM, f"{repeat_mae:.4f} < {OLD_BEST_ABL04_MAE_UM:.4f}"))
    checks.append(check("full candidate gain over DFF positive", full_gain > 0, f"{full_gain:.2f}%"))
    checks.append(check("seed repeat gain over DFF positive", repeat_gain > 0, f"{repeat_gain:.2f}%"))
    checks.append(check("full candidate DFF win rate at least 4/7", full_win >= 4 / 7, f"{full_win:.4f}"))
    checks.append(check("seed repeat DFF win rate at least 4/7", repeat_win >= 4 / 7, f"{repeat_win:.4f}"))

    full_low = load_stratum_row(FULL_EVAL_TAG, "low_confidence")
    repeat_low = load_stratum_row(REPEAT_EVAL_TAG, "low_confidence")
    full_low_gain = numeric(full_low, "model_vs_dff_gain_ratio_of_means_percent")
    repeat_low_gain = numeric(repeat_low, "model_vs_dff_gain_ratio_of_means_percent")
    full_low_win = numeric(full_low, "model_vs_dff_win_rate")
    repeat_low_win = numeric(repeat_low, "model_vs_dff_win_rate")
    checks.append(check("full low-confidence gain at least 30 percent", full_low_gain >= 30, f"{full_low_gain:.2f}%"))
    checks.append(check("repeat low-confidence gain at least 30 percent", repeat_low_gain >= 30, f"{repeat_low_gain:.2f}%"))
    checks.append(check("full low-confidence win rate at least 6/7", full_low_win >= 6 / 7, f"{full_low_win:.4f}"))
    checks.append(check("repeat low-confidence win rate at least 6/7", repeat_low_win >= 6 / 7, f"{repeat_low_win:.4f}"))

    report_paths = [
        Path("submission_planning/optical_mechanism_analysis/abl07_full_candidate_evidence_report.md"),
        Path("submission_planning/optical_mechanism_analysis/abl07_seed_repeat_stability_report.md"),
        Path("submission_planning/optical_mechanism_analysis/abl07_confidence_gated_prior_decision_note.md"),
    ]
    for path in report_paths:
        checks.append(check(f"report exists: {path.name}", path.exists(), str(path)))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    passed = not errors
    return {
        "status": "pass" if passed else "fail",
        "date": "2026-06-22",
        "run_id": RUN_ID,
        "eligibility_level": "ABL-07 synthetic evidence audit passed; manuscript-table use still gated by real-stack alignment and claim-safety review" if passed else "Not eligible",
        "claim_eligible": False,
        "main_table_eligible": False,
        "old_best_abl04_mae_um": OLD_BEST_ABL04_MAE_UM,
        "summary": {
            "full_candidate_mae_um": full_mae,
            "seed_repeat_mae_um": repeat_mae,
            "full_candidate_gain_vs_dff_percent": full_gain,
            "seed_repeat_gain_vs_dff_percent": repeat_gain,
            "full_candidate_win_rate_vs_dff": full_win,
            "seed_repeat_win_rate_vs_dff": repeat_win,
            "full_low_confidence_gain_percent": full_low_gain,
            "repeat_low_confidence_gain_percent": repeat_low_gain,
        },
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "interpretation": (
            "ABL-07 has two full synthetic train/eval runs, both below the previous ABL-04 mean MAE threshold, "
            "and both preserve a strong low-confidence focus-region gain. Evidence supports elevating ABL-07 as "
            "the main synthetic candidate, while keeping manuscript claims guarded until real-stack alignment."
        ),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# ABL-07 Confidence-Gated Evidence Audit",
        "",
        f"- Status: {report['status']}",
        f"- Eligibility: {report['eligibility_level']}",
        f"- Date: {report['date']}",
        f"- Claim eligible: {str(report['claim_eligible']).lower()}",
        f"- Main table eligible: {str(report['main_table_eligible']).lower()}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Numeric Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Full candidate mean MAE um | {summary['full_candidate_mae_um']:.4f} |",
        f"| Seed repeat mean MAE um | {summary['seed_repeat_mae_um']:.4f} |",
        f"| Old ABL-04 threshold MAE um | {report['old_best_abl04_mae_um']:.4f} |",
        f"| Full candidate gain vs DFF | {summary['full_candidate_gain_vs_dff_percent']:.2f}% |",
        f"| Seed repeat gain vs DFF | {summary['seed_repeat_gain_vs_dff_percent']:.2f}% |",
        f"| Full low-confidence gain | {summary['full_low_confidence_gain_percent']:.2f}% |",
        f"| Repeat low-confidence gain | {summary['repeat_low_confidence_gain_percent']:.2f}% |",
        "",
        "## Checks",
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
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    report = build_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ABL07_confidence_gated_evidence_audit.json"
    md_path = args.out_dir / "ABL07_confidence_gated_evidence_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"ABL-07 evidence audit: {report['status']}")
    print(f"Eligibility: {report['eligibility_level']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
