"""Preflight ablation training-entry assumptions without running training."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


SRC_DIR = Path("src")
DEFAULT_MATRIX = Path("submission_planning/autonomous_research/templates/ablation_run_matrix_template.csv")
DEFAULT_ABLATION_ROOT = Path("tmp/ablation_results")
DEFAULT_OUT_DIR = Path("tmp/ablation_results/preflight")
CORE_RUN_IDS = [f"ABL-{idx:02d}" for idx in range(5)]

SOURCE_FILES = {
    "base_training_entry": SRC_DIR / "final_dataset_training.py",
    "focus_resunet_entry": SRC_DIR / "train_focus_resunet_loss_experiment.py",
    "generator_and_metrics": SRC_DIR / "simulate_antiglare_highres_samples.py",
}

REQUIRED_SYMBOLS = {
    "base_training_entry": {
        "functions": {"build_dataset", "train_model", "evaluate_one", "write_metrics"},
        "classes": set(),
    },
    "focus_resunet_entry": {
        "functions": {"augment_features", "upgraded_channel_count", "train_model", "evaluate_split", "predict_tiled_upgraded"},
        "classes": {"FocusResUNet", "HybridDFFLoss"},
    },
    "generator_and_metrics": {
        "functions": {"features_for_model", "feature_channel_count", "generate_sample_arrays", "metrics"},
        "classes": set(),
    },
}


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def parse_python(path: Path) -> tuple[ast.Module | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        return ast.parse(path.read_text(encoding="utf-8")), "ok"
    except SyntaxError as exc:
        return None, f"syntax error: {exc}"


def collect_symbols(tree: ast.Module | None) -> dict[str, set[str]]:
    functions: set[str] = set()
    classes: set[str] = set()
    if tree is None:
        return {"functions": functions, "classes": classes}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
    return {"functions": functions, "classes": classes}


def read_matrix(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_files(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(path for path in files if path.is_file())


def inspect_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for role, path in SOURCE_FILES.items():
        tree, parse_status = parse_python(path)
        symbols = collect_symbols(tree)
        expected = REQUIRED_SYMBOLS[role]
        missing_functions = sorted(expected["functions"] - symbols["functions"])
        missing_classes = sorted(expected["classes"] - symbols["classes"])
        rows.append(check(f"{role} exists", path.exists(), str(path)))
        rows.append(check(f"{role} syntax", parse_status == "ok", parse_status))
        rows.append(check(f"{role} required functions", not missing_functions, f"missing={missing_functions}"))
        rows.append(check(f"{role} required classes", not missing_classes, f"missing={missing_classes}"))
        summary[role] = {
            "path": str(path),
            "parse_status": parse_status,
            "functions_found": sorted(symbols["functions"]),
            "classes_found": sorted(symbols["classes"]),
            "missing_functions": missing_functions,
            "missing_classes": missing_classes,
        }
    return rows, summary


def inspect_matrix(matrix_path: Path, ablation_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix_rows = read_matrix(matrix_path)
    by_id = {row.get("run_id", ""): row for row in matrix_rows}
    rows.append(check("ablation run matrix exists", matrix_path.exists(), str(matrix_path)))
    rows.append(check("core run ids in matrix", all(run_id in by_id for run_id in CORE_RUN_IDS), f"missing={sorted(set(CORE_RUN_IDS) - set(by_id))}"))
    workspace_summary: dict[str, Any] = {}
    for run_id in CORE_RUN_IDS:
        root = ablation_root / run_id
        config = root / "run_config.json"
        synthetic_metrics = root / "metrics" / "synthetic_metrics.csv"
        real_metrics = root / "metrics" / "real_no_reference_metrics.csv"
        rows.append(check(f"{run_id} workspace exists", root.exists(), str(root)))
        rows.append(check(f"{run_id} run_config exists", config.exists(), str(config)))
        rows.append(check(f"{run_id} synthetic metrics template exists", synthetic_metrics.exists(), str(synthetic_metrics)))
        rows.append(check(f"{run_id} real metrics template exists", real_metrics.exists(), str(real_metrics)))
        payload: dict[str, Any] = {}
        if config.exists():
            try:
                payload = json.loads(config.read_text(encoding="utf-8"))
                rows.append(check(f"{run_id} scaffold-only status", payload.get("status") == "scaffold_only_no_training_run", str(payload.get("status"))))
                rows.append(check(f"{run_id} claim_eligible false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            except json.JSONDecodeError as exc:
                rows.append(check(f"{run_id} run_config JSON", False, f"{exc}"))
        produced_files = find_files(root, ["*.pt", "*.pth", "*.ckpt", "*.npy", "*.png", "*.jpg", "*.jpeg"])
        rows.append(check(f"{run_id} no accidental training artifacts", not produced_files, f"artifact files={len(produced_files)}", "warning"))
        workspace_summary[run_id] = {
            "matrix_row": by_id.get(run_id, {}),
            "run_config": payload,
            "artifact_files": [str(path) for path in produced_files],
        }
    return rows, {"matrix_rows": matrix_rows, "workspaces": workspace_summary}


def inspect_existing_evidence(ablation_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    schema = ablation_root / "schema_audit" / "ablation_feature_schema_audit.json"
    abl03 = ablation_root / "ABL-03" / "logs" / "abl03_focal_difference_implementation_audit.json"
    mask = ablation_root / "mask_smoke" / "ablation_mask_smoke_test.json"
    rows = [
        check("feature schema audit JSON exists", schema.exists(), str(schema)),
        check("ABL-03 implementation audit JSON exists", abl03.exists(), str(abl03)),
        check("ablation mask smoke JSON exists", mask.exists(), str(mask)),
    ]
    for name, path in {"schema": schema, "abl03": abl03, "mask": mask}.items():
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append(check(f"{name} report status pass", payload.get("status") == "pass", str(payload.get("status"))))
            except json.JSONDecodeError as exc:
                rows.append(check(f"{name} JSON parse", False, f"{exc}"))
    return rows, {"schema": str(schema), "abl03": str(abl03), "mask": str(mask)}


def corrected_plan() -> list[dict[str, str]]:
    return [
        {
            "run_id": "ABL-00",
            "corrected_entry": "src/train_focus_resunet_loss_experiment.py",
            "feature_space": "upgraded_38_channel",
            "implementation_note": "full Focus-ResUNet / S2R-FocusNet candidate",
        },
        {
            "run_id": "ABL-01",
            "corrected_entry": "derived from src/train_focus_resunet_loss_experiment.py",
            "feature_space": "focus-stack-only variant decision needed",
            "implementation_note": "direct image-to-depth needs a lower-prior runner, not just final_dataset_training.py",
        },
        {
            "run_id": "ABL-02",
            "corrected_entry": "derived from src/train_focus_resunet_loss_experiment.py",
            "feature_space": "upgraded_38_channel recommended",
            "implementation_note": "zero prior channels corresponding to DFF/GADFF after augment_features; base mask 18-21 is already smoke-tested",
        },
        {
            "run_id": "ABL-03",
            "corrected_entry": "src/train_focus_resunet_loss_experiment.py",
            "feature_space": "upgraded_38_channel",
            "implementation_note": "zero channels 17-32 to remove focal-difference input signal",
        },
        {
            "run_id": "ABL-04",
            "corrected_entry": "derived from src/train_focus_resunet_loss_experiment.py",
            "feature_space": "upgraded_38_channel recommended",
            "implementation_note": "zero risk cue safely; decide whether GADFF-derived channels remain or are separated",
        },
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    source_checks, source_summary = inspect_sources()
    matrix_checks, matrix_summary = inspect_matrix(args.matrix, args.ablation_root)
    evidence_checks, evidence_summary = inspect_existing_evidence(args.ablation_root)
    checks.extend(source_checks)
    checks.extend(matrix_checks)
    checks.extend(evidence_checks)
    plan = corrected_plan()
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "date": date.today().isoformat(),
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
        "source_summary": source_summary,
        "matrix_summary": matrix_summary,
        "existing_evidence": evidence_summary,
        "corrected_plan": plan,
        "decision": "ready-to-design-minimal-ablation-runner" if not errors else "blocked-before-ablation-runner",
        "interpretation": "Preflight only. No model training, inference, checkpoints, or module-effectiveness claims were produced.",
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ablation Training Entry Preflight",
        "",
        f"- Status: {report['status']}",
        f"- Decision: {report['decision']}",
        f"- Date: {report['date']}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Corrected Plan",
        "",
        "| Run ID | Corrected Entry | Feature Space | Implementation Note |",
        "|---|---|---|---|",
    ]
    for row in report["corrected_plan"]:
        lines.append(f"| {row['run_id']} | `{row['corrected_entry']}` | {row['feature_space']} | {row['implementation_note']} |")
    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            "| Role | Path | Parse | Missing Functions | Missing Classes |",
            "|---|---|---|---|---|",
        ]
    )
    for role, payload in report["source_summary"].items():
        lines.append(
            f"| {role} | `{payload['path']}` | {payload['parse_status']} | "
            f"{payload['missing_functions']} | {payload['missing_classes']} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Status | Severity | Detail |",
            "|---|---|---|---|",
        ]
    )
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ablation_training_entry_preflight.json"
    md_path = args.out_dir / "ablation_training_entry_preflight.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Ablation training-entry preflight: {report['status']} ({report['decision']})")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
