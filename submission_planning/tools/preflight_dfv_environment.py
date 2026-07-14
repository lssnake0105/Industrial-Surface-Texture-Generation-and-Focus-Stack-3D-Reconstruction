"""Preflight local readiness for a future DFV external-baseline smoke test.

This script is intentionally non-invasive. It does not download repositories,
install packages, import DFV code, run models, or create predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_METHOD_ROOT = Path("tmp/external_baseline_results/DFV")
DEFAULT_REPO_ROOT = Path("tmp/external_repos/DFV")
DEFAULT_SAMPLE_DIR = Path("tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底")
DEFAULT_EXPORT_MANIFEST = Path("tmp/external_baseline_data/manifest.csv")


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"name": name, "available": False, "version": "", "detail": "not importable"}
    version = ""
    detail = "importable"
    if name in {"numpy", "PIL", "torch"}:
        try:
            module = __import__(name)
            version = str(getattr(module, "__version__", ""))
            if name == "torch":
                cuda_available = bool(module.cuda.is_available())
                cuda_device_count = int(module.cuda.device_count())
                detail = f"cuda_available={cuda_available}, cuda_device_count={cuda_device_count}"
        except Exception as exc:  # pragma: no cover - defensive environment report
            return {"name": name, "available": False, "version": "", "detail": f"import failed: {exc}"}
    return {"name": name, "available": True, "version": version, "detail": detail}


def file_count(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    method_root = args.method_root
    sample_dir = args.sample_dir
    repo_root = args.repo_root
    predictions_dir = method_root / "predictions"
    evaluation_dir = method_root / "evaluation"
    batch_dir = method_root / "batch_evaluation"
    logs_dir = method_root / "logs"
    preflight_dir = method_root / "preflight"

    modules = {name: module_status(name) for name in ["numpy", "PIL", "torch"]}
    repo_exists = repo_root.exists()
    stack_frames = file_count(sample_dir / "stack", "*.png")
    prediction_files = list(predictions_dir.glob("*.npy")) if predictions_dir.exists() else []

    checks = [
        check("DFV workspace exists", method_root.exists(), str(method_root)),
        check("DFV repository path status recorded", True, f"{repo_root} exists={repo_exists}", "info"),
        check("external data manifest exists", args.export_manifest.exists(), str(args.export_manifest)),
        check("P10 sample dir exists", sample_dir.exists(), str(sample_dir)),
        check("P10 stack has 17 frames", stack_frames == 17, f"frames={stack_frames}"),
        check("P10 height_gt exists", (sample_dir / "height_gt.npy").exists(), str(sample_dir / "height_gt.npy")),
        check("P10 high-risk mask exists", (sample_dir / "masks" / "high_risk_mask.npy").exists(), str(sample_dir / "masks" / "high_risk_mask.npy")),
        check("P10 DFF prior exists", (sample_dir / "priors" / "dff_depth.npy").exists(), str(sample_dir / "priors" / "dff_depth.npy")),
        check("P10 GADFF prior exists", (sample_dir / "priors" / "gadff_depth.npy").exists(), str(sample_dir / "priors" / "gadff_depth.npy")),
        check("DFV predictions dir exists", predictions_dir.exists(), str(predictions_dir)),
        check("DFV evaluation dir exists", evaluation_dir.exists(), str(evaluation_dir)),
        check("DFV batch evaluation dir exists", batch_dir.exists(), str(batch_dir)),
        check("DFV logs dir exists", logs_dir.exists(), str(logs_dir)),
        check("DFV preflight dir creatable", preflight_dir.parent.exists(), str(preflight_dir.parent)),
        check("NumPy importable", modules["numpy"]["available"], modules["numpy"]["detail"]),
        check("PIL importable", modules["PIL"]["available"], modules["PIL"]["detail"]),
        check("PyTorch importable", modules["torch"]["available"], modules["torch"]["detail"], "warning"),
        check("no accidental DFV predictions", not prediction_files, f"prediction files={len(prediction_files)}", "warning"),
    ]
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    if errors:
        decision = "blocked-before-repo-download"
    elif not modules["torch"]["available"]:
        decision = "data-ready-but-pytorch-missing"
    elif not repo_exists:
        decision = "ready-for-repository-download-under-tmp"
    else:
        decision = "ready-for-dfv-code-inventory"
    return {
        "status": "pass" if not errors else "fail",
        "decision": decision,
        "date": date.today().isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "paths": {
            "method_root": str(method_root),
            "repo_root": str(repo_root),
            "sample_dir": str(sample_dir),
            "export_manifest": str(args.export_manifest),
            "predictions_dir": str(predictions_dir),
            "preflight_dir": str(preflight_dir),
        },
        "repo_exists": repo_exists,
        "stack_frame_count": stack_frames,
        "modules": modules,
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DFV Environment Preflight",
        "",
        f"- Status: {report['status']}",
        f"- Decision: {report['decision']}",
        f"- Date: {report['date']}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Python",
        "",
        f"- Executable: `{report['python']['executable']}`",
        f"- Version: `{report['python']['version']}`",
        f"- Platform: `{report['python']['platform']}`",
        "",
        "## Paths",
        "",
        "| Field | Path |",
        "|---|---|",
    ]
    for key, value in report["paths"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Modules",
            "",
            "| Module | Available | Version | Detail |",
            "|---|---|---|---|",
        ]
    )
    for name, payload in report["modules"].items():
        lines.append(f"| {name} | {payload['available']} | {payload['version']} | {payload['detail']} |")
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This preflight is not a DFV result. It only records local readiness for a later repository inventory or loader smoke test.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", type=Path, default=DEFAULT_METHOD_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_EXPORT_MANIFEST)
    args = parser.parse_args()

    report = build_report(args)
    preflight_dir = args.method_root / "preflight"
    log_dir = args.method_root / "logs"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = preflight_dir / "dfv_environment_preflight.json"
    md_path = preflight_dir / "dfv_environment_preflight.md"
    log_path = log_dir / f"{report['date']}_dfv_environment_preflight.md"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    json_path.write_text(payload + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    write_markdown(log_path, report)
    print(f"DFV preflight: {report['status']} ({report['decision']})")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {log_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
