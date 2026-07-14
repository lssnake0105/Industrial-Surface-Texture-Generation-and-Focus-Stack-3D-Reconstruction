"""Audit ABL-07 evidence reports for manuscript-claim safety."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPORTS = [
    Path("submission_planning/optical_mechanism_analysis/abl07_full_candidate_evidence_report.md"),
    Path("submission_planning/optical_mechanism_analysis/abl07_seed_repeat_stability_report.md"),
    Path("submission_planning/optical_mechanism_analysis/abl07_confidence_gated_prior_decision_note.md"),
]
DEFAULT_OUT_DIR = Path("tmp/ablation_results/eligibility_audits")

ERROR_PATTERNS = {
    "external_sota_superiority": [
        re.compile(r"outperform(?:s|ed)?\s+(?:external\s+)?(?:SOTA|state[- ]of[- ]the[- ]art|DFV|DDFFNet|HybridDepth)", re.IGNORECASE),
        re.compile(r"优于\s*(?:外部|SOTA|DFV|DDFFNet|HybridDepth)"),
    ],
    "real_absolute_accuracy": [
        re.compile(r"real(?:-| )stack\s+(?:absolute\s+)?(?:height\s+)?(?:accuracy|MAE|error)", re.IGNORECASE),
        re.compile(r"calibrated\s+real\s+(?:height\s+)?(?:accuracy|MAE|error)", re.IGNORECASE),
        re.compile(r"真实.*(?:绝对|标定).*(?:精度|误差|MAE)"),
    ],
    "final_manuscript_claim": [
        re.compile(r"final(?:ized)?\s+(?:paper|manuscript)\s+claim", re.IGNORECASE),
        re.compile(r"ready\s+for\s+(?:paper|manuscript)\s+table", re.IGNORECASE),
        re.compile(r"可直接.*(?:论文|投稿|主表)"),
    ],
}

WARNING_PATTERNS = {
    "absolute_best_language": [
        re.compile(r"\bbest\b", re.IGNORECASE),
        re.compile(r"最优|最好"),
    ],
    "replace_language": [
        re.compile(r"\breplace(?:s|d)?\b", re.IGNORECASE),
        re.compile(r"替代|取代"),
    ],
}

REQUIRED_TERMS = [
    "claim-ineligible",
    "real-stack",
    "audit",
    "low-confidence",
    "confidence-gated",
]

SAFE_MARKERS = [
    "previous",
    "old",
    "candidate",
    "until",
    "not",
    "avoid",
    "should",
    "still",
    "gated",
    "旧",
    "候选",
    "仍",
    "不",
    "避免",
    "需要",
]


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def collect_text(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    chunks: list[str] = []
    for path in paths:
        exists = path.exists()
        checks.append(check(f"report exists: {path.name}", exists, str(path), "error"))
        if exists:
            chunks.append(f"\n\n--- {path} ---\n" + path.read_text(encoding="utf-8"))
    return "\n".join(chunks), checks


def iter_hits(text: str, patterns: list[re.Pattern[str]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            if not pattern.search(line):
                continue
            lowered = line.lower()
            if any(marker.lower() in lowered for marker in SAFE_MARKERS):
                continue
            hits.append({"line": lineno, "text": line.strip(), "pattern": pattern.pattern})
            break
    return hits


def build_report(paths: list[Path]) -> dict[str, Any]:
    text, checks = collect_text(paths)
    findings: dict[str, list[dict[str, Any]]] = {}
    for category, patterns in ERROR_PATTERNS.items():
        hits = iter_hits(text, patterns)
        findings[category] = hits
        checks.append(check(f"no unsafe {category}", not hits, f"hits={len(hits)}", "error"))
    for category, patterns in WARNING_PATTERNS.items():
        hits = iter_hits(text, patterns)
        findings[category] = hits
        checks.append(check(f"review {category}", True, f"hits={len(hits)}", "warning"))
    lowered = text.lower()
    for term in REQUIRED_TERMS:
        checks.append(check(f"required boundary term: {term}", term.lower() in lowered, term, "error"))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "date": "2026-06-22",
        "reports": [str(path) for path in paths],
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "claim_safe_for_internal_evidence_report": not errors,
        "claim_safe_for_manuscript_table": False,
        "interpretation": "ABL-07 report language is safe for internal evidence tracking. Manuscript-table claims remain guarded until the final manuscript-level claim review; calibrated real-height accuracy remains unsupported without real height ground truth.",
        "checks": checks,
        "findings": findings,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ABL-07 Claim Safety Audit",
        "",
        f"- Status: {report['status']}",
        f"- Date: {report['date']}",
        f"- Internal evidence report safe: {str(report['claim_safe_for_internal_evidence_report']).lower()}",
        f"- Manuscript table safe: {str(report['claim_safe_for_manuscript_table']).lower()}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        report["interpretation"],
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---|---|---|",
    ]
    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['check']} | {status} | {row['severity']} | {row['detail']} |")
    lines.extend(["", "## Findings", ""])
    any_findings = False
    for category, hits in report["findings"].items():
        if not hits:
            continue
        any_findings = True
        lines.append(f"### {category}")
        for hit in hits:
            lines.append(f"- line {hit['line']}: {hit['text']}")
        lines.append("")
    if not any_findings:
        lines.append("- none")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, help="Report path to audit. May be repeated.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    paths = args.report or DEFAULT_REPORTS
    report = build_report(paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ABL07_claim_safety_audit.json"
    md_path = args.out_dir / "ABL07_claim_safety_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"ABL-07 claim safety audit: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
