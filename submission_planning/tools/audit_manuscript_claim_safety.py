"""Audit manuscript claim safety against the current evidence boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MANUSCRIPT = Path("submission_planning/manuscript_draft/s2r_focus_stack_manuscript.tex")
DEFAULT_OUT_DIR = Path("tmp/manuscript_audits")

ERROR_PATTERNS = {
    "external_sota_superiority": [
        re.compile(r"outperform(?:s|ed)?\s+(?:DFV|DDFFNet|HybridDepth|external SOTA)", re.IGNORECASE),
        re.compile(r"surpass(?:es|ed)?\s+(?:DFV|DDFFNet|HybridDepth|external SOTA)", re.IGNORECASE),
        re.compile(r"state[- ]of[- ]the[- ]art\s+(?:performance|results)", re.IGNORECASE),
        re.compile(r"超过\s*(?:DFV|DDFFNet|HybridDepth|外部|SOTA)"),
        re.compile(r"优于\s*(?:DFV|DDFFNet|HybridDepth|外部|SOTA)"),
    ],
    "real_absolute_accuracy": [
        re.compile(r"real(?:-| )world\s+absolute\s+(?:height\s+)?accuracy", re.IGNORECASE),
        re.compile(r"calibrated\s+real\s+(?:height\s+)?(?:accuracy|MAE|error)", re.IGNORECASE),
        re.compile(r"absolute\s+real\s+(?:height\s+)?(?:accuracy|MAE|error)", re.IGNORECASE),
        re.compile(r"真实.*绝对.*(?:精度|误差)"),
    ],
    "broad_generalization": [
        re.compile(r"all\s+industrial\s+surfaces", re.IGNORECASE),
        re.compile(r"all\s+microscop(?:e|y)\s+systems", re.IGNORECASE),
        re.compile(r"全部工业表面"),
        re.compile(r"所有显微"),
    ],
}

WARNING_PATTERNS = {
    "module_effect_without_ablation": [
        re.compile(r"(?:demonstrate|prove|validate)s?\s+the\s+(?:effectiveness|contribution)\s+of\s+(?:each|the)\s+(?:module|component)", re.IGNORECASE),
        re.compile(r"模块.*(?:有效|贡献).*证明"),
    ],
    "domain_randomization_gain": [
        re.compile(r"domain randomization\s+(?:improves|improved|boosts|enhances)", re.IGNORECASE),
        re.compile(r"域随机化.*(?:提升|改善)"),
    ],
    "pseudo_label_gain": [
        re.compile(r"pseudo[- ]label(?:ing)?\s+(?:improves|improved|boosts|enhances)", re.IGNORECASE),
        re.compile(r"伪标签.*(?:提升|改善)"),
    ],
}

REQUIRED_PHRASES = {
    "real_no_reference_boundary": [
        "no-reference",
        "calibrated real height ground truth is unavailable",
    ],
    "external_baseline_pending": [
        "DFV",
        "DDFFNet",
        "not yet reported",
    ],
    "external_not_claimed": [
        "does not claim superiority over external SOTA",
    ],
    "calibrated_validation_future": [
        "calibrated real-height validation",
    ],
}

SAFE_MARKERS = [
    "not yet",
    "does not",
    "do not",
    "without",
    "remain",
    "future",
    "planned",
    "pending",
    "unavailable",
    "not establish",
    "not measure",
    "should not",
    "缺少",
    "未来",
    "不能",
    "不应",
    "尚未",
]


def check(name: str, passed: bool, detail: str, severity: str) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def iter_hits(text: str, patterns: list[re.Pattern[str]]) -> list[dict[str, Any]]:
    hits = []
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


def phrase_present(text: str, options: list[str]) -> bool:
    lowered = text.lower()
    return all(option.lower() in lowered for option in options)


def build_report(manuscript: Path) -> dict[str, Any]:
    text = manuscript.read_text(encoding="utf-8")
    checks = [check("manuscript exists", manuscript.exists(), str(manuscript), "error")]
    findings: dict[str, list[dict[str, Any]]] = {}

    for category, patterns in ERROR_PATTERNS.items():
        hits = iter_hits(text, patterns)
        findings[category] = hits
        checks.append(check(f"no {category}", not hits, f"hits={len(hits)}", "error"))

    for category, patterns in WARNING_PATTERNS.items():
        hits = iter_hits(text, patterns)
        findings[category] = hits
        checks.append(check(f"no unsupported {category}", not hits, f"hits={len(hits)}", "warning"))

    for category, phrases in REQUIRED_PHRASES.items():
        present = phrase_present(text, phrases)
        checks.append(check(f"required boundary phrase: {category}", present, " + ".join(phrases), "error"))

    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "manuscript": str(manuscript),
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
        "findings": findings,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Manuscript Claim Safety Audit",
        "",
        f"- Status: {report['status']}",
        f"- Manuscript: {report['manuscript']}",
        f"- Checks: {report['check_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
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
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_report(args.manuscript)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "manuscript_claim_safety_audit.json"
    md_path = args.out_dir / "manuscript_claim_safety_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Manuscript claim audit: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

