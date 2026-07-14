"""Audit the autonomous research package without modifying project assets."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


RESEARCH_DIR = Path("submission_planning/autonomous_research")
TOOLS_DIR = Path("submission_planning/tools")
DEFAULT_OUT_DIR = Path("tmp/research_package_audits")

REQUIRED_DOCS = {
    "index_and_task_control": [
        "research_index.md",
        "research_task_board.md",
        "experiment_roadmap.md",
        "research_log_2026-06-18_ablation_training_entry_preflight.md",
        "research_log_2026-06-19_minimal_ablation_runner_smoke.md",
        "research_log_2026-06-19_ablation_training_runner_preflight.md",
        "research_log_2026-06-19_ablation_training_runner_dry_run.md",
        "research_log_2026-06-19_ablation_small_training_debug.md",
        "research_log_2026-06-19_development_resume_checkpoint.md",
        "research_log_2026-06-19_ablation_controlled_pilot.md",
        "research_log_2026-06-19_ablation_full_split_debug_eval.md",
        "research_log_2026-06-19_ablation_matched_training_preflight.md",
        "research_log_2026-06-19_ablation_matched_training_smoke.md",
        "research_log_2026-06-19_ablation_full_matched_configuration.md",
        "research_log_2026-06-19_ablation_matched_evaluator_smoke.md",
        "research_log_2026-06-19_ablation_full_candidate_results.md",
        "research_log_2026-06-19_ablation_longer_repeat.md",
        "recovery_breakpoint_2026-06-19_ablation_debug_to_pilot.md",
        "recovery_breakpoint_2026-06-19_matched_training_preflight_to_smoke.md",
        "recovery_breakpoint_2026-06-19_matched_smoke_to_full_config.md",
        "recovery_breakpoint_2026-06-19_full_config_to_matched_evaluator.md",
        "recovery_breakpoint_2026-06-19_matched_evaluator_to_full_training.md",
        "recovery_breakpoint_2026-06-19_after_full_candidate_eval.md",
        "recovery_breakpoint_2026-06-19_after_longer_repeat.md",
    ],
    "sota_and_claim_safety": [
        "external_baseline_feasibility.md",
        "claim_evidence_matrix.md",
        "external_sota_next_decision_log.md",
        "external_sota_eligibility_audit.md",
        "depth_anything_v2_auxiliary_protocol.md",
        "submission_gap_closure_plan.md",
        "research_package_integrity_audit.md",
        "manuscript_claim_safety_audit.md",
    ],
    "external_execution": [
        "baseline_adapter_spec.md",
        "external_baseline_data_preflight.md",
        "dfv_ddffnet_integration_protocol.md",
        "external_baseline_workspace_scaffold.md",
        "dfv_environment_preflight.md",
    ],
    "manuscript_planning": [
        "manuscript_blueprint.md",
        "latex_manuscript_assembly_plan.md",
        "abstract_draft_en.md",
        "introduction_outline_en.md",
        "method_outline_en.md",
        "experiments_outline_en.md",
        "discussion_outline_en.md",
    ],
    "experiment_design": [
        "ablation_design.md",
        "ablation_execution_protocol.md",
        "ablation_feature_schema_audit.md",
        "abl03_focal_difference_implementation_audit.md",
        "ablation_mask_smoke_test.md",
        "ablation_training_entry_preflight.md",
        "minimal_ablation_runner_smoke.md",
        "ablation_training_runner_preflight.md",
        "ablation_training_runner_dry_run.md",
        "ablation_small_training_debug.md",
        "ablation_controlled_pilot.md",
        "ablation_full_split_debug_evaluation.md",
        "ablation_matched_training_preflight.md",
        "ablation_matched_training_smoke.md",
        "ablation_full_matched_training_configuration.md",
        "ablation_matched_full_split_evaluator_smoke.md",
        "ablation_matched_full_candidate_results.md",
        "ablation_matched_longer_repeat_results.md",
        "supervisor_update_2026-06-19.md",
        "supervisor_experiment_report_2026-06-19.md",
        "failure_analysis_plan.md",
        "figure_table_plan.md",
        "submission_readiness_checklist.md",
    ],
}

REQUIRED_TOOLS = [
    "preflight_external_baseline_data.py",
    "export_one_external_baseline_sample.py",
    "preflight_dfv_environment.py",
    "smoke_test_external_baseline_package.py",
    "evaluate_external_prediction.py",
    "evaluate_external_prediction_batch.py",
    "audit_external_sota_eligibility.py",
    "scaffold_external_baseline_workspace.py",
    "audit_manuscript_claim_safety.py",
    "scaffold_ablation_workspace.py",
    "audit_ablation_feature_schema.py",
    "audit_abl03_focal_difference_implementation.py",
    "smoke_test_ablation_masks.py",
    "preflight_ablation_training_entry.py",
    "run_ablation_variant_smoke.py",
    "preflight_ablation_training_runner.py",
    "run_ablation_variant_training.py",
    "audit_ablation_pilot_eligibility.py",
    "evaluate_ablation_full_split_metrics.py",
    "audit_ablation_full_split_eligibility.py",
    "preflight_ablation_matched_training.py",
    "audit_ablation_matched_smoke_eligibility.py",
    "preflight_ablation_full_matched_configuration.py",
    "evaluate_ablation_matched_full_split_metrics.py",
    "audit_ablation_matched_training_eligibility.py",
    "scaffold_depth_anything_v2_auxiliary_workspace.py",
]

METHODS = ["DFV", "DDFFNet"]
ABLATION_RUN_IDS = [f"ABL-{idx:02d}" for idx in range(7)]
FORBIDDEN_PATTERN = re.compile("\u4e0d\u662f" + r".*" + "\u800c\u662f")
UNSUPPORTED_CLAIM_PATTERNS = [
    re.compile(r"outperform(?:s|ed)?\s+(?:DFV|DDFFNet)", re.IGNORECASE),
    re.compile(r"exceed(?:s|ed)?\s+(?:DFV|DDFFNet)", re.IGNORECASE),
    re.compile(r"超过\s*(?:DFV|DDFFNet)"),
    re.compile(r"优于\s*(?:DFV|DDFFNet)"),
]
SAFE_CLAIM_CONTEXT_MARKERS = [
    "not yet",
    "not allowed",
    "not supported",
    "no claim",
    "cannot",
    "do not",
    "pending",
    "若",
    "如果",
    "不能",
    "不应",
    "不允许",
    "缺少",
    "需要新增实验",
    "禁止",
    "claim that",
]


def check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": severity, "detail": detail}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def required_doc_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, names in REQUIRED_DOCS.items():
        for name in names:
            path = RESEARCH_DIR / name
            rows.append(check(f"required doc: {category}/{name}", path.exists(), str(path)))
    return rows


def tool_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_TOOLS:
        path = TOOLS_DIR / name
        rows.append(check(f"required tool: {name}", path.exists(), str(path)))
        if path.exists():
            try:
                ast.parse(read_text(path))
                rows.append(check(f"tool syntax: {name}", True, "AST parse ok"))
            except SyntaxError as exc:
                rows.append(check(f"tool syntax: {name}", False, f"{exc}"))
    cache_paths = list(TOOLS_DIR.rglob("__pycache__")) if TOOLS_DIR.exists() else []
    rows.append(check("no tool __pycache__", not cache_paths, f"cache dirs={len(cache_paths)}"))
    return rows


def text_safety_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    forbidden_hits = []
    unsupported_hits = []
    for path in sorted(Path("submission_planning").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".tex", ".txt"}:
            continue
        text = read_text(path)
        if FORBIDDEN_PATTERN.search(text):
            forbidden_hits.append(str(path))
        if has_unsupported_superiority_claim(text):
            unsupported_hits.append(str(path))
    rows.append(check("no forbidden Chinese pattern", not forbidden_hits, f"hits={forbidden_hits}"))
    rows.append(
        check(
            "no unsupported DFV/DDFFNet superiority claim",
            not unsupported_hits,
            f"hits={unsupported_hits}",
        )
    )
    return rows


def has_unsupported_superiority_claim(text: str) -> bool:
    for line in text.splitlines():
        if not any(pattern.search(line) for pattern in UNSUPPORTED_CLAIM_PATTERNS):
            continue
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in SAFE_CLAIM_CONTEXT_MARKERS):
            continue
        return True
    return False


def method_workspace_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        root = Path("tmp/external_baseline_results") / method
        run_config = root / "run_config.json"
        manifest = root / "prediction_manifest.csv"
        logs_dir = root / "logs"
        predictions_dir = root / "predictions"
        preflight_dir = root / "preflight"
        preflight_md = preflight_dir / "dfv_environment_preflight.md"
        preflight_json = preflight_dir / "dfv_environment_preflight.json"
        rows.extend(
            [
                check(f"{method} workspace exists", root.exists(), str(root)),
                check(f"{method} run_config exists", run_config.exists(), str(run_config)),
                check(f"{method} prediction manifest exists", manifest.exists(), str(manifest)),
                check(f"{method} logs dir exists", logs_dir.exists(), str(logs_dir)),
                check(f"{method} predictions dir exists", predictions_dir.exists(), str(predictions_dir)),
            ]
        )
        if method == "DFV":
            rows.append(check("DFV environment preflight report exists", preflight_md.exists(), str(preflight_md)))
            rows.append(check("DFV environment preflight JSON exists", preflight_json.exists(), str(preflight_json)))
            if preflight_json.exists():
                try:
                    preflight_payload = json.loads(preflight_json.read_text(encoding="utf-8"))
                    rows.append(check("DFV preflight status pass", preflight_payload.get("status") == "pass", str(preflight_payload.get("status"))))
                    rows.append(check("DFV preflight has no errors", preflight_payload.get("error_count") == 0, str(preflight_payload.get("error_count"))))
                except json.JSONDecodeError as exc:
                    rows.append(check("DFV preflight JSON parse", False, f"{exc}"))
        if run_config.exists():
            try:
                payload = json.loads(run_config.read_text(encoding="utf-8"))
                status_ok = payload.get("status") == "scaffold_only_no_external_prediction"
                eligible_ok = payload.get("main_table_eligible") is False
                rows.append(check(f"{method} status is scaffold-only", status_ok, str(payload.get("status"))))
                rows.append(check(f"{method} main_table_eligible false", eligible_ok, str(payload.get("main_table_eligible"))))
            except json.JSONDecodeError as exc:
                rows.append(check(f"{method} run_config JSON", False, f"{exc}"))
        prediction_files = sorted(predictions_dir.glob("*.npy")) if predictions_dir.exists() else []
        rows.append(
            check(
                f"{method} no accidental prediction files",
                not prediction_files,
                f"prediction files={len(prediction_files)}",
                severity="warning",
            )
        )
    return rows


def data_package_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = Path("tmp/external_baseline_data/manifest.csv")
    sample_root = Path("tmp/external_baseline_data/samples")
    rows.append(check("external data manifest exists", manifest.exists(), str(manifest)))
    rows.append(check("external data samples dir exists", sample_root.exists(), str(sample_root)))
    if manifest.exists():
        text = read_text(manifest)
        rows.append(check("P10 sample in manifest", "test_V谷_P10_宽谷粗糙平底" in text, "P10 sample expected"))
    return rows


def ablation_workspace_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path("tmp/ablation_results")
    schema_report = root / "schema_audit" / "ablation_feature_schema_audit.md"
    schema_json = root / "schema_audit" / "ablation_feature_schema_audit.json"
    abl03_report = root / "ABL-03" / "logs" / "abl03_focal_difference_implementation_audit.md"
    abl03_json = root / "ABL-03" / "logs" / "abl03_focal_difference_implementation_audit.json"
    mask_report = root / "mask_smoke" / "ablation_mask_smoke_test.md"
    mask_json = root / "mask_smoke" / "ablation_mask_smoke_test.json"
    entry_report = root / "preflight" / "ablation_training_entry_preflight.md"
    entry_json = root / "preflight" / "ablation_training_entry_preflight.json"
    runner_smoke_report = root / "runner_smoke" / "minimal_ablation_runner_smoke.md"
    runner_smoke_json = root / "runner_smoke" / "minimal_ablation_runner_smoke.json"
    training_preflight_report = root / "training_runner_preflight" / "ablation_training_runner_preflight.md"
    training_preflight_json = root / "training_runner_preflight" / "ablation_training_runner_preflight.json"
    training_dry_run_report = root / "training_runner_dry_run" / "ablation_training_runner_dry_run_summary.md"
    training_dry_run_json = root / "training_runner_dry_run" / "ablation_training_runner_dry_run_summary.json"
    abl01_decision = root / "ABL-01" / "logs" / "2026-06-19_lower_prior_architecture_decision.md"
    abl00_dry_run = root / "ABL-00" / "logs" / "2026-06-19_training_runner_dry_run.md"
    abl03_dry_run = root / "ABL-03" / "logs" / "2026-06-19_training_runner_dry_run.md"
    small_train_report = root / "training_runner_small_train" / "ablation_training_runner_small_train_summary.md"
    small_train_json = root / "training_runner_small_train" / "ablation_training_runner_small_train_summary.json"
    pilot_report = root / "training_runner_controlled_pilot" / "2026-06-19_controlled_pilot_summary.md"
    pilot_json = root / "training_runner_controlled_pilot" / "2026-06-19_controlled_pilot_summary.json"
    pilot_eligibility_report = root / "eligibility_audits" / "ABL_pilot_eligibility.md"
    pilot_eligibility_json = root / "eligibility_audits" / "ABL_pilot_eligibility.json"
    full_split_report = root / "full_split_debug_eval" / "2026-06-19_full_split_debug_eval_summary.md"
    full_split_json = root / "full_split_debug_eval" / "2026-06-19_full_split_debug_eval_summary.json"
    full_split_per_sample_csv = root / "full_split_debug_eval" / "2026-06-19_full_split_debug_eval_per_sample_metrics.csv"
    full_split_summary_csv = root / "full_split_debug_eval" / "2026-06-19_full_split_debug_eval_method_summary_metrics.csv"
    full_split_eligibility_report = root / "eligibility_audits" / "ABL_full_split_eligibility.md"
    full_split_eligibility_json = root / "eligibility_audits" / "ABL_full_split_eligibility.json"
    matched_preflight_report = root / "matched_training_preflight" / "2026-06-19_matched_training_preflight.md"
    matched_preflight_json = root / "matched_training_preflight" / "2026-06-19_matched_training_preflight.json"
    matched_smoke_report = root / "training_runner_matched_smoke" / "2026-06-19_matched_training_smoke_summary.md"
    matched_smoke_json = root / "training_runner_matched_smoke" / "2026-06-19_matched_training_smoke_summary.json"
    matched_smoke_eligibility_report = root / "eligibility_audits" / "ABL_matched_smoke_eligibility.md"
    matched_smoke_eligibility_json = root / "eligibility_audits" / "ABL_matched_smoke_eligibility.json"
    full_matched_config_report = root / "matched_training_full_config" / "2026-06-19_matched_training_full_candidate_config_preflight.md"
    full_matched_config_json = root / "matched_training_full_config" / "2026-06-19_matched_training_full_candidate_config_preflight.json"
    matched_evaluator_smoke_report = root / "matched_full_split_eval" / "2026-06-19_matched_evaluator_smoke" / "2026-06-19_matched_evaluator_smoke_summary.md"
    matched_evaluator_smoke_json = root / "matched_full_split_eval" / "2026-06-19_matched_evaluator_smoke" / "2026-06-19_matched_evaluator_smoke_summary.json"
    matched_full_training_report = root / "training_runner_matched_full_candidate" / "2026-06-19_matched_training_full_candidate_summary.md"
    matched_full_training_json = root / "training_runner_matched_full_candidate" / "2026-06-19_matched_training_full_candidate_summary.json"
    matched_full_eval_report = root / "matched_full_split_eval" / "2026-06-19_matched_full_candidate_eval" / "2026-06-19_matched_full_candidate_eval_summary.md"
    matched_full_eval_json = root / "matched_full_split_eval" / "2026-06-19_matched_full_candidate_eval" / "2026-06-19_matched_full_candidate_eval_summary.json"
    matched_full_eligibility_report = root / "eligibility_audits" / "ABL_matched_training_eligibility.md"
    matched_full_eligibility_json = root / "eligibility_audits" / "ABL_matched_training_eligibility.json"
    matched_longer_training_report = root / "training_runner_matched_longer_repeat" / "2026-06-19_matched_training_longer_repeat_summary.md"
    matched_longer_training_json = root / "training_runner_matched_longer_repeat" / "2026-06-19_matched_training_longer_repeat_summary.json"
    matched_longer_eval_report = root / "matched_full_split_eval" / "2026-06-19_matched_longer_repeat_eval" / "2026-06-19_matched_longer_repeat_eval_summary.md"
    matched_longer_eval_json = root / "matched_full_split_eval" / "2026-06-19_matched_longer_repeat_eval" / "2026-06-19_matched_longer_repeat_eval_summary.json"
    rows.append(check("ablation workspace root exists", root.exists(), str(root)))
    rows.append(check("ablation feature schema report exists", schema_report.exists(), str(schema_report)))
    rows.append(check("ablation feature schema JSON exists", schema_json.exists(), str(schema_json)))
    rows.append(check("ABL-03 focal-difference audit report exists", abl03_report.exists(), str(abl03_report)))
    rows.append(check("ABL-03 focal-difference audit JSON exists", abl03_json.exists(), str(abl03_json)))
    rows.append(check("ablation mask smoke report exists", mask_report.exists(), str(mask_report)))
    rows.append(check("ablation mask smoke JSON exists", mask_json.exists(), str(mask_json)))
    rows.append(check("ablation training-entry preflight report exists", entry_report.exists(), str(entry_report)))
    rows.append(check("ablation training-entry preflight JSON exists", entry_json.exists(), str(entry_json)))
    rows.append(check("minimal ablation runner smoke report exists", runner_smoke_report.exists(), str(runner_smoke_report)))
    rows.append(check("minimal ablation runner smoke JSON exists", runner_smoke_json.exists(), str(runner_smoke_json)))
    rows.append(check("ablation training-runner preflight report exists", training_preflight_report.exists(), str(training_preflight_report)))
    rows.append(check("ablation training-runner preflight JSON exists", training_preflight_json.exists(), str(training_preflight_json)))
    rows.append(check("ablation training-runner dry-run summary exists", training_dry_run_report.exists(), str(training_dry_run_report)))
    rows.append(check("ablation training-runner dry-run JSON exists", training_dry_run_json.exists(), str(training_dry_run_json)))
    rows.append(check("ABL-01 lower-prior decision log exists", abl01_decision.exists(), str(abl01_decision)))
    rows.append(check("ABL-00 training-runner dry-run log exists", abl00_dry_run.exists(), str(abl00_dry_run)))
    rows.append(check("ABL-03 training-runner dry-run log exists", abl03_dry_run.exists(), str(abl03_dry_run)))
    rows.append(check("ablation small-training debug summary exists", small_train_report.exists(), str(small_train_report)))
    rows.append(check("ablation small-training debug JSON exists", small_train_json.exists(), str(small_train_json)))
    rows.append(check("ablation controlled pilot summary exists", pilot_report.exists(), str(pilot_report)))
    rows.append(check("ablation controlled pilot JSON exists", pilot_json.exists(), str(pilot_json)))
    rows.append(check("ablation pilot eligibility report exists", pilot_eligibility_report.exists(), str(pilot_eligibility_report)))
    rows.append(check("ablation pilot eligibility JSON exists", pilot_eligibility_json.exists(), str(pilot_eligibility_json)))
    rows.append(check("ablation full-split debug summary exists", full_split_report.exists(), str(full_split_report)))
    rows.append(check("ablation full-split debug JSON exists", full_split_json.exists(), str(full_split_json)))
    rows.append(check("ablation full-split per-sample CSV exists", full_split_per_sample_csv.exists(), str(full_split_per_sample_csv)))
    rows.append(check("ablation full-split method summary CSV exists", full_split_summary_csv.exists(), str(full_split_summary_csv)))
    rows.append(check("ablation full-split eligibility report exists", full_split_eligibility_report.exists(), str(full_split_eligibility_report)))
    rows.append(check("ablation full-split eligibility JSON exists", full_split_eligibility_json.exists(), str(full_split_eligibility_json)))
    rows.append(check("ablation matched training preflight report exists", matched_preflight_report.exists(), str(matched_preflight_report)))
    rows.append(check("ablation matched training preflight JSON exists", matched_preflight_json.exists(), str(matched_preflight_json)))
    rows.append(check("ablation matched training smoke summary exists", matched_smoke_report.exists(), str(matched_smoke_report)))
    rows.append(check("ablation matched training smoke JSON exists", matched_smoke_json.exists(), str(matched_smoke_json)))
    rows.append(check("ablation matched smoke eligibility report exists", matched_smoke_eligibility_report.exists(), str(matched_smoke_eligibility_report)))
    rows.append(check("ablation matched smoke eligibility JSON exists", matched_smoke_eligibility_json.exists(), str(matched_smoke_eligibility_json)))
    rows.append(check("ablation full matched config report exists", full_matched_config_report.exists(), str(full_matched_config_report)))
    rows.append(check("ablation full matched config JSON exists", full_matched_config_json.exists(), str(full_matched_config_json)))
    rows.append(check("ablation matched evaluator smoke report exists", matched_evaluator_smoke_report.exists(), str(matched_evaluator_smoke_report)))
    rows.append(check("ablation matched evaluator smoke JSON exists", matched_evaluator_smoke_json.exists(), str(matched_evaluator_smoke_json)))
    rows.append(check("ablation matched full candidate training report exists", matched_full_training_report.exists(), str(matched_full_training_report)))
    rows.append(check("ablation matched full candidate training JSON exists", matched_full_training_json.exists(), str(matched_full_training_json)))
    rows.append(check("ablation matched full candidate eval report exists", matched_full_eval_report.exists(), str(matched_full_eval_report)))
    rows.append(check("ablation matched full candidate eval JSON exists", matched_full_eval_json.exists(), str(matched_full_eval_json)))
    rows.append(check("ablation matched full candidate eligibility report exists", matched_full_eligibility_report.exists(), str(matched_full_eligibility_report)))
    rows.append(check("ablation matched full candidate eligibility JSON exists", matched_full_eligibility_json.exists(), str(matched_full_eligibility_json)))
    rows.append(check("ablation matched longer repeat training report exists", matched_longer_training_report.exists(), str(matched_longer_training_report)))
    rows.append(check("ablation matched longer repeat training JSON exists", matched_longer_training_json.exists(), str(matched_longer_training_json)))
    rows.append(check("ablation matched longer repeat eval report exists", matched_longer_eval_report.exists(), str(matched_longer_eval_report)))
    rows.append(check("ablation matched longer repeat eval JSON exists", matched_longer_eval_json.exists(), str(matched_longer_eval_json)))
    if entry_json.exists():
        try:
            payload = json.loads(entry_json.read_text(encoding="utf-8"))
            rows.append(check("ablation training-entry preflight status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation training-entry preflight has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            corrected_plan = payload.get("corrected_plan", [])
            corrected_ids = {row.get("run_id") for row in corrected_plan if isinstance(row, dict)}
            core_ids = {f"ABL-{idx:02d}" for idx in range(5)}
            rows.append(check("ablation training-entry corrected plan covers ABL-00..04", core_ids.issubset(corrected_ids), f"covered={sorted(corrected_ids)}"))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation training-entry preflight JSON parse", False, f"{exc}"))
    if runner_smoke_json.exists():
        try:
            payload = json.loads(runner_smoke_json.read_text(encoding="utf-8"))
            rows.append(check("minimal ablation runner smoke status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("minimal ablation runner smoke has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            variant_ids = {row.get("run_id") for row in payload.get("variant_summaries", []) if isinstance(row, dict)}
            core_ids = {f"ABL-{idx:02d}" for idx in range(5)}
            rows.append(check("minimal ablation runner smoke covers ABL-00..04", core_ids.issubset(variant_ids), f"covered={sorted(variant_ids)}"))
            statuses = {row.get("run_id"): row.get("status") for row in payload.get("variant_summaries", []) if isinstance(row, dict)}
            rows.append(check("ABL-01 recorded as no-forward design", statuses.get("ABL-01") == "design_recorded_no_forward", str(statuses.get("ABL-01"))))
        except json.JSONDecodeError as exc:
            rows.append(check("minimal ablation runner smoke JSON parse", False, f"{exc}"))
    if training_preflight_json.exists():
        try:
            payload = json.loads(training_preflight_json.read_text(encoding="utf-8"))
            rows.append(check("ablation training-runner preflight status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation training-runner preflight has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            modes = {row.get("run_id"): row.get("runner_mode") for row in payload.get("variant_plans", []) if isinstance(row, dict)}
            rows.append(check("ABL-01 lower-prior runner mode recorded", modes.get("ABL-01") == "lower_prior_focus_stack_only", str(modes.get("ABL-01"))))
            trainable = {row.get("run_id"): row.get("trainable_now") for row in payload.get("variant_plans", []) if isinstance(row, dict)}
            rows.append(check("ABL-01 training gated", trainable.get("ABL-01") is False, str(trainable.get("ABL-01"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation training-runner preflight JSON parse", False, f"{exc}"))
    if training_dry_run_json.exists():
        try:
            payload = json.loads(training_dry_run_json.read_text(encoding="utf-8"))
            rows.append(check("ablation training-runner dry-run status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation training-runner dry-run has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation training-runner dry-run covers ABL-00/03", {"ABL-00", "ABL-03"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            reports = {row.get("run_id"): row for row in payload.get("reports", []) if isinstance(row, dict)}
            rows.append(check("ABL-00 dry-run status passed", reports.get("ABL-00", {}).get("status") == "dry_run_passed", str(reports.get("ABL-00", {}).get("status"))))
            rows.append(check("ABL-03 dry-run status passed", reports.get("ABL-03", {}).get("status") == "dry_run_passed", str(reports.get("ABL-03", {}).get("status"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation training-runner dry-run JSON parse", False, f"{exc}"))
    if small_train_json.exists():
        try:
            payload = json.loads(small_train_json.read_text(encoding="utf-8"))
            rows.append(check("ablation small-training debug status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation small-training debug has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation small-training debug covers ABL-00/03", {"ABL-00", "ABL-03"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            reports = {row.get("run_id"): row for row in payload.get("reports", []) if isinstance(row, dict)}
            for run_id in ["ABL-00", "ABL-03"]:
                checkpoint = Path(str(reports.get(run_id, {}).get("checkpoint", "")))
                metrics_csv = Path(str(reports.get(run_id, {}).get("metrics_csv", "")))
                rows.append(check(f"{run_id} debug checkpoint under tmp", str(checkpoint).startswith(str(root / run_id / "checkpoints")), str(checkpoint)))
                rows.append(check(f"{run_id} debug metrics under tmp", str(metrics_csv).startswith(str(root / run_id / "metrics")), str(metrics_csv)))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation small-training debug JSON parse", False, f"{exc}"))
    if pilot_json.exists():
        try:
            payload = json.loads(pilot_json.read_text(encoding="utf-8"))
            rows.append(check("ablation controlled pilot status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation controlled pilot has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation controlled pilot run_kind", payload.get("run_kind") == "controlled_pilot", str(payload.get("run_kind"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation controlled pilot covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            reports = {row.get("run_id"): row for row in payload.get("reports", []) if isinstance(row, dict)}
            for run_id in ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]:
                checkpoint = Path(str(reports.get(run_id, {}).get("checkpoint", "")))
                metrics_csv = Path(str(reports.get(run_id, {}).get("metrics_csv", "")))
                rows.append(check(f"{run_id} pilot checkpoint under tmp", str(checkpoint).startswith(str(root / run_id / "checkpoints")), str(checkpoint)))
                rows.append(check(f"{run_id} pilot metrics under tmp", str(metrics_csv).startswith(str(root / run_id / "metrics")), str(metrics_csv)))
                rows.append(check(f"{run_id} pilot status completed", reports.get(run_id, {}).get("status") == "controlled_pilot_debug_completed", str(reports.get(run_id, {}).get("status"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation controlled pilot JSON parse", False, f"{exc}"))
    if pilot_eligibility_json.exists():
        try:
            payload = json.loads(pilot_eligibility_json.read_text(encoding="utf-8"))
            rows.append(check("ablation pilot eligibility status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation pilot eligibility debug-only", payload.get("eligibility_level") == "Debug-only; not manuscript main-table evidence", str(payload.get("eligibility_level"))))
            rows.append(check("ablation pilot claim_eligible false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation pilot main_table_eligible false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            rows.append(check("ablation pilot eligibility has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation pilot eligibility JSON parse", False, f"{exc}"))
    if full_split_json.exists():
        try:
            payload = json.loads(full_split_json.read_text(encoding="utf-8"))
            rows.append(check("ablation full-split debug status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation full-split debug has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation full-split debug sample count", payload.get("sample_count") == 7, str(payload.get("sample_count"))))
            rows.append(check("ablation full-split debug claim false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation full-split debug main table false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation full-split debug covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            summary = payload.get("summary", [])
            rows.append(check("ablation full-split debug summary rows", len(summary) == 4, f"rows={len(summary)}"))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation full-split debug JSON parse", False, f"{exc}"))
    if full_split_eligibility_json.exists():
        try:
            payload = json.loads(full_split_eligibility_json.read_text(encoding="utf-8"))
            rows.append(check("ablation full-split eligibility status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(
                check(
                    "ablation full-split eligibility diagnostic-only",
                    payload.get("eligibility_level") == "Diagnostic full-split metrics only; not manuscript ablation evidence",
                    str(payload.get("eligibility_level")),
                )
            )
            rows.append(check("ablation full-split eligibility claim false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation full-split eligibility main table false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            rows.append(check("ablation full-split eligibility has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation full-split eligibility JSON parse", False, f"{exc}"))
    if matched_preflight_json.exists():
        try:
            payload = json.loads(matched_preflight_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched training preflight status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched training preflight has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched training preflight claim false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation matched training preflight main table false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            split_counts = payload.get("split_counts", {})
            rows.append(
                check(
                    "ablation matched training split counts",
                    split_counts == {"train": 27, "validation": 10, "test": 7},
                    str(split_counts),
                )
            )
            variant_plans = payload.get("variant_plans", [])
            run_ids = {row.get("run_id") for row in variant_plans if isinstance(row, dict)}
            rows.append(check("ablation matched training preflight covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            trainable = {row.get("run_id"): row.get("trainable_now") for row in variant_plans if isinstance(row, dict)}
            rows.append(check("ablation matched training core variants trainable", all(trainable.get(run_id) is True for run_id in ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]), str(trainable)))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched training preflight JSON parse", False, f"{exc}"))
    if matched_smoke_json.exists():
        try:
            payload = json.loads(matched_smoke_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched smoke status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched smoke has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched smoke run_kind", payload.get("run_kind") == "matched_smoke", str(payload.get("run_kind"))))
            rows.append(check("ablation matched smoke split counts", payload.get("split_counts") == {"train": 27, "validation": 10, "test": 7}, str(payload.get("split_counts"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation matched smoke covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            reports = {row.get("run_id"): row for row in payload.get("reports", []) if isinstance(row, dict)}
            for run_id in ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]:
                checkpoint = Path(str(reports.get(run_id, {}).get("checkpoint", "")))
                metrics_csv = Path(str(reports.get(run_id, {}).get("metrics_csv", "")))
                rows.append(check(f"{run_id} matched smoke checkpoint under tmp", str(checkpoint).startswith(str(root / run_id / "checkpoints")), str(checkpoint)))
                rows.append(check(f"{run_id} matched smoke checkpoint exists", checkpoint.exists(), str(checkpoint)))
                rows.append(check(f"{run_id} matched smoke history under tmp", str(metrics_csv).startswith(str(root / run_id / "metrics")), str(metrics_csv)))
                rows.append(check(f"{run_id} matched smoke history exists", metrics_csv.exists(), str(metrics_csv)))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched smoke JSON parse", False, f"{exc}"))
    if matched_smoke_eligibility_json.exists():
        try:
            payload = json.loads(matched_smoke_eligibility_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched smoke eligibility status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(
                check(
                    "ablation matched smoke eligibility smoke-only",
                    payload.get("eligibility_level") == "Matched smoke only; not manuscript ablation evidence",
                    str(payload.get("eligibility_level")),
                )
            )
            rows.append(check("ablation matched smoke eligibility claim false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation matched smoke eligibility main table false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            rows.append(check("ablation matched smoke eligibility has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched smoke eligibility JSON parse", False, f"{exc}"))
    if full_matched_config_json.exists():
        try:
            payload = json.loads(full_matched_config_json.read_text(encoding="utf-8"))
            rows.append(check("ablation full matched config status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation full matched config has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation full matched config has no warnings", payload.get("warning_count") == 0, str(payload.get("warning_count"))))
            rows.append(check("ablation full matched config claim false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation full matched config main table false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            config_payload = payload.get("full_matched_training_config", {})
            rows.append(check("ablation full matched config tag", config_payload.get("tag") == "2026-06-19_matched_training_full_candidate", str(config_payload.get("tag"))))
            rows.append(check("ablation full matched config run_kind", config_payload.get("run_kind") == "matched_full_candidate", str(config_payload.get("run_kind"))))
            rows.append(check("ablation full matched config split counts", config_payload.get("split_counts") == {"train": 27, "validation": 10, "test": 7}, str(config_payload.get("split_counts"))))
            rows.append(check("ablation full matched config epochs", config_payload.get("epochs") == 4, str(config_payload.get("epochs"))))
            rows.append(check("ablation full matched train patches", config_payload.get("train_patches_per_epoch") == 128, str(config_payload.get("train_patches_per_epoch"))))
            rows.append(check("ablation full matched validation patches", config_payload.get("validation_patches_per_epoch") == 32, str(config_payload.get("validation_patches_per_epoch"))))
            rows.append(check("ablation full matched batch size", config_payload.get("batch_size") == 1, str(config_payload.get("batch_size"))))
            rows.append(check("ablation full matched uses all train samples", config_payload.get("max_train_samples") == 0, str(config_payload.get("max_train_samples"))))
            rows.append(check("ablation full matched uses all validation samples", config_payload.get("max_val_samples") == 0, str(config_payload.get("max_val_samples"))))
            rows.append(check("ablation full matched output under tmp", config_payload.get("output_scope") == "tmp/ablation_results/<run_id>/", str(config_payload.get("output_scope"))))
            variant_plans = payload.get("variant_plans", [])
            run_ids = {row.get("run_id") for row in variant_plans if isinstance(row, dict)}
            rows.append(check("ablation full matched config covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            trainable = {row.get("run_id"): row.get("trainable_in_current_runner") for row in variant_plans if isinstance(row, dict)}
            rows.append(check("ablation full matched core variants trainable", all(trainable.get(run_id) is True for run_id in ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]), str(trainable)))
            evaluator_plan = payload.get("evaluator_plan", {})
            rows.append(
                check(
                    "ablation full matched evaluator next tool recorded",
                    "evaluate_ablation_matched_full_split_metrics.py" in str(evaluator_plan.get("required_next_tool", "")),
                    str(evaluator_plan.get("required_next_tool")),
                )
            )
            rows.append(check("ablation full matched evaluator test samples", evaluator_plan.get("required_test_samples") == 7, str(evaluator_plan.get("required_test_samples"))))
            eligibility_plan = payload.get("eligibility_plan", {})
            rows.append(
                check(
                    "ablation full matched eligibility audit recorded",
                    eligibility_plan.get("required_audit") == "audit_ablation_matched_training_eligibility.py",
                    str(eligibility_plan.get("required_audit")),
                )
            )
            rows.append(check("ablation full matched eligibility claim false", eligibility_plan.get("claim_eligible_before_audit") is False, str(eligibility_plan.get("claim_eligible_before_audit"))))
            rows.append(check("ablation full matched eligibility main table false", eligibility_plan.get("main_table_eligible_before_audit") is False, str(eligibility_plan.get("main_table_eligible_before_audit"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation full matched config JSON parse", False, f"{exc}"))
    if matched_evaluator_smoke_json.exists():
        try:
            payload = json.loads(matched_evaluator_smoke_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched evaluator smoke status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched evaluator smoke sample count", payload.get("sample_count") == 1, str(payload.get("sample_count"))))
            rows.append(check("ablation matched evaluator smoke flag", payload.get("smoke_evaluation") is True, str(payload.get("smoke_evaluation"))))
            rows.append(check("ablation matched evaluator smoke has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched evaluator smoke rows", len(payload.get("summary", [])) == 4, f"rows={len(payload.get('summary', []))}"))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched evaluator smoke JSON parse", False, f"{exc}"))
    if matched_full_training_json.exists():
        try:
            payload = json.loads(matched_full_training_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched full training status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched full training run_kind", payload.get("run_kind") == "matched_full_candidate", str(payload.get("run_kind"))))
            rows.append(check("ablation matched full training tag", payload.get("tag") == "2026-06-19_matched_training_full_candidate", str(payload.get("tag"))))
            rows.append(check("ablation matched full training has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched full training split counts", payload.get("split_counts") == {"train": 27, "validation": 10, "test": 7}, str(payload.get("split_counts"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation matched full training covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            reports = {row.get("run_id"): row for row in payload.get("reports", []) if isinstance(row, dict)}
            for run_id in ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]:
                checkpoint = Path(str(reports.get(run_id, {}).get("checkpoint", "")))
                history = reports.get(run_id, {}).get("history", [])
                rows.append(check(f"{run_id} full candidate checkpoint exists", checkpoint.exists(), str(checkpoint)))
                rows.append(check(f"{run_id} full candidate checkpoint under tmp", str(checkpoint).startswith(str(root / run_id / "checkpoints")), str(checkpoint)))
                rows.append(check(f"{run_id} full candidate history rows", len(history) == 4, f"rows={len(history)}"))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched full training JSON parse", False, f"{exc}"))
    if matched_full_eval_json.exists():
        try:
            payload = json.loads(matched_full_eval_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched full eval status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched full eval tag", payload.get("tag") == "2026-06-19_matched_full_candidate_eval", str(payload.get("tag"))))
            rows.append(check("ablation matched full eval checkpoint tag", payload.get("checkpoint_tag") == "2026-06-19_matched_training_full_candidate", str(payload.get("checkpoint_tag"))))
            rows.append(check("ablation matched full eval sample count", payload.get("sample_count") == 7, str(payload.get("sample_count"))))
            rows.append(check("ablation matched full eval is not smoke", payload.get("smoke_evaluation") is False, str(payload.get("smoke_evaluation"))))
            rows.append(check("ablation matched full eval rows", len(payload.get("summary", [])) == 4, f"rows={len(payload.get('summary', []))}"))
            rows.append(check("ablation matched full eval has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched full eval claim false before audit", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
            rows.append(check("ablation matched full eval main table false before audit", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched full eval JSON parse", False, f"{exc}"))
    if matched_full_eligibility_json.exists():
        try:
            payload = json.loads(matched_full_eligibility_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched full eligibility status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched full eligibility has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched full eligibility checks", payload.get("check_count") == 104, str(payload.get("check_count"))))
            rows.append(check("ablation matched full eligibility claim true", payload.get("claim_eligible") is True, str(payload.get("claim_eligible"))))
            rows.append(check("ablation matched full eligibility main table true", payload.get("main_table_eligible") is True, str(payload.get("main_table_eligible"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched full eligibility JSON parse", False, f"{exc}"))
    if matched_longer_training_json.exists():
        try:
            payload = json.loads(matched_longer_training_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched longer training status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched longer training run_kind", payload.get("run_kind") == "matched_longer_repeat", str(payload.get("run_kind"))))
            rows.append(check("ablation matched longer training tag", payload.get("tag") == "2026-06-19_matched_training_longer_repeat", str(payload.get("tag"))))
            rows.append(check("ablation matched longer training has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
            rows.append(check("ablation matched longer training split counts", payload.get("split_counts") == {"train": 27, "validation": 10, "test": 7}, str(payload.get("split_counts"))))
            run_ids = set(payload.get("run_ids", []))
            rows.append(check("ablation matched longer training covers ABL-00/02/03/04", {"ABL-00", "ABL-02", "ABL-03", "ABL-04"}.issubset(run_ids), f"run_ids={sorted(run_ids)}"))
            reports = {row.get("run_id"): row for row in payload.get("reports", []) if isinstance(row, dict)}
            for run_id in ["ABL-00", "ABL-02", "ABL-03", "ABL-04"]:
                checkpoint = Path(str(reports.get(run_id, {}).get("checkpoint", "")))
                history = reports.get(run_id, {}).get("history", [])
                rows.append(check(f"{run_id} longer repeat checkpoint exists", checkpoint.exists(), str(checkpoint)))
                rows.append(check(f"{run_id} longer repeat checkpoint under tmp", str(checkpoint).startswith(str(root / run_id / "checkpoints")), str(checkpoint)))
                rows.append(check(f"{run_id} longer repeat history rows", len(history) == 8, f"rows={len(history)}"))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched longer training JSON parse", False, f"{exc}"))
    if matched_longer_eval_json.exists():
        try:
            payload = json.loads(matched_longer_eval_json.read_text(encoding="utf-8"))
            rows.append(check("ablation matched longer eval status pass", payload.get("status") == "pass", str(payload.get("status"))))
            rows.append(check("ablation matched longer eval tag", payload.get("tag") == "2026-06-19_matched_longer_repeat_eval", str(payload.get("tag"))))
            rows.append(check("ablation matched longer eval checkpoint tag", payload.get("checkpoint_tag") == "2026-06-19_matched_training_longer_repeat", str(payload.get("checkpoint_tag"))))
            rows.append(check("ablation matched longer eval sample count", payload.get("sample_count") == 7, str(payload.get("sample_count"))))
            rows.append(check("ablation matched longer eval is not smoke", payload.get("smoke_evaluation") is False, str(payload.get("smoke_evaluation"))))
            rows.append(check("ablation matched longer eval rows", len(payload.get("summary", [])) == 4, f"rows={len(payload.get('summary', []))}"))
            rows.append(check("ablation matched longer eval has no errors", payload.get("error_count") == 0, str(payload.get("error_count"))))
        except json.JSONDecodeError as exc:
            rows.append(check("ablation matched longer eval JSON parse", False, f"{exc}"))
    for run_id in ABLATION_RUN_IDS:
        run_root = root / run_id
        config = run_root / "run_config.json"
        synthetic_metrics = run_root / "metrics" / "synthetic_metrics.csv"
        real_metrics = run_root / "metrics" / "real_no_reference_metrics.csv"
        rows.extend(
            [
                check(f"{run_id} workspace exists", run_root.exists(), str(run_root)),
                check(f"{run_id} run_config exists", config.exists(), str(config)),
                check(f"{run_id} synthetic metrics template exists", synthetic_metrics.exists(), str(synthetic_metrics)),
                check(f"{run_id} real metrics template exists", real_metrics.exists(), str(real_metrics)),
            ]
        )
        if config.exists():
            try:
                payload = json.loads(config.read_text(encoding="utf-8"))
                allowed_status = payload.get("status") in {
                    "scaffold_only_no_training_run",
                    "small_training_debug_run",
                    "controlled_pilot_debug_run",
                    "matched_training_smoke_run",
                    "matched_training_full_candidate_run",
                    "matched_training_longer_repeat_run",
                }
                rows.append(check(f"{run_id} safe status", allowed_status, str(payload.get("status"))))
                rows.append(check(f"{run_id} claim_eligible false", payload.get("claim_eligible") is False, str(payload.get("claim_eligible"))))
                rows.append(check(f"{run_id} main_table_eligible false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
            except json.JSONDecodeError as exc:
                rows.append(check(f"{run_id} run_config JSON", False, f"{exc}"))
    return rows


def supervisor_report_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    report = RESEARCH_DIR / "supervisor_experiment_report_2026-06-19.md"
    figures_dir = RESEARCH_DIR / "report_figures_2026-06-19"
    required_figures = [
        "fig2_experiment_budget_timeline.png",
        "fig3_ablation_metrics_candidate_vs_repeat.png",
        "fig4_longer_repeat_improvement.png",
        "fig5_per_sample_longer_repeat_heatmap.png",
        "fig6_test_sample_conditions.png",
        "sample_id_mapping.csv",
    ]
    rows.append(check("supervisor experiment report exists", report.exists(), str(report)))
    rows.append(check("supervisor report figures dir exists", figures_dir.exists(), str(figures_dir)))
    if report.exists():
        text = read_text(report)
        rows.append(check("supervisor report mentions S2R framing", "simulation-to-real" in text.lower(), "simulation-to-real expected"))
        rows.append(check("supervisor report mentions DFF/GADFF prior", "DFF/GADFF prior" in text, "prior contribution expected"))
        rows.append(check("supervisor report includes longer repeat result", "109.2209" in text and "75.4572" in text, "longer repeat key metrics expected"))
        rows.append(check("supervisor report includes evidence boundary", "证据边界" in text, "evidence boundary section expected"))
        rows.append(check("supervisor report excludes pure workflow figure", "fig1_research_workflow.png" not in text, "workflow should be text-only"))
    for name in required_figures:
        path = figures_dir / name
        rows.append(check(f"supervisor report artifact exists: {name}", path.exists(), str(path)))
        if path.exists():
            rows.append(check(f"supervisor report artifact nonempty: {name}", path.stat().st_size > 0, f"bytes={path.stat().st_size}"))
    return rows


def foundation_depth_auxiliary_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path("tmp/foundation_depth_auxiliary/DepthAnythingV2")
    run_config = root / "run_config.json"
    input_manifest = root / "input_manifest.csv"
    logs_dir = root / "logs"
    predictions_dir = root / "predictions"
    visualizations_dir = root / "visualizations"
    rows.extend(
        [
            check("DepthAnythingV2 auxiliary workspace exists", root.exists(), str(root)),
            check("DepthAnythingV2 run_config exists", run_config.exists(), str(run_config)),
            check("DepthAnythingV2 input_manifest exists", input_manifest.exists(), str(input_manifest)),
            check("DepthAnythingV2 logs dir exists", logs_dir.exists(), str(logs_dir)),
            check("DepthAnythingV2 predictions dir exists", predictions_dir.exists(), str(predictions_dir)),
            check("DepthAnythingV2 visualizations dir exists", visualizations_dir.exists(), str(visualizations_dir)),
        ]
    )
    if run_config.exists():
        try:
            payload = json.loads(run_config.read_text(encoding="utf-8"))
            rows.append(check("DepthAnythingV2 scaffold-only status", payload.get("status") == "scaffold_only_no_model_run", str(payload.get("status"))))
            rows.append(check("DepthAnythingV2 auxiliary_only true", payload.get("auxiliary_only") is True, str(payload.get("auxiliary_only"))))
            rows.append(check("DepthAnythingV2 main_table_eligible false", payload.get("main_table_eligible") is False, str(payload.get("main_table_eligible"))))
        except json.JSONDecodeError as exc:
            rows.append(check("DepthAnythingV2 run_config JSON", False, f"{exc}"))
    if input_manifest.exists():
        text = read_text(input_manifest)
        rows.append(check("DepthAnythingV2 manifest marked auxiliary", "auxiliary_qualitative_only" in text, "comparison scope expected"))
        rows.append(check("DepthAnythingV2 manifest excludes main table", "false" in text.lower(), "main_table_eligible=false expected"))
    prediction_files = sorted(predictions_dir.glob("*.npy")) if predictions_dir.exists() else []
    visualization_files = sorted(visualizations_dir.glob("*.png")) if visualizations_dir.exists() else []
    rows.append(
        check(
            "DepthAnythingV2 no accidental prediction files",
            not prediction_files,
            f"prediction files={len(prediction_files)}",
            severity="warning",
        )
    )
    rows.append(
        check(
            "DepthAnythingV2 no accidental visualization files",
            not visualization_files,
            f"visualization files={len(visualization_files)}",
            severity="warning",
        )
    )
    return rows


def manuscript_depth_anything_checks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manuscript = Path("submission_planning/manuscript_draft/s2r_focus_stack_manuscript.tex")
    references = Path("submission_planning/manuscript_draft/references.bib")
    rows.append(check("manuscript draft exists", manuscript.exists(), str(manuscript)))
    rows.append(check("manuscript references exist", references.exists(), str(references)))
    if manuscript.exists():
        text = read_text(manuscript)
        rows.append(check("manuscript cites Depth Anything V2", "depth_anything_v2_2024" in text, "citation key expected"))
        rows.append(check("manuscript marks Depth Anything V2 auxiliary", "auxiliary" in text.lower() and "main sota table" in text.lower(), "auxiliary boundary expected"))
        rows.append(check("manuscript mentions single-frame Depth Anything V2 protocol", "single frame" in text.lower() or "single-frame" in text.lower(), "single-frame boundary expected"))
    if references.exists():
        text = read_text(references)
        rows.append(check("references contain one Depth Anything V2 key", text.count("depth_anything_v2_2024") == 1, "single bib key expected"))
        rows.append(check("references include arXiv 2406.09414", "2406.09414" in text, "arXiv identifier expected"))
    return rows


def build_report() -> dict[str, Any]:
    checks = []
    checks.extend(required_doc_checks())
    checks.extend(tool_checks())
    checks.extend(text_safety_checks())
    checks.extend(method_workspace_checks())
    checks.extend(data_package_checks())
    checks.extend(ablation_workspace_checks())
    checks.extend(supervisor_report_checks())
    checks.extend(foundation_depth_auxiliary_checks())
    checks.extend(manuscript_depth_anything_checks())
    errors = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    return {
        "status": "pass" if not errors else "fail",
        "check_count": len(checks),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Research Package Integrity Audit",
        "",
        f"- Status: {report['status']}",
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
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    report = build_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "research_package_integrity_audit.json"
    md_path = args.out_dir / "research_package_integrity_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Research package audit: {report['status']}")
    print(f"Checks: {report['check_count']}, errors: {report['error_count']}, warnings: {report['warning_count']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
