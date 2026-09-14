from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from cv_agent.ingestion import ingest_request
from cv_agent.renderer import PDFRenderer
from cv_agent.schemas import (
    GroundednessAudit,
    GroundednessFinding,
    IngestionRequest,
    ModelSettings,
    PipelineSnapshot,
)
from cv_agent.validation import PackageValidator, ProfileValidator, repair_profile_grounding


ROOT = Path(__file__).resolve().parents[1]


def _audit(snapshot: PipelineSnapshot) -> GroundednessAudit:
    locations = ["resume.professional_summary"]
    for role_index, role in enumerate(snapshot.document_package.resume.experience):
        locations.extend(
            f"resume.experience[{role_index}].bullets[{bullet_index}]"
            for bullet_index, _ in enumerate(role.bullets)
        )
    for paragraph_index, paragraph in enumerate(
        snapshot.document_package.cover_letter.paragraphs
    ):
        locations.extend(
            f"cover_letter.paragraphs[{paragraph_index}].candidate_claims[{claim_index}]"
            for claim_index, _ in enumerate(paragraph.candidate_claims)
        )
    return GroundednessAudit(
        passed=True,
        findings=[
            GroundednessFinding(
                location=location,
                supported=True,
                explanation="Candidate claim is supported.",
                supporting_achievement_ids=["a1", "a2", "a3", "a4", "a5"],
            )
            for location in locations
        ],
    )


def test_master_profile_exact_evidence_is_valid(snapshot: PipelineSnapshot) -> None:
    bundle = ingest_request(
        IngestionRequest(
            jd_path=ROOT / "examples" / "job_description.md",
            jd_text=None,
            cv_paths=[
                ROOT / "examples" / "candidate_primary.md",
                ROOT / "examples" / "candidate_history.txt",
            ],
            max_chars_per_document=120_000,
        )
    )
    assert ProfileValidator().validate(snapshot.master_profile, bundle) == []


def test_empty_groundedness_audit_is_valid_when_document_has_no_claims() -> None:
    audit = GroundednessAudit(passed=True, findings=[])

    assert audit.passed
    assert audit.findings == []


def test_profile_grounding_repair_prunes_and_relinks_skills_and_clears_bad_date(
    snapshot: PipelineSnapshot,
) -> None:
    bundle = ingest_request(
        IngestionRequest(
            jd_path=ROOT / "examples" / "job_description.md",
            jd_text=None,
            cv_paths=[
                ROOT / "examples" / "candidate_primary.md",
                ROOT / "examples" / "candidate_history.txt",
            ],
            max_chars_per_document=120_000,
        )
    )
    payload = deepcopy(snapshot.master_profile.model_dump())
    payload["skills"][0]["evidence_ids"] = ["e_name"]
    payload["skills"].append(
        {"name": "ImaginaryDB", "category": "Data", "evidence_ids": ["e_skills"]}
    )
    payload["education"][0]["graduation_date"] = "May 2099"
    profile = snapshot.master_profile.model_validate(payload)

    repaired, counts = repair_profile_grounding(profile, bundle)

    assert repaired.skills[0].name == "Python"
    assert repaired.skills[0].evidence_ids
    assert all(skill.name != "ImaginaryDB" for skill in repaired.skills)
    assert repaired.education[0].graduation_date is None
    assert counts["skills_relinked"] == 1
    assert counts["skills_pruned"] == 1
    assert counts["education_values_cleared"] == 1
    assert ProfileValidator().validate(repaired, bundle) == []


def test_profile_grounding_repair_recovers_widespread_miscitations(
    snapshot: PipelineSnapshot,
) -> None:
    bundle = ingest_request(
        IngestionRequest(
            jd_path=ROOT / "examples" / "job_description.md",
            jd_text=None,
            cv_paths=[
                ROOT / "examples" / "candidate_primary.md",
                ROOT / "examples" / "candidate_history.txt",
            ],
            max_chars_per_document=120_000,
        )
    )
    payload = deepcopy(snapshot.master_profile.model_dump())
    payload["contact_evidence_ids"] = ["e_name"]
    for fact in payload["summary_facts"]:
        fact["evidence_ids"] = ["e_name"]
    for role in payload["experience"]:
        role["evidence_ids"] = ["e_name"]
        role["start_date"] = "2099"
        for achievement in role["achievements"]:
            achievement["evidence_ids"] = ["e_name"]
    payload["experience"][0]["achievements"][0]["statement"] = payload[
        "experience"
    ][0]["achievements"][0]["statement"].replace("42%", "73%")
    for skill in payload["skills"]:
        skill["evidence_ids"] = ["e_name"]
    payload["education"][0]["evidence_ids"] = ["e_name"]
    payload["education"][0]["graduation_date"] = "2099"
    profile = snapshot.master_profile.model_validate(payload)

    repaired, counts = repair_profile_grounding(profile, bundle)

    assert counts["achievements_snapped"] == 1
    assert counts["skills_relinked"] == len(snapshot.master_profile.skills)
    assert repaired.experience[0].achievements[0].statement.startswith(
        "Reduced data pipeline latency by 42%"
    )
    assert ProfileValidator().validate(repaired, bundle) == []


def test_valid_package_passes_all_deterministic_gates(snapshot: PipelineSnapshot) -> None:
    settings = ModelSettings(
        model="test-model",
        temperature=0.2,
        max_correction_retries=2,
        keyword_coverage_threshold=0.65,
        metric_density_threshold=0.60,
        groundedness_threshold=1.0,
    )
    renderer = PDFRenderer()
    resume_html, cover_html = renderer.render_html(snapshot.document_package)
    result = PackageValidator(settings).validate(
        snapshot.document_package,
        snapshot.master_profile,
        snapshot.job_requirements,
        template_issues=renderer.audit_html(resume_html, cover_html),
        groundedness_audit=_audit(snapshot),
        expected_letter_date="2026-09-12",
    )
    assert result.passed, result.issues
    assert result.metric_density == 0.8
    assert result.groundedness_ratio == 1.0
    assert result.ats_integrity_passed


def test_common_strong_action_verbs_are_accepted(snapshot: PipelineSnapshot) -> None:
    payload = deepcopy(snapshot.document_package.model_dump())
    payload["resume"]["experience"][0]["bullets"][0]["text"] = (
        "Architected data pipeline improvements that reduced latency by 42% through "
        "Apache Spark partitioning and Airflow orchestration."
    )
    package = snapshot.document_package.model_validate(payload)
    settings = ModelSettings(
        model="test-model",
        temperature=0.2,
        max_correction_retries=2,
        keyword_coverage_threshold=0.65,
        metric_density_threshold=0.60,
        groundedness_threshold=1.0,
    )

    result = PackageValidator(settings).validate(
        package,
        snapshot.master_profile,
        snapshot.job_requirements,
        groundedness_audit=_audit(snapshot),
        expected_letter_date="2026-09-12",
    )

    assert "weak_bullet_opening" not in {issue.code for issue in result.issues}


def test_keyword_gate_excludes_unsupported_jd_terms(snapshot: PipelineSnapshot) -> None:
    requirements = snapshot.job_requirements.model_copy(
        update={
            "hard_skills": [*snapshot.job_requirements.hard_skills, "Rust", "Kubernetes"],
            "tools": [*snapshot.job_requirements.tools, "Snowflake"],
            "keywords": [*snapshot.job_requirements.keywords, "machine learning"],
        }
    )
    settings = ModelSettings(
        model="test-model",
        temperature=0.2,
        max_correction_retries=2,
        keyword_coverage_threshold=0.65,
        metric_density_threshold=0.60,
        groundedness_threshold=1.0,
    )

    result = PackageValidator(settings).validate(
        snapshot.document_package,
        snapshot.master_profile,
        requirements,
        groundedness_audit=_audit(snapshot),
        expected_letter_date="2026-09-12",
    )

    assert result.passed, result.issues
    assert "Rust" not in result.missing_keywords
    assert "Kubernetes" not in result.missing_keywords
    assert "Snowflake" not in result.missing_keywords
    assert "machine learning" not in result.missing_keywords


def test_invented_metric_fails_closed(snapshot: PipelineSnapshot) -> None:
    payload = deepcopy(snapshot.document_package.model_dump())
    payload["resume"]["experience"][0]["bullets"][0]["text"] = (
        "Reduced data pipeline latency by 73% by redesigning Apache Spark partitioning "
        "and Airflow orchestration."
    )
    payload["resume"]["experience"][0]["bullets"][0]["measurement"] = "73%"
    package = snapshot.document_package.model_validate(payload)
    settings = ModelSettings(
        model="test-model",
        temperature=0.2,
        max_correction_retries=2,
        keyword_coverage_threshold=0.65,
        metric_density_threshold=0.60,
        groundedness_threshold=1.0,
    )
    result = PackageValidator(settings).validate(
        package,
        snapshot.master_profile,
        snapshot.job_requirements,
        groundedness_audit=_audit(snapshot),
        expected_letter_date="2026-09-12",
    )
    assert not result.passed
    assert "unsupported_metric" in {issue.code for issue in result.issues}


def test_external_audit_disagreement_is_advisory_when_local_grounding_passes(
    snapshot: PipelineSnapshot,
) -> None:
    good_audit = _audit(snapshot)
    findings = list(good_audit.findings)
    findings[0] = findings[0].model_copy(
        update={
            "supported": False,
            "explanation": "Simulated false negative from the external auditor.",
        }
    )
    external_audit = GroundednessAudit(passed=False, findings=findings)

    result = PackageValidator(snapshot.model_settings).validate(
        snapshot.document_package,
        snapshot.master_profile,
        snapshot.job_requirements,
        groundedness_audit=external_audit,
        expected_letter_date="2026-09-12",
    )

    assert result.passed
    assert result.groundedness_ratio == 1.0
    audit_issue = next(issue for issue in result.issues if issue.code == "llm_groundedness")
    assert audit_issue.severity == "warning"
