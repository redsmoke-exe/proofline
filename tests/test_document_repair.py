from __future__ import annotations

from copy import deepcopy
from datetime import date

from cv_agent.document_repair import repair_document_package
from cv_agent.schemas import (
    DocumentPackage,
    ModelSettings,
    PipelineSnapshot,
    ProfileCertification,
    ProfileProject,
)
from cv_agent.validation import PackageValidator


def _corrupted_package(snapshot: PipelineSnapshot) -> DocumentPackage:
    payload = deepcopy(snapshot.document_package.model_dump())
    payload["resume"]["contact"]["full_name"] = "Invented Person"
    payload["resume"]["target_title"] = "Imaginary Platform Wizard"
    payload["resume"]["professional_summary"] = {
        "text": "Created an unsupported platform improvement of 73%.",
        "achievement_ids": ["a1"],
    }
    payload["resume"]["skills"] = ["ImaginaryDB", "Python", "python"]
    payload["resume"]["experience"][0]["company"] = "Invented Company"
    payload["resume"]["experience"][0]["start_date"] = "January 2099"
    payload["resume"]["experience"][0]["bullets"][0].update(
        {
            "text": "Raised an invented outcome by 73% using ImaginaryDB.",
            "accomplishment": "Raised an invented outcome",
            "measurement": "73%",
            "method": "ImaginaryDB",
            "achievement_ids": ["a5"],
        }
    )
    payload["resume"]["education"][0]["institution"] = "Invented University"
    payload["cover_letter"]["candidate"]["email"] = "fake@example.invalid"
    payload["cover_letter"]["letter_date"] = "2099-01-01"
    payload["cover_letter"]["recipient_company"] = "Wrong Company"
    payload["cover_letter"]["salutation"] = "To whom it may concern"
    payload["cover_letter"]["signatory"] = "Someone Else"
    payload["cover_letter"]["paragraphs"] = [
        {
            "text": "This deliberately ungrounded paragraph is long enough for the schema.",
            "candidate_claims": [],
            "matched_requirements": ["imaginary requirement"],
        },
        {
            "text": "I produced an unsupported outcome of 73 percent for a client.",
            "candidate_claims": [
                {
                    "text": "I produced an unsupported outcome of 73 percent.",
                    "achievement_ids": ["missing-achievement"],
                }
            ],
            "matched_requirements": ["imaginary requirement"],
        },
        {
            "text": "Thank you for considering this intentionally invalid application.",
            "candidate_claims": [],
            "matched_requirements": ["imaginary requirement"],
        },
    ]
    return DocumentPackage.model_validate(payload)


def test_repair_removes_drift_and_produces_locally_grounded_package(
    snapshot: PipelineSnapshot,
) -> None:
    repaired, counts = repair_document_package(
        _corrupted_package(snapshot),
        snapshot.master_profile,
        snapshot.job_requirements,
        date(2026, 9, 12),
    )

    assert repaired.resume.contact == snapshot.master_profile.contact
    assert repaired.cover_letter.candidate == snapshot.master_profile.contact
    assert repaired.resume.target_title == snapshot.job_requirements.target_title
    assert repaired.cover_letter.recipient_company == "Acme Cloud"
    assert repaired.cover_letter.letter_date == "2026-09-12"
    assert repaired.cover_letter.salutation == "Dear Hiring Team,"
    assert repaired.cover_letter.signatory == "Alex Morgan"
    assert repaired.resume.professional_summary is not None
    assert "Data platform engineer" in repaired.resume.professional_summary.text
    assert "Reduced data pipeline latency" in repaired.resume.professional_summary.text
    assert repaired.resume.professional_summary.achievement_ids == ["f_summary", "a1"]
    assert repaired.resume.strategy is not None
    assert repaired.resume.strategy.career_stage == "experienced"
    assert repaired.resume.skill_groups
    assert "ImaginaryDB" not in repaired.resume.skills
    assert len({item.casefold() for item in repaired.resume.skills}) == len(
        repaired.resume.skills
    )
    assert repaired.resume.experience[0].company == "Northstar Analytics"
    assert repaired.resume.experience[0].start_date == "January 2022"
    assert repaired.resume.experience[0].bullets[0].text in {
        item.statement
        for item in snapshot.master_profile.experience[0].achievements
    }
    assert all(
        all(metric in bullet.text for metric in bullet.metrics)
        for role in repaired.resume.experience
        for bullet in role.bullets
    )
    assert repaired.resume.education[0].institution == "University of Texas at Dallas"
    for paragraph in repaired.cover_letter.paragraphs:
        for claim in paragraph.candidate_claims:
            assert claim.text in paragraph.text

    settings = ModelSettings(
        model="test-model",
        keyword_coverage_threshold=0.65,
        metric_density_threshold=0.60,
        groundedness_threshold=1.0,
    )
    validation = PackageValidator(settings).validate(
        repaired,
        snapshot.master_profile,
        snapshot.job_requirements,
        expected_letter_date="2026-09-12",
    )
    assert validation.passed, validation.issues
    assert validation.groundedness_ratio == 1.0
    assert counts["summary_repaired"] == 1
    assert counts["cover_letter_repaired"] == 1


def test_repair_falls_back_when_all_model_records_are_unknown_and_is_deterministic(
    snapshot: PipelineSnapshot,
) -> None:
    payload = deepcopy(snapshot.document_package.model_dump())
    payload["resume"]["experience"] = [
        {
            "profile_experience_id": "unknown-role",
            "company": "Unknown Employer",
            "title": "Unknown Role",
            "location": None,
            "start_date": None,
            "end_date": None,
            "bullets": [
                {
                    "text": "Created a completely unsupported result for testing.",
                    "accomplishment": "Created a result",
                    "measurement": "unsupported",
                    "method": "testing",
                    "achievement_ids": ["unknown-achievement"],
                }
            ],
        }
    ]
    payload["resume"]["education"] = [
        {
            "profile_education_id": "unknown-education",
            "institution": "Unknown University",
            "degree": "Unknown Degree",
            "field_of_study": None,
            "graduation_date": None,
            "location": None,
        }
    ]
    package = DocumentPackage.model_validate(payload)

    first, first_counts = repair_document_package(
        package,
        snapshot.master_profile,
        snapshot.job_requirements,
        "2026-09-12",
    )
    second, second_counts = repair_document_package(
        package,
        snapshot.master_profile,
        snapshot.job_requirements,
        "2026-09-12",
    )

    assert first == second
    assert first_counts == second_counts
    assert first_counts["experience_fallback_used"] == 1
    assert first_counts["education_fallback_used"] == 1
    assert [item.profile_experience_id for item in first.resume.experience] == [
        item.profile_experience_id
        for item in snapshot.master_profile.experience
        if item.achievements
    ]
    assert [item.profile_education_id for item in first.resume.education] == [
        item.profile_education_id for item in snapshot.master_profile.education
    ]


def test_repair_canonicalizes_selected_projects_and_certifications(
    snapshot: PipelineSnapshot,
) -> None:
    profile = snapshot.master_profile.model_copy(
        update={
            "projects": [
                ProfileProject(
                    profile_project_id="project_1",
                    name="Source Project",
                    description="Built a Python data quality dashboard for operations.",
                    technologies=["Python", "SQL"],
                    evidence_ids=["e_a1"],
                )
            ],
            "certifications": [
                ProfileCertification(
                    profile_certification_id="cert_1",
                    name="Source Certification",
                    issuer="Source Issuer",
                    date="2024",
                    evidence_ids=["e_a1"],
                )
            ],
        }
    )
    payload = deepcopy(snapshot.document_package.model_dump())
    payload["resume"]["projects"] = [
        {
            "profile_project_id": "project_1",
            "name": "Invented Project Name",
            "description": {
                "text": "Invented project claim with 73% improvement.",
                "achievement_ids": ["unknown"],
            },
            "technologies": ["ImaginaryDB"],
        },
        {
            "profile_project_id": "unknown_project",
            "name": "Unknown Project",
            "description": {
                "text": "Another unsupported project description.",
                "achievement_ids": ["unknown"],
            },
            "technologies": [],
        },
    ]
    payload["resume"]["certifications"] = [
        {
            "profile_certification_id": "cert_1",
            "name": "Invented Certification Name",
            "issuer": "Wrong Issuer",
            "date": "2099",
        },
        {
            "profile_certification_id": "unknown_cert",
            "name": "Unknown Certification",
            "issuer": None,
            "date": None,
        },
    ]

    repaired, _ = repair_document_package(
        DocumentPackage.model_validate(payload),
        profile,
        snapshot.job_requirements,
        "2026-09-12",
    )

    assert len(repaired.resume.projects) == 1
    assert repaired.resume.projects[0].name == "Source Project"
    assert repaired.resume.projects[0].description.model_dump() == {
        "text": "Built a Python data quality dashboard for operations.",
        "achievement_ids": ["project_1"],
    }
    assert repaired.resume.projects[0].technologies == ["Python", "SQL"]
    assert len(repaired.resume.certifications) == 1
    assert repaired.resume.certifications[0].name == "Source Certification"
    assert repaired.resume.certifications[0].issuer == "Source Issuer"
    assert repaired.resume.certifications[0].date == "2024"


def test_repair_supports_student_resume_without_summary_or_experience(
    snapshot: PipelineSnapshot,
) -> None:
    profile = snapshot.master_profile.model_copy(
        update={
            "contact": snapshot.master_profile.contact.model_copy(
                update={"location": "Bengaluru, India"}
            ),
            "summary_facts": [],
            "experience": [],
        }
    )

    repaired, _ = repair_document_package(
        snapshot.document_package,
        profile,
        snapshot.job_requirements,
        "2026-09-12",
    )

    assert repaired.resume.professional_summary is None
    assert repaired.resume.experience == []
    assert repaired.resume.strategy is not None
    assert repaired.resume.strategy.career_stage == "student"
    assert repaired.resume.strategy.page_size == "A4"
    assert repaired.resume.strategy.section_order[:2] == ["education", "skills"]
    assert all(
        not paragraph.candidate_claims
        for paragraph in repaired.cover_letter.paragraphs
    )

    validation = PackageValidator(ModelSettings(model="test-model")).validate(
        repaired,
        profile,
        snapshot.job_requirements,
        expected_letter_date="2026-09-12",
    )
    assert validation.passed, validation.issues
    assert validation.groundedness_ratio == 1.0
