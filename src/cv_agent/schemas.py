"""Strict data contracts for every CV Agent pipeline boundary."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields and trims string values."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


DocumentKind = Literal["txt", "md", "pdf", "docx"]
CareerStage = Literal["student", "early_career", "experienced", "career_changer"]
TargetMarket = Literal["india", "uk", "us", "global"]
PageSize = Literal["A4", "Letter"]
ResumeSection = Literal[
    "summary",
    "skills",
    "experience",
    "projects",
    "certifications",
    "education",
]
ProjectType = Literal["personal", "academic", "open_source", "work", "other"]
CredentialStatus = Literal["active", "expired", "in_progress", "unknown"]
CredentialType = Literal[
    "professional_certification",
    "certificate",
    "skill_badge",
    "license",
    "training",
    "unknown",
]
EducationStatus = Literal["completed", "expected", "coursework", "unknown"]
ContactLinkType = Literal["linkedin", "github", "portfolio", "website", "other"]


class IngestionRequest(StrictModel):
    jd_path: Path | None
    jd_text: str | None
    cv_paths: list[Path] = Field(min_length=1)
    max_chars_per_document: int = Field(default=120_000, ge=1_000, le=1_000_000)

    @model_validator(mode="after")
    def require_one_jd_source(self) -> "IngestionRequest":
        if (self.jd_path is None) == (self.jd_text is None):
            raise ValueError("provide exactly one of jd_path or jd_text")
        if self.jd_text is not None and not self.jd_text.strip():
            raise ValueError("jd_text cannot be blank")
        return self


class DocumentText(StrictModel):
    document_id: str = Field(min_length=8)
    source_path: Path
    kind: DocumentKind
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: str = Field(min_length=1)
    character_count: int = Field(ge=1)


class IngestionBundle(StrictModel):
    job_description: DocumentText
    candidate_documents: list[DocumentText] = Field(min_length=1)


class EvidenceRecord(StrictModel):
    """An exact quotation from an ingested candidate document."""

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=8)
    quote: str = Field(min_length=3)


class GroundedFact(StrictModel):
    fact_id: str = Field(min_length=1)
    statement: str = Field(min_length=3)
    evidence_ids: list[str] = Field(min_length=1)


class ContactLink(StrictModel):
    type: ContactLinkType
    label: str = Field(min_length=1)
    url: str = Field(min_length=3)


def _contact_link_from_string(value: str) -> dict[str, str]:
    normalized = value.casefold()
    if "linkedin.com" in normalized:
        kind, label = "linkedin", "LinkedIn"
    elif "github.com" in normalized:
        kind, label = "github", "GitHub"
    elif normalized.startswith(("http://", "https://")):
        kind, label = "portfolio", "Portfolio"
    else:
        kind, label = "other", value
    return {"type": kind, "label": label, "url": value}


class ContactInfo(StrictModel):
    full_name: str = Field(min_length=1)
    email: str | None
    phone: str | None
    location: str | None
    links: list[ContactLink]

    @model_validator(mode="before")
    @classmethod
    def migrate_plain_links(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        links = payload.get("links", [])
        if isinstance(links, list):
            payload["links"] = [
                _contact_link_from_string(item) if isinstance(item, str) else item
                for item in links
            ]
        return payload


class ProfileAchievement(StrictModel):
    achievement_id: str = Field(min_length=1)
    statement: str = Field(min_length=3)
    evidence_ids: list[str] = Field(min_length=1)


class ProfileExperience(StrictModel):
    profile_experience_id: str = Field(min_length=1)
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    location: str | None
    start_date: str | None
    end_date: str | None
    evidence_ids: list[str] = Field(min_length=1)
    achievements: list[ProfileAchievement]


class ProfileSkill(StrictModel):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ProfileEducation(StrictModel):
    profile_education_id: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    field_of_study: str | None
    graduation_date: str | None
    location: str | None
    gpa: str | None
    honors: list[str]
    coursework: list[str]
    thesis: str | None
    completion_status: EducationStatus
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_education_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("gpa", None)
        payload.setdefault("honors", [])
        payload.setdefault("coursework", [])
        payload.setdefault("thesis", None)
        payload.setdefault("completion_status", "unknown")
        return payload


class ProfileProject(StrictModel):
    profile_project_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=3)
    technologies: list[str]
    role: str | None
    project_type: ProjectType
    start_date: str | None
    end_date: str | None
    repository_url: str | None
    demo_url: str | None
    employer_profile_experience_id: str | None
    achievements: list[ProfileAchievement]
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_project_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("role", None)
        payload.setdefault("project_type", "other")
        payload.setdefault("start_date", None)
        payload.setdefault("end_date", None)
        payload.setdefault("repository_url", None)
        payload.setdefault("demo_url", None)
        payload.setdefault("employer_profile_experience_id", None)
        payload.setdefault("achievements", [])
        return payload


class ProfileCertification(StrictModel):
    profile_certification_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    issuer: str | None
    date: str | None
    earned_date: str | None
    expiry_date: str | None
    status: CredentialStatus
    credential_id: str | None
    verification_url: str | None
    credential_type: CredentialType
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_certification_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("earned_date", payload.get("date"))
        payload.setdefault("expiry_date", None)
        payload.setdefault("status", "unknown")
        payload.setdefault("credential_id", None)
        payload.setdefault("verification_url", None)
        payload.setdefault("credential_type", "unknown")
        return payload


class MasterProfile(StrictModel):
    contact: ContactInfo
    contact_evidence_ids: list[str] = Field(min_length=1)
    evidence_catalog: list[EvidenceRecord] = Field(min_length=1)
    summary_facts: list[GroundedFact]
    experience: list[ProfileExperience]
    skills: list[ProfileSkill]
    education: list[ProfileEducation]
    projects: list[ProfileProject]
    certifications: list[ProfileCertification]


class JobRequirements(StrictModel):
    source_document_id: str = Field(min_length=8)
    target_title: str = Field(min_length=1)
    company_name: str | None
    competencies: list[str]
    hard_skills: list[str]
    tools: list[str]
    methodologies: list[str]
    responsibilities: list[str]
    pain_points: list[str]
    keywords: list[str] = Field(min_length=1)


class GroundedText(StrictModel):
    text: str = Field(min_length=3)
    achievement_ids: list[str] = Field(min_length=1)


class ResumeBullet(StrictModel):
    """A rendered, sourced achievement with flexible evidence components."""

    text: str = Field(min_length=12)
    action: str | None
    object: str | None
    method: str | None
    scope: str | None
    outcome: str | None
    metrics: list[str]
    attribution: str | None
    # Retained as nullable migration aliases for existing snapshots/API consumers.
    accomplishment: str | None
    measurement: str | None
    achievement_ids: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_bullet_components(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        text = str(payload.get("text", "")).strip()
        accomplishment = payload.get("accomplishment")
        measurement = payload.get("measurement")
        payload.setdefault("action", text.split(maxsplit=1)[0] if text else None)
        payload.setdefault("object", accomplishment)
        payload.setdefault("method", None)
        payload.setdefault("scope", None)
        payload.setdefault("outcome", measurement)
        payload.setdefault("metrics", [measurement] if measurement else [])
        payload.setdefault("attribution", None)
        payload.setdefault("accomplishment", payload.get("object"))
        payload.setdefault("measurement", payload.get("outcome"))
        return payload


class SkillGroup(StrictModel):
    category: str = Field(min_length=1)
    skills: list[str] = Field(min_length=1)


class ResumeExperience(StrictModel):
    profile_experience_id: str = Field(min_length=1)
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    location: str | None
    start_date: str | None
    end_date: str | None
    bullets: list[ResumeBullet] = Field(min_length=1)


class ResumeEducation(StrictModel):
    profile_education_id: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    field_of_study: str | None
    graduation_date: str | None
    location: str | None
    gpa: str | None
    honors: list[str]
    coursework: list[str]
    thesis: str | None
    completion_status: EducationStatus

    @model_validator(mode="before")
    @classmethod
    def migrate_education_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("gpa", None)
        payload.setdefault("honors", [])
        payload.setdefault("coursework", [])
        payload.setdefault("thesis", None)
        payload.setdefault("completion_status", "unknown")
        return payload


class ResumeProject(StrictModel):
    profile_project_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: GroundedText
    technologies: list[str]
    role: str | None
    project_type: ProjectType
    start_date: str | None
    end_date: str | None
    repository_url: str | None
    demo_url: str | None
    employer_profile_experience_id: str | None
    bullets: list[ResumeBullet]

    @model_validator(mode="before")
    @classmethod
    def migrate_project_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("role", None)
        payload.setdefault("project_type", "other")
        payload.setdefault("start_date", None)
        payload.setdefault("end_date", None)
        payload.setdefault("repository_url", None)
        payload.setdefault("demo_url", None)
        payload.setdefault("employer_profile_experience_id", None)
        payload.setdefault("bullets", [])
        return payload


class ResumeCertification(StrictModel):
    profile_certification_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    issuer: str | None
    date: str | None
    earned_date: str | None
    expiry_date: str | None
    status: CredentialStatus
    credential_id: str | None
    verification_url: str | None
    credential_type: CredentialType

    @model_validator(mode="before")
    @classmethod
    def migrate_certification_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("earned_date", payload.get("date"))
        payload.setdefault("expiry_date", None)
        payload.setdefault("status", "unknown")
        payload.setdefault("credential_id", None)
        payload.setdefault("verification_url", None)
        payload.setdefault("credential_type", "unknown")
        return payload


class ResumeStrategy(StrictModel):
    career_stage: CareerStage
    target_market: TargetMarket
    page_size: PageSize
    target_pages: int = Field(ge=1, le=2)
    include_summary: bool
    section_order: list[ResumeSection] = Field(min_length=1)
    strongest_evidence_sections: list[ResumeSection]
    unsupported_requirements: list[str]
    evidence_gap_questions: list[str]


class ATSResume(StrictModel):
    contact: ContactInfo
    target_title: str = Field(min_length=1)
    professional_summary: GroundedText | None
    skills: list[str]
    skill_groups: list[SkillGroup]
    experience: list[ResumeExperience]
    education: list[ResumeEducation]
    projects: list[ResumeProject]
    certifications: list[ResumeCertification]
    strategy: ResumeStrategy | None

    @model_validator(mode="before")
    @classmethod
    def migrate_resume_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        skills = payload.get("skills", [])
        payload.setdefault(
            "skill_groups",
            [{"category": "Technical Skills", "skills": skills}] if skills else [],
        )
        payload.setdefault("strategy", None)
        return payload


class CoverLetterParagraph(StrictModel):
    text: str = Field(min_length=20)
    candidate_claims: list[GroundedText]
    matched_requirements: list[str] = Field(min_length=1)


class CoverLetter(StrictModel):
    candidate: ContactInfo
    letter_date: str = Field(min_length=1)
    recipient_company: str = Field(min_length=1)
    salutation: str = Field(min_length=1)
    paragraphs: list[CoverLetterParagraph] = Field(min_length=3, max_length=4)
    closing: str = Field(min_length=1)
    signatory: str = Field(min_length=1)


class DocumentPackage(StrictModel):
    resume: ATSResume
    cover_letter: CoverLetter


Severity = Literal["error", "warning"]


class ValidationIssue(StrictModel):
    code: str = Field(min_length=1)
    severity: Severity
    message: str = Field(min_length=1)
    location: str = Field(min_length=1)
    retryable: bool


class GroundednessFinding(StrictModel):
    location: str = Field(min_length=1)
    supported: bool
    explanation: str = Field(min_length=1)
    supporting_achievement_ids: list[str]


class GroundednessAudit(StrictModel):
    passed: bool
    findings: list[GroundednessFinding]

    @model_validator(mode="after")
    def result_matches_findings(self) -> "GroundednessAudit":
        locations = [finding.location for finding in self.findings]
        if len(locations) != len(set(locations)):
            raise ValueError("groundedness finding locations must be unique")
        if self.passed != all(finding.supported for finding in self.findings):
            raise ValueError("passed must equal whether all findings are supported")
        return self


class ValidationResult(StrictModel):
    passed: bool
    keyword_coverage_ratio: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str]
    missing_keywords: list[str]
    metric_density: float = Field(ge=0.0, le=1.0)
    groundedness_ratio: float = Field(ge=0.0, le=1.0)
    ats_integrity_passed: bool
    issues: list[ValidationIssue]


class RenderedArtifact(StrictModel):
    kind: Literal["resume", "cover_letter"]
    pdf_path: Path
    html_path: Path
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(ge=1)


class RenderManifest(StrictModel):
    template_version: str
    rendered_on: date
    artifacts: list[RenderedArtifact] = Field(min_length=2, max_length=2)


class ModelSettings(StrictModel):
    provider: Literal["openai", "gemini", "openrouter"] = "openai"
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.1, le=0.3)
    max_correction_retries: int = Field(default=2, ge=0, le=2)
    keyword_coverage_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Retained for configuration/snapshot compatibility. Metric usage is telemetry,
    # not a quality threshold: truthful qualitative outcomes are valid evidence.
    metric_density_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    groundedness_threshold: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def enforce_retry_contract(self) -> "ModelSettings":
        if self.max_correction_retries > 2:
            raise ValueError("max_correction_retries cannot exceed 2")
        return self


class PipelineResult(StrictModel):
    master_profile: MasterProfile
    job_requirements: JobRequirements
    document_package: DocumentPackage
    groundedness_audit: GroundednessAudit
    validation: ValidationResult
    render_manifest: RenderManifest
    model_settings: ModelSettings
    correction_attempts: int = Field(ge=0, le=2)


class PipelineSnapshot(StrictModel):
    """Serializable audit record written next to generated artifacts."""

    master_profile: MasterProfile
    job_requirements: JobRequirements
    document_package: DocumentPackage
    groundedness_audit: GroundednessAudit
    validation: ValidationResult
    model_settings: ModelSettings
    correction_attempts: int = Field(ge=0, le=2)


class CorrectionContext(StrictModel):
    attempt: int = Field(ge=1, le=2)
    issues: list[ValidationIssue] = Field(min_length=1)


class GenerateRequest(StrictModel):
    ingestion: IngestionRequest
    output_dir: Path
    letter_date: date
