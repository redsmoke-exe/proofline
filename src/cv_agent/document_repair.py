"""Deterministic, truth-preserving repair for generated candidate documents.

The language model is useful for selecting and arranging material, but it is not the
authority for candidate facts.  This module projects its ``DocumentPackage`` back onto
the exact records in ``MasterProfile`` and constructs a conservative cover letter.  It
therefore cannot introduce a skill, identity field, metric, or achievement that was not
already accepted at the profile boundary.
"""

from __future__ import annotations

import re
from datetime import date
from collections.abc import Callable
from typing import TypeVar

from .schemas import (
    ATSResume,
    CoverLetter,
    CoverLetterParagraph,
    DocumentPackage,
    GroundedText,
    JobRequirements,
    MasterProfile,
    ProfileAchievement,
    ResumeBullet,
    ResumeCertification,
    ResumeEducation,
    ResumeExperience,
    ResumeProject,
    ResumeStrategy,
    SkillGroup,
)
from .strategy import build_resume_strategy


_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$€£₹]\s*)?\d[\d,.]*(?:\.\d+)?\s*"
    r"(?:%|percent|x|k|m|b|million|billion)?",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z][a-z0-9+#.-]{1,}", re.IGNORECASE)
_METHOD_SEPARATOR_RE = re.compile(
    r"\s+(by|through|using|via|with)\s+",
    re.IGNORECASE,
)
_T = TypeVar("_T")


class DocumentRepairError(ValueError):
    """Raised only when the current schemas cannot represent the supplied profile."""


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _WORD_RE.findall(value)}


def _dedupe(values: list[_T], key: Callable[[_T], object]) -> list[_T]:
    result: list[_T] = []
    seen: set[object] = set()
    for value in values:
        marker = key(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _require_representable_profile(profile: MasterProfile) -> None:
    """Explain structural impossibilities instead of fabricating schema filler."""

    missing: list[str] = []
    if not profile.summary_facts and not any(
        role.achievements for role in profile.experience
    ) and not profile.projects and not profile.skills and not profile.education:
        missing.append("at least one grounded skill, education, summary, achievement, or project")
    if missing:
        raise DocumentRepairError(
            "The current ATSResume schema cannot represent this profile without "
            "inventing content; missing " + ", ".join(missing) + "."
        )


def _requirement_values(requirements: JobRequirements) -> list[str]:
    return _dedupe(
        [
            *requirements.responsibilities,
            *requirements.pain_points,
            *requirements.competencies,
            *requirements.hard_skills,
            *requirements.tools,
            *requirements.methodologies,
            *requirements.keywords,
            requirements.target_title,
        ],
        lambda item: _normalized(item),
    )


def _safe_requirement_values(requirements: JobRequirements) -> list[str]:
    """Prefer JD phrases without numbers, which must not look like candidate metrics."""

    values = _requirement_values(requirements)
    safe = [value for value in values if not _NUMBER_RE.search(value)]
    return safe or [requirements.target_title]


def _claim_catalog(profile: MasterProfile) -> dict[str, str]:
    result = {fact.fact_id: fact.statement for fact in profile.summary_facts}
    for role in profile.experience:
        for achievement in role.achievements:
            result.setdefault(achievement.achievement_id, achievement.statement)
    for project in profile.projects:
        result.setdefault(project.profile_project_id, project.description)
        for achievement in project.achievements:
            result.setdefault(achievement.achievement_id, achievement.statement)
    return result


def _ranked_claim_ids(profile: MasterProfile, requirements: JobRequirements) -> list[str]:
    requirement_tokens = _tokens(" ".join(_requirement_values(requirements)))
    candidates: list[tuple[int, int, int, str]] = []
    def add(claim_id: str, statement: str, kind_priority: int) -> None:
        relevance = len(_tokens(statement) & requirement_tokens)
        metric_bonus = 1 if _NUMBER_RE.search(statement) else 0
        candidates.append((relevance, metric_bonus, kind_priority, claim_id))

    # Stable sort preserves source order for equally relevant claims.
    for fact in profile.summary_facts:
        add(fact.fact_id, fact.statement, 2)
    for role in profile.experience:
        for achievement in role.achievements:
            add(achievement.achievement_id, achievement.statement, 1)
    for project in profile.projects:
        add(project.profile_project_id, project.description, 0)
        for achievement in project.achievements:
            add(achievement.achievement_id, achievement.statement, 0)
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in candidates]


def _canonical_summary(
    package: DocumentPackage,
    profile: MasterProfile,
    requirements: JobRequirements,
    strategy: ResumeStrategy,
) -> GroundedText | None:
    if not strategy.include_summary:
        return None
    claims = _claim_catalog(profile)
    if not claims:
        return None
    model_summary = package.resume.professional_summary
    cited_ids = _dedupe(
        [
            item
            for item in (model_summary.achievement_ids if model_summary else [])
            if item in claims
        ],
        lambda item: item,
    )
    if model_summary is not None and cited_ids:
        evidence = " ".join(claims[item] for item in cited_ids)
        unsupported_numbers = {
            item.group(0).strip() for item in _NUMBER_RE.finditer(model_summary.text)
        } - {item.group(0).strip() for item in _NUMBER_RE.finditer(evidence)}
        lexical_support = len(_tokens(model_summary.text) & _tokens(evidence)) / max(
            len(_tokens(model_summary.text)), 1
        )
        if not unsupported_numbers and lexical_support >= 0.25:
            return GroundedText(text=model_summary.text, achievement_ids=cited_ids)

    selected_ids = _ranked_claim_ids(profile, requirements)[:2]
    return GroundedText(
        text=" ".join(claims[item] for item in selected_ids),
        achievement_ids=selected_ids,
    )


def _canonical_skills(
    profile: MasterProfile,
    requirements: JobRequirements,
    strategy: ResumeStrategy,
) -> tuple[list[str], list[SkillGroup]]:
    by_normalized_name = {
        _normalized(skill.name): skill.name
        for skill in profile.skills
    }
    ordered: list[str] = []
    for requirement in [
        *requirements.hard_skills,
        *requirements.tools,
        *requirements.methodologies,
        *requirements.keywords,
    ]:
        skill = by_normalized_name.get(_normalized(requirement))
        if skill is not None:
            ordered.append(skill)
    requirement_tokens = _tokens(" ".join(_requirement_values(requirements)))
    remaining = sorted(
        profile.skills,
        key=lambda skill: len(_tokens(skill.name) & requirement_tokens),
        reverse=True,
    )
    ordered.extend(skill.name for skill in remaining)
    max_skills = 32 if strategy.target_pages == 2 else 24
    skills = _dedupe(ordered, lambda item: _normalized(item))[:max_skills]
    profile_by_name = {
        _normalized(skill.name): skill for skill in profile.skills
    }
    groups: dict[str, list[str]] = {}
    for name in skills:
        skill = profile_by_name.get(_normalized(name))
        if skill is None:
            continue
        groups.setdefault(skill.category, []).append(skill.name)
    skill_groups = [
        SkillGroup(
            category=category,
            skills=_dedupe(items, lambda item: _normalized(item)),
        )
        for category, items in groups.items()
        if items
    ]
    return skills, skill_groups


def _achievement_for_untrusted_bullet(
    text: str,
    achievements: list[ProfileAchievement],
) -> ProfileAchievement:
    claim_tokens = _tokens(text)
    return max(
        enumerate(achievements),
        key=lambda item: (
            len(claim_tokens & _tokens(item[1].statement)),
            bool(_NUMBER_RE.search(item[1].statement)),
            -item[0],
        ),
    )[1]


def _bullet_from_achievement(achievement: ProfileAchievement) -> ResumeBullet:
    statement = achievement.statement.strip()
    separator = _METHOD_SEPARATOR_RE.search(statement)
    if separator:
        accomplishment = statement[: separator.start()].strip(" .;:") or statement
        method = statement[separator.start() :].strip(" .;:") or statement
    else:
        accomplishment = statement
        method = statement
    measurements = [match.group(0).strip() for match in _NUMBER_RE.finditer(statement)]
    measurement = ", ".join(measurements) if measurements else None
    words = statement.split(maxsplit=1)
    action = words[0].strip(" .;:") if words else None
    object_text = words[1].strip(" .;:") if len(words) > 1 else accomplishment
    return ResumeBullet(
        text=statement,
        action=action,
        object=object_text,
        scope=None,
        outcome=measurement,
        metrics=measurements,
        attribution=None,
        accomplishment=accomplishment,
        measurement=measurement,
        method=method if separator else None,
        achievement_ids=[achievement.achievement_id],
    )


def _canonical_experience(
    package: DocumentPackage,
    profile: MasterProfile,
    requirements: JobRequirements,
    strategy: ResumeStrategy,
) -> tuple[list[ResumeExperience], bool]:
    profile_by_id = {role.profile_experience_id: role for role in profile.experience}
    model_by_id = {
        role.profile_experience_id: role
        for role in package.resume.experience
        if role.profile_experience_id in profile_by_id
    }
    selected_ids = _dedupe(
        [
            role.profile_experience_id
            for role in package.resume.experience
            if role.profile_experience_id in profile_by_id
            and profile_by_id[role.profile_experience_id].achievements
        ],
        lambda item: item,
    )
    used_fallback = not selected_ids
    if not selected_ids:
        selected_ids = [
            role.profile_experience_id
            for role in profile.experience
            if role.achievements
        ]

    max_roles = 6 if strategy.target_pages == 2 else 4
    result: list[ResumeExperience] = []
    requirement_tokens = _tokens(" ".join(_requirement_values(requirements)))
    linked_projects: dict[str, list[ProfileAchievement]] = {}
    for project in profile.projects:
        employer_id = project.employer_profile_experience_id
        if project.project_type != "work" or not employer_id:
            continue
        project_claims = list(project.achievements) or [
            ProfileAchievement(
                achievement_id=project.profile_project_id,
                statement=project.description,
                evidence_ids=project.evidence_ids,
            )
        ]
        linked_projects.setdefault(employer_id, []).extend(project_claims)

    for role_position, role_id in enumerate(selected_ids[:max_roles]):
        source = profile_by_id[role_id]
        source_achievements = {
            achievement.achievement_id: achievement
            for achievement in source.achievements
        }
        selected_achievements: list[ProfileAchievement] = []
        model_role = model_by_id.get(role_id)
        if model_role is not None:
            for bullet in model_role.bullets:
                valid_ids = [
                    item for item in bullet.achievement_ids if item in source_achievements
                ]
                if valid_ids:
                    selected_achievements.extend(
                        source_achievements[item] for item in valid_ids
                    )
                else:
                    selected_achievements.append(
                        _achievement_for_untrusted_bullet(
                            bullet.text,
                            source.achievements,
                        )
                    )
        if not selected_achievements:
            selected_achievements = list(source.achievements)
        selected_achievements = _dedupe(
            [*selected_achievements, *linked_projects.get(role_id, [])],
            lambda item: item.achievement_id,
        )
        selected_achievements.sort(
            key=lambda item: len(_tokens(item.statement) & requirement_tokens),
            reverse=True,
        )
        max_bullets = 5 if strategy.target_pages == 2 and role_position == 0 else 4
        if role_position > 0:
            max_bullets = min(max_bullets, 3)
        selected_achievements = selected_achievements[:max_bullets]
        result.append(
            ResumeExperience(
                profile_experience_id=source.profile_experience_id,
                company=source.company,
                title=source.title,
                location=source.location,
                start_date=source.start_date,
                end_date=source.end_date,
                bullets=[
                    _bullet_from_achievement(achievement)
                    for achievement in selected_achievements
                ],
            )
        )
    return result, used_fallback


def _canonical_education(
    package: DocumentPackage,
    profile: MasterProfile,
) -> tuple[list[ResumeEducation], bool]:
    by_id = {item.profile_education_id: item for item in profile.education}
    selected_ids = _dedupe(
        [
            item.profile_education_id
            for item in package.resume.education
            if item.profile_education_id in by_id
        ],
        lambda item: item,
    )
    used_fallback = not selected_ids
    if not selected_ids:
        selected_ids = [item.profile_education_id for item in profile.education]
    return (
        [
            ResumeEducation(
                profile_education_id=by_id[item].profile_education_id,
                institution=by_id[item].institution,
                degree=by_id[item].degree,
                field_of_study=by_id[item].field_of_study,
                graduation_date=by_id[item].graduation_date,
                location=by_id[item].location,
                gpa=by_id[item].gpa,
                honors=list(by_id[item].honors),
                coursework=list(by_id[item].coursework),
                thesis=by_id[item].thesis,
                completion_status=by_id[item].completion_status,
            )
            for item in selected_ids
        ],
        used_fallback,
    )


def _canonical_projects(
    package: DocumentPackage,
    profile: MasterProfile,
    strategy: ResumeStrategy,
) -> list[ResumeProject]:
    by_id = {item.profile_project_id: item for item in profile.projects}
    selected_ids = _dedupe(
        [
            item.profile_project_id
            for item in package.resume.projects
            if item.profile_project_id in by_id
        ],
        lambda item: item,
    )
    if not selected_ids:
        selected_ids = [item.profile_project_id for item in profile.projects]
    selected_ids = [
        item
        for item in selected_ids
        if not (
            by_id[item].project_type == "work"
            and by_id[item].employer_profile_experience_id
        )
    ][: 4 if strategy.target_pages == 2 else 2]
    result: list[ResumeProject] = []
    for item in selected_ids:
        source = by_id[item]
        result.append(
            ResumeProject(
                profile_project_id=source.profile_project_id,
                name=source.name,
                description=GroundedText(
                    text=source.description,
                    achievement_ids=[source.profile_project_id],
                ),
                technologies=list(source.technologies),
                role=source.role,
                project_type=source.project_type,
                start_date=source.start_date,
                end_date=source.end_date,
                repository_url=source.repository_url,
                demo_url=source.demo_url,
                employer_profile_experience_id=source.employer_profile_experience_id,
                bullets=[
                    _bullet_from_achievement(achievement)
                    for achievement in source.achievements[:3]
                ],
            )
        )
    return result


def _canonical_certifications(
    package: DocumentPackage,
    profile: MasterProfile,
) -> list[ResumeCertification]:
    by_id = {
        item.profile_certification_id: item for item in profile.certifications
    }
    selected_ids = _dedupe(
        [
            item.profile_certification_id
            for item in package.resume.certifications
            if item.profile_certification_id in by_id
        ],
        lambda item: item,
    )
    if not selected_ids:
        selected_ids = [
            item.profile_certification_id for item in profile.certifications
        ]
    return [
        ResumeCertification(
            profile_certification_id=by_id[item].profile_certification_id,
            name=by_id[item].name,
            issuer=by_id[item].issuer,
            date=by_id[item].date,
            earned_date=by_id[item].earned_date,
            expiry_date=by_id[item].expiry_date,
            status=by_id[item].status,
            credential_id=by_id[item].credential_id,
            verification_url=by_id[item].verification_url,
            credential_type=by_id[item].credential_type,
        )
        for item in selected_ids
    ]


def _canonical_cover_letter(
    profile: MasterProfile,
    requirements: JobRequirements,
    expected_letter_date: str,
) -> CoverLetter:
    company = requirements.company_name or "Hiring Organization"
    requirement_values = _safe_requirement_values(requirements)
    ranked_claim_ids = _ranked_claim_ids(profile, requirements)
    claim_catalog = _claim_catalog(profile)
    # Prefer role achievements in the letter; a summary fact is the truthful fallback.
    achievement_ids = {
        achievement.achievement_id
        for role in profile.experience
        for achievement in role.achievements
    }
    cover_claim_ids = [item for item in ranked_claim_ids if item in achievement_ids]
    if not cover_claim_ids:
        cover_claim_ids = ranked_claim_ids
    first_claim_id = cover_claim_ids[0] if cover_claim_ids else None
    second_claim_id = cover_claim_ids[1] if len(cover_claim_ids) > 1 else None

    introduction_requirement = requirement_values[0]
    evidence_requirement = requirement_values[min(1, len(requirement_values) - 1)]
    closing_requirement = requirement_values[min(2, len(requirement_values) - 1)]
    paragraphs = [
        CoverLetterParagraph(
            text=(
                f"I am applying for the {requirements.target_title} role at {company}. "
                f"The opportunity to contribute to {introduction_requirement} closely "
                "matches my professional focus."
            ),
            candidate_claims=[],
            matched_requirements=[introduction_requirement],
        ),
    ]
    if first_claim_id is not None:
        first_claim = claim_catalog[first_claim_id]
        paragraphs.append(
            CoverLetterParagraph(
                text=(
                    f"{first_claim} This experience would help me contribute to "
                    f"{evidence_requirement}."
                ),
                candidate_claims=[
                    GroundedText(text=first_claim, achievement_ids=[first_claim_id])
                ],
                matched_requirements=[evidence_requirement],
            )
        )
    else:
        paragraphs.append(
            CoverLetterParagraph(
                text=(
                    f"I am particularly interested in the team's work on "
                    f"{evidence_requirement} and the outcomes expected from this role."
                ),
                candidate_claims=[],
                matched_requirements=[evidence_requirement],
            )
        )
    if second_claim_id is not None:
        second_claim = claim_catalog[second_claim_id]
        closing_text = (
            f"{second_claim} I would welcome the opportunity to bring this experience "
            f"to {company} and support {closing_requirement}."
        )
        closing_claims = [
            GroundedText(text=second_claim, achievement_ids=[second_claim_id])
        ]
    else:
        closing_text = (
            f"I would welcome the opportunity to bring my experience to {company} and "
            f"support {closing_requirement}. Thank you for your consideration."
        )
        closing_claims = []
    paragraphs.append(
        CoverLetterParagraph(
            text=closing_text,
            candidate_claims=closing_claims,
            matched_requirements=[closing_requirement],
        )
    )
    return CoverLetter(
        candidate=profile.contact,
        letter_date=expected_letter_date,
        recipient_company=company,
        salutation="Dear Hiring Team,",
        paragraphs=paragraphs,
        closing="Sincerely,",
        signatory=profile.contact.full_name,
    )


def repair_document_package(
    package: DocumentPackage,
    profile: MasterProfile,
    requirements: JobRequirements,
    expected_letter_date: str | date,
    strategy: ResumeStrategy | None = None,
) -> tuple[DocumentPackage, dict[str, int]]:
    """Return a schema-valid package projected onto exact, grounded profile records.

    Model ordering choices are retained when their referenced records exist. Unknown and
    duplicate references are discarded. If no selected experience or education record is
    usable, the function deterministically falls back to grounded profile records. Every
    rendered candidate claim is copied from the profile verbatim.
    """

    _require_representable_profile(profile)
    letter_date = (
        expected_letter_date.isoformat()
        if isinstance(expected_letter_date, date)
        else expected_letter_date
    )
    if not letter_date.strip():
        raise DocumentRepairError("expected_letter_date cannot be blank")

    strategy = strategy or build_resume_strategy(profile, requirements)
    experience, experience_fallback = _canonical_experience(
        package, profile, requirements, strategy
    )
    education, education_fallback = _canonical_education(package, profile)
    skills, skill_groups = _canonical_skills(profile, requirements, strategy)
    repaired = DocumentPackage(
        resume=ATSResume(
            contact=profile.contact,
            target_title=requirements.target_title,
            professional_summary=_canonical_summary(
                package, profile, requirements, strategy
            ),
            skills=skills,
            skill_groups=skill_groups,
            experience=experience,
            education=education,
            projects=_canonical_projects(package, profile, strategy),
            certifications=_canonical_certifications(package, profile),
            strategy=strategy,
        ),
        cover_letter=_canonical_cover_letter(
            profile,
            requirements,
            letter_date,
        ),
    )

    original = package.model_dump(mode="json")
    result = repaired.model_dump(mode="json")
    counts = {
        "resume_contact_repaired": int(original["resume"]["contact"] != result["resume"]["contact"]),
        "target_title_repaired": int(original["resume"]["target_title"] != result["resume"]["target_title"]),
        "summary_repaired": int(original["resume"]["professional_summary"] != result["resume"]["professional_summary"]),
        "skills_repaired": int(original["resume"]["skills"] != result["resume"]["skills"]),
        "experience_records_repaired": sum(
            1
            for index, role in enumerate(result["resume"]["experience"])
            if index >= len(original["resume"]["experience"])
            or role != original["resume"]["experience"][index]
        ),
        "education_records_repaired": sum(
            1
            for index, item in enumerate(result["resume"]["education"])
            if index >= len(original["resume"]["education"])
            or item != original["resume"]["education"][index]
        ),
        "projects_repaired": int(original["resume"]["projects"] != result["resume"]["projects"]),
        "certifications_repaired": int(original["resume"]["certifications"] != result["resume"]["certifications"]),
        "cover_letter_repaired": int(original["cover_letter"] != result["cover_letter"]),
        "experience_fallback_used": int(experience_fallback),
        "education_fallback_used": int(education_fallback),
    }
    return repaired, counts
