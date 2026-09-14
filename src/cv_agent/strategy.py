"""Deterministic resume strategy and evidence-gap planning.

The planner changes presentation only. It never upgrades a candidate claim or treats a
job-description requirement as candidate experience.
"""

from __future__ import annotations

import re

from .schemas import JobRequirements, MasterProfile, ResumeSection, ResumeStrategy


_WORD_RE = re.compile(r"[a-z][a-z0-9+#.-]{1,}", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<!\w)(?:[$€£₹]\s*)?\d[\d,.]*(?:\.\d+)?", re.IGNORECASE)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _WORD_RE.findall(value)}


def _profile_text(profile: MasterProfile) -> str:
    values: list[str] = [
        *(fact.statement for fact in profile.summary_facts),
        *(skill.name for skill in profile.skills),
    ]
    for role in profile.experience:
        values.extend([role.title, role.company])
        values.extend(item.statement for item in role.achievements)
    for project in profile.projects:
        values.extend([project.name, project.description, *project.technologies])
        values.extend(item.statement for item in project.achievements)
    for certification in profile.certifications:
        values.extend(
            value
            for value in [certification.name, certification.issuer]
            if value
        )
    return " ".join(values)


def _is_supported(requirement: str, profile_text: str) -> bool:
    requirement_normalized = _normalized(requirement)
    profile_normalized = _normalized(profile_text)
    if requirement_normalized in profile_normalized:
        return True
    requirement_tokens = _tokens(requirement)
    return bool(requirement_tokens) and requirement_tokens <= _tokens(profile_text)


def _target_market(location: str | None) -> tuple[str, str]:
    value = _normalized(location or "")
    india_signals = {
        "india", "bengaluru", "bangalore", "delhi", "gurugram", "gurgaon",
        "hyderabad", "mumbai", "pune", "chennai", "kolkata", "noida",
    }
    uk_signals = {"united kingdom", " uk", "london", "manchester", "edinburgh", "glasgow"}
    if any(signal in value for signal in india_signals):
        return "india", "A4"
    if any(signal in f" {value}" for signal in uk_signals):
        return "uk", "A4"
    us_states = {
        "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi",
        "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi",
        "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc",
        "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut",
        "vt", "va", "wa", "wv", "wi", "wy", "dc",
    }
    location_parts = {
        part.strip().casefold().split()[0]
        for part in (location or "").split(",")[1:]
        if part.strip()
    }
    if (
        bool(location_parts & us_states)
        or "united states" in value
        or " usa" in f" {value}"
        or " u.s." in f" {value}"
    ):
        return "us", "Letter"
    return "global", "A4"


def _career_stage(profile: MasterProfile, requirements: JobRequirements) -> str:
    if not profile.experience:
        return "student" if profile.education else "career_changer"

    requirement_text = " ".join(
        [
            *requirements.hard_skills,
            *requirements.tools,
            *requirements.methodologies,
            *requirements.responsibilities,
            *requirements.keywords,
        ]
    )
    requirement_tokens = _tokens(requirement_text)
    experience_tokens = _tokens(
        " ".join(
            [
                *(role.title for role in profile.experience),
                *(
                    item.statement
                    for role in profile.experience
                    for item in role.achievements
                ),
            ]
        )
    )
    project_tokens = _tokens(
        " ".join(
            [
                *(project.description for project in profile.projects),
                *(technology for project in profile.projects for technology in project.technologies),
            ]
        )
    )
    experience_overlap = len(requirement_tokens & experience_tokens)
    project_overlap = len(requirement_tokens & project_tokens)
    if profile.projects and project_overlap >= max(2, experience_overlap * 2):
        return "career_changer"
    if len(profile.experience) == 1 and sum(
        len(role.achievements) for role in profile.experience
    ) <= 3:
        return "early_career"
    return "experienced"


def _evidence_gap_questions(
    profile: MasterProfile,
    unsupported_requirements: list[str],
) -> list[str]:
    questions = [
        (
            f"Do you have demonstrable experience with {requirement}? If yes, provide "
            "where you used it, your contribution, and the outcome?"
        )
        for requirement in unsupported_requirements[:3]
    ]

    achievements = [
        item.statement for role in profile.experience for item in role.achievements
    ]
    if achievements and any(not _NUMBER_RE.search(item) for item in achievements):
        questions.append(
            "For any unquantified achievement, can you verify useful scale such as users, "
            "requests, data volume, time saved, reliability, cost, or delivery frequency?"
        )
    if profile.projects and any(
        not (project.repository_url or project.demo_url) for project in profile.projects
    ):
        questions.append(
            "Do any selected projects have a public repository or live demo URL that a recruiter can verify?"
        )
    if profile.projects and any(not project.achievements for project in profile.projects):
        questions.append(
            "For each important project, what did you personally implement and how was it tested, benchmarked, or used?"
        )
    if profile.certifications and any(
        certification.status == "unknown" or not certification.verification_url
        for certification in profile.certifications
    ):
        questions.append(
            "For each relevant credential, is it active, expired, or in progress, and is there a public verification URL?"
        )
    return list(dict.fromkeys(questions))[:8]


def build_resume_strategy(
    profile: MasterProfile,
    requirements: JobRequirements,
) -> ResumeStrategy:
    """Return a source-aware, career-stage-specific rendering plan."""

    career_stage = _career_stage(profile, requirements)
    market, page_size = _target_market(profile.contact.location)
    profile_text = _profile_text(profile)
    requested = list(
        dict.fromkeys(
            [
                *requirements.hard_skills,
                *requirements.tools,
                *requirements.methodologies,
                *requirements.responsibilities,
                *requirements.keywords,
            ]
        )
    )
    unsupported = [item for item in requested if not _is_supported(item, profile_text)]
    include_summary = bool(profile.summary_facts) and career_stage in {
        "experienced",
        "career_changer",
    }

    if career_stage == "student":
        order: list[ResumeSection] = [
            "education", "skills", "projects", "experience", "certifications",
        ]
    elif career_stage == "career_changer":
        order = [
            "summary", "skills", "certifications", "projects", "experience", "education",
        ]
    elif career_stage == "early_career":
        order = [
            "summary", "skills", "experience", "projects", "education", "certifications",
        ]
    else:
        order = [
            "summary", "skills", "experience", "certifications", "projects", "education",
        ]
    if not include_summary and "summary" in order:
        order.remove("summary")

    available: dict[ResumeSection, bool] = {
        "summary": include_summary,
        "skills": bool(profile.skills),
        "experience": any(role.achievements for role in profile.experience),
        "projects": bool(profile.projects),
        "certifications": bool(profile.certifications),
        "education": bool(profile.education),
    }
    section_order = [section for section in order if available[section]]
    evidence_sizes = {
        "summary": len(profile.summary_facts),
        "skills": len(profile.skills),
        "experience": sum(len(role.achievements) for role in profile.experience),
        "projects": len(profile.projects),
        "certifications": len(profile.certifications),
        "education": len(profile.education),
    }
    strongest = sorted(
        section_order,
        key=lambda section: (evidence_sizes[section], -section_order.index(section)),
        reverse=True,
    )[:3]
    content_volume = evidence_sizes["experience"] + (evidence_sizes["projects"] * 2)
    target_pages = 2 if career_stage == "experienced" and content_volume > 8 else 1

    return ResumeStrategy(
        career_stage=career_stage,
        target_market=market,
        page_size=page_size,
        target_pages=target_pages,
        include_summary=include_summary,
        section_order=section_order or ["skills"],
        strongest_evidence_sections=strongest,
        unsupported_requirements=unsupported[:12],
        evidence_gap_questions=_evidence_gap_questions(profile, unsupported),
    )
