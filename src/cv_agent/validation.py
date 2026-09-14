"""Deterministic ATS, metric, keyword, provenance, and truthfulness gates."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .ingestion import normalize_text
from .schemas import (
    DocumentPackage,
    EvidenceRecord,
    GroundednessAudit,
    IngestionBundle,
    JobRequirements,
    MasterProfile,
    ModelSettings,
    ValidationIssue,
    ValidationResult,
)
from .strategy import build_resume_strategy


_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$€£₹]\s*)?\d[\d,.]*(?:\.\d+)?\s*(?:%|percent|x|k|m|b|million|billion)?",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z][a-z0-9+#.-]{1,}", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}
_ACTION_VERBS = {
    "accelerated",
    "administered",
    "advised",
    "analyzed",
    "architected",
    "authored",
    "achieved",
    "automated",
    "built",
    "championed",
    "coached",
    "collaborated",
    "configured",
    "consolidated",
    "coordinated",
    "created",
    "cut",
    "decreased",
    "deployed",
    "delivered",
    "developed",
    "designed",
    "directed",
    "doubled",
    "drove",
    "eliminated",
    "enabled",
    "engineered",
    "enhanced",
    "established",
    "executed",
    "expanded",
    "facilitated",
    "founded",
    "generated",
    "grew",
    "guided",
    "hardened",
    "identified",
    "implemented",
    "improved",
    "increased",
    "integrated",
    "introduced",
    "launched",
    "led",
    "maintained",
    "managed",
    "migrated",
    "modernized",
    "negotiated",
    "operated",
    "optimized",
    "orchestrated",
    "overhauled",
    "partnered",
    "pioneered",
    "planned",
    "produced",
    "programmed",
    "redesigned",
    "reduced",
    "refactored",
    "resolved",
    "revamped",
    "saved",
    "scaled",
    "secured",
    "shipped",
    "simplified",
    "spearheaded",
    "standardized",
    "streamlined",
    "strengthened",
    "supervised",
    "supported",
    "trained",
    "transformed",
    "tripled",
    "upgraded",
    "validated",
    # Present-tense forms are valid for genuinely ongoing work in a current role.
    "administer", "advise", "analyze", "architect", "author", "automate", "build",
    "champion", "coach", "collaborate", "configure", "coordinate", "create", "deploy",
    "deliver", "develop", "design", "direct", "drive", "enable", "engineer", "enhance",
    "execute", "facilitate", "generate", "guide", "harden", "identify", "implement",
    "improve", "increase", "integrate", "lead", "maintain", "manage", "migrate", "operate",
    "optimize", "orchestrate", "partner", "plan", "produce", "program", "reduce", "refactor",
    "resolve", "scale", "secure", "ship", "simplify", "standardize", "streamline", "support",
    "train", "upgrade", "validate",
}


def _normalized(value: str) -> str:
    return " ".join(normalize_text(value).casefold().split())


def _numbers(value: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group(0).casefold()) for match in _NUMBER_RE.finditer(value)}


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(value)
        if token.casefold() not in _STOP_WORDS
    }


def _overlap_ratio(claim: str, evidence: str) -> float:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & _tokens(evidence)) / len(claim_tokens)


def _issue(
    code: str,
    message: str,
    location: str,
    *,
    retryable: bool = True,
    severity: str = "error",
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        location=location,
        retryable=retryable,
    )


def _referenced_text(ids: Iterable[str], achievement_text: dict[str, str]) -> str:
    return " ".join(achievement_text[item] for item in ids if item in achievement_text)


def repair_profile_grounding(
    profile: MasterProfile,
    bundle: IngestionBundle,
) -> tuple[MasterProfile, dict[str, int]]:
    """Rebuild claim citations from exact source text without inventing content.

    Small/free models often extract useful facts but attach the wrong evidence IDs. This
    pass searches exact source lines and short line windows, re-cites supported claims,
    snaps unsupported achievement paraphrases to their closest exact source line, and
    removes only claims that still cannot pass the same deterministic provenance rules.
    """

    source_text = {
        document.document_id: _normalized(document.text)
        for document in bundle.candidate_documents
    }
    evidence_by_id: dict[str, EvidenceRecord] = {}
    evidence_by_quote: dict[tuple[str, str], EvidenceRecord] = {}
    for item in profile.evidence_catalog:
        if (
            item.evidence_id not in evidence_by_id
            and item.document_id in source_text
            and _normalized(item.quote) in source_text[item.document_id]
        ):
            evidence_by_id[item.evidence_id] = item
            evidence_by_quote[(item.document_id, _normalized(item.quote))] = item

    evidence_added = 0
    used_ids: set[str] = set()

    def add_candidate(document_id: str, quote: str) -> None:
        nonlocal evidence_added
        quote = quote.strip(" -*•\t")
        normalized_quote = _normalized(quote)
        if len(normalized_quote) < 3 or normalized_quote not in source_text[document_id]:
            return
        key = (document_id, normalized_quote)
        if key in evidence_by_quote:
            return
        digest = hashlib.sha256(f"{document_id}\0{normalized_quote}".encode()).hexdigest()
        evidence_id = f"auto_{digest[:16]}"
        suffix = 16
        while evidence_id in evidence_by_id:
            suffix += 2
            evidence_id = f"auto_{digest[:suffix]}"
        record = EvidenceRecord(
            evidence_id=evidence_id,
            document_id=document_id,
            quote=quote,
        )
        evidence_by_id[evidence_id] = record
        evidence_by_quote[key] = record
        evidence_added += 1

    for document in bundle.candidate_documents:
        normalized = normalize_text(document.text)
        lines = [line.strip(" -*•\t") for line in normalized.splitlines() if line.strip()]
        for index in range(len(lines)):
            for width in (1, 2, 3):
                window = " ".join(lines[index : index + width])
                if window and len(window) <= 1_200:
                    add_candidate(document.document_id, window)
        flattened = " ".join(lines)
        for sentence in re.split(r"(?<=[.!?])\s+", flattened):
            if len(sentence) <= 1_200:
                add_candidate(document.document_id, sentence)

    candidates = list(evidence_by_id.values())

    def cited_text(evidence_ids: list[str]) -> str:
        return " ".join(
            evidence_by_id[item].quote for item in evidence_ids if item in evidence_by_id
        )

    def is_supported(claim: str, evidence_ids: list[str]) -> bool:
        evidence = cited_text(evidence_ids)
        return (
            bool(evidence_ids)
            and not (_numbers(claim) - _numbers(evidence))
            and _overlap_ratio(claim, evidence) >= 0.25
        )

    def select_evidence(claim: str, current_ids: list[str]) -> list[str] | None:
        current = list(dict.fromkeys(item for item in current_ids if item in evidence_by_id))
        if is_supported(claim, current):
            used_ids.update(current)
            return current

        claim_tokens = _tokens(claim)
        claim_numbers = _numbers(claim)
        selected: list[str] = []
        covered_tokens: set[str] = set()
        covered_numbers: set[str] = set()
        remaining = candidates.copy()
        for _ in range(8):
            best: EvidenceRecord | None = None
            best_gain = 0.0
            for candidate in remaining:
                quote_tokens = _tokens(candidate.quote)
                new_tokens = (claim_tokens & quote_tokens) - covered_tokens
                new_numbers = (claim_numbers & _numbers(candidate.quote)) - covered_numbers
                precision = len(claim_tokens & quote_tokens) / max(len(quote_tokens), 1)
                gain = len(new_tokens) + (len(new_numbers) * 4) + precision
                if gain > best_gain:
                    best = candidate
                    best_gain = gain
            if best is None or best_gain <= 0:
                break
            selected.append(best.evidence_id)
            covered_tokens.update(claim_tokens & _tokens(best.quote))
            covered_numbers.update(claim_numbers & _numbers(best.quote))
            remaining.remove(best)
            if is_supported(claim, selected):
                used_ids.update(selected)
                return selected
        return None

    def closest_exact_quote(claim: str) -> EvidenceRecord | None:
        ranked = [
            (
                (_overlap_ratio(claim, item.quote) * 0.75)
                + (
                    len(_tokens(claim) & _tokens(item.quote))
                    / max(len(_tokens(item.quote)), 1)
                    * 0.25
                ),
                item,
            )
            for item in candidates
            if _overlap_ratio(claim, item.quote) >= 0.45
        ]
        return max(ranked, key=lambda item: item[0])[1] if ranked else None

    counts = {
        "evidence_added": evidence_added,
        "contact_relinked": 0,
        "summary_facts_pruned": 0,
        "roles_pruned": 0,
        "role_values_cleared": 0,
        "achievements_relinked": 0,
        "achievements_snapped": 0,
        "achievements_pruned": 0,
        "skills_relinked": 0,
        "skills_pruned": 0,
        "education_values_cleared": 0,
        "education_pruned": 0,
        "projects_pruned": 0,
        "certifications_pruned": 0,
    }

    contact_ids = select_evidence(
        profile.contact.full_name,
        profile.contact_evidence_ids,
    )
    contact_updates: dict[str, object] = {}
    if contact_ids is None:
        contact_ids = list(
            dict.fromkeys(
                item for item in profile.contact_evidence_ids if item in evidence_by_id
            )
        )
        used_ids.update(contact_ids)
    else:
        contact_values = [profile.contact.full_name]
        for field_name in ("email", "phone", "location"):
            value = getattr(profile.contact, field_name)
            if value is None:
                contact_updates[field_name] = None
                continue
            trial_ids = select_evidence(" ".join([*contact_values, value]), contact_ids)
            if trial_ids is None:
                contact_updates[field_name] = None
            else:
                contact_updates[field_name] = value
                contact_values.append(value)
                contact_ids = trial_ids
        links = []
        for link in profile.contact.links:
            trial_ids = select_evidence(" ".join([*contact_values, link.url]), contact_ids)
            if trial_ids is not None:
                links.append(link)
                contact_values.append(link.url)
                contact_ids = trial_ids
        contact_updates["links"] = links
        if contact_ids != profile.contact_evidence_ids:
            counts["contact_relinked"] += 1
    contact = profile.contact.model_copy(update=contact_updates)

    summary_facts = []
    for fact in profile.summary_facts:
        evidence_ids = select_evidence(fact.statement, fact.evidence_ids)
        if evidence_ids is None:
            counts["summary_facts_pruned"] += 1
        else:
            summary_facts.append(fact.model_copy(update={"evidence_ids": evidence_ids}))

    experience = []
    for role in profile.experience:
        required_values = [role.company, role.title]
        kept_optional: dict[str, str | None] = {}
        evidence_ids = select_evidence(" ".join(required_values), role.evidence_ids)
        if evidence_ids is None:
            counts["roles_pruned"] += 1
            continue
        for field_name in ("location", "start_date", "end_date"):
            value = getattr(role, field_name)
            if value is None:
                kept_optional[field_name] = None
                continue
            trial = " ".join([*required_values, *[v for v in kept_optional.values() if v], value])
            trial_ids = select_evidence(trial, evidence_ids)
            if trial_ids is None:
                kept_optional[field_name] = None
                counts["role_values_cleared"] += 1
            else:
                kept_optional[field_name] = value
                evidence_ids = trial_ids

        achievements = []
        for achievement in role.achievements:
            achievement_ids = select_evidence(
                achievement.statement,
                achievement.evidence_ids,
            )
            if achievement_ids is not None:
                if achievement_ids != achievement.evidence_ids:
                    counts["achievements_relinked"] += 1
                achievements.append(
                    achievement.model_copy(update={"evidence_ids": achievement_ids})
                )
                continue
            exact = closest_exact_quote(achievement.statement)
            if exact is not None:
                used_ids.add(exact.evidence_id)
                achievements.append(
                    achievement.model_copy(
                        update={"statement": exact.quote, "evidence_ids": [exact.evidence_id]}
                    )
                )
                counts["achievements_snapped"] += 1
            else:
                counts["achievements_pruned"] += 1
        experience.append(
            role.model_copy(
                update={
                    "evidence_ids": evidence_ids,
                    "achievements": achievements,
                    **kept_optional,
                }
            )
        )

    skills = []
    for skill in profile.skills:
        evidence_ids = select_evidence(skill.name, skill.evidence_ids)
        if evidence_ids is None:
            counts["skills_pruned"] += 1
        else:
            if evidence_ids != skill.evidence_ids:
                counts["skills_relinked"] += 1
            skills.append(skill.model_copy(update={"evidence_ids": evidence_ids}))

    education = []
    for item in profile.education:
        required_values = [item.institution, item.degree]
        evidence_ids = select_evidence(" ".join(required_values), item.evidence_ids)
        if evidence_ids is None:
            counts["education_pruned"] += 1
            continue
        kept_optional: dict[str, object] = {}
        for field_name in (
            "field_of_study", "graduation_date", "location", "gpa", "thesis"
        ):
            value = getattr(item, field_name)
            if value is None:
                kept_optional[field_name] = None
                continue
            trial = " ".join([*required_values, *[v for v in kept_optional.values() if v], value])
            trial_ids = select_evidence(trial, evidence_ids)
            if trial_ids is None:
                kept_optional[field_name] = None
                counts["education_values_cleared"] += 1
            else:
                kept_optional[field_name] = value
                evidence_ids = trial_ids
        for field_name in ("honors", "coursework"):
            kept_values: list[str] = []
            for value in getattr(item, field_name):
                trial_ids = select_evidence(value, evidence_ids)
                if trial_ids is None:
                    counts["education_values_cleared"] += 1
                else:
                    kept_values.append(value)
                    evidence_ids = trial_ids
            kept_optional[field_name] = kept_values
        completion_status = item.completion_status
        if completion_status != "unknown":
            status_ids = select_evidence(
                completion_status.replace("_", " "), evidence_ids
            )
            if status_ids is None:
                completion_status = "unknown"
            else:
                evidence_ids = list(dict.fromkeys([*evidence_ids, *status_ids]))
        education.append(
            item.model_copy(
                update={
                    "evidence_ids": evidence_ids,
                    "completion_status": completion_status,
                    **kept_optional,
                }
            )
        )

    projects = []
    for project in profile.projects:
        evidence_ids = select_evidence(
            " ".join([project.name, project.description]), project.evidence_ids
        )
        if evidence_ids is None:
            counts["projects_pruned"] += 1
        else:
            technologies: list[str] = []
            for technology in project.technologies:
                trial_ids = select_evidence(technology, evidence_ids)
                if trial_ids is not None:
                    technologies.append(technology)
                    evidence_ids = list(dict.fromkeys([*evidence_ids, *trial_ids]))
            optional_updates: dict[str, str | None] = {}
            for field_name in (
                "role", "start_date", "end_date", "repository_url", "demo_url"
            ):
                value = getattr(project, field_name)
                trial_ids = select_evidence(value, evidence_ids) if value else None
                if value is not None and trial_ids is not None:
                    optional_updates[field_name] = value
                    evidence_ids = list(dict.fromkeys([*evidence_ids, *trial_ids]))
                else:
                    optional_updates[field_name] = None
            employer_id = project.employer_profile_experience_id
            if employer_id not in {role.profile_experience_id for role in experience}:
                employer_id = None
            achievements = []
            for achievement in project.achievements:
                achievement_ids = select_evidence(
                    achievement.statement, achievement.evidence_ids
                )
                if achievement_ids is not None:
                    achievements.append(
                        achievement.model_copy(update={"evidence_ids": achievement_ids})
                    )
            projects.append(
                project.model_copy(
                    update={
                        "evidence_ids": evidence_ids,
                        "technologies": technologies,
                        "achievements": achievements,
                        "employer_profile_experience_id": employer_id,
                        **optional_updates,
                    }
                )
            )

    certifications = []
    for certification in profile.certifications:
        evidence_ids = select_evidence(certification.name, certification.evidence_ids)
        if evidence_ids is None:
            counts["certifications_pruned"] += 1
        else:
            updates: dict[str, str | None] = {}
            for field_name in (
                "issuer", "date", "earned_date", "expiry_date", "credential_id",
                "verification_url",
            ):
                value = getattr(certification, field_name)
                trial_ids = select_evidence(value, evidence_ids) if value else None
                if value is not None and trial_ids is not None:
                    updates[field_name] = value
                    evidence_ids = list(dict.fromkeys([*evidence_ids, *trial_ids]))
                else:
                    updates[field_name] = None
            status = certification.status
            if status != "unknown":
                status_ids = select_evidence(status.replace("_", " "), evidence_ids)
                if status_ids is None:
                    status = "unknown"
                else:
                    evidence_ids = list(dict.fromkeys([*evidence_ids, *status_ids]))
            credential_type = certification.credential_type
            if credential_type != "unknown":
                credential_type_ids = select_evidence(
                    credential_type.replace("_", " "), evidence_ids
                )
                if credential_type_ids is None:
                    credential_type = "unknown"
                else:
                    evidence_ids = list(
                        dict.fromkeys([*evidence_ids, *credential_type_ids])
                    )
            certifications.append(
                certification.model_copy(
                    update={
                        "evidence_ids": evidence_ids,
                        "status": status,
                        "credential_type": credential_type,
                        **updates,
                    }
                )
            )

    repaired = profile.model_copy(
        update={
            "contact": contact,
            "contact_evidence_ids": contact_ids,
            "evidence_catalog": [
                item for evidence_id, item in evidence_by_id.items() if evidence_id in used_ids
            ],
            "summary_facts": summary_facts,
            "experience": experience,
            "skills": skills,
            "education": education,
            "projects": projects,
            "certifications": certifications,
        }
    )
    return repaired, counts


class ProfileValidator:
    """Fail closed when the master profile cannot be traced to exact source quotes."""

    def validate(self, profile: MasterProfile, bundle: IngestionBundle) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        source_text = {
            document.document_id: _normalized(document.text)
            for document in bundle.candidate_documents
        }
        evidence_by_id = {item.evidence_id: item for item in profile.evidence_catalog}
        if len(evidence_by_id) != len(profile.evidence_catalog):
            issues.append(_issue("duplicate_evidence_id", "Evidence IDs must be unique.", "master_profile"))

        for index, evidence in enumerate(profile.evidence_catalog):
            location = f"master_profile.evidence_catalog[{index}]"
            document = source_text.get(evidence.document_id)
            if document is None:
                issues.append(
                    _issue(
                        "unknown_document",
                        f"Evidence references unknown document {evidence.document_id!r}.",
                        location,
                    )
                )
            elif _normalized(evidence.quote) not in document:
                issues.append(
                    _issue(
                        "inexact_evidence_quote",
                        "Evidence quote is not an exact normalized substring of its source.",
                        location,
                    )
                )

        def check_claim(claim: str, evidence_ids: list[str], location: str) -> None:
            missing = sorted(set(evidence_ids) - set(evidence_by_id))
            if missing:
                issues.append(
                    _issue(
                        "unknown_evidence",
                        f"Claim references missing evidence IDs: {', '.join(missing)}.",
                        location,
                    )
                )
                return
            evidence_text = " ".join(evidence_by_id[item].quote for item in evidence_ids)
            unsupported_numbers = _numbers(claim) - _numbers(evidence_text)
            if unsupported_numbers:
                issues.append(
                    _issue(
                        "unsupported_metric",
                        f"Numbers absent from cited source: {', '.join(sorted(unsupported_numbers))}.",
                        location,
                    )
                )
            if _overlap_ratio(claim, evidence_text) < 0.25:
                issues.append(
                    _issue(
                        "weak_source_overlap",
                        "Claim has insufficient lexical support in its exact evidence quotes.",
                        location,
                    )
                )

        contact_claim = " ".join(
            value
            for value in [
                profile.contact.full_name,
                profile.contact.email,
                profile.contact.phone,
                profile.contact.location,
                *(link.url for link in profile.contact.links),
            ]
            if value
        )
        check_claim(contact_claim, profile.contact_evidence_ids, "master_profile.contact")

        for index, fact in enumerate(profile.summary_facts):
            check_claim(fact.statement, fact.evidence_ids, f"master_profile.summary_facts[{index}]")

        role_keys: set[tuple[str, str, str | None, str | None]] = set()
        achievement_ids: set[str] = set()
        for role_index, role in enumerate(profile.experience):
            role_location = f"master_profile.experience[{role_index}]"
            key = (
                role.company.casefold(),
                role.title.casefold(),
                role.start_date,
                role.end_date,
            )
            if key in role_keys:
                issues.append(
                    _issue(
                        "duplicate_role",
                        "Duplicate company/title/timeline remains after aggregation.",
                        role_location,
                    )
                )
            role_keys.add(key)
            check_claim(
                " ".join(
                    value
                    for value in [role.company, role.title, role.location, role.start_date, role.end_date]
                    if value
                ),
                role.evidence_ids,
                role_location,
            )
            for bullet_index, achievement in enumerate(role.achievements):
                if achievement.achievement_id in achievement_ids:
                    issues.append(
                        _issue(
                            "duplicate_achievement_id",
                            f"Duplicate achievement ID {achievement.achievement_id!r}.",
                            f"{role_location}.achievements[{bullet_index}]",
                        )
                    )
                achievement_ids.add(achievement.achievement_id)
                check_claim(
                    achievement.statement,
                    achievement.evidence_ids,
                    f"{role_location}.achievements[{bullet_index}]",
                )

        for index, skill in enumerate(profile.skills):
            check_claim(skill.name, skill.evidence_ids, f"master_profile.skills[{index}]")
        for index, education in enumerate(profile.education):
            check_claim(
                " ".join(
                    value
                    for value in [
                        education.institution,
                        education.degree,
                        education.field_of_study,
                        education.graduation_date,
                        education.location,
                        education.gpa,
                        *education.honors,
                        *education.coursework,
                        education.thesis,
                    ]
                    if value
                ),
                education.evidence_ids,
                f"master_profile.education[{index}]",
            )
            if education.completion_status != "unknown":
                check_claim(
                    education.completion_status.replace("_", " "),
                    education.evidence_ids,
                    f"master_profile.education[{index}].completion_status",
                )
        profile_role_ids = {
            role.profile_experience_id for role in profile.experience
        }
        for index, project in enumerate(profile.projects):
            check_claim(
                " ".join(
                    value
                    for value in [
                        project.name,
                        project.description,
                        *project.technologies,
                        project.role,
                        project.start_date,
                        project.end_date,
                        project.repository_url,
                        project.demo_url,
                    ]
                    if value
                ),
                project.evidence_ids,
                f"master_profile.projects[{index}]",
            )
            if (
                project.employer_profile_experience_id is not None
                and project.employer_profile_experience_id not in profile_role_ids
            ):
                issues.append(
                    _issue(
                        "unknown_project_employer",
                        "A work project references an unknown employment record.",
                        f"master_profile.projects[{index}].employer_profile_experience_id",
                    )
                )
            for achievement_index, achievement in enumerate(project.achievements):
                if achievement.achievement_id in achievement_ids:
                    issues.append(
                        _issue(
                            "duplicate_achievement_id",
                            f"Duplicate achievement ID {achievement.achievement_id!r}.",
                            f"master_profile.projects[{index}].achievements[{achievement_index}]",
                        )
                    )
                achievement_ids.add(achievement.achievement_id)
                check_claim(
                    achievement.statement,
                    achievement.evidence_ids,
                    f"master_profile.projects[{index}].achievements[{achievement_index}]",
                )
        for index, certification in enumerate(profile.certifications):
            check_claim(
                " ".join(
                    value
                    for value in [
                        certification.name,
                        certification.issuer,
                        certification.date,
                        certification.earned_date,
                        certification.expiry_date,
                        certification.credential_id,
                        certification.verification_url,
                    ]
                    if value
                ),
                certification.evidence_ids,
                f"master_profile.certifications[{index}]",
            )
            for field_name in ("status", "credential_type"):
                value = getattr(certification, field_name)
                if value != "unknown":
                    check_claim(
                        value.replace("_", " "),
                        certification.evidence_ids,
                        f"master_profile.certifications[{index}].{field_name}",
                    )
        return issues


class PackageValidator:
    """Validate generated content before any PDF is accepted as final."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    def validate(
        self,
        package: DocumentPackage,
        profile: MasterProfile,
        requirements: JobRequirements,
        template_issues: list[ValidationIssue] | None = None,
        groundedness_audit: GroundednessAudit | None = None,
        expected_letter_date: str | None = None,
    ) -> ValidationResult:
        issues = list(template_issues or [])
        resume = package.resume
        cover = package.cover_letter

        if resume.contact != profile.contact or cover.candidate != profile.contact:
            issues.append(
                _issue(
                    "contact_drift",
                    "Rendered contact details must exactly match the master profile.",
                    "document_package",
                )
            )
        if resume.target_title.casefold() != requirements.target_title.casefold():
            issues.append(
                _issue(
                    "target_title_mismatch",
                    "Resume target title must match the extracted JD title.",
                    "resume.target_title",
                )
            )
        expected_company = requirements.company_name or "Hiring Organization"
        if cover.recipient_company.casefold() != expected_company.casefold():
            issues.append(
                _issue(
                    "company_mismatch",
                    "Cover-letter company must match the company extracted from the JD.",
                    "cover_letter.recipient_company",
                )
            )
        if cover.salutation != "Dear Hiring Team,":
            issues.append(
                _issue(
                    "salutation_drift",
                    'Cover-letter salutation must be exactly "Dear Hiring Team,".',
                    "cover_letter.salutation",
                )
            )
        if cover.signatory != profile.contact.full_name:
            issues.append(
                _issue(
                    "signatory_drift",
                    "Cover-letter signatory must exactly match the candidate name.",
                    "cover_letter.signatory",
                )
            )
        if expected_letter_date is not None and cover.letter_date != expected_letter_date:
            issues.append(
                _issue(
                    "letter_date_drift",
                    "Cover-letter date must exactly match the supplied date.",
                    "cover_letter.letter_date",
                )
            )

        cover_text = " ".join(paragraph.text for paragraph in cover.paragraphs)
        if _normalized(requirements.target_title) not in _normalized(cover_text):
            issues.append(
                _issue(
                    "cover_target_title_missing",
                    "Cover letter must explicitly name the target role.",
                    "cover_letter.paragraphs",
                )
            )
        if _normalized(expected_company) not in _normalized(cover_text):
            issues.append(
                _issue(
                    "cover_company_missing",
                    "Cover letter must explicitly name the target company.",
                    "cover_letter.paragraphs",
                )
            )

        profile_skills = {item.name.casefold() for item in profile.skills}
        if len({skill.casefold() for skill in resume.skills}) != len(resume.skills):
            issues.append(
                _issue(
                    "duplicate_skill",
                    "Resume skill list contains case-insensitive duplicates.",
                    "resume.skills",
                )
            )
        unknown_skills = sorted(skill for skill in resume.skills if skill.casefold() not in profile_skills)
        if unknown_skills:
            issues.append(
                _issue(
                    "invented_skill",
                    f"Skills are not present in the master profile: {', '.join(unknown_skills)}.",
                    "resume.skills",
                )
            )
        grouped_skills = [
            skill for group in resume.skill_groups for skill in group.skills
        ]
        if {item.casefold() for item in grouped_skills} != {
            item.casefold() for item in resume.skills
        }:
            issues.append(
                _issue(
                    "skill_group_mismatch",
                    "Categorized skills must contain exactly the selected grounded skills.",
                    "resume.skill_groups",
                )
            )
        expected_strategy = build_resume_strategy(profile, requirements)
        if resume.strategy is not None and resume.strategy != expected_strategy:
            issues.append(
                _issue(
                    "strategy_drift",
                    "Resume strategy must match the deterministic career-stage plan.",
                    "resume.strategy",
                )
            )

        experience_by_id = {item.profile_experience_id: item for item in profile.experience}
        achievement_text: dict[str, str] = {}
        role_achievement_ids: dict[str, set[str]] = {}
        for role in profile.experience:
            role_achievement_ids[role.profile_experience_id] = {
                item.achievement_id for item in role.achievements
            }
            achievement_text.update(
                {item.achievement_id: item.statement for item in role.achievements}
            )
        for fact in profile.summary_facts:
            achievement_text[fact.fact_id] = fact.statement
        for project in profile.projects:
            achievement_text[project.profile_project_id] = project.description
            achievement_text.update(
                {
                    achievement.achievement_id: achievement.statement
                    for achievement in project.achievements
                }
            )
            if (
                project.project_type == "work"
                and project.employer_profile_experience_id in role_achievement_ids
            ):
                role_achievement_ids[project.employer_profile_experience_id].add(
                    project.profile_project_id
                )
                role_achievement_ids[project.employer_profile_experience_id].update(
                    achievement.achievement_id for achievement in project.achievements
                )

        local_grounding_checks: list[bool] = []
        required_audit_locations: list[str] = []

        def check_grounded_text(text: str, ids: list[str], location: str) -> None:
            missing = sorted(set(ids) - set(achievement_text))
            if missing:
                local_grounding_checks.append(False)
                issues.append(
                    _issue(
                        "unknown_achievement",
                        f"Claim references unknown profile IDs: {', '.join(missing)}.",
                        location,
                    )
                )
                return
            evidence = _referenced_text(ids, achievement_text)
            supported_numbers = not (_numbers(text) - _numbers(evidence))
            lexical_support = _overlap_ratio(text, evidence) >= 0.25
            local_grounding_checks.append(supported_numbers and lexical_support)
            if not supported_numbers:
                issues.append(
                    _issue(
                        "unsupported_metric",
                        "Claim contains a number absent from its cited profile achievements.",
                        location,
                    )
                )
            if not lexical_support:
                issues.append(
                    _issue(
                        "weak_claim_grounding",
                        "Claim has insufficient overlap with cited profile achievements.",
                        location,
                    )
                )

        if resume.professional_summary is not None:
            check_grounded_text(
                resume.professional_summary.text,
                resume.professional_summary.achievement_ids,
                "resume.professional_summary",
            )
            required_audit_locations.append("resume.professional_summary")
        bullets = []
        for role_index, role in enumerate(resume.experience):
            location = f"resume.experience[{role_index}]"
            source_role = experience_by_id.get(role.profile_experience_id)
            if source_role is None:
                issues.append(
                    _issue("unknown_role", "Resume role is absent from the master profile.", location)
                )
                continue
            identity = (
                role.company,
                role.title,
                role.location,
                role.start_date,
                role.end_date,
            )
            source_identity = (
                source_role.company,
                source_role.title,
                source_role.location,
                source_role.start_date,
                source_role.end_date,
            )
            if identity != source_identity:
                issues.append(
                    _issue(
                        "role_identity_drift",
                        "Company, title, location, and dates must exactly match the master profile.",
                        location,
                    )
                )
            allowed_ids = role_achievement_ids[role.profile_experience_id]
            for bullet_index, bullet in enumerate(role.bullets):
                bullet_location = f"{location}.bullets[{bullet_index}]"
                required_audit_locations.append(bullet_location)
                bullets.append(bullet)
                disallowed = sorted(set(bullet.achievement_ids) - allowed_ids)
                if disallowed:
                    issues.append(
                        _issue(
                            "cross_role_evidence",
                            f"Bullet cites achievements outside its role: {', '.join(disallowed)}.",
                            bullet_location,
                        )
                    )
                check_grounded_text(bullet.text, bullet.achievement_ids, bullet_location)
                first_word = next(iter(_WORD_RE.findall(bullet.text)), "").casefold()
                if first_word not in _ACTION_VERBS:
                    issues.append(
                        _issue(
                            "weak_bullet_opening",
                            f"Bullet begins with {first_word or 'no word'!r}; use a strong "
                            "action verb in the tense appropriate to the work.",
                            bullet_location,
                            severity="warning",
                        )
                    )
                for component_name, component in (
                    ("action", bullet.action),
                    ("object", bullet.object),
                    ("method", bullet.method),
                    ("scope", bullet.scope),
                    ("outcome", bullet.outcome),
                    ("attribution", bullet.attribution),
                ):
                    if component and _overlap_ratio(component, bullet.text) < 0.35:
                        issues.append(
                            _issue(
                                "achievement_component_missing",
                                f"Bullet text does not clearly express its {component_name} component.",
                                bullet_location,
                                severity="warning",
                            )
                        )
                for metric in bullet.metrics:
                    if _normalized(metric) not in _normalized(bullet.text):
                        issues.append(
                            _issue(
                                "metric_component_missing",
                                "A structured metric must appear in the rendered bullet text.",
                                bullet_location,
                                severity="warning",
                            )
                        )

        education_by_id = {item.profile_education_id: item for item in profile.education}
        for index, education in enumerate(resume.education):
            source = education_by_id.get(education.profile_education_id)
            if source is None or (
                education.institution,
                education.degree,
                education.field_of_study,
                education.graduation_date,
                education.location,
                education.gpa,
                education.honors,
                education.coursework,
                education.thesis,
                education.completion_status,
            ) != (
                source.institution,
                source.degree,
                source.field_of_study,
                source.graduation_date,
                source.location,
                source.gpa,
                source.honors,
                source.coursework,
                source.thesis,
                source.completion_status,
            ):
                issues.append(
                    _issue(
                        "education_drift",
                        "Education must be copied exactly from a master-profile record.",
                        f"resume.education[{index}]",
                    )
                )

        project_by_id = {item.profile_project_id: item for item in profile.projects}
        for index, project in enumerate(resume.projects):
            source = project_by_id.get(project.profile_project_id)
            project_location = f"resume.projects[{index}]"
            if source is None or (
                project.name,
                project.technologies,
                project.role,
                project.project_type,
                project.start_date,
                project.end_date,
                project.repository_url,
                project.demo_url,
                project.employer_profile_experience_id,
            ) != (
                source.name,
                source.technologies,
                source.role,
                source.project_type,
                source.start_date,
                source.end_date,
                source.repository_url,
                source.demo_url,
                source.employer_profile_experience_id,
            ):
                issues.append(
                    _issue(
                        "project_drift",
                        "Project identity must match the master profile.",
                        project_location,
                    )
                )
            check_grounded_text(
                project.description.text,
                project.description.achievement_ids,
                f"{project_location}.description",
            )
            required_audit_locations.append(f"{project_location}.description")
            allowed_project_ids = (
                {
                    source.profile_project_id,
                    *(item.achievement_id for item in source.achievements),
                }
                if source is not None
                else set()
            )
            for bullet_index, bullet in enumerate(project.bullets):
                bullet_location = f"{project_location}.bullets[{bullet_index}]"
                if set(bullet.achievement_ids) - allowed_project_ids:
                    issues.append(
                        _issue(
                            "cross_project_evidence",
                            "Project bullet cites evidence from another profile record.",
                            bullet_location,
                        )
                    )
                check_grounded_text(
                    bullet.text, bullet.achievement_ids, bullet_location
                )
                required_audit_locations.append(bullet_location)

        certification_by_id = {
            item.profile_certification_id: item for item in profile.certifications
        }
        for index, certification in enumerate(resume.certifications):
            source = certification_by_id.get(certification.profile_certification_id)
            if source is None or (
                certification.name,
                certification.issuer,
                certification.date,
                certification.earned_date,
                certification.expiry_date,
                certification.status,
                certification.credential_id,
                certification.verification_url,
                certification.credential_type,
            ) != (
                source.name,
                source.issuer,
                source.date,
                source.earned_date,
                source.expiry_date,
                source.status,
                source.credential_id,
                source.verification_url,
                source.credential_type,
            ):
                issues.append(
                    _issue(
                        "certification_drift",
                        "Certification must match the master profile exactly.",
                        f"resume.certifications[{index}]",
                    )
                )

        requirement_text = " ".join(
            [
                requirements.target_title,
                expected_company,
                *requirements.competencies,
                *requirements.hard_skills,
                *requirements.tools,
                *requirements.methodologies,
                *requirements.responsibilities,
                *requirements.pain_points,
                *requirements.keywords,
            ]
        )
        for index, paragraph in enumerate(cover.paragraphs):
            for matched_requirement in paragraph.matched_requirements:
                if _overlap_ratio(matched_requirement, requirement_text) < 0.50:
                    issues.append(
                        _issue(
                            "unknown_matched_requirement",
                            "Cover paragraph maps to a requirement not supported by the JD extraction.",
                            f"cover_letter.paragraphs[{index}].matched_requirements",
                        )
                    )
            for claim_index, claim in enumerate(paragraph.candidate_claims):
                claim_location = (
                    f"cover_letter.paragraphs[{index}].candidate_claims[{claim_index}]"
                )
                required_audit_locations.append(claim_location)
                if _normalized(claim.text) not in _normalized(paragraph.text):
                    issues.append(
                        _issue(
                            "unmapped_cover_claim",
                            "Declared candidate claim must appear verbatim in the paragraph.",
                            claim_location,
                        )
                    )
                check_grounded_text(
                    claim.text,
                    claim.achievement_ids,
                    claim_location,
                )
            claim_numbers = set().union(
                *(_numbers(claim.text) for claim in paragraph.candidate_claims)
            ) if paragraph.candidate_claims else set()
            # Numbers that belong to the employer/JD (for example, "Fortune 500")
            # or the supplied letter date are not candidate claims and do not need a
            # profile achievement citation.
            contextual_numbers = _numbers(requirement_text) | _numbers(cover.letter_date)
            uncited_numbers = _numbers(paragraph.text) - claim_numbers - contextual_numbers
            if uncited_numbers:
                local_grounding_checks.append(False)
                issues.append(
                    _issue(
                        "uncited_cover_metric",
                        "Numeric cover-letter text must be captured in a grounded candidate claim.",
                        f"cover_letter.paragraphs[{index}]",
                    )
                )

        resume_text = resume.model_dump_json(
            exclude={
                "strategy": True,
                "professional_summary": {"achievement_ids"},
            }
        )
        candidate_text = " ".join(
            [
                *(fact.statement for fact in profile.summary_facts),
                *(role.title for role in profile.experience),
                *(
                    achievement.statement
                    for role in profile.experience
                    for achievement in role.achievements
                ),
                *(skill.name for skill in profile.skills),
                *(
                    " ".join(
                        value
                        for value in [
                            education.institution,
                            education.degree,
                            education.field_of_study,
                        ]
                        if value
                    )
                    for education in profile.education
                ),
                *(
                    " ".join([project.name, project.description, *project.technologies])
                    for project in profile.projects
                ),
                *(
                    " ".join(
                        value
                        for value in [certification.name, certification.issuer]
                        if value
                    )
                    for certification in profile.certifications
                ),
            ]
        )
        normalized_candidate = _normalized(candidate_text)
        requested_keywords = [
            *requirements.keywords,
            *requirements.hard_skills,
            *requirements.tools,
            *requirements.methodologies,
        ]
        # ATS alignment must never force the generator to claim a JD keyword that the
        # candidate sources do not support. The advertised title is safe as a target;
        # every other scored keyword must already occur in the grounded master profile.
        supported_keywords = [
            keyword
            for keyword in requested_keywords
            if _normalized(keyword) in normalized_candidate
        ]
        keyword_pool = []
        seen_keywords: set[str] = set()
        for keyword in [requirements.target_title, *supported_keywords]:
            normalized_keyword = _normalized(keyword)
            if normalized_keyword and normalized_keyword not in seen_keywords:
                keyword_pool.append(keyword.strip())
                seen_keywords.add(normalized_keyword)
        normalized_resume = _normalized(resume_text)
        matched_keywords = [
            keyword for keyword in keyword_pool if _normalized(keyword) in normalized_resume
        ]
        missing_keywords = [keyword for keyword in keyword_pool if keyword not in matched_keywords]
        keyword_coverage = len(matched_keywords) / len(keyword_pool) if keyword_pool else 1.0
        if keyword_coverage < self.settings.keyword_coverage_threshold:
            issues.append(
                _issue(
                    "keyword_coverage",
                    f"Keyword coverage {keyword_coverage:.1%} is below "
                    f"{self.settings.keyword_coverage_threshold:.1%}; missing: "
                    f"{', '.join(missing_keywords[:12])}.",
                    "resume",
                    severity="warning",
                )
            )

        metric_bullets = sum(bool(_numbers(bullet.text)) for bullet in bullets)
        metric_density = metric_bullets / len(bullets) if bullets else 0.0
        # This is descriptive telemetry only. There is no evidence-backed universal
        # percentage of bullets that must contain numbers, so it never drives repair.

        # The model-based audit is a useful second opinion, but deterministic profile
        # IDs, numeric checks, lexical support, and cross-role checks remain the hard
        # provenance boundary. A flaky or overly conservative external auditor must
        # not veto a package that passes those local checks.
        if groundedness_audit is not None:
            reviewed_locations = {
                finding.location for finding in groundedness_audit.findings
            }
            missing_reviews = sorted(
                set(required_audit_locations) - reviewed_locations
            )
            unexpected_reviews = sorted(
                reviewed_locations - set(required_audit_locations)
            )
            if missing_reviews or unexpected_reviews:
                details = []
                if missing_reviews:
                    details.append(f"missing: {', '.join(missing_reviews)}")
                if unexpected_reviews:
                    details.append(f"unexpected: {', '.join(unexpected_reviews)}")
                issues.append(
                    _issue(
                        "incomplete_groundedness_audit",
                        "Audit locations do not match the claim manifest ("
                        + "; ".join(details)
                        + ").",
                        "groundedness_audit",
                        severity="warning",
                    )
                )
            for finding in groundedness_audit.findings:
                if not finding.supported:
                    issues.append(
                        _issue(
                            "llm_groundedness",
                            finding.explanation,
                            finding.location,
                            severity="warning",
                        )
                    )

        all_grounding_checks = local_grounding_checks
        groundedness_ratio = (
            sum(all_grounding_checks) / len(all_grounding_checks)
            if all_grounding_checks
            else 1.0
        )
        if groundedness_ratio < self.settings.groundedness_threshold:
            issues.append(
                _issue(
                    "groundedness_ratio",
                    f"Groundedness {groundedness_ratio:.1%} is below the required "
                    f"{self.settings.groundedness_threshold:.1%}.",
                    "document_package",
                )
            )

        ats_integrity_passed = not any(
            issue.severity == "error"
            and issue.code
            in {
                "template_integrity",
                "missing_standard_section",
                "unsafe_markup",
            }
            for issue in issues
        )
        passed = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(
            passed=passed,
            keyword_coverage_ratio=keyword_coverage,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            metric_density=metric_density,
            groundedness_ratio=groundedness_ratio,
            ats_integrity_passed=ats_integrity_passed,
            issues=issues,
        )


def validation_feedback(result: ValidationResult) -> str:
    """Produce concise, targeted repair instructions for the next model call."""

    lines = [
        "Revise only unsupported or deficient content. Never add an uncited fact or number."
    ]
    for issue in result.issues:
        if issue.severity == "error" and issue.retryable:
            lines.append(f"- [{issue.code}] {issue.location}: {issue.message}")
    return "\n".join(lines)
