"""Extensible state-machine orchestration for the end-to-end agent workflow."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .agent import ResumeAgents
from .document_repair import DocumentRepairError, repair_document_package
from .ingestion import ingest_request
from .logging_config import log_event
from .renderer import PDFRenderer
from .schemas import (
    GenerateRequest,
    GroundednessAudit,
    GroundednessFinding,
    IngestionBundle,
    JobRequirements,
    MasterProfile,
    PipelineResult,
    PipelineSnapshot,
    ValidationIssue,
    ValidationResult,
)
from .strategy import build_resume_strategy
from .validation import PackageValidator, ProfileValidator


LOGGER = logging.getLogger(__name__)
_CACHE_LIMIT = 8
_PERSISTENT_CACHE_VERSION = "grounded-stage-cache-v2"
_DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "output" / "cache"
_PROFILE_CACHE: OrderedDict[tuple[str, ...], MasterProfile] = OrderedDict()
_JD_CACHE: OrderedDict[tuple[str, ...], JobRequirements] = OrderedDict()
_CACHE_LOCK = Lock()
CacheModel = TypeVar("CacheModel", bound=BaseModel)


def _cache_get(cache: OrderedDict, key: tuple[str, ...]) -> Any | None:
    with _CACHE_LOCK:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value


def _cache_put(cache: OrderedDict, key: tuple[str, ...], value: Any) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_LIMIT:
            cache.popitem(last=False)


def _cache_delete(cache: OrderedDict, key: tuple[str, ...]) -> None:
    with _CACHE_LOCK:
        cache.pop(key, None)


def _clear_memory_caches() -> None:
    """Clear process-local caches; intended for deterministic tests and shutdown hooks."""

    with _CACHE_LOCK:
        _PROFILE_CACHE.clear()
        _JD_CACHE.clear()


class PipelineStage(str, Enum):
    INGESTED = "ingested"
    PROFILE_AGGREGATED = "profile_aggregated"
    JD_ANALYZED = "jd_analyzed"
    RESUME_STRATEGIZED = "resume_strategized"
    DOCUMENTS_DRAFTED = "documents_drafted"
    VALIDATED = "validated"
    RENDERED = "rendered"


class PipelineHook(Protocol):
    def __call__(self, stage: PipelineStage, payload: Any) -> None: ...


class QualityGateError(RuntimeError):
    def __init__(self, message: str, issues: list[ValidationIssue]) -> None:
        super().__init__(message)
        self.issues = issues


def _deterministic_groundedness_audit(package: Any) -> GroundednessAudit:
    """Build the audit artifact from claims that passed deterministic provenance."""

    claims: list[tuple[str, list[str]]] = []
    if package.resume.professional_summary is not None:
        claims.append(
            (
                "resume.professional_summary",
                package.resume.professional_summary.achievement_ids,
            )
        )
    for role_index, role in enumerate(package.resume.experience):
        claims.extend(
            (
                f"resume.experience[{role_index}].bullets[{bullet_index}]",
                bullet.achievement_ids,
            )
            for bullet_index, bullet in enumerate(role.bullets)
        )
    claims.extend(
        (
            f"resume.projects[{index}].description",
            project.description.achievement_ids,
        )
        for index, project in enumerate(package.resume.projects)
    )
    for project_index, project in enumerate(package.resume.projects):
        claims.extend(
            (
                f"resume.projects[{project_index}].bullets[{bullet_index}]",
                bullet.achievement_ids,
            )
            for bullet_index, bullet in enumerate(project.bullets)
        )
    for paragraph_index, paragraph in enumerate(package.cover_letter.paragraphs):
        claims.extend(
            (
                f"cover_letter.paragraphs[{paragraph_index}].candidate_claims[{claim_index}]",
                claim.achievement_ids,
            )
            for claim_index, claim in enumerate(paragraph.candidate_claims)
        )
    return GroundednessAudit(
        passed=True,
        findings=[
            GroundednessFinding(
                location=location,
                supported=True,
                explanation="Passed deterministic profile-ID, numeric, and lexical checks.",
                supporting_achievement_ids=list(dict.fromkeys(ids)),
            )
            for location, ids in claims
        ],
    )


class ResumePipeline:
    def __init__(
        self,
        agents: ResumeAgents,
        profile_validator: ProfileValidator,
        package_validator: PackageValidator,
        renderer: PDFRenderer,
        hooks: list[PipelineHook] | None = None,
        enable_external_audit: bool | None = None,
        persistent_cache_dir: Path | None = None,
    ) -> None:
        self.agents = agents
        self.profile_validator = profile_validator
        self.package_validator = package_validator
        self.renderer = renderer
        self.hooks = hooks or []
        if enable_external_audit is None:
            configured = os.getenv("ENABLE_EXTERNAL_GROUNDEDNESS_AUDIT", "false")
            enable_external_audit = configured.strip().casefold() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.enable_external_audit = enable_external_audit
        configured_cache = os.getenv("CV_AGENT_CACHE_DIR")
        self.persistent_cache_dir = (
            persistent_cache_dir
            or (Path(configured_cache).expanduser() if configured_cache else _DEFAULT_CACHE_ROOT)
        ).resolve()

    def _persistent_cache_path(
        self,
        namespace: str,
        key: tuple[str, ...],
    ) -> Path:
        digest = hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()
        return self.persistent_cache_dir / namespace / f"{digest}.json"

    def _persistent_cache_get(
        self,
        namespace: str,
        key: tuple[str, ...],
        model: type[CacheModel],
    ) -> CacheModel | None:
        path = self._persistent_cache_path(namespace, key)
        with _CACHE_LOCK:
            if not path.is_file():
                return None
            try:
                value = model.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                path.unlink(missing_ok=True)
                log_event(LOGGER, "persistent_cache_entry_discarded", namespace=namespace)
                return None
        log_event(LOGGER, "persistent_cache_hit", namespace=namespace)
        return value

    def _persistent_cache_put(
        self,
        namespace: str,
        key: tuple[str, ...],
        value: BaseModel,
    ) -> None:
        path = self._persistent_cache_path(namespace, key)
        with _CACHE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.parent.chmod(0o700)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    prefix=f".{path.stem}-",
                    suffix=".tmp",
                    dir=path.parent,
                    delete=False,
                ) as temporary:
                    temporary.write(value.model_dump_json(indent=2))
                    temporary_path = Path(temporary.name)
                temporary_path.chmod(0o600)
                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        log_event(LOGGER, "persistent_cache_written", namespace=namespace)

    def _persistent_cache_delete(
        self,
        namespace: str,
        key: tuple[str, ...],
    ) -> None:
        path = self._persistent_cache_path(namespace, key)
        with _CACHE_LOCK:
            path.unlink(missing_ok=True)

    def _resolve_profile(
        self,
        bundle: IngestionBundle,
        cache_key: tuple[str, ...],
    ) -> MasterProfile:
        profile = _cache_get(_PROFILE_CACHE, cache_key)
        if profile is None:
            profile = self._persistent_cache_get(
                "profiles",
                cache_key,
                MasterProfile,
            )
            if profile is not None:
                _cache_put(_PROFILE_CACHE, cache_key, profile)
        profile_issues: list[ValidationIssue] = []
        if profile is not None:
            profile_issues = self.profile_validator.validate(profile, bundle)
            log_event(LOGGER, "profile_cache_hit", document_count=len(bundle.candidate_documents))
            if not profile_issues:
                return profile
            _cache_delete(_PROFILE_CACHE, cache_key)
            self._persistent_cache_delete("profiles", cache_key)
            log_event(
                LOGGER,
                "invalid_profile_cache_entry_evicted",
                issue_count=len(profile_issues),
            )

        profile_feedback: str | None = None
        max_retries = min(self.package_validator.settings.max_correction_retries, 1)
        for attempt in range(max_retries + 1):
            profile = self.agents.aggregate_profile(bundle, profile_feedback)
            profile_issues = self.profile_validator.validate(profile, bundle)
            if not profile_issues:
                _cache_put(_PROFILE_CACHE, cache_key, profile)
                self._persistent_cache_put("profiles", cache_key, profile)
                return profile
            if attempt < max_retries:
                profile_feedback = self._issues_as_feedback(profile_issues)
                log_event(
                    LOGGER,
                    "profile_correction_requested",
                    attempt=attempt + 1,
                    issue_count=len(profile_issues),
                )
        raise QualityGateError(
            "Master profile failed provenance checks after the allowed corrections.",
            profile_issues,
        )

    def _resolve_requirements(
        self,
        bundle: IngestionBundle,
        cache_key: tuple[str, ...],
    ) -> JobRequirements:
        requirements = _cache_get(_JD_CACHE, cache_key)
        if requirements is None:
            requirements = self._persistent_cache_get(
                "job_requirements",
                cache_key,
                JobRequirements,
            )
            if requirements is not None:
                _cache_put(_JD_CACHE, cache_key, requirements)
        if requirements is not None:
            log_event(LOGGER, "job_requirements_cache_hit")
        else:
            requirements = self.agents.extract_job_requirements(bundle)
        if requirements.source_document_id != bundle.job_description.document_id:
            requirements = requirements.model_copy(
                update={"source_document_id": bundle.job_description.document_id}
            )
            log_event(LOGGER, "job_requirements_source_id_repaired")
        _cache_put(_JD_CACHE, cache_key, requirements)
        self._persistent_cache_put("job_requirements", cache_key, requirements)
        return requirements

    def _emit(self, stage: PipelineStage, payload: Any, started: float) -> float:
        now = time.perf_counter()
        log_event(
            LOGGER,
            "pipeline_stage_completed",
            stage=stage.value,
            elapsed_ms=round((now - started) * 1000, 2),
        )
        for hook in self.hooks:
            hook(stage, payload)
        return now

    def run(self, request: GenerateRequest) -> PipelineResult:
        stage_started = time.perf_counter()
        bundle = ingest_request(request.ingestion)
        stage_started = self._emit(PipelineStage.INGESTED, bundle, stage_started)

        profile_cache_key = (
            _PERSISTENT_CACHE_VERSION,
            *sorted(document.sha256 for document in bundle.candidate_documents),
        )
        jd_cache_key = (
            _PERSISTENT_CACHE_VERSION,
            bundle.job_description.sha256,
        )

        # Candidate aggregation and JD analysis are independent after ingestion. They
        # are both required, so waiting for them concurrently reduces the critical path
        # without weakening any validation boundary.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline-analysis") as executor:
            profile_future = executor.submit(
                self._resolve_profile,
                bundle,
                profile_cache_key,
            )
            requirements_future = executor.submit(
                self._resolve_requirements,
                bundle,
                jd_cache_key,
            )
            profile = profile_future.result()
            requirements = requirements_future.result()

        stage_started = self._emit(
            PipelineStage.PROFILE_AGGREGATED, profile, stage_started
        )
        stage_started = self._emit(PipelineStage.JD_ANALYZED, requirements, stage_started)

        strategy = build_resume_strategy(profile, requirements)
        stage_started = self._emit(
            PipelineStage.RESUME_STRATEGIZED, strategy, stage_started
        )

        raw_package = self.agents.generate_documents(
            profile,
            requirements,
            request.letter_date,
            strategy=strategy,
        )
        try:
            package, repair_counts = repair_document_package(
                raw_package,
                profile,
                requirements,
                request.letter_date,
                strategy,
            )
        except DocumentRepairError as exc:
            raise QualityGateError(
                "The supplied CV does not contain enough grounded material to build "
                "the required resume sections without inventing content.",
                [
                    ValidationIssue(
                        code="insufficient_grounded_profile",
                        severity="error",
                        message=str(exc),
                        location="master_profile",
                        retryable=False,
                    )
                ],
            ) from exc
        correction_attempts = int(any(repair_counts.values()))
        if correction_attempts:
            log_event(
                LOGGER,
                "document_package_repaired",
                **repair_counts,
            )
        stage_started = self._emit(
            PipelineStage.DOCUMENTS_DRAFTED, package, stage_started
        )

        _, _, template_issues = self.renderer.preflight(package)
        validation = self.package_validator.validate(
            package,
            profile,
            requirements,
            template_issues=template_issues,
            expected_letter_date=request.letter_date.isoformat(),
        )
        if not validation.passed:
            raise QualityGateError(
                "Document package failed deterministic truth or template checks.",
                validation.issues,
            )

        # The optional LLM audit is intentionally advisory and runs only after local
        # checks pass. It can be enabled for paid/reliable models without making a
        # timeout or false negative discard an otherwise verified package.
        if self.enable_external_audit:
            try:
                external_audit = self.agents.audit_groundedness(
                    profile,
                    requirements,
                    package,
                )
                log_event(
                    LOGGER,
                    "external_groundedness_audit_completed",
                    passed=external_audit.passed,
                    finding_count=len(external_audit.findings),
                    unsupported_count=sum(
                        not finding.supported for finding in external_audit.findings
                    ),
                )
            except Exception as exc:
                log_event(
                    LOGGER,
                    "external_groundedness_audit_unavailable",
                    error_type=exc.__class__.__name__,
                    status_code=getattr(exc, "status_code", None),
                )

        audit = _deterministic_groundedness_audit(package)
        validation = self.package_validator.validate(
            package,
            profile,
            requirements,
            template_issues=template_issues,
            groundedness_audit=audit,
            expected_letter_date=request.letter_date.isoformat(),
        )
        stage_started = self._emit(PipelineStage.VALIDATED, validation, stage_started)

        if not validation.passed:
            raise RuntimeError("pipeline reached an invalid terminal state")

        manifest = self.renderer.render(package, request.output_dir, request.letter_date)
        resume_pages = next(
            (
                artifact.page_count
                for artifact in manifest.artifacts
                if artifact.kind == "resume"
            ),
            0,
        )
        if resume_pages > strategy.target_pages:
            validation = validation.model_copy(
                update={
                    "issues": [
                        *validation.issues,
                        ValidationIssue(
                            code="page_budget_exceeded",
                            severity="warning",
                            message=(
                                f"Resume rendered to {resume_pages} pages; the "
                                f"{strategy.career_stage} plan targets {strategy.target_pages}."
                            ),
                            location="rendered_resume",
                            retryable=False,
                        ),
                    ]
                }
            )
        stage_started = self._emit(PipelineStage.RENDERED, manifest, stage_started)
        snapshot = PipelineSnapshot(
            master_profile=profile,
            job_requirements=requirements,
            document_package=package,
            groundedness_audit=audit,
            validation=validation,
            model_settings=self.package_validator.settings,
            correction_attempts=correction_attempts,
        )
        self._write_snapshot(snapshot, request.output_dir / "pipeline_snapshot.json")
        return PipelineResult(
            master_profile=profile,
            job_requirements=requirements,
            document_package=package,
            groundedness_audit=audit,
            validation=validation,
            render_manifest=manifest,
            model_settings=self.package_validator.settings,
            correction_attempts=correction_attempts,
        )

    @staticmethod
    def _issues_as_feedback(issues: list[ValidationIssue]) -> str:
        return "\n".join(
            f"- [{issue.code}] {issue.location}: {issue.message}" for issue in issues
        )

    @staticmethod
    def _write_snapshot(snapshot: PipelineSnapshot, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        temporary.chmod(0o600)
        temporary.replace(path)
        log_event(LOGGER, "pipeline_snapshot_written", path=str(path))
