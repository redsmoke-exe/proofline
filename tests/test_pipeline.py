from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from cv_agent.ingestion import ingest_request
from cv_agent.pipeline import ResumePipeline, _clear_memory_caches
from cv_agent.renderer import PDFRenderer
from cv_agent.schemas import (
    DocumentPackage,
    GenerateRequest,
    GroundednessAudit,
    GroundednessFinding,
    IngestionRequest,
    ModelSettings,
    PipelineSnapshot,
)
from cv_agent.validation import PackageValidator, ProfileValidator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_pipeline_caches() -> None:
    _clear_memory_caches()
    yield
    _clear_memory_caches()


class FakeAgents:
    def __init__(self, snapshot: PipelineSnapshot) -> None:
        self.snapshot = snapshot
        self.llm = SimpleNamespace(settings=snapshot.model_settings)
        self.aggregate_calls = 0
        self.requirement_calls = 0
        self.generate_calls = 0
        self.audit_calls = 0

    def aggregate_profile(self, *_: object, **__: object):
        self.aggregate_calls += 1
        return self.snapshot.master_profile

    def extract_job_requirements(self, *_: object, **__: object):
        self.requirement_calls += 1
        return self.snapshot.job_requirements

    def generate_documents(self, *_: object, **__: object) -> DocumentPackage:
        self.generate_calls += 1
        if self.generate_calls > 1:
            return self.snapshot.document_package
        payload = deepcopy(self.snapshot.document_package.model_dump())
        payload["resume"]["experience"][0]["bullets"][0]["text"] = (
            "Reduced data pipeline latency by 73% by redesigning Apache Spark partitioning "
            "and Airflow orchestration."
        )
        payload["resume"]["experience"][0]["bullets"][0]["measurement"] = "73%"
        return DocumentPackage.model_validate(payload)

    def audit_groundedness(self, *_: object, **__: object) -> GroundednessAudit:
        self.audit_calls += 1
        package = self.snapshot.document_package
        locations = ["resume.professional_summary"]
        for role_index, role in enumerate(package.resume.experience):
            locations.extend(
                f"resume.experience[{role_index}].bullets[{bullet_index}]"
                for bullet_index, _ in enumerate(role.bullets)
            )
        for paragraph_index, paragraph in enumerate(package.cover_letter.paragraphs):
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
                    explanation="supported",
                    supporting_achievement_ids=["a1"],
                )
                for location in locations
            ],
        )


def test_pipeline_repairs_failed_quality_gate(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    settings = ModelSettings(
        model="test-model",
        temperature=0.2,
        max_correction_retries=2,
        keyword_coverage_threshold=0.65,
        metric_density_threshold=0.60,
        groundedness_threshold=1.0,
    )
    agents = FakeAgents(snapshot)
    pipeline = ResumePipeline(
        agents=agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(settings),
        renderer=PDFRenderer(),
        persistent_cache_dir=tmp_path / "cache",
    )
    result = pipeline.run(
        GenerateRequest(
            ingestion=IngestionRequest(
                jd_path=ROOT / "examples" / "job_description.md",
                jd_text=None,
                cv_paths=[
                    ROOT / "examples" / "candidate_primary.md",
                    ROOT / "examples" / "candidate_history.txt",
                ],
                max_chars_per_document=120_000,
            ),
            output_dir=tmp_path,
            letter_date=date(2026, 9, 12),
        )
    )
    assert result.validation.passed
    assert result.correction_attempts == 1
    assert agents.generate_calls == 1
    assert agents.audit_calls == 0
    assert (tmp_path / "resume.pdf").is_file()
    assert (tmp_path / "cover_letter.pdf").is_file()
    assert (tmp_path / "pipeline_snapshot.json").stat().st_mode & 0o777 == 0o600


def test_pipeline_reuses_validated_profile_and_jd_analysis(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    settings = snapshot.model_settings
    agents = FakeAgents(snapshot)
    pipeline = ResumePipeline(
        agents=agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(settings),
        renderer=PDFRenderer(),
        persistent_cache_dir=tmp_path / "cache",
    )
    ingestion = IngestionRequest(
        jd_path=ROOT / "examples" / "job_description.md",
        jd_text=None,
        cv_paths=[
            ROOT / "examples" / "candidate_primary.md",
            ROOT / "examples" / "candidate_history.txt",
        ],
        max_chars_per_document=120_000,
    )

    pipeline.run(
        GenerateRequest(
            ingestion=ingestion,
            output_dir=tmp_path / "first",
            letter_date=date(2026, 9, 12),
        )
    )
    pipeline.run(
        GenerateRequest(
            ingestion=ingestion,
            output_dir=tmp_path / "second",
            letter_date=date(2026, 9, 12),
        )
    )

    assert agents.aggregate_calls == 1
    assert agents.requirement_calls == 1


def test_persistent_profile_and_jd_cache_survive_pipeline_restart(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    ingestion = IngestionRequest(
        jd_path=ROOT / "examples" / "job_description.md",
        jd_text=None,
        cv_paths=[
            ROOT / "examples" / "candidate_primary.md",
            ROOT / "examples" / "candidate_history.txt",
        ],
        max_chars_per_document=120_000,
    )
    first_agents = FakeAgents(snapshot)
    first_pipeline = ResumePipeline(
        agents=first_agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(snapshot.model_settings),
        renderer=PDFRenderer(),
        persistent_cache_dir=tmp_path / "cache",
    )
    first_pipeline.run(
        GenerateRequest(
            ingestion=ingestion,
            output_dir=tmp_path / "first",
            letter_date=date(2026, 9, 12),
        )
    )

    _clear_memory_caches()
    second_agents = FakeAgents(snapshot)
    second_pipeline = ResumePipeline(
        agents=second_agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(snapshot.model_settings),
        renderer=PDFRenderer(),
        persistent_cache_dir=tmp_path / "cache",
    )
    second_pipeline.run(
        GenerateRequest(
            ingestion=ingestion,
            output_dir=tmp_path / "second",
            letter_date=date(2026, 9, 12),
        )
    )

    assert first_agents.aggregate_calls == 1
    assert first_agents.requirement_calls == 1
    assert second_agents.aggregate_calls == 0
    assert second_agents.requirement_calls == 0
    assert second_agents.generate_calls == 1
    cache_files = list((tmp_path / "cache").rglob("*.json"))
    assert len(cache_files) == 2
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in cache_files)


def test_profile_and_job_analysis_start_in_parallel(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    barrier = Barrier(2)

    class ParallelAgents(FakeAgents):
        def aggregate_profile(self, *_: object, **__: object):
            self.aggregate_calls += 1
            barrier.wait(timeout=2)
            return self.snapshot.master_profile

        def extract_job_requirements(self, *_: object, **__: object):
            self.requirement_calls += 1
            barrier.wait(timeout=2)
            return self.snapshot.job_requirements

    agents = ParallelAgents(snapshot)
    pipeline = ResumePipeline(
        agents=agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(snapshot.model_settings),
        renderer=PDFRenderer(),
        persistent_cache_dir=tmp_path / "cache",
    )
    result = pipeline.run(
        GenerateRequest(
            ingestion=IngestionRequest(
                jd_path=ROOT / "examples" / "job_description.md",
                jd_text=None,
                cv_paths=[
                    ROOT / "examples" / "candidate_primary.md",
                    ROOT / "examples" / "candidate_history.txt",
                ],
                max_chars_per_document=120_000,
            ),
            output_dir=tmp_path / "output",
            letter_date=date(2026, 9, 12),
        )
    )
    assert result.validation.passed
    assert agents.aggregate_calls == 1
    assert agents.requirement_calls == 1


def test_external_audit_timeout_is_advisory(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    class FailingAuditAgents(FakeAgents):
        def audit_groundedness(self, *_: object, **__: object) -> GroundednessAudit:
            self.audit_calls += 1
            raise TimeoutError("simulated independent-auditor timeout")

    agents = FailingAuditAgents(snapshot)
    pipeline = ResumePipeline(
        agents=agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(snapshot.model_settings),
        renderer=PDFRenderer(),
        enable_external_audit=True,
        persistent_cache_dir=tmp_path / "cache",
    )

    result = pipeline.run(
        GenerateRequest(
            ingestion=IngestionRequest(
                jd_path=ROOT / "examples" / "job_description.md",
                jd_text=None,
                cv_paths=[
                    ROOT / "examples" / "candidate_primary.md",
                    ROOT / "examples" / "candidate_history.txt",
                ],
                max_chars_per_document=120_000,
            ),
            output_dir=tmp_path,
            letter_date=date(2026, 9, 12),
        )
    )

    assert agents.audit_calls == 1
    assert result.validation.passed
    assert result.groundedness_audit.passed
    assert (tmp_path / "resume.pdf").is_file()
    assert (tmp_path / "cover_letter.pdf").is_file()


def test_pipeline_repairs_job_description_source_identifier(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    class WrongSourceAgents(FakeAgents):
        def extract_job_requirements(self, *_: object, **__: object):
            self.requirement_calls += 1
            return self.snapshot.job_requirements.model_copy(
                update={"source_document_id": "wrong-source-id"}
            )

    agents = WrongSourceAgents(snapshot)
    pipeline = ResumePipeline(
        agents=agents,  # type: ignore[arg-type]
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(snapshot.model_settings),
        renderer=PDFRenderer(),
        persistent_cache_dir=tmp_path / "cache",
    )
    result = pipeline.run(
        GenerateRequest(
            ingestion=IngestionRequest(
                jd_path=ROOT / "examples" / "job_description.md",
                jd_text=None,
                cv_paths=[
                    ROOT / "examples" / "candidate_primary.md",
                    ROOT / "examples" / "candidate_history.txt",
                ],
                max_chars_per_document=120_000,
            ),
            output_dir=tmp_path,
            letter_date=date(2026, 9, 12),
        )
    )

    bundle = ingest_request(
        IngestionRequest(
            jd_path=ROOT / "examples" / "job_description.md",
            jd_text=None,
            cv_paths=[ROOT / "examples" / "candidate_primary.md"],
            max_chars_per_document=120_000,
        )
    )
    assert result.job_requirements.source_document_id == bundle.job_description.document_id
