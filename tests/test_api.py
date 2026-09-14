from __future__ import annotations

from datetime import date
from pathlib import Path
import re

from fastapi.testclient import TestClient

import cv_agent.api as api_module
from cv_agent.api import app
from cv_agent.agent import ProviderAPIError, ProviderPoolError
from cv_agent.ingestion import ingest_request
from cv_agent.pipeline import QualityGateError
from cv_agent.renderer import PDFRenderer
from cv_agent.schemas import PipelineResult, PipelineSnapshot, ValidationIssue


client = TestClient(app)


def test_health_reports_backend_readiness() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["api_key_configured"], bool)
    assert payload["provider"] in {"openai", "gemini", "openrouter"}
    assert payload["model"]
    assert isinstance(payload["fallback_models"], list)


def test_cors_allows_local_studio() -> None:
    response = client.options(
        "/api/v1/generate",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_artifact_route_rejects_untrusted_run_identifier() -> None:
    response = client.get("/api/v1/runs/not-a-run/resume.pdf")
    assert response.status_code == 404


def test_credit_exhaustion_has_actionable_api_error() -> None:
    class ExhaustedCreditsError(RuntimeError):
        status_code = 429
        body = {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}

    status, message = api_module._public_generation_error(ExhaustedCreditsError())
    assert status == 402
    assert "quota or credits are exhausted" in message


def test_provider_timeout_has_actionable_api_error() -> None:
    class APITimeoutError(RuntimeError):
        pass

    status, message = api_module._public_generation_error(APITimeoutError())
    assert status == 504
    assert "took too long" in message


def test_openrouter_error_envelope_timeout_is_not_reported_as_structured_output() -> None:
    status, message = api_module._public_generation_error(
        ProviderAPIError(
            "OpenRouter",
            {"code": 504, "message": "Upstream idle timeout exceeded"},
        )
    )
    assert status == 504
    assert "upstream inference timed out" in message
    assert "structured" not in message


def test_all_provider_timeouts_have_actionable_error() -> None:
    failure = ProviderPoolError(
        [
            (
                "gemini",
                ProviderAPIError("Gemini", {"code": 504, "message": "timeout"}),
            ),
            (
                "openrouter",
                ProviderAPIError("OpenRouter", {"code": 504, "message": "timeout"}),
            ),
        ]
    )
    status, message = api_module._public_generation_error(failure)
    assert status == 504
    assert message == "All configured LLM providers timed out. Retry the request."


def test_generate_contract_returns_real_pdf_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = PipelineSnapshot.model_validate_json(
        (api_module.EXAMPLES_ROOT / "sample_snapshot.json").read_text(encoding="utf-8")
    )

    class LocalPipeline:
        def run(self, request):
            manifest = PDFRenderer().render(
                snapshot.document_package,
                request.output_dir,
                date(2026, 9, 12),
            )
            return PipelineResult(
                master_profile=snapshot.master_profile,
                job_requirements=snapshot.job_requirements,
                document_package=snapshot.document_package,
                groundedness_audit=snapshot.groundedness_audit,
                validation=snapshot.validation,
                render_manifest=manifest,
                model_settings=snapshot.model_settings,
                correction_attempts=snapshot.correction_attempts,
            )

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(api_module, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(api_module, "_build_pipeline", lambda: LocalPipeline())

    response = client.post(
        "/api/v1/generate",
        data={"use_sample_profile": "true", "letter_date": "2026-09-12"},
        files={"jd_file": ("job.md", b"Data Platform Engineer with Python and SQL", "text/markdown")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["scores"]["evidence_strength"] == 100
    assert payload["documents"]["resume"]["contact"]["full_name"] == "Alex Morgan"
    assert payload["artifacts"]["resume"]["pages"] == 1

    for download_key, kind in (
        ("resume_pdf", "resume"),
        ("cover_letter_pdf", "cover-letter"),
    ):
        pdf = client.get(payload["downloads"][download_key])
        assert pdf.status_code == 200
        assert pdf.headers["content-type"] == "application/pdf"
        disposition = pdf.headers["content-disposition"]
        assert re.fullmatch(
            rf'attachment; filename="{kind}-[a-f0-9]{{8}}-[a-f0-9]{{8}}\.pdf"',
            disposition,
        )
        repeated = client.get(payload["downloads"][download_key])
        assert repeated.headers["content-disposition"] != disposition
        assert pdf.content.startswith(b"%PDF")


def test_hosting_proxy_secret_blocks_direct_api_use(monkeypatch) -> None:
    monkeypatch.setenv("PROXY_AUTH_SECRET", "test-proxy-secret")

    blocked = client.get("/api/v1/runs/00000000000000000000000000000000/resume.pdf")
    allowed = client.get(
        "/api/v1/runs/00000000000000000000000000000000/resume.pdf",
        headers={"X-Proofline-Proxy-Secret": "test-proxy-secret"},
    )

    assert blocked.status_code == 401
    assert allowed.status_code == 404


def test_generate_reports_document_ingestion_as_actionable_422(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class IngestionOnlyPipeline:
        def run(self, request):
            ingest_request(request.ingestion)
            raise AssertionError("corrupt input should fail during ingestion")

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(api_module, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(api_module, "_build_pipeline", lambda: IngestionOnlyPipeline())

    response = client.post(
        "/api/v1/generate",
        data={"use_sample_profile": "true", "letter_date": "2026-09-12"},
        files={"jd_file": ("broken.pdf", b"this is not a PDF", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "A supplied document could not be parsed. Re-export it as PDF or DOCX, "
        "or upload its contents as TXT or Markdown."
    )
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_generate_rejects_oversized_pasted_job_description(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(api_module, "RUNS_ROOT", tmp_path / "runs")

    response = client.post(
        "/api/v1/generate",
        data={
            "use_sample_profile": "true",
            "letter_date": "2026-09-12",
            "jd_text": "x" * (api_module.MAX_DOCUMENT_CHARS + 1),
        },
    )

    assert response.status_code == 422
    assert "120,000-character limit" in response.json()["detail"]
    assert not (tmp_path / "runs").exists()


def test_generate_reports_quality_gate_issues(tmp_path: Path, monkeypatch) -> None:
    class FailingPipeline:
        def run(self, request):
            raise QualityGateError(
                "Document package failed the quality gate after allowed corrections.",
                [
                    ValidationIssue(
                        code="keyword_coverage",
                        severity="error",
                        message="Keyword coverage is below the configured threshold.",
                        location="resume",
                        retryable=True,
                    )
                ],
            )

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(api_module, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(api_module, "_build_pipeline", lambda: FailingPipeline())

    response = client.post(
        "/api/v1/generate",
        data={"use_sample_profile": "true", "letter_date": "2026-09-12"},
        files={"jd_file": ("job.md", b"Python and SQL platform role", "text/markdown")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "Document package failed the quality gate after allowed corrections."
    assert detail["issues"][0]["code"] == "keyword_coverage"
    assert detail["issues"][0]["location"] == "resume"
