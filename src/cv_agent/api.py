"""HTTP API exposing the resume pipeline to the React studio."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import uuid
from secrets import compare_digest
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .agent import ResumeAgents, build_reliable_structured_client
from .logging_config import configure_logging, log_event
from .ingestion import IngestionError, normalize_text
from .pipeline import QualityGateError, ResumePipeline
from .renderer import PDFRenderer
from .schemas import GenerateRequest, IngestionRequest, ModelSettings, PipelineResult
from .strategy import build_resume_strategy
from .validation import PackageValidator, ProfileValidator


APPLICATION_ROOT = Path(os.getenv("CV_AGENT_APP_ROOT", Path.cwd())).expanduser().resolve()
RUNS_ROOT = Path(
    os.getenv("CV_AGENT_RUNS_DIR", APPLICATION_ROOT / "output" / "runs")
).expanduser().resolve()
EXAMPLES_ROOT = Path(
    os.getenv("CV_AGENT_EXAMPLES_DIR", APPLICATION_ROOT / "examples")
).expanduser().resolve()
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_CV_FILES = 8
MAX_DOCUMENT_CHARS = 120_000
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
PIPELINE_CONCURRENCY = asyncio.Semaphore(2)

load_dotenv(APPLICATION_ROOT / ".env")
configure_logging(os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)


def _provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"openai", "gemini", "openrouter"}:
        raise HTTPException(status_code=503, detail=f"Unsupported LLM_PROVIDER: {provider}")
    return provider


def _model_for(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("LLM_MODEL") or os.getenv(
            "GEMINI_MODEL", "gemini-3-flash-preview"
        )
    if provider == "openrouter":
        return os.getenv("LLM_MODEL") or os.getenv(
            "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
        )
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.6")


def _fallback_models_for(provider: str) -> list[str]:
    if provider == "openrouter":
        configured = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    elif provider == "gemini":
        configured = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.7-flash,gemini-3.6-flash",
        )
    else:
        return []
    primary = _model_for(provider)
    return list(
        dict.fromkeys(
            model.strip()
            for model in configured.split(",")
            if model.strip() and model.strip() != primary
        )
    )


def _api_key_name(provider: str) -> str:
    if provider == "gemini":
        return "GEMINI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"


def _configured_providers() -> list[str]:
    return [
        provider
        for provider in ("gemini", "openrouter", "openai")
        if os.getenv(_api_key_name(provider))
    ]


def _parallel_providers_enabled() -> bool:
    return os.getenv("LLM_PARALLEL_PROVIDERS", "true").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _execution_provider_ready(provider: str) -> bool:
    if _parallel_providers_enabled():
        return any(
            os.getenv(_api_key_name(candidate))
            for candidate in ("gemini", "openrouter")
        )
    return bool(os.getenv(_api_key_name(provider)))


def _provider_label(provider: str) -> str:
    if provider == "gemini":
        return "Gemini"
    if provider == "openrouter":
        return "OpenRouter"
    return "OpenAI"


def _build_pipeline() -> ResumePipeline:
    provider = _provider()
    settings = ModelSettings(
        provider=provider,
        model=_model_for(provider),
        temperature=0.2,
        max_correction_retries=2,
        keyword_coverage_threshold=0.65,
        groundedness_threshold=1.0,
    )
    return ResumePipeline(
        agents=ResumeAgents(build_reliable_structured_client(settings)),
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(settings),
        renderer=PDFRenderer(),
    )


async def _persist_upload(upload: UploadFile, destination_dir: Path) -> Path:
    filename = Path(upload.filename or "document.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or 'unknown'}")
    destination = destination_dir / f"{uuid.uuid4().hex[:10]}-{filename}"
    size = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                stream.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"{filename} exceeds the 10 MB limit")
            stream.write(chunk)
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"{filename} is empty")
    return destination


def _alignment_components(result: PipelineResult) -> dict[str, int]:
    validation = result.validation
    resume = result.document_package.resume
    strategy = resume.strategy or build_resume_strategy(
        result.master_profile, result.job_requirements
    )
    present = {
        "summary": resume.professional_summary is not None,
        "skills": bool(resume.skills),
        "experience": bool(resume.experience),
        "projects": bool(resume.projects),
        "certifications": bool(resume.certifications),
        "education": bool(resume.education),
    }
    expected_sections = strategy.section_order
    completeness = (
        sum(present[section] for section in expected_sections) / len(expected_sections)
        if expected_sections
        else 1.0
    )
    return {
        "supported_requirement_coverage": round(validation.keyword_coverage_ratio * 100),
        "evidence_strength": round(validation.groundedness_ratio * 100),
        "parseability": 100 if validation.ats_integrity_passed else 0,
        "content_completeness": round(completeness * 100),
    }


def _application_alignment(result: PipelineResult) -> int:
    components = _alignment_components(result)
    weighted = (
        components["supported_requirement_coverage"] * 0.40
        + components["evidence_strength"] * 0.35
        + components["parseability"] * 0.15
        + components["content_completeness"] * 0.10
    )
    return round(weighted)


def _public_generation_error(exc: Exception) -> tuple[int, str]:
    provider = _provider()
    label = _provider_label(provider)
    key_name = _api_key_name(provider)
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    exception_name = exc.__class__.__name__
    if exception_name == "ProviderPoolError":
        if status == 504:
            return 504, "All configured LLM providers timed out. Retry the request."
        if status == 429:
            return 429, "All configured LLM providers are rate-limited. Retry shortly."
        return 503, "All configured LLM providers failed. Retry shortly."
    if status in {408, 504}:
        return 504, f"{label} upstream inference timed out. Retry the request."
    if exception_name in {"APITimeoutError", "ReadTimeout", "TimeoutException"}:
        return 504, f"{label} took too long to respond. Retry or choose a faster model."
    if exception_name in {"APIConnectionError", "ConnectError"}:
        return 503, f"Could not connect to {label}. Check the network and try again."
    if exception_name == "StructuredOutputError":
        return 502, f"{label} returned an invalid structured response. Retry the request."
    if status == 400:
        suffix = f" ({param})" if param else ""
        return 502, f"{label} rejected an agent request{suffix}: {message or 'invalid request'}"
    if status == 401:
        return 503, f"{label} authentication failed. Check {key_name}."
    if status in {403, 404}:
        return 503, f"The configured {label} model is unavailable to this API project."
    if code == "credit_balance_exhausted" or error_type == "insufficient_quota":
        return 402, f"{label} API quota or credits are exhausted. Check billing/quota for the configured project, then try again."
    if status == 429:
        return 429, f"{label} rate limit reached. Wait briefly and try again."
    if isinstance(status, int) and status >= 500:
        return 502, f"{label} is temporarily unavailable. Try again shortly."
    return 500, f"Generation failed: {exception_name}"


def _public_ingestion_error(exc: IngestionError) -> str:
    """Translate parser failures without exposing temporary server paths."""

    message = str(exc).lower()
    if "password-protected pdf" in message:
        return "A supplied PDF is password-protected. Remove the password and upload it again."
    if "no extractable text" in message:
        return (
            "A supplied document contains no extractable text. Upload a text-based PDF, "
            "DOCX, TXT, or Markdown file; scanned PDFs need OCR first."
        )
    if "has " in message and " characters; limit is " in message:
        return (
            f"A supplied document exceeds the {MAX_DOCUMENT_CHARS:,}-character text limit. "
            "Split or shorten it and try again."
        )
    if "unable to decode text file" in message:
        return "A supplied text file could not be decoded. Save it as UTF-8 and upload it again."
    if "failed to parse pdf" in message or "failed to parse word document" in message:
        return (
            "A supplied document could not be parsed. Re-export it as PDF or DOCX, "
            "or upload its contents as TXT or Markdown."
        )
    return "A supplied document could not be read. Check the file and upload it again."


def _resume_markdown(result: PipelineResult) -> str:
    resume = result.document_package.resume
    contact = " · ".join(
        value
        for value in [
            resume.contact.location,
            resume.contact.email,
            resume.contact.phone,
            *(f"{link.label}: {link.url}" for link in resume.contact.links),
        ]
        if value
    )
    lines = [
        f"# {resume.contact.full_name}",
        f"## {resume.target_title}",
        "",
        contact,
        "",
    ]
    order = (
        resume.strategy.section_order
        if resume.strategy is not None
        else ["summary", "skills", "experience", "education", "projects", "certifications"]
    )
    for section in order:
        if section == "summary" and resume.professional_summary is not None:
            lines.extend(["### Professional Summary", resume.professional_summary.text, ""])
        elif section == "skills" and resume.skills:
            lines.append("### Technical Skills")
            groups = resume.skill_groups or []
            if groups:
                lines.extend(
                    f"- **{group.category}:** {', '.join(group.skills)}" for group in groups
                )
            else:
                lines.append(" · ".join(resume.skills))
            lines.append("")
        elif section == "experience" and resume.experience:
            lines.append("### Professional Experience")
            for role in resume.experience:
                dates = "–".join(value for value in [role.start_date, role.end_date] if value)
                lines.extend(
                    [
                        f"**{role.title} — {role.company}**",
                        dates,
                        *[f"- {bullet.text}" for bullet in role.bullets],
                        "",
                    ]
                )
        elif section == "projects" and resume.projects:
            lines.append("### Projects")
            for project in resume.projects:
                links = " · ".join(
                    value
                    for value in [project.repository_url, project.demo_url]
                    if value
                )
                lines.extend(
                    [
                        f"**{project.name}** — {', '.join(project.technologies)}",
                        links,
                        project.description.text,
                        *[f"- {bullet.text}" for bullet in project.bullets],
                        "",
                    ]
                )
        elif section == "certifications" and resume.certifications:
            lines.append("### Certifications")
            for certification in resume.certifications:
                date_value = certification.earned_date or certification.date
                values = [certification.name, certification.issuer, date_value]
                lines.append("- " + " — ".join(value for value in values if value))
            lines.append("")
        elif section == "education" and resume.education:
            lines.append("### Education")
            for education in resume.education:
                degree = ", ".join(
                    value for value in [education.degree, education.field_of_study] if value
                )
                lines.append(f"- {degree} — {education.institution}")
            lines.append("")
    return "\n".join(lines)


def _cover_markdown(result: PipelineResult) -> str:
    letter = result.document_package.cover_letter
    lines = [f"# Cover letter — {letter.signatory}", "", letter.letter_date, "", letter.salutation, ""]
    lines.extend(paragraph.text for paragraph in letter.paragraphs)
    lines.extend(["", letter.closing, letter.signatory])
    return "\n\n".join(lines)


def _result_payload(run_id: str, result: PipelineResult) -> dict[str, Any]:
    validation = result.validation
    requirements = result.job_requirements
    package = result.document_package
    artifact_pages = {artifact.kind: artifact.page_count for artifact in result.render_manifest.artifacts}
    strategy = package.resume.strategy or build_resume_strategy(
        result.master_profile, requirements
    )
    achievement_audit = []
    for role in package.resume.experience:
        for bullet in role.bullets[:2]:
            achievement_audit.append(
                {
                    "action": bullet.action,
                    "contribution": bullet.object or bullet.accomplishment or bullet.text,
                    "outcome": bullet.outcome or bullet.measurement or "Qualitative source-backed outcome",
                    "method": bullet.method,
                    "source": f"{role.company} · {role.title}",
                }
            )
    for project in package.resume.projects:
        project_claims = project.bullets[:2]
        if not project_claims:
            achievement_audit.append(
                {
                    "action": None,
                    "contribution": project.description.text,
                    "outcome": "Source-backed project evidence",
                    "method": ", ".join(project.technologies) or None,
                    "source": f"Project · {project.name}",
                }
            )
        for bullet in project_claims:
            achievement_audit.append(
                {
                    "action": bullet.action,
                    "contribution": bullet.object or bullet.accomplishment or bullet.text,
                    "outcome": bullet.outcome or bullet.measurement or "Qualitative source-backed outcome",
                    "method": bullet.method,
                    "source": f"Project · {project.name}",
                }
            )
    components = _alignment_components(result)
    return {
        "run_id": run_id,
        "status": "complete",
        "target": {"title": requirements.target_title, "company": requirements.company_name},
        "scores": {
            "application_alignment": _application_alignment(result),
            **components,
        },
        "analysis": {
            "matched_keywords": validation.matched_keywords,
            "missing_keywords": validation.missing_keywords,
            "requirements": [
                *requirements.hard_skills,
                *requirements.tools,
                *requirements.methodologies,
                *requirements.responsibilities,
            ],
            "evidence": [fact.statement for fact in result.master_profile.summary_facts]
            + [
                achievement.statement
                for role in result.master_profile.experience
                for achievement in role.achievements
            ]
            + [project.description for project in result.master_profile.projects]
            + [
                achievement.statement
                for project in result.master_profile.projects
                for achievement in project.achievements
            ],
            "guardrails": {
                "groundedness_passed": result.groundedness_audit.passed,
                "ats_integrity_passed": validation.ats_integrity_passed,
                "correction_attempts": result.correction_attempts,
                "issues": [issue.model_dump(mode="json") for issue in validation.issues],
            },
            "strategy": strategy.model_dump(mode="json"),
            "evidence_gap_questions": strategy.evidence_gap_questions,
            "score_explanation": (
                "Internal application-alignment diagnostic; it is not an employer ATS score "
                "or a guarantee of progression."
            ),
            "achievement_audit": achievement_audit,
        },
        "documents": package.model_dump(mode="json"),
        "markdown": {"resume": _resume_markdown(result), "cover_letter": _cover_markdown(result)},
        "downloads": {
            "resume_pdf": f"/api/v1/runs/{run_id}/resume.pdf",
            "cover_letter_pdf": f"/api/v1/runs/{run_id}/cover_letter.pdf",
        },
        "artifacts": {
            "resume": {"pages": artifact_pages.get("resume", 0)},
            "cover_letter": {"pages": artifact_pages.get("cover_letter", 0)},
        },
    }


app = FastAPI(
    title="Proofline CV Agent API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)


def _allowed_origins() -> list[str]:
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    configured = [
        value.strip().rstrip("/")
        for value in os.getenv("CORS_ORIGINS", "").split(",")
        if value.strip()
    ]
    return list(dict.fromkeys([*defaults, *configured]))


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Proofline-Proxy-Secret"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def require_hosting_proxy(request: Request, call_next):
    """Optionally prevent direct use of a publicly hosted provider-backed API."""

    expected = os.getenv("PROXY_AUTH_SECRET", "").strip()
    if expected and request.url.path.startswith("/api/v1/"):
        supplied = request.headers.get("X-Proofline-Proxy-Secret", "")
        if not supplied or not compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, Any]:
    provider = _provider()
    configured_providers = _configured_providers()
    return {
        "status": "ok",
        "provider": provider,
        "model": _model_for(provider),
        "fallback_models": _fallback_models_for(provider),
        "api_key_configured": bool(os.getenv(_api_key_name(provider))),
        "configured_providers": configured_providers,
        "parallel_providers": (
            _parallel_providers_enabled()
            and "gemini" in configured_providers
            and "openrouter" in configured_providers
        ),
    }


@app.post("/api/v1/generate")
async def generate(
    jd_text: Annotated[str | None, Form()] = None,
    jd_file: Annotated[UploadFile | None, File()] = None,
    cv_files: Annotated[list[UploadFile] | None, File()] = None,
    use_sample_profile: Annotated[bool, Form()] = False,
    letter_date: Annotated[date | None, Form()] = None,
) -> dict[str, Any]:
    provider = _provider()
    key_name = _api_key_name(provider)
    if not _execution_provider_ready(provider):
        raise HTTPException(status_code=503, detail=f"{key_name} is not configured on the backend")
    uploads = cv_files or []
    if len(uploads) > MAX_CV_FILES:
        raise HTTPException(status_code=422, detail=f"A maximum of {MAX_CV_FILES} CV files is supported")
    if not uploads and not use_sample_profile:
        raise HTTPException(status_code=422, detail="Upload at least one CV source")
    normalized_jd_text = normalize_text(jd_text) if jd_text is not None else None
    if bool(normalized_jd_text) == bool(jd_file):
        raise HTTPException(status_code=422, detail="Provide exactly one job description source")
    if normalized_jd_text is not None and len(normalized_jd_text) > MAX_DOCUMENT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The pasted job description exceeds the {MAX_DOCUMENT_CHARS:,}-character "
                "limit. Shorten it or upload a smaller document."
            ),
        )

    effective_letter_date = letter_date or date.today()

    run_id = uuid.uuid4().hex
    run_root = RUNS_ROOT / run_id
    upload_root = run_root / "uploads"
    output_root = run_root / "artifacts"
    upload_root.mkdir(parents=True, exist_ok=False)
    try:
        jd_path = await _persist_upload(jd_file, upload_root) if jd_file else None
        cv_paths = [await _persist_upload(upload, upload_root) for upload in uploads]
        if use_sample_profile and not cv_paths:
            cv_paths = [EXAMPLES_ROOT / "candidate_primary.md", EXAMPLES_ROOT / "candidate_history.txt"]
        request = GenerateRequest(
            ingestion=IngestionRequest(
                jd_path=jd_path,
                jd_text=normalized_jd_text,
                cv_paths=cv_paths,
                max_chars_per_document=MAX_DOCUMENT_CHARS,
            ),
            output_dir=output_root,
            letter_date=effective_letter_date,
        )
        async with PIPELINE_CONCURRENCY:
            result = await run_in_threadpool(_build_pipeline().run, request)
        return _result_payload(run_id, result)
    except HTTPException:
        shutil.rmtree(run_root, ignore_errors=True)
        raise
    except QualityGateError as exc:
        shutil.rmtree(run_root, ignore_errors=True)
        issues = [issue.model_dump(mode="json") for issue in exc.issues]
        log_event(
            LOGGER,
            "quality_gate_failed",
            run_id=run_id,
            issue_count=len(exc.issues),
            issues=[
                {
                    "code": issue.code,
                    "location": issue.location,
                    "retryable": issue.retryable,
                    "severity": issue.severity,
                }
                for issue in exc.issues[:20]
            ],
        )
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "issues": issues},
        ) from exc
    except IngestionError as exc:
        shutil.rmtree(run_root, ignore_errors=True)
        log_event(
            LOGGER,
            "ingestion_failed",
            run_id=run_id,
            exception_type=exc.__class__.__name__,
        )
        raise HTTPException(status_code=422, detail=_public_ingestion_error(exc)) from exc
    except Exception as exc:
        shutil.rmtree(run_root, ignore_errors=True)
        status_code, detail = _public_generation_error(exc)
        provider_status = getattr(exc, "status_code", None)
        log_event(
            LOGGER,
            "generation_failed",
            run_id=run_id,
            provider=provider,
            exception_type=exc.__class__.__name__,
            provider_status_code=provider_status if isinstance(provider_status, int) else None,
            response_status_code=status_code,
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.get("/api/v1/runs/{run_id}/{artifact_name}")
def download_artifact(run_id: str, artifact_name: str) -> FileResponse:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    allowed = {
        "resume.pdf": ("resume.pdf", "application/pdf"),
        "cover_letter.pdf": ("cover_letter.pdf", "application/pdf"),
    }
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found")
    stored_filename, media_type = allowed[artifact_name]
    path = RUNS_ROOT / run_id / "artifacts" / stored_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    document_kind = "resume" if artifact_name == "resume.pdf" else "cover-letter"
    unique_filename = f"{document_kind}-{run_id[:8]}-{uuid.uuid4().hex[:8]}.pdf"
    return FileResponse(path, media_type=media_type, filename=unique_filename)


def run() -> None:
    """Run the local API server."""
    import uvicorn

    uvicorn.run("cv_agent.api:app", host="127.0.0.1", port=8000, reload=False)
