"""Command-line entrypoint for generation, ingestion inspection, and safe re-rendering."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from .agent import ResumeAgents, build_reliable_structured_client
from .ingestion import ingest_request
from .logging_config import configure_logging
from .pipeline import QualityGateError, ResumePipeline
from .renderer import PDFRenderer
from .schemas import (
    GenerateRequest,
    IngestionRequest,
    ModelSettings,
    PipelineSnapshot,
)
from .validation import PackageValidator, ProfileValidator


def _add_ingestion_arguments(parser: argparse.ArgumentParser) -> None:
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", type=Path, help="Path to .txt, .md, .pdf, or .docx JD")
    jd_group.add_argument("--jd-text", help="Raw job-description text")
    parser.add_argument(
        "--cv",
        type=Path,
        action="append",
        required=True,
        help="Candidate CV path; repeat for multiple files",
    )
    parser.add_argument("--max-chars", type=int, default=120_000)


def _ingestion_from_args(args: argparse.Namespace) -> IngestionRequest:
    return IngestionRequest(
        jd_path=args.jd,
        jd_text=args.jd_text,
        cv_paths=args.cv,
        max_chars_per_document=args.max_chars,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv-agent",
        description="Generate grounded, ATS-optimized resume and cover-letter PDFs.",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Run the complete agent pipeline")
    _add_ingestion_arguments(generate)
    generate.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    generate.add_argument(
        "--provider",
        choices=["openai", "gemini", "openrouter"],
        default=os.getenv("LLM_PROVIDER", "gemini"),
    )
    generate.add_argument("--model", default=None)
    generate.add_argument("--temperature", type=float, default=0.2)
    generate.add_argument("--max-correction-retries", type=int, default=2)
    generate.add_argument("--keyword-threshold", type=float, default=0.65)
    # Accepted for backward-compatible scripts but intentionally ignored. Verified
    # metrics are useful telemetry, never a content quota.
    generate.add_argument(
        "--metric-threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    generate.add_argument("--letter-date", type=date.fromisoformat, default=date.today())

    inspect = subparsers.add_parser("inspect", help="Parse inputs without calling an LLM")
    _add_ingestion_arguments(inspect)

    render = subparsers.add_parser(
        "render", help="Re-render a previously validated pipeline snapshot"
    )
    render.add_argument("--snapshot", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    key_names = {
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key_name = key_names[args.provider]
    parallel_enabled = os.getenv("LLM_PARALLEL_PROVIDERS", "true").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    pooled_provider_ready = parallel_enabled and any(
        os.getenv(name) for name in ("GEMINI_API_KEY", "OPENROUTER_API_KEY")
    )
    if not os.getenv(key_name) and not pooled_provider_ready:
        print(f"error: {key_name} is not set", file=sys.stderr)
        return 2
    model = args.model
    if model is None:
        if args.provider == "gemini":
            model = os.getenv("LLM_MODEL") or os.getenv(
                "GEMINI_MODEL", "gemini-3-flash-preview"
            )
        elif args.provider == "openrouter":
            model = os.getenv("LLM_MODEL") or os.getenv(
                "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
            )
        else:
            model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.6")
    settings = ModelSettings(
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        max_correction_retries=args.max_correction_retries,
        keyword_coverage_threshold=args.keyword_threshold,
        groundedness_threshold=1.0,
    )
    client = build_reliable_structured_client(settings)
    pipeline = ResumePipeline(
        agents=ResumeAgents(client),
        profile_validator=ProfileValidator(),
        package_validator=PackageValidator(settings),
        renderer=PDFRenderer(),
    )
    request = GenerateRequest(
        ingestion=_ingestion_from_args(args),
        output_dir=args.output_dir,
        letter_date=args.letter_date,
    )
    try:
        result = pipeline.run(request)
    except QualityGateError as exc:
        print(f"quality gate failed: {exc}", file=sys.stderr)
        for issue in exc.issues:
            print(f"- [{issue.code}] {issue.location}: {issue.message}", file=sys.stderr)
        return 3
    summary = {
        "passed": result.validation.passed,
        "keyword_coverage": result.validation.keyword_coverage_ratio,
        "verified_metric_usage": result.validation.metric_density,
        "groundedness": result.validation.groundedness_ratio,
        "resume_strategy": result.document_package.resume.strategy.model_dump()
        if result.document_package.resume.strategy
        else None,
        "correction_attempts": result.correction_attempts,
        "artifacts": [
            {
                "kind": item.kind,
                "path": str(item.pdf_path),
                "pages": item.page_count,
                "sha256": item.sha256,
            }
            for item in result.render_manifest.artifacts
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    bundle = ingest_request(_ingestion_from_args(args))
    payload = {
        "job_description": {
            "document_id": bundle.job_description.document_id,
            "source": str(bundle.job_description.source_path),
            "kind": bundle.job_description.kind,
            "characters": bundle.job_description.character_count,
        },
        "candidate_documents": [
            {
                "document_id": item.document_id,
                "source": str(item.source_path),
                "kind": item.kind,
                "characters": item.character_count,
            }
            for item in bundle.candidate_documents
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_render(args: argparse.Namespace) -> int:
    snapshot = PipelineSnapshot.model_validate_json(
        args.snapshot.read_text(encoding="utf-8")
    )
    renderer = PDFRenderer()
    resume_html, cover_html = renderer.render_html(snapshot.document_package)
    fresh_validation = PackageValidator(snapshot.model_settings).validate(
        snapshot.document_package,
        snapshot.master_profile,
        snapshot.job_requirements,
        template_issues=renderer.audit_html(resume_html, cover_html),
        groundedness_audit=snapshot.groundedness_audit,
        expected_letter_date=snapshot.document_package.cover_letter.letter_date,
    )
    # Page-count warnings are added only after a PDF has been rendered. They are
    # not part of the pre-render truth/template validation reproduced above.
    comparable_snapshot_validation = snapshot.validation.model_copy(
        update={
            "issues": [
                issue
                for issue in snapshot.validation.issues
                if issue.code != "page_budget_exceeded"
            ]
        }
    )
    if (
        not snapshot.validation.passed
        or not fresh_validation.passed
        or comparable_snapshot_validation != fresh_validation
    ):
        print(
            "error: refusing to render a failed, stale, or modified validation snapshot",
            file=sys.stderr,
        )
        return 3
    manifest = renderer.render(
        snapshot.document_package,
        args.output_dir,
        date.today(),
    )
    print(manifest.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "inspect":
        return _run_inspect(args)
    if args.command == "render":
        return _run_render(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
