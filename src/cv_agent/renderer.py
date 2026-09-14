"""Deterministic Jinja/HTML/CSS rendering and WeasyPrint PDF compilation."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pypdf import PdfReader, PdfWriter

from .logging_config import log_event
from .schemas import (
    DocumentPackage,
    RenderManifest,
    RenderedArtifact,
    ValidationIssue,
)


LOGGER = logging.getLogger(__name__)
TEMPLATE_VERSION = "ats-adaptive-single-column-v3"
_FORBIDDEN_TAG_RE = re.compile(r"<\s*(table|img|svg|canvas|iframe)\b", re.IGNORECASE)
_FORBIDDEN_CSS_RE = re.compile(
    r"(?:display\s*:\s*(?:grid|flex)|column-count\s*:|columns\s*:|float\s*:)",
    re.IGNORECASE,
)


class RenderingError(RuntimeError):
    pass


class PDFRenderer:
    def __init__(self, template_dir: Path | None = None) -> None:
        self.template_dir = template_dir or Path(__file__).resolve().parent / "templates"
        self.environment = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(("html", "xml")),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )
        self.css = (self.template_dir / "styles.css").read_text(encoding="utf-8")

    def render_html(self, package: DocumentPackage) -> tuple[str, str]:
        page_size = (
            package.resume.strategy.page_size
            if package.resume.strategy is not None
            else "Letter"
        )
        resume_html = self.environment.get_template("resume.html").render(
            resume=package.resume,
            css=self.css,
            template_version=TEMPLATE_VERSION,
            page_size=page_size,
        )
        cover_html = self.environment.get_template("cover_letter.html").render(
            letter=package.cover_letter,
            css=self.css,
            template_version=TEMPLATE_VERSION,
            page_size=page_size,
        )
        return resume_html, cover_html

    def audit_html(self, resume_html: str, cover_html: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        combined = f"{resume_html}\n{cover_html}"
        forbidden_tag = _FORBIDDEN_TAG_RE.search(combined)
        if forbidden_tag:
            issues.append(
                ValidationIssue(
                    code="template_integrity",
                    severity="error",
                    message=f"Forbidden ATS-hostile tag: {forbidden_tag.group(1)}.",
                    location="rendered_html",
                    retryable=False,
                )
            )
        forbidden_css = _FORBIDDEN_CSS_RE.search(combined)
        if forbidden_css:
            issues.append(
                ValidationIssue(
                    code="template_integrity",
                    severity="error",
                    message=f"Forbidden multi-column CSS: {forbidden_css.group(0)}.",
                    location="rendered_html",
                    retryable=False,
                )
            )
        allowed_headings = {
            "Professional Summary",
            "Technical Skills",
            "Professional Experience",
            "Projects",
            "Certifications",
            "Education",
        }
        headings = set(re.findall(r"<h2>([^<]+)</h2>", resume_html))
        if not headings:
            issues.append(
                ValidationIssue(
                    code="missing_standard_section",
                    severity="error",
                    message="Resume must contain at least one standard evidence section.",
                    location="resume_html",
                    retryable=False,
                )
            )
        for heading in sorted(headings - allowed_headings):
            issues.append(
                ValidationIssue(
                    code="nonstandard_section",
                    severity="error",
                    message=f"Resume contains a nonstandard section heading: {heading}.",
                    location="resume_html",
                    retryable=False,
                )
            )
        return issues

    def preflight(self, package: DocumentPackage) -> tuple[str, str, list[ValidationIssue]]:
        resume_html, cover_html = self.render_html(package)
        return resume_html, cover_html, self.audit_html(resume_html, cover_html)

    def render(
        self,
        package: DocumentPackage,
        output_dir: Path,
        rendered_on: date,
    ) -> RenderManifest:
        self._configure_native_runtime()
        try:
            from weasyprint import HTML
        except ImportError as exc:
            raise RenderingError(
                "WeasyPrint is required to render PDFs; install requirements.txt"
            ) from exc

        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_html, cover_html, issues = self.preflight(package)
        if issues:
            messages = "; ".join(issue.message for issue in issues)
            raise RenderingError(f"template preflight failed: {messages}")

        outputs = (
            ("resume", resume_html, output_dir / "resume.html", output_dir / "resume.pdf"),
            (
                "cover_letter",
                cover_html,
                output_dir / "cover_letter.html",
                output_dir / "cover_letter.pdf",
            ),
        )
        artifacts: list[RenderedArtifact] = []
        for kind, html_text, html_path, pdf_path in outputs:
            html_path.write_text(html_text, encoding="utf-8", newline="\n")
            with tempfile.NamedTemporaryFile(
                prefix=f".{kind}-",
                suffix=".pdf",
                dir=output_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                HTML(string=html_text, base_url=str(self.template_dir)).write_pdf(
                    str(temporary_path),
                    pdf_tags=True,
                )
                self._normalize_pdf_metadata(temporary_path, pdf_path)
            finally:
                temporary_path.unlink(missing_ok=True)

            payload = pdf_path.read_bytes()
            reader = PdfReader(str(pdf_path))
            page_count = len(reader.pages)
            extracted_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            expected_identity = (
                package.resume.contact.full_name
                if kind == "resume"
                else package.cover_letter.signatory
            )
            if len(extracted_text.strip()) < 40 or expected_identity not in extracted_text:
                raise RenderingError(
                    f"{kind} PDF failed plain-text parseability verification"
                )
            artifacts.append(
                RenderedArtifact(
                    kind=kind,  # type: ignore[arg-type]
                    pdf_path=pdf_path,
                    html_path=html_path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    page_count=page_count,
                )
            )
            log_event(
                LOGGER,
                "artifact_rendered",
                artifact_kind=kind,
                pdf_path=str(pdf_path),
                page_count=page_count,
                bytes=len(payload),
            )
        return RenderManifest(
            template_version=TEMPLATE_VERSION,
            rendered_on=rendered_on,
            artifacts=artifacts,
        )

    @staticmethod
    def _configure_native_runtime() -> None:
        """Make common Homebrew libraries discoverable without shell-specific setup."""

        cache_dir = Path(tempfile.gettempdir()) / "cv-agent-font-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        if sys.platform != "darwin" or os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"):
            return
        candidates = [Path("/opt/homebrew/lib"), Path("/usr/local/lib")]
        available = [str(path) for path in candidates if path.is_dir()]
        if available:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(available)

    @staticmethod
    def _normalize_pdf_metadata(source: Path, destination: Path) -> None:
        """Atomically rewrite volatile metadata while preserving page content."""

        reader = PdfReader(str(source))
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        writer.add_metadata(
            {
                "/Title": "ATS Candidate Document",
                "/Author": "CV Agent",
                "/Creator": TEMPLATE_VERSION,
                "/Producer": "CV Agent deterministic renderer",
                "/CreationDate": "D:20000101000000Z",
                "/ModDate": "D:20000101000000Z",
            }
        )
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-normalized-",
            suffix=".pdf",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            normalized_path = Path(temporary.name)
            writer.write(temporary)
        try:
            os.replace(normalized_path, destination)
        finally:
            normalized_path.unlink(missing_ok=True)
