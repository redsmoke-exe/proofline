from __future__ import annotations

from datetime import date
from pathlib import Path

from pypdf import PdfReader

from cv_agent.renderer import PDFRenderer
from cv_agent.schemas import PipelineSnapshot
from cv_agent.document_repair import repair_document_package


def test_renderer_writes_parseable_single_column_pdfs(
    tmp_path: Path, snapshot: PipelineSnapshot
) -> None:
    renderer = PDFRenderer()
    manifest = renderer.render(snapshot.document_package, tmp_path, date(2026, 9, 12))
    assert len(manifest.artifacts) == 2
    resume_pdf = tmp_path / "resume.pdf"
    cover_pdf = tmp_path / "cover_letter.pdf"
    assert resume_pdf.is_file() and cover_pdf.is_file()

    resume_text = "\n".join(page.extract_text() or "" for page in PdfReader(resume_pdf).pages)
    assert "PROFESSIONAL SUMMARY" in resume_text
    assert "PROFESSIONAL EXPERIENCE" in resume_text
    assert "EDUCATION" in resume_text
    assert "42%" in resume_text
    assert "Alex Morgan" in resume_text

    cover_text = "\n".join(page.extract_text() or "" for page in PdfReader(cover_pdf).pages)
    assert "Dear Hiring Team" in cover_text
    assert "Acme Cloud" in cover_text


def test_static_css_is_embedded_without_html_escaping(snapshot: PipelineSnapshot) -> None:
    resume_html, cover_html = PDFRenderer().render_html(snapshot.document_package)
    assert 'font-family: Arial, "Liberation Sans", sans-serif' in resume_html
    assert 'content: " | "' in cover_html
    assert ">Data Platform Engineer</p>" in resume_html
    assert "Target role:" not in resume_html
    assert "&#34;" not in resume_html + cover_html


def test_student_strategy_renders_a4_without_optional_sections(
    snapshot: PipelineSnapshot,
) -> None:
    profile = snapshot.master_profile.model_copy(
        update={
            "contact": snapshot.master_profile.contact.model_copy(
                update={"location": "Bengaluru, India"}
            ),
            "summary_facts": [],
            "experience": [],
        }
    )
    package, _ = repair_document_package(
        snapshot.document_package,
        profile,
        snapshot.job_requirements,
        "2026-09-12",
    )

    resume_html, _, issues = PDFRenderer().preflight(package)

    assert issues == []
    assert 'class="resume-document page-a4"' in resume_html
    assert "PROFESSIONAL SUMMARY" not in resume_html
    assert "PROFESSIONAL EXPERIENCE" not in resume_html
    assert "<h2>Education</h2>" in resume_html
