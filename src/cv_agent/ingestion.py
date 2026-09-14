"""Universal, defensive text ingestion for JD and candidate documents."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from .schemas import DocumentKind, DocumentText, IngestionBundle, IngestionRequest


SUPPORTED_EXTENSIONS: dict[str, DocumentKind] = {
    ".txt": "txt",
    ".md": "md",
    ".pdf": "pdf",
    ".docx": "docx",
}


class IngestionError(ValueError):
    """Raised when a source cannot be safely converted into useful text."""


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace while preserving readable paragraphs."""

    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _iter_docx_blocks(parent: DocxDocument) -> Iterable[Paragraph | Table]:
    """Yield paragraphs and tables in document order."""

    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    for child in parent.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


class UniversalDocumentReader:
    """Read supported files with size limits and deterministic normalization."""

    def __init__(self, max_file_bytes: int = 25 * 1024 * 1024) -> None:
        self.max_file_bytes = max_file_bytes

    def read(self, path: Path, max_chars: int = 120_000) -> DocumentText:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise IngestionError(f"source is not a readable file: {source}")
        if source.stat().st_size > self.max_file_bytes:
            raise IngestionError(
                f"source exceeds {self.max_file_bytes} byte limit: {source}"
            )

        suffix = source.suffix.lower()
        kind = SUPPORTED_EXTENSIONS.get(suffix)
        if kind is None:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise IngestionError(f"unsupported extension {suffix!r}; expected {allowed}")

        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if kind in {"txt", "md"}:
            extracted = self._decode_text(payload, source)
        elif kind == "pdf":
            extracted = self._read_pdf(source)
        else:
            extracted = self._read_docx(source)

        normalized = normalize_text(extracted)
        if not normalized:
            raise IngestionError(f"no extractable text found in {source}")
        if len(normalized) > max_chars:
            raise IngestionError(
                f"extracted text from {source} has {len(normalized)} characters; "
                f"limit is {max_chars}. Split the document or raise the configured limit."
            )
        return DocumentText(
            document_id=f"doc_{digest[:16]}",
            source_path=source,
            kind=kind,
            sha256=digest,
            text=normalized,
            character_count=len(normalized),
        )

    def read_raw_text(self, text: str, label: str = "job-description.txt") -> DocumentText:
        normalized = normalize_text(text)
        if not normalized:
            raise IngestionError("raw job description is empty")
        payload = normalized.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return DocumentText(
            document_id=f"doc_{digest[:16]}",
            source_path=Path(label),
            kind="txt",
            sha256=digest,
            text=normalized,
            character_count=len(normalized),
        )

    @staticmethod
    def _decode_text(payload: bytes, source: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise IngestionError(f"unable to decode text file: {source}")

    @staticmethod
    def _read_pdf(source: Path) -> str:
        try:
            reader = PdfReader(str(source))
            if reader.is_encrypted and not reader.decrypt(""):
                raise IngestionError(f"password-protected PDF is unsupported: {source}")
            pages: list[str] = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text(extraction_mode="layout") or ""
                except TypeError:
                    text = page.extract_text() or ""
                pages.append(f"[Page {page_number}]\n{text}")
            return "\n\n".join(pages)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"failed to parse PDF {source}: {exc}") from exc

    @staticmethod
    def _read_docx(source: Path) -> str:
        try:
            document = Document(str(source))
            blocks: list[str] = []
            for block in _iter_docx_blocks(document):
                if isinstance(block, Paragraph):
                    if block.text.strip():
                        blocks.append(block.text)
                else:
                    for row in block.rows:
                        cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if cells:
                            blocks.append(" | ".join(cells))

            for section in document.sections:
                for container in (section.header, section.footer):
                    for paragraph in container.paragraphs:
                        if paragraph.text.strip():
                            blocks.append(paragraph.text)
            return "\n".join(dict.fromkeys(blocks))
        except Exception as exc:
            raise IngestionError(f"failed to parse Word document {source}: {exc}") from exc


def ingest_request(
    request: IngestionRequest,
    reader: UniversalDocumentReader | None = None,
) -> IngestionBundle:
    """Resolve an ingestion request into a normalized bundle."""

    reader = reader or UniversalDocumentReader()
    if request.jd_path is not None:
        job_description = reader.read(request.jd_path, request.max_chars_per_document)
    else:
        assert request.jd_text is not None
        job_description = reader.read_raw_text(request.jd_text)

    candidate_documents = [
        reader.read(path, request.max_chars_per_document) for path in request.cv_paths
    ]
    seen: set[str] = set()
    unique_documents: list[DocumentText] = []
    for document in candidate_documents:
        if document.sha256 not in seen:
            unique_documents.append(document)
            seen.add(document.sha256)
    if not unique_documents:
        raise IngestionError("no unique candidate documents were provided")
    return IngestionBundle(
        job_description=job_description,
        candidate_documents=unique_documents,
    )
