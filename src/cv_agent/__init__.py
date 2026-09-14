"""Production ATS resume and cover-letter agent."""

from .agent import (
    GeminiStructuredClient,
    HedgedStructuredClient,
    OpenAIStructuredClient,
    OpenRouterStructuredClient,
    ResumeAgents,
    build_reliable_structured_client,
    build_structured_client,
)
from .ingestion import UniversalDocumentReader, ingest_request
from .pipeline import QualityGateError, ResumePipeline
from .renderer import PDFRenderer
from .strategy import build_resume_strategy

__all__ = [
    "OpenAIStructuredClient",
    "GeminiStructuredClient",
    "HedgedStructuredClient",
    "PDFRenderer",
    "OpenRouterStructuredClient",
    "QualityGateError",
    "ResumeAgents",
    "ResumePipeline",
    "UniversalDocumentReader",
    "build_reliable_structured_client",
    "build_resume_strategy",
    "build_structured_client",
    "ingest_request",
]

__version__ = "1.0.0"
