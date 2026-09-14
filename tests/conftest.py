from __future__ import annotations

from pathlib import Path

import pytest

from cv_agent.schemas import PipelineSnapshot


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshot() -> PipelineSnapshot:
    return PipelineSnapshot.model_validate_json(
        (ROOT / "examples" / "sample_snapshot.json").read_text(encoding="utf-8")
    )
