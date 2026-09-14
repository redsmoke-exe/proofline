"""Convenience runner for source checkouts; installed users can call `cv-agent`."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_agent.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
