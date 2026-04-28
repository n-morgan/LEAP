"""
runners/base.py — ModelRunner protocol and shared utilities.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelRunner(Protocol):
    """Any object with a .run() method can be used as a runner."""

    @property
    def model_slug(self) -> str:
        """Filesystem-safe identifier used for the output folder name."""
        ...

    def run(
        self,
        document_markdown: str,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Run extraction on ``document_markdown`` using ``system_prompt``.
        Returns a list of raw policy dicts (no DSPy validation).
        """
        ...


def slugify(s: str) -> str:
    """Replace non-alphanumeric characters with hyphens for safe folder names."""
    return re.sub(r"[^a-zA-Z0-9._-]", "-", s).strip("-")
