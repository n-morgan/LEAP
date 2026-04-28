"""
runners/anthropic.py — Anthropic Messages API runner.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from rlm_pipeline import _parse_rlm_output
from runners.base import slugify


class AnthropicRunner:
    """Anthropic Messages API extraction."""

    def __init__(
        self,
        model_name: str = "claude-opus-4-6",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[Any] = None

    @property
    def model_slug(self) -> str:
        return slugify(f"anthropic_{self.model_name}")

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        response = self._get_client().messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT:\n{document_markdown}\n\n"
                        "Extract and classify all climate policies from the document "
                        "as a JSON list."
                    ),
                }
            ],
        )
        raw = response.content[0].text if response.content else ""
        return _parse_rlm_output(raw)
