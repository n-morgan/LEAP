"""
runners/openai.py — Direct OpenAI Chat Completions runner.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from rlm_pipeline import _parse_rlm_output
from runners.base import slugify


class OpenAIRunner:
    """Direct OpenAI Chat Completions extraction (no recursion)."""

    def __init__(
        self,
        model_name: str = "gpt-5.4",
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self._client: Optional[Any] = None

    @property
    def model_slug(self) -> str:
        return slugify(f"openai_{self.model_name}")

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT:\n{document_markdown}\n\n"
                        "Extract and classify all climate policies from the document "
                        "as a JSON list."
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_rlm_output(raw)
