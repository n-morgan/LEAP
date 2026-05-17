"""
runners/openai.py — Direct OpenAI Chat Completions runner.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..rlm_pipeline import _DEFAULT_EXPERT_KNOWLEDGE_PATH, _parse_rlm_output, parse_document
from .base import slugify


class OpenAIRunner:
    """Direct OpenAI Chat Completions extraction (no recursion)."""

    def __init__(
        self,
        model_name: str = "gpt-5.4",
        temperature: float = 0.0,
        expert_knowledge_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.expert_knowledge_path = (
            expert_knowledge_path or _DEFAULT_EXPERT_KNOWLEDGE_PATH
        )
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
        if self.expert_knowledge_path and os.path.exists(self.expert_knowledge_path):
            expert_knowledge = parse_document(self.expert_knowledge_path)
            user_content = (
                f"DOCUMENT:\n{document_markdown}\n\n"
                f"EXTRACTION CRITERIA:\n{expert_knowledge}\n\n"
                "Extract and classify all climate policies from the document "
                "as a JSON list."
            )
        else:
            user_content = (
                f"DOCUMENT:\n{document_markdown}\n\n"
                "Extract and classify all climate policies from the document "
                "as a JSON list."
            )

        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as e:
            err = str(e).lower()
            if "context_length_exceeded" in err or "too many tokens" in err or "maximum context" in err:
                print(
                    f"  [WARN] Context length exceeded for {self.model_name} "
                    f"— recording as empty extraction (do not truncate)."
                )
                return []
            # Some models (e.g. gpt-5.5) only support the default temperature
            if "temperature" in err and "unsupported" in err:
                print(
                    f"  [WARN] {self.model_name} does not support temperature={self.temperature}"
                    f" — retrying with default temperature."
                )
                response = self._get_client().chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
            else:
                raise
        raw = response.choices[0].message.content or ""
        return _parse_rlm_output(raw)
