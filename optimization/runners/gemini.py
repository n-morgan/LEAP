"""
runners/gemini.py — Google Generative AI runner.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from rlm_pipeline import _DEFAULT_EXPERT_KNOWLEDGE_PATH, _parse_rlm_output, parse_document
from runners.base import slugify


class GeminiRunner:
    """Google Generative AI extraction."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        expert_knowledge_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.expert_knowledge_path = (
            expert_knowledge_path or _DEFAULT_EXPERT_KNOWLEDGE_PATH
        )

    @property
    def model_slug(self) -> str:
        return slugify(f"gemini_{self.model_name}")

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(temperature=self.temperature),
        )

        if self.expert_knowledge_path and os.path.exists(self.expert_knowledge_path):
            expert_knowledge = parse_document(self.expert_knowledge_path)
            prompt = (
                f"DOCUMENT:\n{document_markdown}\n\n"
                f"EXTRACTION CRITERIA:\n{expert_knowledge}\n\n"
                "Extract and classify all climate policies from the document "
                "as a JSON list."
            )
        else:
            prompt = (
                f"DOCUMENT:\n{document_markdown}\n\n"
                "Extract and classify all climate policies from the document "
                "as a JSON list."
            )

        try:
            response = model.generate_content(prompt)
        except Exception as e:
            err = str(e).lower()
            if (
                "too large" in err
                or "token" in err
                or "context" in err
                or "resource_exhausted" in err
                or "payload size" in err
            ):
                print(
                    f"  [WARN] Context length exceeded for {self.model_name} "
                    f"— recording as empty extraction (do not truncate)."
                )
                return []
            raise
        raw = response.text or ""
        return _parse_rlm_output(raw)
