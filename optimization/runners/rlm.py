"""
runners/rlm.py — RLM recursive pipeline runner.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

# Ensure the parent optimization/ directory is on the path so rlm_pipeline
# can be imported regardless of where the script is invoked from.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from rlm_pipeline import _DEFAULT_EXPERT_KNOWLEDGE_PATH, run_rlm_for_optimizer
from runners.base import slugify


class RLMRunner:
    """
    RLM recursive pipeline (run_rlm_for_optimizer from rlm_pipeline.py).

    Supports expert_knowledge_path for grounding criteria injection.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.4",
        sub_model_name: str | None = None,
        expert_knowledge_path: str | None = None,
        max_iterations: int = 50,
        trace_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.sub_model_name = sub_model_name or model_name
        self.expert_knowledge_path = (
            expert_knowledge_path or _DEFAULT_EXPERT_KNOWLEDGE_PATH
        )
        self.max_iterations = max_iterations
        self.trace_dir = trace_dir

    @property
    def model_slug(self) -> str:
        return slugify(f"rlm_{self.model_name}")

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        # run_rlm_for_optimizer takes a document path, not markdown directly.
        # Write to a temp file so the existing API is satisfied.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(document_markdown)
            tmp_path = tmp.name
        try:
            return run_rlm_for_optimizer(
                prompt_string=system_prompt,
                document_path=tmp_path,
                trace_dir=self.trace_dir,
                expert_knowledge_path=self.expert_knowledge_path,
                model_name=self.model_name,
                sub_model_name=self.sub_model_name,
                max_iterations=self.max_iterations,
            )
        finally:
            os.unlink(tmp_path)
