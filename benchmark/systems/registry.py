"""
systems/registry.py — System registry

Maps ``runner`` strings from config files to runner classes from the
optimization/runners/ implementation. Add a new system here to include it
in the benchmark without touching eval.py or benchmark.py.

Supported runner types
----------------------
    rlm      — Recursive Language Model (RLM) pipeline
    openai   — Direct OpenAI Chat Completions
    anthropic — Anthropic Messages API
    gemini   — Google Generative AI
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

# Resolve optimization/ so runner imports work regardless of cwd.
_OPT = pathlib.Path(__file__).resolve().parent.parent.parent / "optimization"
if str(_OPT) not in sys.path:
    sys.path.insert(0, str(_OPT))

from runners import RLMRunner, OpenAIRunner, AnthropicRunner, GeminiRunner  # noqa: E402
from rlm_pipeline import _DEFAULT_EXPERT_KNOWLEDGE_PATH                      # noqa: E402

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RUNNER_REGISTRY: dict[str, type] = {
    "rlm":       RLMRunner,
    "openai":    OpenAIRunner,
    "anthropic": AnthropicRunner,
    "gemini":    GeminiRunner,
}


def build_runner(config: dict[str, Any]) -> Any:
    """Instantiate a runner from a parsed config dict.

    Args:
        config: Parsed YAML config with at minimum ``runner`` and ``model``.

    Returns:
        A configured runner instance ready to call ``.run()``.
    """
    runner_type = config.get("runner", "").lower()
    if runner_type not in RUNNER_REGISTRY:
        raise ValueError(
            f"Unknown runner {runner_type!r}. "
            f"Valid: {sorted(RUNNER_REGISTRY)}"
        )

    model = config["model"]
    ekp   = config.get("expert_knowledge_path") or _DEFAULT_EXPERT_KNOWLEDGE_PATH

    if runner_type == "rlm":
        return RLMRunner(
            model_name=model,
            expert_knowledge_path=ekp,
            max_iterations=config.get("max_iterations", 50),
        )

    if runner_type == "openai":
        return OpenAIRunner(model_name=model, expert_knowledge_path=ekp)

    if runner_type == "anthropic":
        return AnthropicRunner(model_name=model, expert_knowledge_path=ekp)

    if runner_type == "gemini":
        return GeminiRunner(model_name=model, expert_knowledge_path=ekp)

    # Unreachable given the registry check above, but keeps mypy happy.
    raise ValueError(f"Unhandled runner type: {runner_type!r}")
