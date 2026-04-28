"""
runners — LEAP model runner package.

Add a new backend by creating a module in this folder and registering it in
RUNNER_CLASSES below.
"""

from runners.base import ModelRunner, slugify
from runners.rlm import RLMRunner
from runners.openai import OpenAIRunner
from runners.anthropic import AnthropicRunner
from runners.gemini import GeminiRunner

RUNNER_CLASSES: dict[str, type] = {
    "rlm":       RLMRunner,
    "openai":    OpenAIRunner,
    "anthropic": AnthropicRunner,
    "gemini":    GeminiRunner,
}

__all__ = [
    "ModelRunner",
    "slugify",
    "RLMRunner",
    "OpenAIRunner",
    "AnthropicRunner",
    "GeminiRunner",
    "RUNNER_CLASSES",
]
