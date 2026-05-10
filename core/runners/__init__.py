"""
runners — LEAP model runner package.

Add a new backend by creating a module in this folder and registering it in
RUNNER_CLASSES below.
"""

from .base import ModelRunner, slugify
from .rlm import RLMRunner
from .openai import OpenAIRunner
from .anthropic import AnthropicRunner
from .gemini import GeminiRunner

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
