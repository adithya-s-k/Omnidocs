"""
API backend configuration for Qwen3-VL layout detection.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QwenLayoutAPIConfig(BaseModel):
    """
    API backend configuration for Qwen layout detection.

    This backend uses OpenAI-compatible APIs (OpenRouter, Novita AI, etc.)
    for serverless inference without local GPU.
    Requires: openai

    Example:
        >>> import os
        >>> config = QwenLayoutAPIConfig(
        ...     model="qwen/qwen3-vl-8b-instruct",
        ...     api_key=os.environ["OPENROUTER_API_KEY"],
        ...     base_url="https://openrouter.ai/api/v1",
        ... )
    """

    model: str = Field(
        default="qwen/qwen3-vl-8b-instruct",
        description="API model identifier. Format varies by provider. OpenRouter: 'qwen/qwen3-vl-8b-instruct'",
    )
    api_key: str = Field(
        ...,
        description="API key for authentication. Required.",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="API base URL. "
        "OpenRouter: 'https://openrouter.ai/api/v1', "
        "Novita AI: 'https://api.novita.ai/v3/openai'",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        description="Request timeout in seconds.",
    )
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional headers to send with requests. Useful for provider-specific headers.",
    )

    model_config = ConfigDict(extra="forbid")
