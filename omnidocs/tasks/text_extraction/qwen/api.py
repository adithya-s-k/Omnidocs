"""
API backend configuration for Qwen3-VL text extraction.

Uses litellm for provider-agnostic inference (OpenRouter, Gemini, Azure, etc.).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QwenTextAPIConfig(BaseModel):
    """
    API backend configuration for Qwen text extraction.

    Uses litellm for provider-agnostic API access. Supports OpenRouter,
    Gemini, Azure, OpenAI, and any other litellm-compatible provider.

    API keys can be passed directly or read from environment variables.

    Example:
        ```python
        # OpenRouter (reads OPENROUTER_API_KEY from env)
        config = QwenTextAPIConfig(
            model="openrouter/qwen/qwen3-vl-8b-instruct",
        )

        # With explicit key
        config = QwenTextAPIConfig(
            model="openrouter/qwen/qwen3-vl-8b-instruct",
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base="https://openrouter.ai/api/v1",
        )
        ```
    """

    model: str = Field(
        default="openrouter/qwen/qwen3-vl-8b-instruct",
        description="Model identifier in litellm format with provider prefix. "
        "Examples: 'openrouter/qwen/qwen3-vl-8b-instruct', 'openrouter/qwen/qwen3-vl-32b-instruct'",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication. If None, litellm reads from environment variables.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override base URL. Usually not needed — litellm knows provider endpoints.",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=131072,
        description="Maximum number of tokens to generate. "
        "Text extraction typically needs more tokens than layout detection.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )
    timeout: int = Field(
        default=180,
        ge=10,
        description="Request timeout in seconds. Text extraction may need longer timeouts for complex documents.",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version string. Required for Azure OpenAI.",
    )
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional headers to send with requests. Useful for provider-specific headers.",
    )

    model_config = ConfigDict(extra="forbid")
