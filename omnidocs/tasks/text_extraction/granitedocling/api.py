"""
API backend configuration for Granite Docling text extraction.

Uses litellm for provider-agnostic inference (OpenRouter, Gemini, Azure, etc.).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GraniteDoclingTextAPIConfig(BaseModel):
    """
    Configuration for Granite Docling text extraction via API.

    Uses litellm for provider-agnostic API access. Supports OpenRouter,
    Gemini, Azure, OpenAI, and any other litellm-compatible provider.

    API keys can be passed directly or read from environment variables.

    Example:
        ```python
        # OpenRouter
        config = GraniteDoclingTextAPIConfig(
            model="openrouter/ibm-granite/granite-docling-258M",
        )
        ```
    """

    model: str = Field(
        default="openrouter/ibm-granite/granite-docling-258M",
        description="Model identifier in litellm format with provider prefix.",
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
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    timeout: int = Field(
        default=180,
        ge=10,
        description="Request timeout in seconds",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version string. Required for Azure OpenAI.",
    )
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional HTTP headers",
    )

    model_config = ConfigDict(extra="forbid")
