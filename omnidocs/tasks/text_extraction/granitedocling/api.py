"""API backend configuration for Granite Docling text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GraniteDoclingTextAPIConfig(BaseModel):
    """Configuration for Granite Docling text extraction via API.

    Uses OpenAI-compatible API endpoints (LiteLLM, OpenRouter, etc.).
    """

    model: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="API model identifier",
    )
    api_key: str = Field(
        ...,
        description="API key for authentication. Required.",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="API base URL",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=16384,
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
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional HTTP headers",
    )

    model_config = ConfigDict(extra="forbid")
