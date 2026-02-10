"""
VLM API configuration for provider-agnostic VLM inference via litellm.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VLMAPIConfig(BaseModel):
    """
    Provider-agnostic VLM API configuration using litellm.

    Supports any provider litellm supports: Gemini, OpenRouter, Azure,
    OpenAI, Anthropic, etc. The model string uses litellm format with
    provider prefix (e.g., "gemini/gemini-2.5-flash").

    API keys can be passed directly or read from environment variables
    (GOOGLE_API_KEY, OPENROUTER_API_KEY, AZURE_API_KEY, OPENAI_API_KEY, etc.).

    Example:
        ```python
        # Gemini (reads GOOGLE_API_KEY from env)
        config = VLMAPIConfig(model="gemini/gemini-2.5-flash")

        # OpenRouter with explicit key
        config = VLMAPIConfig(
            model="openrouter/qwen/qwen3-vl-8b-instruct",
            api_key="sk-...",
        )

        # Azure OpenAI
        config = VLMAPIConfig(
            model="azure/gpt-4o",
            api_base="https://my-deployment.openai.azure.com/",
        )
        ```
    """

    model: str = Field(
        ...,
        description="Model identifier in litellm format with provider prefix. "
        "Examples: 'gemini/gemini-2.5-flash', 'openrouter/qwen/qwen3-vl-8b-instruct', "
        "'azure/gpt-4o', 'openai/gpt-4o'",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication. If None, litellm reads from "
        "environment variables (GOOGLE_API_KEY, OPENROUTER_API_KEY, etc.).",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override base URL. Usually not needed — litellm knows provider endpoints from the model prefix.",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=131072,
        description="Maximum number of tokens to generate.",
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
        description="Request timeout in seconds.",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version string. Required for Azure OpenAI (e.g., '2025-01-01-preview').",
    )
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional headers to send with requests.",
    )

    model_config = ConfigDict(extra="forbid")
