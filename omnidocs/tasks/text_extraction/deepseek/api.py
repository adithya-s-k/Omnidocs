"""
API backend configuration for DeepSeek-OCR text extraction.

Primary provider: Novita AI
  https://novita.ai/models/model-detail/deepseek-deepseek-ocr

Note: DeepSeek-OCR-2 API availability may vary by provider — check
novita.ai for updated model slugs as providers onboard the new version.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DeepSeekOCRTextAPIConfig(BaseModel):
    """
    API backend configuration for DeepSeek-OCR / DeepSeek-OCR-2 text extraction.

    Uses litellm for provider-agnostic API access.
    Primary provider: Novita AI (official hosting).

    Example:
        ```python
        # Novita AI (reads NOVITA_API_KEY from env)
        config = DeepSeekOCRTextAPIConfig(
            model="novita/deepseek/deepseek-ocr",
        )
        ```
    """

    model: str = Field(
        default="novita/deepseek/deepseek-ocr",
        description="Model identifier in litellm format. "
        "Novita AI: 'novita/deepseek/deepseek-ocr'. "
        "Check novita.ai for DeepSeek-OCR-2 slug once available.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key. If None, reads from env (e.g. NOVITA_API_KEY).",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Override provider base URL.",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=131072,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for greedy decoding.",
    )
    timeout: int = Field(
        default=180,
        ge=10,
        description="Request timeout in seconds.",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version string (for Azure OpenAI).",
    )
    extra_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional request headers.",
    )

    model_config = ConfigDict(extra="forbid")
