"""API backend configuration for GLM-OCR text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GLMOCRAPIConfig(BaseModel):
    """
    API backend configuration for GLM-OCR.

    GLM-OCR is not yet widely available via public APIs.
    Use a self-hosted vLLM server or ZhipuAI if/when they expose it.

    Example:
```python
        # Self-hosted vLLM server
        config = GLMOCRAPIConfig(
            model="zai-org/GLM-OCR",
            api_base="http://localhost:8000/v1",
            api_key="token-abc",
        )
```
    """

    model: str = Field(
        default="zai-org/GLM-OCR",
        description="Model identifier. For litellm: 'openai/zai-org/GLM-OCR' for self-hosted.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="API base URL (e.g. self-hosted vLLM server).",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        description="Request timeout in seconds.",
    )
    api_version: Optional[str] = Field(default=None)
    extra_headers: Optional[dict[str, str]] = Field(default=None)

    model_config = ConfigDict(extra="forbid")
