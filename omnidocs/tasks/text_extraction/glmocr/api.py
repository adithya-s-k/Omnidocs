"""API backend configuration for GLM-OCR text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GLMOCRAPIConfig(BaseModel):
    """
        API backend configuration for GLM-OCR.

        Primary provider: ZhipuAI / BigModel (official) — get key at open.bigmodel.cn.

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
        default="openai/glm-ocr",
        description="Model in litellm openai/ format matching --served-model-name.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key.",
    )
    api_base: str = Field(
        default=None,
        description="Base URL of vLLM server (e.g. http://localhost:8192/v1).",
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
    repetition_penalty: float = Field(
        default=1.05,
        ge=1.0,
        le=2.0,
        description="Repetition penalty to prevent looping at temperature=0.0.",
    )
    api_version: Optional[str] = Field(default=None)
    extra_headers: Optional[dict[str, str]] = Field(default=None)

    model_config = ConfigDict(extra="forbid")
