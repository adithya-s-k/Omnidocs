"""API backend configuration for LightOn text extraction."""

from pydantic import BaseModel, ConfigDict, Field


class LightOnTextAPIConfig(BaseModel):
    """
    API backend config for LightOn text extraction.

    Connects to VLLM OpenAI-compatible API server running LightOn model.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import LightOnTextExtractor
        from omnidocs.tasks.text_extraction.lighton import LightOnTextAPIConfig

        extractor = LightOnTextExtractor(
            backend=LightOnTextAPIConfig(
                api_base="http://localhost:8000/v1"
            )
        )
        result = extractor.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="lightonai/LightOnOCR-2-1B",
        description="Model name on the API server",
    )
    api_base: str = Field(
        default="http://localhost:8000/v1",
        description="API base URL (OpenAI-compatible)",
    )
    api_key: str = Field(
        default="sk-no-key-needed",
        description="API key (dummy value for local servers)",
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate",
    )
