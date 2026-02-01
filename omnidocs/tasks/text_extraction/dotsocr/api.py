"""API backend configuration for Dots OCR (VLLM online server)."""

from typing import Optional

from pydantic import BaseModel, Field


class DotsOCRAPIConfig(BaseModel):
    """
    API backend configuration for Dots OCR.

    This config is for accessing a deployed VLLM server via OpenAI-compatible API.
    Typically used with modal_dotsocr_vllm_online.py deployment.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

        config = DotsOCRAPIConfig(
                model="dotsocr",
                api_base="https://your-modal-app.modal.run/v1",
                api_key="optional-key",
            )
        extractor = DotsOCRTextExtractor(backend=config)
        ```
    """

    model: str = Field(default="dotsocr", description="Model identifier for the API endpoint")
    api_base: str = Field(
        ...,
        description="Base URL for the VLLM server API (e.g., https://app.modal.run/v1)",
    )
    api_key: Optional[str] = Field(default=None, description="API key for authentication (if required)")
    timeout: int = Field(default=120, ge=10, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retry attempts")

    class Config:
        extra = "forbid"  # Raise error on unknown params
