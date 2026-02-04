"""API backend configuration for MinerU VL layout detection."""

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MinerUVLLayoutAPIConfig(BaseModel):
    """
    API backend config for MinerU VL layout detection.

    Example:
        ```python
        from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutAPIConfig

        detector = MinerUVLLayoutDetector(
            backend=MinerUVLLayoutAPIConfig(
                server_url="https://your-server.modal.run"
            )
        )
        result = detector.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    server_url: str = Field(
        ...,
        description="VLLM server URL",
    )
    model_name: str = Field(
        default="mineru-vl",
        description="Model name as configured on the server",
    )
    timeout: int = Field(
        default=300,
        ge=10,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries on failure",
    )
    layout_image_size: Tuple[int, int] = Field(
        default=(1036, 1036),
        description="Resize image to this size for layout detection",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for authentication",
    )
