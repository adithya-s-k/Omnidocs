"""
Pydantic models for structured extraction outputs.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StructuredOutput(BaseModel):
    """
    Output from structured extraction.

    Contains the extracted data as a validated Pydantic model instance,
    along with metadata about the extraction.
    """

    data: BaseModel = Field(
        ...,
        description="Extracted structured data matching the provided schema.",
    )
    raw_output: Optional[str] = Field(
        default=None,
        description="Raw model output before parsing.",
    )
    image_width: Optional[int] = Field(
        default=None,
        ge=1,
        description="Width of the source image in pixels.",
    )
    image_height: Optional[int] = Field(
        default=None,
        ge=1,
        description="Height of the source image in pixels.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the model used for extraction.",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
