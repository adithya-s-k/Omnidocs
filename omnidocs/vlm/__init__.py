"""
VLM - Shared Vision-Language Model infrastructure.

Provides provider-agnostic VLM inference via litellm.

Example:
    ```python
    from omnidocs.vlm import VLMAPIConfig, vlm_completion

    config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
    result = vlm_completion(config, "Extract text from this image", image)
    ```
"""

from .client import vlm_completion, vlm_structured_completion
from .config import VLMAPIConfig

__all__ = [
    "VLMAPIConfig",
    "vlm_completion",
    "vlm_structured_completion",
]
