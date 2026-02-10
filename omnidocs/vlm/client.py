"""
VLM completion utilities using litellm for provider-agnostic inference.
"""

import base64
import io
from typing import Any, Dict

from PIL import Image
from pydantic import BaseModel

from .config import VLMAPIConfig


def _encode_image(image: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _build_kwargs(config: VLMAPIConfig, messages: list) -> Dict[str, Any]:
    """Build litellm.completion kwargs from config."""
    # Azure newer models (GPT-4o, GPT-5) require max_completion_tokens
    token_key = "max_completion_tokens" if config.model.startswith("azure/") else "max_tokens"
    kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        token_key: config.max_tokens,
        "temperature": config.temperature,
        "timeout": config.timeout,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_version:
        kwargs["api_version"] = config.api_version
    if config.extra_headers:
        kwargs["extra_headers"] = config.extra_headers
    return kwargs


def vlm_completion(config: VLMAPIConfig, prompt: str, image: Image.Image) -> str:
    """
    Send image + prompt to any VLM via litellm. Returns raw text.

    Args:
        config: VLM API configuration.
        prompt: Text prompt to send with the image.
        image: PIL Image to send.

    Returns:
        Raw text response from the model.
    """
    import litellm

    b64 = _encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]

    kwargs = _build_kwargs(config, messages)
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


def vlm_structured_completion(
    config: VLMAPIConfig,
    prompt: str,
    image: Image.Image,
    response_schema: type[BaseModel],
) -> BaseModel:
    """
    Send image + prompt, get structured Pydantic output.

    Tries two strategies:
    1. litellm's native response_format (works with OpenAI, Gemini, etc.)
    2. Fallback: prompt-based JSON extraction for providers that don't
       support response_format (OpenRouter, some open-source models)

    Args:
        config: VLM API configuration.
        prompt: Text prompt to send with the image.
        image: PIL Image to send.
        response_schema: Pydantic model class for structured output.

    Returns:
        Validated instance of response_schema.
    """
    import json

    import litellm

    b64 = _encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]

    # Strategy 1: Try response_format (native structured output)
    kwargs = _build_kwargs(config, messages)
    kwargs["response_format"] = response_schema
    try:
        response = litellm.completion(**kwargs)
        raw = response.choices[0].message.content
        return response_schema.model_validate_json(raw)
    except Exception:
        pass

    # Strategy 2: Fallback — prompt for JSON, parse manually
    schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
    json_prompt = (
        f"{prompt}\n\n"
        f"Respond with ONLY valid JSON matching this schema (no markdown fencing, no extra text):\n"
        f"{schema_json}"
    )
    fallback_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]
    kwargs = _build_kwargs(config, fallback_messages)
    response = litellm.completion(**kwargs)
    raw = response.choices[0].message.content

    # Strip markdown fencing if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    return response_schema.model_validate_json(text)
