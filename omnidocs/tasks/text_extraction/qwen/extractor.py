"""
Qwen3-VL text extractor.

A Vision-Language Model for extracting text from document images
as structured HTML or Markdown.

Supports PyTorch, VLLM, MLX, and API backends.

Example:
    >>> from omnidocs.tasks.text_extraction import QwenTextExtractor
    >>> from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    >>>
    >>> extractor = QwenTextExtractor(
    ...     backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
    ... )
    >>> result = extractor.extract(image, output_format="markdown")
    >>> print(result.content)
"""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Union

import numpy as np
from PIL import Image

from ..base import BaseTextExtractor
from ..models import OutputFormat, TextOutput

if TYPE_CHECKING:
    from .api import QwenTextAPIConfig
    from .mlx import QwenTextMLXConfig
    from .pytorch import QwenTextPyTorchConfig
    from .vllm import QwenTextVLLMConfig

# Union type for all supported backends
QwenTextBackendConfig = Union[
    "QwenTextPyTorchConfig",
    "QwenTextVLLMConfig",
    "QwenTextMLXConfig",
    "QwenTextAPIConfig",
]

# Qwen3-VL document parsing prompts
QWEN_PROMPTS = {
    "html": "qwenvl html",
    "markdown": "qwenvl markdown",
}


def _get_model_cache_dir() -> Path:
    """
    Get model cache directory from environment or default.

    Checks OMNIDOCS_MODEL_CACHE environment variable first,
    falls back to ~/.omnidocs/models.
    """
    cache_dir = os.environ.get("OMNIDOCS_MODEL_CACHE", os.path.expanduser("~/.omnidocs/models"))
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_html_output(html_output: str) -> str:
    """
    Remove bounding box attributes from HTML output.

    Qwen3-VL HTML output includes data-bbox attributes with coordinates.
    This function removes them for cleaner output.
    """
    cleaned = re.sub(r'\s*data-bbox="[^"]*"', "", html_output)
    cleaned = re.sub(r"\s+>", ">", cleaned)
    return cleaned


def _clean_markdown_output(md_output: str) -> str:
    """
    Remove coordinate annotations from Markdown output.

    Qwen3-VL Markdown format includes:
    - <!-- Table (x1, y1, x2, y2) --> before tables
    - <!-- Image (x1, y1, x2, y2) --> for image placeholders
    """
    cleaned = re.sub(r"<!--\s*(Table|Image)\s*\([^)]+\)\s*-->\n?", "", md_output)
    return cleaned


def _extract_plain_text(output: str, output_format: str) -> str:
    """Extract plain text from HTML or Markdown output."""
    if output_format == "html":
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", output)
    else:
        # Remove markdown formatting
        text = re.sub(r"```[^`]*```", "", output)  # Code blocks
        text = re.sub(r"<!--[^>]+-->", "", text)  # Comments
        text = re.sub(r"\|[-:]+\|", "", text)  # Table separators
        text = re.sub(r"[#*_`]", "", text)  # Formatting chars

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class QwenTextExtractor(BaseTextExtractor):
    """
    Qwen3-VL Vision-Language Model text extractor.

    Extracts text from document images and outputs as structured
    HTML or Markdown. Uses Qwen3-VL's built-in document parsing prompts.

    Supports PyTorch, VLLM, MLX, and API backends.

    Example:
        >>> from omnidocs.tasks.text_extraction import QwenTextExtractor
        >>> from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
        >>>
        >>> # Initialize with PyTorch backend
        >>> extractor = QwenTextExtractor(
        ...     backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
        ... )
        >>>
        >>> # Extract as Markdown
        >>> result = extractor.extract(image, output_format="markdown")
        >>> print(result.content)
        >>>
        >>> # Extract as HTML
        >>> result = extractor.extract(image, output_format="html")
        >>> print(result.content)
    """

    def __init__(self, backend: QwenTextBackendConfig):
        """
        Initialize Qwen text extractor.

        Args:
            backend: Backend configuration. One of:
                - QwenTextPyTorchConfig: PyTorch/HuggingFace backend
                - QwenTextVLLMConfig: VLLM high-throughput backend
                - QwenTextMLXConfig: MLX backend for Apple Silicon
                - QwenTextAPIConfig: API backend (OpenRouter, etc.)
        """
        self.backend_config = backend
        self._backend: Any = None
        self._processor: Any = None
        self._loaded = False

        # Backend-specific helpers
        self._process_vision_info: Any = None
        self._sampling_params_class: Any = None
        self._mlx_config: Any = None
        self._apply_chat_template: Any = None
        self._generate: Any = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load appropriate backend based on config type."""
        config_type = type(self.backend_config).__name__

        if config_type == "QwenTextPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "QwenTextVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "QwenTextMLXConfig":
            self._load_mlx_backend()
        elif config_type == "QwenTextAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(
                f"Unknown backend config: {config_type}. "
                f"Expected one of: QwenTextPyTorchConfig, QwenTextVLLMConfig, "
                f"QwenTextMLXConfig, QwenTextAPIConfig"
            )

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requires torch, transformers, and qwen-vl-utils. "
                "Install with: uv add torch transformers accelerate qwen-vl-utils"
            ) from e

        config = self.backend_config
        cache_dir = _get_model_cache_dir()

        # Resolve device
        device = self._resolve_device(config.device)

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": config.torch_dtype if config.torch_dtype != "auto" else "auto",
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": str(cache_dir),
        }
        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._backend = AutoModelForImageTextToText.from_pretrained(config.model, **model_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            cache_dir=str(cache_dir),
        )
        self._process_vision_info = process_vision_info
        self._device = device

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend."""
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "VLLM backend requires vllm, torch, transformers, and qwen-vl-utils. "
                "Install with: uv add vllm torch transformers qwen-vl-utils"
            ) from e

        config = self.backend_config
        cache_dir = _get_model_cache_dir()

        # Use config download_dir or default cache
        download_dir = config.download_dir or str(cache_dir)

        self._backend = LLM(
            model=config.model,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            download_dir=download_dir,
            disable_custom_all_reduce=config.disable_custom_all_reduce,
        )
        self._processor = AutoProcessor.from_pretrained(config.model, cache_dir=str(cache_dir))
        self._process_vision_info = process_vision_info
        self._sampling_params_class = SamplingParams

    def _load_mlx_backend(self) -> None:
        """Load MLX backend (Apple Silicon)."""
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as e:
            raise ImportError("MLX backend requires mlx and mlx-vlm. Install with: uv add mlx mlx-vlm") from e

        config = self.backend_config

        self._backend, self._processor = load(config.model)
        self._mlx_config = load_config(config.model)
        self._apply_chat_template = apply_chat_template
        self._generate = generate

    def _load_api_backend(self) -> None:
        """Load API backend."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("API backend requires openai. Install with: uv add openai") from e

        config = self.backend_config

        client_kwargs: Dict[str, Any] = {
            "base_url": config.base_url,
            "api_key": config.api_key,
        }
        if config.extra_headers:
            client_kwargs["default_headers"] = config.extra_headers

        self._backend = OpenAI(**client_kwargs)

    def _resolve_device(self, device: str) -> str:
        """Resolve device, auto-detecting if needed."""
        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                return "mps"
            elif device in ("cuda", "mps"):
                # Requested GPU but not available, fall back to CPU
                return "cpu"
            return device
        except ImportError:
            return "cpu"

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        """
        Extract text from an image.

        Args:
            image: Input image as:
                - PIL.Image.Image: PIL image object
                - np.ndarray: Numpy array (HWC format, RGB)
                - str or Path: Path to image file
            output_format: Desired output format:
                - "html": Structured HTML with div elements
                - "markdown": Markdown format

        Returns:
            TextOutput containing extracted text content

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image format or output_format is not supported
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if output_format not in ("html", "markdown"):
            raise ValueError(f"Invalid output_format: {output_format}. Expected 'html' or 'markdown'.")

        # Prepare image
        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Get prompt for output format
        prompt = QWEN_PROMPTS[output_format]

        # Run inference based on backend
        config_type = type(self.backend_config).__name__
        if config_type == "QwenTextPyTorchConfig":
            raw_output = self._infer_pytorch(pil_image, prompt)
        elif config_type == "QwenTextVLLMConfig":
            raw_output = self._infer_vllm(pil_image, prompt)
        elif config_type == "QwenTextMLXConfig":
            raw_output = self._infer_mlx(pil_image, prompt)
        elif config_type == "QwenTextAPIConfig":
            raw_output = self._infer_api(pil_image, prompt)
        else:
            raise RuntimeError(f"Unknown backend: {config_type}")

        # Clean output
        if output_format == "html":
            cleaned_output = _clean_html_output(raw_output)
        else:
            cleaned_output = _clean_markdown_output(raw_output)

        # Extract plain text
        plain_text = _extract_plain_text(raw_output, output_format)

        return TextOutput(
            content=cleaned_output,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            plain_text=plain_text,
            image_width=width,
            image_height=height,
            model_name=f"Qwen3-VL ({type(self.backend_config).__name__})",
        )

    # ============= Backend-specific inference methods =============

    def _infer_pytorch(self, image: Image.Image, prompt: str) -> str:
        """Run inference with PyTorch backend."""
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
                image.save(f, format="PNG")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._process_vision_info(messages)

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._backend.device)

            config = self.backend_config
            generated_ids = self._backend.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )

            # Trim to only new tokens
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

            return self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )[0]
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _infer_vllm(self, image: Image.Image, prompt: str) -> str:
        """Run inference with VLLM backend."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, _, _ = self._process_vision_info(messages, return_video_kwargs=True)
        mm_data = {"image": image_inputs} if image_inputs else {}

        config = self.backend_config
        sampling_params = self._sampling_params_class(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        outputs = self._backend.generate(
            [{"prompt": text, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text

    def _infer_mlx(self, image: Image.Image, prompt: str) -> str:
        """Run inference with MLX backend."""
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
                image.save(f, format="PNG")

            formatted_prompt = self._apply_chat_template(self._processor, self._mlx_config, prompt, num_images=1)

            config = self.backend_config
            result = self._generate(
                self._backend,
                self._processor,
                formatted_prompt,
                [temp_path],
                max_tokens=config.max_tokens,
                temp=config.temperature,
                verbose=False,
            )

            return result.text if hasattr(result, "text") else str(result)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _infer_api(self, image: Image.Image, prompt: str) -> str:
        """Run inference with API backend."""
        import base64
        import io

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/png;base64,{img_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        config = self.backend_config
        response = self._backend.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
        )

        return response.choices[0].message.content
