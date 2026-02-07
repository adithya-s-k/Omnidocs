"""Granite Docling text extractor with multi-backend support."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from PIL import Image

from omnidocs.cache import add_reference, get_cache_key, get_cached, set_cached
from omnidocs.tasks.text_extraction.base import BaseTextExtractor
from omnidocs.tasks.text_extraction.models import OutputFormat, TextOutput
from omnidocs.utils.cache import get_model_cache_dir

if TYPE_CHECKING:
    from .api import GraniteDoclingTextAPIConfig
    from .mlx import GraniteDoclingTextMLXConfig
    from .pytorch import GraniteDoclingTextPyTorchConfig
    from .vllm import GraniteDoclingTextVLLMConfig

GraniteDoclingTextBackendConfig = Union[
    "GraniteDoclingTextPyTorchConfig",
    "GraniteDoclingTextVLLMConfig",
    "GraniteDoclingTextMLXConfig",
    "GraniteDoclingTextAPIConfig",
]

# Granite Docling prompt
GRANITE_DOCLING_PROMPT = "Convert this page to docling."


class GraniteDoclingTextExtractor(BaseTextExtractor):
    """
    Granite Docling text extractor supporting PyTorch, VLLM, MLX, and API backends.

    Granite Docling is IBM's compact vision-language model optimized for document
    conversion. It outputs DocTags format which is converted to Markdown using
    the docling_core library.

    Example:
        >>> from omnidocs.tasks.text_extraction.granitedocling import (
        ...     GraniteDoclingTextExtractor,
        ...     GraniteDoclingTextPyTorchConfig,
        ... )
        >>> config = GraniteDoclingTextPyTorchConfig(device="cuda")
        >>> extractor = GraniteDoclingTextExtractor(backend=config)
        >>> result = extractor.extract(image, output_format="markdown")
        >>> print(result.content)
    """

    def __init__(self, backend: GraniteDoclingTextBackendConfig):
        """
        Initialize Granite Docling extractor with backend configuration.

        Args:
            backend: Backend configuration (PyTorch, VLLM, MLX, or API config)
        """
        self.backend_config = backend
        self._backend: Any = None
        self._processor: Any = None
        self._loaded: bool = False

        # Backend-specific helpers
        self._mlx_config: Any = None
        self._apply_chat_template: Any = None
        self._generate: Any = None
        self._sampling_params_class: Any = None
        self._device: str = "cpu"

        self._load_model()

    def _load_model(self) -> None:
        """Load model based on backend config type.

        Uses unified model cache with reference counting to share models.
        """
        config_type = type(self.backend_config).__name__

        # Check cache first (skip API backend which has no model to cache)
        if config_type != "GraniteDoclingTextAPIConfig":
            cache_key = get_cache_key(self.backend_config)
            self._cache_key = cache_key
            cached = get_cached(cache_key)
            if cached is not None:
                self._backend, self._processor = cached
                add_reference(cache_key, self)
                # Re-import lightweight helpers needed for inference
                if config_type == "GraniteDoclingTextVLLMConfig":
                    from vllm import SamplingParams

                    self._sampling_params_class = SamplingParams
                elif config_type == "GraniteDoclingTextMLXConfig":
                    from mlx_vlm import generate
                    from mlx_vlm.prompt_utils import apply_chat_template
                    from mlx_vlm.utils import load_config

                    self._mlx_config = load_config(self.backend_config.model)
                    self._apply_chat_template = apply_chat_template
                    self._generate = generate
                self._loaded = True
                return

        if config_type == "GraniteDoclingTextPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "GraniteDoclingTextVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "GraniteDoclingTextMLXConfig":
            self._load_mlx_backend()
        elif config_type == "GraniteDoclingTextAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(
                f"Unknown backend config: {config_type}. "
                "Expected one of: GraniteDoclingTextPyTorchConfig, "
                "GraniteDoclingTextVLLMConfig, GraniteDoclingTextMLXConfig, "
                "GraniteDoclingTextAPIConfig"
            )

        # Cache the loaded model (skip API)
        if config_type != "GraniteDoclingTextAPIConfig":
            set_cached(cache_key, (self._backend, self._processor), owner=self)

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch backend with HuggingFace transformers."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requires torch and transformers. Install with: uv add torch transformers accelerate"
            ) from e

        config = self.backend_config
        cache_dir = get_model_cache_dir(config.cache_dir)

        # Resolve device
        if config.device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        elif config.device == "mps" and not torch.backends.mps.is_available():
            self._device = "cpu"
        else:
            self._device = config.device

        # Model kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": str(cache_dir),
        }

        if config.device_map:
            model_kwargs["device_map"] = config.device_map

        if config.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._processor = AutoProcessor.from_pretrained(config.model, cache_dir=str(cache_dir))
        self._backend = AutoModelForImageTextToText.from_pretrained(config.model, **model_kwargs)

        if config.device_map is None:
            self._backend = self._backend.to(self._device)

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend for high-throughput inference."""
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "VLLM backend requires vllm and transformers. Install with: uv add vllm transformers"
            ) from e

        config = self.backend_config
        cache_dir = get_model_cache_dir()

        # Use config download_dir or default cache
        download_dir = config.download_dir or str(cache_dir)

        llm_kwargs: dict[str, Any] = {
            "model": config.model,
            "revision": config.revision,  # IMPORTANT: "untied" for VLLM
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "disable_custom_all_reduce": config.disable_custom_all_reduce,
            "limit_mm_per_prompt": {"image": config.limit_mm_per_prompt},
            "download_dir": download_dir,
        }

        if config.enforce_eager:
            llm_kwargs["enforce_eager"] = True

        # Fast boot configuration
        if config.fast_boot:
            try:
                from vllm.config.compilation import (
                    CompilationConfig,
                    CompilationMode,
                    CUDAGraphMode,
                )

                llm_kwargs["compilation_config"] = CompilationConfig(
                    mode=CompilationMode.NONE,
                    cudagraph_mode=CUDAGraphMode.NONE,
                )
            except ImportError:
                pass  # Older VLLM versions

        self._backend = LLM(**llm_kwargs)
        self._processor = AutoProcessor.from_pretrained(config.model, cache_dir=str(cache_dir))
        self._sampling_params_class = SamplingParams

    def _load_mlx_backend(self) -> None:
        """Load MLX backend for Apple Silicon."""
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as e:
            raise ImportError("MLX backend requires mlx and mlx-vlm. Install with: uv add mlx mlx-vlm") from e

        config = self.backend_config

        # Set HF_HOME if cache_dir is specified (MLX respects HF_HOME)
        if config.cache_dir:
            import os

            os.environ["HF_HOME"] = config.cache_dir

        self._backend, self._processor = load(config.model)
        self._mlx_config = load_config(config.model)
        self._apply_chat_template = apply_chat_template
        self._generate = generate

    def _load_api_backend(self) -> None:
        """Load API backend for remote inference."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("API backend requires openai. Install with: uv add openai") from e

        config = self.backend_config
        client_kwargs: dict[str, Any] = {
            "base_url": config.base_url,
            "api_key": config.api_key,
        }
        if config.extra_headers:
            client_kwargs["default_headers"] = config.extra_headers

        self._backend = OpenAI(**client_kwargs)

    def _convert_doctags_to_markdown(self, doctags: str, image: Image.Image) -> str:
        """Convert DocTags output to Markdown using docling_core."""
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.document import DocTagsDocument

            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
            return doc.export_to_markdown()
        except Exception:
            # Fallback: return raw doctags if conversion fails
            return doctags

    def _prepare_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Convert input to PIL Image."""
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        return pil_image

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        """
        Extract text from an image using Granite Docling.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            output_format: Output format ("markdown" or "html")

        Returns:
            TextOutput with extracted content
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if output_format not in ("html", "markdown"):
            raise ValueError(f"Invalid output_format: {output_format}")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Dispatch to backend-specific inference
        config_type = type(self.backend_config).__name__

        if config_type == "GraniteDoclingTextPyTorchConfig":
            raw_output = self._infer_pytorch(pil_image)
        elif config_type == "GraniteDoclingTextVLLMConfig":
            raw_output = self._infer_vllm(pil_image)
        elif config_type == "GraniteDoclingTextMLXConfig":
            raw_output = self._infer_mlx(pil_image)
        elif config_type == "GraniteDoclingTextAPIConfig":
            raw_output = self._infer_api(pil_image)
        else:
            raise RuntimeError(f"Unknown backend: {config_type}")

        # Convert DocTags to Markdown
        markdown_output = self._convert_doctags_to_markdown(raw_output, pil_image)

        # For HTML output, wrap in basic HTML structure
        if output_format == "html":
            content = f"<html><body>\n{markdown_output}\n</body></html>"
        else:
            content = markdown_output

        return TextOutput(
            content=content,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            plain_text=self._extract_plain_text(markdown_output),
            image_width=width,
            image_height=height,
            model_name=f"Granite-Docling-258M ({config_type.replace('Config', '')})",
        )

    def _infer_pytorch(self, image: Image.Image) -> str:
        """PyTorch inference using HuggingFace transformers."""
        import torch

        config = self.backend_config

        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": GRANITE_DOCLING_PROMPT},
                ],
            },
        ]

        # Apply chat template
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process inputs
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(self._backend.device)

        # Generate
        with torch.no_grad():
            generated_ids = self._backend.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else None,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode (exclude prompt)
        prompt_length = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, prompt_length:]
        doctags = self._processor.batch_decode(output_ids, skip_special_tokens=False)[0].strip()

        return doctags

    def _infer_vllm(self, image: Image.Image) -> str:
        """VLLM inference for high-throughput."""
        config = self.backend_config

        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": GRANITE_DOCLING_PROMPT},
                ],
            },
        ]

        # Apply chat template
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        # Sampling params
        sampling_params = self._sampling_params_class(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            skip_special_tokens=False,
        )

        # Generate
        outputs = self._backend.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text

    def _infer_mlx(self, image: Image.Image) -> str:
        """MLX inference for Apple Silicon."""
        import os
        import tempfile

        config = self.backend_config

        # Save image temporarily (mlx_vlm requires file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            temp_path = f.name

        try:
            formatted_prompt = self._apply_chat_template(
                self._processor,
                self._mlx_config,
                GRANITE_DOCLING_PROMPT,
                num_images=1,
            )

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
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _infer_api(self, image: Image.Image) -> str:
        """API inference via OpenAI-compatible endpoint."""
        import base64
        from io import BytesIO

        config = self.backend_config

        # Encode image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Create request
        response = self._backend.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                        {"type": "text", "text": GRANITE_DOCLING_PROMPT},
                    ],
                }
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
        )

        return response.choices[0].message.content

    def _extract_plain_text(self, content: str) -> str:
        """Extract plain text from markdown content."""
        # Remove markdown formatting
        text = re.sub(r"#+ ", "", content)  # Headers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # Italic
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # Links
        text = re.sub(r"```[\s\S]*?```", "", text)  # Code blocks
        text = re.sub(r"`(.+?)`", r"\1", text)  # Inline code
        text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple newlines

        return text.strip()
