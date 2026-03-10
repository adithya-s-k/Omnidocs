"""LightOn text extractor with multi-backend support.

LightOn OCR is optimized for document text extraction and recognition.
Supports multiple backends: PyTorch, VLLM, MLX, and API.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from PIL import Image

from ....cache import add_reference, get_cache_key, get_cached
from ....utils.cache import get_model_cache_dir
from ..base import BaseTextExtractor
from ..models import OutputFormat, TextOutput
from .utils import simple_post_process

if TYPE_CHECKING:
    from .mlx import LightOnTextMLXConfig
    from .pytorch import LightOnTextPyTorchConfig
    from .vllm import LightOnTextVLLMConfig

# Type alias for all backend configs
LightOnTextBackendConfig = Union[
    "LightOnTextPyTorchConfig",
    "LightOnTextVLLMConfig",
    "LightOnTextMLXConfig",
]


class LightOnTextExtractor(BaseTextExtractor):
    """
    LightOn text extractor with multi-backend support.

    LightOn OCR is optimized for document text extraction with multi-lingual capabilities.

    Supports multiple backends:
    - PyTorch (HuggingFace Transformers)
    - VLLM (high-throughput GPU)
    - MLX (Apple Silicon)
    - API (VLLM OpenAI-compatible server)

    Example:
        ```python
        from omnidocs.tasks.text_extraction import LightOnTextExtractor
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        # PyTorch backend
        extractor = LightOnTextExtractor(
            backend=LightOnTextPyTorchConfig(device="cuda", torch_dtype="bfloat16")
        )
        result = extractor.extract(image)
        print(result.content)

        # VLLM backend for high-throughput inference
        from omnidocs.tasks.text_extraction.lighton import LightOnTextVLLMConfig

        extractor = LightOnTextExtractor(
            backend=LightOnTextVLLMConfig(gpu_memory_utilization=0.85)
        )
        result = extractor.extract(image)
        ```
    """

    def __init__(self, backend: LightOnTextBackendConfig):
        """
        Initialize LightOn text extractor.

        Args:
            backend: Backend configuration (PyTorch, VLLM, MLX, or API)
        """
        self.backend_config = backend
        self._client = None
        self._processor = None
        self._loaded = False
        self._sampling_params_class = None
        self._mlx_config = None
        self._apply_chat_template = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model based on backend config.

        Uses unified model cache with reference counting to share models.
        """
        config_type = type(self.backend_config).__name__

        # Check cache first (except for API backend which has no model to cache)
        cache_key = get_cache_key(self.backend_config)
        self._cache_key = cache_key
        cached = get_cached(cache_key)
        if cached is not None:
            client_data = cached
            if isinstance(client_data, tuple):
                self._client, self._processor = client_data
            else:
                self._client = client_data
            add_reference(cache_key, self)
            self._loaded = True
            return

        # Load model based on backend type
        if config_type == "LightOnTextPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "LightOnTextVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "LightOnTextMLXConfig":
            self._load_mlx_backend()
        else:
            raise TypeError(f"Unknown backend config: {config_type}")

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        import torch
        from transformers import AutoProcessor, LightOnOcrForConditionalGeneration

        config = self.backend_config
        cache_dir = get_model_cache_dir(config.cache_dir)

        # Determine device
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.bfloat16 if device == "cuda" else torch.float32,
        }
        dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        # Load model
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": dtype,
            "cache_dir": cache_dir,
        }

        if device == "cuda" and config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif device == "cuda":
            model_kwargs["attn_implementation"] = "sdpa"

        if config.device_map:
            model_kwargs["device_map"] = config.device_map

        print(f"Loading LightOn model from {config.model}...")
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            config.model,
            **model_kwargs,
        )

        if not config.device_map:
            model = model.to(device)
        model = model.eval()

        processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            cache_dir=cache_dir,
        )

        self._client = _TransformersClient(model, processor, config.max_new_tokens)

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend."""    
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        config = self.backend_config
        cache_dir = get_model_cache_dir()
        download_dir = config.download_dir or str(cache_dir)

        self._client = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            trust_remote_code=config.trust_remote_code,
            enforce_eager=config.enforce_eager,
            download_dir=download_dir,
            disable_custom_all_reduce=config.disable_custom_all_reduce,
            limit_mm_per_prompt={"image": 1},
        )
        self._processor = AutoProcessor.from_pretrained(  # ← load once here
            config.model,
            cache_dir=str(cache_dir),
        )
        self._sampling_params_class = SamplingParams

    # def _load_mlx_backend(self) -> None:
    #     """Load MLX backend."""
    #     from mlx_vlm import load

    #     config = self.backend_config
    #     print(f"Loading MLX model from {config.model}...")
    #     model, processor = load(config.model, cache_dir=config.cache_dir)
    #     self._client = _MLXClient(model, processor, config.max_tokens)

    def _load_mlx_backend(self) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        config = self.backend_config
        if config.cache_dir:
            import os
            os.environ["HF_HOME"] = config.cache_dir

        model, processor = load(config.model)
        self._client = _MLXClient(model, processor, config.max_tokens)
        self._mlx_config = load_config(config.model)
        self._apply_chat_template = apply_chat_template

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        """
        Extract text from an image.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            output_format: Desired output format ('html' or 'markdown')

        Returns:
            TextOutput with extracted text content
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        # Prepare image
        image_obj = self._prepare_image(image)

        # Run extraction
        config_type = type(self.backend_config).__name__

        if config_type == "LightOnTextPyTorchConfig":
            raw_output = self._extract_pytorch(image_obj)
        elif config_type == "LightOnTextVLLMConfig":
            raw_output = self._extract_vllm(image_obj)
        elif config_type == "LightOnTextMLXConfig":
            raw_output = self._extract_mlx(image_obj)
        else:
            raise TypeError(f"Unknown backend: {config_type}")

        # Post-process output
        content = simple_post_process(raw_output)

        # Convert to desired format
        if output_format == "html":
            content = self._markdown_to_html(content)

        return TextOutput(
            content=content,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            plain_text=content if output_format == "markdown" else self._html_to_text(content),
            image_width=image_obj.width,
            image_height=image_obj.height,
            model_name="lighton-ocr",
        )

    def _extract_pytorch(self, image: Image.Image) -> str:
        """Extract text using PyTorch backend."""
        # Format conversation for LightOn
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe this document."},
                ],
            }
        ]

        # Apply chat template
        prompt = self._client.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare inputs
        inputs = self._client.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        # Move to device and generate
        device = self._client.model.device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            output_ids = self._client.model.generate(
                **inputs,
                max_new_tokens=self._client.max_new_tokens,
            )

        # Decode
        raw_output = self._client.processor.decode(
            output_ids[0],
            skip_special_tokens=False,
        )

        return raw_output

    def _extract_vllm(self, image: Image.Image) -> str:
        """Extract text using VLLM backend."""
        from transformers import AutoProcessor

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe this document."},
                ],
            }
        ]

        # Build prompt the same way PyTorch backend does
        if self._processor is None:
            cache_dir = get_model_cache_dir()
            self._processor = AutoProcessor.from_pretrained(
                self.backend_config.model,
                cache_dir=str(cache_dir),
            )

        prompt = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = self._sampling_params_class(
            max_tokens=self.backend_config.max_tokens,
            temperature=0.2,
            top_p=0.9,
        )

        outputs = self._client.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text

    # def _extract_mlx(self, image: Image.Image) -> str:
    #     """Extract text using MLX backend."""
    #     from mlx_vlm import generate

    #     prompt = "Transcribe this document."
    #     response = generate(
    #         model=self._client.model,
    #         processor=self._client.processor,
    #         prompt=prompt,
    #         image=image,
    #         max_tokens=self._client.max_tokens,
    #     )

    #     return response

    def _extract_mlx(self, image: Image.Image) -> str:
        import os
        import tempfile
        from mlx_vlm import generate

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            image.save(f, format="PNG")

        try:
            prompt = self._apply_chat_template(
                self._client.processor,
                self._mlx_config,
                "Transcribe this document.",
                num_images=1,
            )
            result = generate(
                self._client.model,
                self._client.processor,
                prompt,
                [temp_path],
                max_tokens=self._client.max_tokens,
                verbose=False,
            )
            return result.text if hasattr(result, "text") else str(result)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML (basic conversion)."""
        # Basic markdown to HTML conversion
        html = markdown_text.replace("\n", "<br>\n")
        return f"<div>{html}</div>"

    def _html_to_text(self, html_text: str) -> str:
        """Convert HTML to plain text."""
        from html.parser import HTMLParser

        class MLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.strict = False
                self.convert_charrefs = True
                self.text = []

            def handle_data(self, d):
                self.text.append(d)

            def get_data(self):
                return "".join(self.text)

        stripper = MLStripper()
        stripper.feed(html_text)
        return stripper.get_data()


class _TransformersClient:
    """Internal client wrapper for PyTorch/HuggingFace backend."""

    def __init__(self, model, processor, max_new_tokens: int):
        """Initialize wrapper."""
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens


class _MLXClient:
    """Internal client wrapper for MLX backend."""

    def __init__(self, model, processor, max_tokens: int):
        """Initialize wrapper."""
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens
