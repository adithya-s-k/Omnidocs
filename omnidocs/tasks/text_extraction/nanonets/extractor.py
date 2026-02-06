"""
Nanonets OCR2-3B text extractor.

A Vision-Language Model for extracting text from document images
with support for tables (HTML), equations (LaTeX), and image captions.

Supports PyTorch and VLLM backends.

Example:
    ```python
    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

    extractor = NanonetsTextExtractor(
            backend=NanonetsTextPyTorchConfig()
        )
    result = extractor.extract(image)
    print(result.content)
    ```
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Union

import numpy as np
from PIL import Image

from omnidocs.utils.cache import get_model_cache_dir

from ..base import BaseTextExtractor
from ..models import OutputFormat, TextOutput

if TYPE_CHECKING:
    from .mlx import NanonetsTextMLXConfig
    from .pytorch import NanonetsTextPyTorchConfig
    from .vllm import NanonetsTextVLLMConfig

# Union type for all supported backends
NanonetsTextBackendConfig = Union[
    "NanonetsTextPyTorchConfig",
    "NanonetsTextVLLMConfig",
    "NanonetsTextMLXConfig",
]

# Nanonets OCR2 extraction prompt
NANONETS_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, "
    "add a small description of the image inside the <img></img> tag; otherwise, "
    "add the image caption inside <img></img>. "
    "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. "
    "Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)


class NanonetsTextExtractor(BaseTextExtractor):
    """
    Nanonets OCR2-3B Vision-Language Model text extractor.

    Extracts text from document images with support for:
    - Tables (output as HTML)
    - Equations (output as LaTeX)
    - Image captions (wrapped in <img></img> tags)
    - Watermarks (wrapped in <watermark></watermark> tags)
    - Page numbers (wrapped in <page_number></page_number> tags)
    - Checkboxes (using ☐ and ☑ symbols)

    Supports PyTorch, VLLM, and MLX backends.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import NanonetsTextExtractor
        from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

        # Initialize with PyTorch backend
        extractor = NanonetsTextExtractor(
                backend=NanonetsTextPyTorchConfig()
            )

        # Extract text
        result = extractor.extract(image)
        print(result.content)
        ```
    """

    def __init__(self, backend: NanonetsTextBackendConfig):
        """
        Initialize Nanonets text extractor.

        Args:
            backend: Backend configuration. One of:
                - NanonetsTextPyTorchConfig: PyTorch/HuggingFace backend
                - NanonetsTextVLLMConfig: VLLM high-throughput backend
                - NanonetsTextMLXConfig: MLX backend for Apple Silicon
        """
        self.backend_config = backend
        self._backend: Any = None
        self._processor: Any = None
        self._loaded = False

        # Backend-specific helpers
        self._process_vision_info: Any = None
        self._sampling_params_class: Any = None
        self._device: str = "cpu"

        # MLX-specific helpers
        self._mlx_config: Any = None
        self._apply_chat_template: Any = None
        self._generate: Any = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load appropriate backend based on config type."""
        config_type = type(self.backend_config).__name__

        if config_type == "NanonetsTextPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "NanonetsTextVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "NanonetsTextMLXConfig":
            self._load_mlx_backend()
        else:
            raise TypeError(
                f"Unknown backend config: {config_type}. "
                f"Expected one of: NanonetsTextPyTorchConfig, NanonetsTextVLLMConfig, NanonetsTextMLXConfig"
            )

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requires torch and transformers. Install with: uv add torch transformers accelerate"
            ) from e

        config = self.backend_config
        cache_dir = get_model_cache_dir(config.cache_dir)

        # Resolve device
        self._device = self._resolve_device(config.device)

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": (config.torch_dtype if config.torch_dtype != "auto" else "auto"),
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": str(cache_dir),
        }
        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "sdpa"

        self._backend = AutoModelForImageTextToText.from_pretrained(config.model, **model_kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            cache_dir=str(cache_dir),
        )

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend."""
        # Set VLLM_USE_V1=0 for stability
        os.environ["VLLM_USE_V1"] = "0"

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
        cache_dir = get_model_cache_dir()

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

        Note: Nanonets OCR2 produces a unified output format that includes
        tables as HTML and equations as LaTeX inline. The output_format
        parameter is accepted for API compatibility but does not change
        the output structure.

        Args:
            image: Input image as:
                - PIL.Image.Image: PIL image object
                - np.ndarray: Numpy array (HWC format, RGB)
                - str or Path: Path to image file
            output_format: Accepted for API compatibility (default: "markdown")

        Returns:
            TextOutput containing extracted text content

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image format is not supported
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Run inference based on backend
        config_type = type(self.backend_config).__name__
        if config_type == "NanonetsTextPyTorchConfig":
            raw_output = self._infer_pytorch(pil_image)
        elif config_type == "NanonetsTextVLLMConfig":
            raw_output = self._infer_vllm(pil_image)
        elif config_type == "NanonetsTextMLXConfig":
            raw_output = self._infer_mlx(pil_image)
        else:
            raise RuntimeError(f"Unknown backend: {config_type}")

        # Clean output
        cleaned_output = raw_output.replace("<|im_end|>", "").strip()

        return TextOutput(
            content=cleaned_output,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            plain_text=cleaned_output,
            image_width=width,
            image_height=height,
            model_name=f"Nanonets-OCR2-3B ({type(self.backend_config).__name__})",
        )

    def _infer_pytorch(self, image: Image.Image) -> str:
        """Run inference with PyTorch backend."""
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": NANONETS_PROMPT}]}]
        prompt_full = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self._processor(
            text=[prompt_full],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self._backend.device)

        input_len = inputs["input_ids"].shape[-1]

        config = self.backend_config
        with torch.no_grad():
            do_sample = config.temperature > 0.0
            output_ids = self._backend.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=do_sample,
                temperature=config.temperature if do_sample else None,
            )

        # Decode only new tokens
        generation = output_ids[0][input_len:]
        return self._processor.decode(generation, skip_special_tokens=True)

    def _infer_vllm(self, image: Image.Image) -> str:
        """Run inference with VLLM backend."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": NANONETS_PROMPT},
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

    def _infer_mlx(self, image: Image.Image) -> str:
        """Run inference with MLX backend."""
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
                image.save(f, format="PNG")

            formatted_prompt = self._apply_chat_template(
                self._processor, self._mlx_config, NANONETS_PROMPT, num_images=1
            )

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
