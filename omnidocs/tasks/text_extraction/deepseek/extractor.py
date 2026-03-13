"""
DeepSeek-OCR / DeepSeek-OCR-2 text extractor.

DeepSeek-OCR (Oct 2024, arXiv:2510.18234) — v1, MIT, 3B params
DeepSeek-OCR-2 (Jan 2026, arXiv:2601.20552) — v2, Apache 2.0, 3B params, "Visual Causal Flow"

Both models share the same inference interface:
  - AutoModel + AutoTokenizer (NOT AutoModelForCausalLM / AutoProcessor)
  - model.infer(tokenizer, prompt, image_file, output_path, ...) for PyTorch
  - Grounding prompt format: "<image>\\n<|grounding|>Convert the document to markdown."

Supported backends: PyTorch, VLLM (official upstream support), MLX, API.

GitHub:
  v1: https://github.com/deepseek-ai/DeepSeek-OCR
  v2: https://github.com/deepseek-ai/DeepSeek-OCR-2

Example:
    ```python
    from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
    from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

    extractor = DeepSeekOCRTextExtractor(
        backend=DeepSeekOCRTextVLLMConfig()  # DeepSeek-OCR-2, VLLM
    )
    result = extractor.extract(image)
    print(result.content)
    ```
"""
import base64
import io
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import litellm
import numpy as np
from PIL import Image

from omnidocs.utils.cache import get_model_cache_dir

from ....cache import add_reference, get_cache_key, get_cached, set_cached
from ..base import BaseTextExtractor
from ..models import OutputFormat, TextOutput

if TYPE_CHECKING:
    from .api import DeepSeekOCRTextAPIConfig
    from .mlx import DeepSeekOCRTextMLXConfig
    from .pytorch import DeepSeekOCRTextPyTorchConfig
    from .vllm import DeepSeekOCRTextVLLMConfig

DeepSeekOCRTextBackendConfig = Union[
    "DeepSeekOCRTextPyTorchConfig",
    "DeepSeekOCRTextVLLMConfig",
    "DeepSeekOCRTextMLXConfig",
    "DeepSeekOCRTextAPIConfig",
]

# Prompt variants — from the official DeepSeek-OCR-2 README
DEEPSEEK_PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr": "<image>\n<|grounding|>OCR this image.",
    "free": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
}


class DeepSeekOCRTextExtractor(BaseTextExtractor):
    """
    DeepSeek-OCR / DeepSeek-OCR-2 text extractor.

    High-accuracy OCR model that reads complex real-world documents (PDFs, forms,
    tables, handwritten/noisy text) and outputs clean Markdown. Uses a hybrid
    vision encoder + causal text decoder — output is structured by the model itself
    rather than post-processed from bounding boxes.

    DeepSeek-OCR-2 ("Visual Causal Flow") is the default — released Jan 2026.

    Supports PyTorch, VLLM (recommended), MLX, and API backends.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
        from omnidocs.tasks.text_extraction.deepseek import (
            DeepSeekOCRTextPyTorchConfig,
            DeepSeekOCRTextVLLMConfig,
        )

        # VLLM — ~2500 tokens/s on A100 (recommended for production)
        extractor = DeepSeekOCRTextExtractor(
            backend=DeepSeekOCRTextVLLMConfig()
        )
        result = extractor.extract(image)
        print(result.content)

        # PyTorch with crop_mode for dense pages
        extractor = DeepSeekOCRTextExtractor(
            backend=DeepSeekOCRTextPyTorchConfig(crop_mode=True)
        )
        ```
    """

    def __init__(self, backend: DeepSeekOCRTextBackendConfig):
        """
        Initialize DeepSeek-OCR extractor.

        Args:
            backend: Backend config. One of:
                - DeepSeekOCRTextPyTorchConfig (local GPU)
                - DeepSeekOCRTextVLLMConfig (recommended, high-throughput)
                - DeepSeekOCRTextMLXConfig (Apple Silicon)
                - DeepSeekOCRTextAPIConfig (Novita AI)
        """
        self.backend_config = backend
        self._backend: Any = None   # model
        self._processor: Any = None  # tokenizer
        self._loaded = False
        self._device: str = "cpu"

        # VLLM helpers
        self._sampling_params_class: Any = None

        # MLX helpers
        self._mlx_config: Any = None
        self._apply_chat_template: Any = None
        self._generate: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """Load appropriate backend with unified model cache."""
        config_type = type(self.backend_config).__name__

        if config_type != "DeepSeekOCRTextAPIConfig":
            cache_key = get_cache_key(self.backend_config)
            self._cache_key = cache_key
            cached = get_cached(cache_key)
            if cached is not None:
                self._backend, self._processor = cached
                add_reference(cache_key, self)
                if config_type == "DeepSeekOCRTextVLLMConfig":
                    from vllm import SamplingParams
                    self._sampling_params_class = SamplingParams
                elif config_type == "DeepSeekOCRTextMLXConfig":
                    from mlx_vlm import generate
                    from mlx_vlm.prompt_utils import apply_chat_template
                    from mlx_vlm.utils import load_config
                    self._mlx_config = load_config(self.backend_config.model)
                    self._apply_chat_template = apply_chat_template
                    self._generate = generate
                self._loaded = True
                return

        dispatch = {
            "DeepSeekOCRTextPyTorchConfig": self._load_pytorch_backend,
            "DeepSeekOCRTextVLLMConfig": self._load_vllm_backend,
            "DeepSeekOCRTextMLXConfig": self._load_mlx_backend,
            "DeepSeekOCRTextAPIConfig": self._load_api_backend,
        }
        loader = dispatch.get(config_type)
        if loader is None:
            raise TypeError(
                f"Unknown backend config: {config_type}. Expected one of: "
                + ", ".join(dispatch.keys())
            )
        loader()

        if config_type != "DeepSeekOCRTextAPIConfig":
            set_cached(cache_key, (self._backend, self._processor), owner=self)

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch backend.

        Both DeepSeek-OCR and DeepSeek-OCR-2 use AutoModel + AutoTokenizer.
        The model provides a .infer() method that handles tiling and output assembly.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requires torch and transformers==4.46.3. "
                "Also install: einops addict easydict. "
                "Optionally: flash-attn==2.7.3 --no-build-isolation (CUDA 11.8). "
                "Install with: uv add torch 'transformers==4.46.3' einops addict easydict"
            ) from e

        config = self.backend_config
        cache_dir = get_model_cache_dir(config.cache_dir)
        self._device = self._resolve_device(config.device)

        attn_impl = "flash_attention_2" if config.use_flash_attention else "eager"

        self._backend = AutoModel.from_pretrained(
            config.model,
            _attn_implementation=attn_impl,
            trust_remote_code=config.trust_remote_code,
            use_safetensors=True,
            cache_dir=str(cache_dir),
        ).eval().to(self._device).to(
            {"bfloat16": __import__("torch").bfloat16,
             "float16": __import__("torch").float16,
             "float32": __import__("torch").float32,
             "auto": __import__("torch").bfloat16}[config.torch_dtype]
        )

        self._processor = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            cache_dir=str(cache_dir),
        )

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend (~2500 tokens/s on A100)."""
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        except ImportError as e:
            raise ImportError(
                "VLLM backend requires vllm>=0.11.1 and transformers==4.46.3. "
                "Install with: uv add vllm torch 'transformers==4.46.3'"
            ) from e

        config = self.backend_config
        cache_dir = get_model_cache_dir()
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
            enable_prefix_caching=config.enable_prefix_caching,   # must be False
            mm_processor_cache_gb=config.mm_processor_cache_gb,   # must be 0
            logits_processors=[NGramPerReqLogitsProcessor],        # required for v1
        )
        self._processor = AutoTokenizer.from_pretrained(config.model, cache_dir=str(cache_dir))
        self._sampling_params_class = SamplingParams

    def _load_mlx_backend(self) -> None:
        """Load MLX backend (Apple Silicon only)."""
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as e:
            raise ImportError(
                "MLX backend requires mlx-vlm. Install with: uv add mlx mlx-vlm"
            ) from e

        config = self.backend_config
        if config.cache_dir:
            os.environ["HF_HOME"] = config.cache_dir

        self._backend, self._processor = load(config.model)
        self._mlx_config = load_config(config.model)
        self._apply_chat_template = apply_chat_template
        self._generate = generate

    def _load_api_backend(self) -> None:
        """API backend — no model to load."""
        pass

    def _resolve_device(self, device: str) -> str:
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device in ("cuda",):
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
        Extract text from a document image.

        DeepSeek-OCR always outputs Markdown-structured text.
        The output_format parameter is accepted for API compatibility.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            output_format: Accepted for API compatibility (default: "markdown")

        Returns:
            TextOutput with extracted Markdown content
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        config_type = type(self.backend_config).__name__
        dispatch = {
            "DeepSeekOCRTextPyTorchConfig": self._infer_pytorch,
            "DeepSeekOCRTextVLLMConfig": self._infer_vllm,
            "DeepSeekOCRTextMLXConfig": self._infer_mlx,
            "DeepSeekOCRTextAPIConfig": self._infer_api,
        }
        raw_output = dispatch[config_type](pil_image)
        cleaned = raw_output.strip()

        return TextOutput(
            content=cleaned,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            plain_text=cleaned,
            image_width=width,
            image_height=height,
            model_name=f"DeepSeek-OCR ({self.backend_config.model}, {config_type})",
        )

    # ============= Backend inference =============

    def _infer_pytorch(self, image: Image.Image) -> str:
        config = self.backend_config
        prompt = DEEPSEEK_PROMPTS["markdown"]
        import contextlib

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.png")
            out_path = os.path.join(tmpdir, "output")
            os.makedirs(out_path, exist_ok=True)
            image.save(img_path)

            with open(os.devnull, "w") as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                self._backend.infer(
                    self._processor,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=out_path,
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                    save_results=True,
                )

            # Walk entire output tree — model saves .mmd files in subdirectories
            texts = []
            for root, _, files in os.walk(out_path):
                for fname in sorted(files):
                    if fname.endswith(".mmd"):  # fix: was .mdd
                        with open(os.path.join(root, fname), encoding="utf-8") as f:
                            texts.append(f.read())

            return "\n\n".join(texts)

    def _infer_vllm(self, image: Image.Image) -> str:
        """VLLM inference."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = DEEPSEEK_PROMPTS["markdown"]
        # DeepSeek-OCR vLLM uses the grounding prompt directly
        mm_data = {"image": [image]}
        config = self.backend_config

        sampling_params = self._sampling_params_class(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            skip_special_tokens=config.skip_special_tokens,
            extra_args=dict(
                ngram_size=config.ngram_size,
                window_size=config.ngram_window_size,
                whitelist_token_ids={128821, 128822},  # <td>, </td> tokens
            ),
        )
        outputs = self._backend.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text

    def _infer_mlx(self, image: Image.Image) -> str:
        """MLX inference (Apple Silicon)."""
        prompt = DEEPSEEK_PROMPTS["markdown"]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            image.save(f, format="PNG")

        try:
            formatted_prompt = self._apply_chat_template(
                self._processor, self._mlx_config, prompt, num_images=1
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
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # def _infer_api(self, image: Image.Image) -> str:
    #     """API inference via litellm."""
    #     from omnidocs.vlm import VLMAPIConfig, vlm_completion

    #     config = self.backend_config
    #     vlm_config = VLMAPIConfig(
    #         model=config.model,
    #         api_key=config.api_key,
    #         api_base=config.api_base,
    #         max_tokens=config.max_tokens,
    #         temperature=config.temperature,
    #         timeout=config.timeout,
    #         api_version=config.api_version,
    #         extra_headers=config.extra_headers,
    #     )
    #     return vlm_completion(vlm_config, DEEPSEEK_PROMPTS["markdown"], image)
    def _infer_api(self, image: Image.Image) -> str:
        """API inference via litellm (Novita/OpenAI-compatible)."""
        config = self.backend_config

        # Encode image
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # API format: image first, then text (no <image> token prefix)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                ],
            }
        ]

        kwargs = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.timeout,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["api_base"] = config.api_base

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
