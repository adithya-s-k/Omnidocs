"""
GLM-OCR text extractor.

GLM-OCR from zai-org (Feb 2026) — 0.9B OCR-specialist model.
Architecture: CogViT visual encoder (0.4B) + GLM decoder (0.5B).
Scores #1 on OmniDocBench V1.5 (94.62).

Key differences from GLM-V:
  - Uses AutoModelForImageTextToText (NOT Glm4vForConditionalGeneration)
  - Uses AutoProcessor with direct image input (no chat template URL trick)
  - Much smaller (0.9B vs 9B) — faster, lower VRAM
  - Requires transformers>=5.3.0
  - No <think> tokens, no <|begin_of_box|> — clean output
"""
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from PIL import Image

from omnidocs.utils.cache import get_model_cache_dir

from ....cache import add_reference, get_cache_key, get_cached, set_cached
from ..base import BaseTextExtractor
from ..models import OutputFormat, TextOutput

if TYPE_CHECKING:
    from .api import GLMOCRAPIConfig
    from .pytorch import GLMOCRPyTorchConfig
    from .vllm import GLMOCRVLLMConfig

GLMOCRBackendConfig = Union[
    "GLMOCRPyTorchConfig",
    "GLMOCRVLLMConfig",
    "GLMOCRAPIConfig",
]

GLMOCR_PROMPT = "Text Recognition:"


class GLMOCRTextExtractor(BaseTextExtractor):
    """
    GLM-OCR text extractor (zai-org/GLM-OCR, 0.9B, Feb 2026).

    Purpose-built OCR model, #1 on OmniDocBench V1.5.
    Faster and cheaper than GLM-V for pure document OCR tasks.

    Example:
```python
        from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        extractor = GLMOCRTextExtractor(backend=GLMOCRPyTorchConfig())
        result = extractor.extract(image)
        print(result.content)
```
    """

    def __init__(self, backend: GLMOCRBackendConfig):
        self.backend_config = backend
        self._backend: Any = None
        self._processor: Any = None
        self._loaded = False
        self._sampling_params_class: Any = None
        self._load_model()

    def _load_model(self) -> None:
        config_type = type(self.backend_config).__name__

        if config_type != "GLMOCRAPIConfig":
            cache_key = get_cache_key(self.backend_config)
            self._cache_key = cache_key
            cached = get_cached(cache_key)
            if cached is not None:
                self._backend, self._processor = cached
                add_reference(cache_key, self)
                if config_type == "GLMOCRVLLMConfig":
                    from vllm import SamplingParams
                    self._sampling_params_class = SamplingParams
                self._loaded = True
                return

        dispatch = {
            "GLMOCRPyTorchConfig": self._load_pytorch_backend,
            "GLMOCRVLLMConfig": self._load_vllm_backend,
            "GLMOCRAPIConfig": self._load_api_backend,
        }
        loader = dispatch.get(config_type)
        if loader is None:
            raise TypeError(f"Unknown backend config: {config_type}")
        loader()

        if config_type != "GLMOCRAPIConfig":
            set_cached(cache_key, (self._backend, self._processor), owner=self)

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "GLM-OCR requires transformers>=5.3.0. "
                "Install with: uv add 'transformers>=5.3.0'"
            ) from e

        config = self.backend_config
        cache_dir = get_model_cache_dir(config.cache_dir)

        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(config.torch_dtype, "auto")

        model_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "device_map": config.device_map,
            "cache_dir": str(cache_dir),
        }
        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._backend = AutoModelForImageTextToText.from_pretrained(
            config.model, **model_kwargs
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            config.model, cache_dir=str(cache_dir)
        )

    def _load_vllm_backend(self) -> None:
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "VLLM backend requires vllm>=0.17.0 and transformers>=5.3.0."
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
        )
        self._processor = AutoProcessor.from_pretrained(
            config.model, cache_dir=str(cache_dir)
        )
        self._sampling_params_class = SamplingParams

    def _load_api_backend(self) -> None:
        pass

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        config_type = type(self.backend_config).__name__
        dispatch = {
            "GLMOCRPyTorchConfig": self._infer_pytorch,
            "GLMOCRVLLMConfig": self._infer_vllm,
            "GLMOCRAPIConfig": self._infer_api,
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
            model_name=f"GLM-OCR ({self.backend_config.model}, {config_type})",
        )

    def _infer_pytorch(self, image: Image.Image) -> str:
        import tempfile
        import torch

        config = self.backend_config

        # GLM-OCR uses the same chat template pattern as GLM-V:
        # image passed as {"type": "image", "url": <local path>}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            image.save(f, format="PNG")

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": temp_path},
                        {"type": "text", "text": GLMOCR_PROMPT},
                    ],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._backend.device)

            input_len = inputs["input_ids"].shape[1]
            do_sample = config.temperature > 0.0

            with torch.no_grad():
                generated_ids = self._backend.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=do_sample,
                    temperature=config.temperature if do_sample else None,
                )

            return self._processor.decode(
                generated_ids[0][input_len:],
                skip_special_tokens=True,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    def _infer_vllm(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # GLM-OCR requires the chat template to embed image placeholder tokens.
        # Passing a raw prompt string bypasses this and causes vLLM's multimodal
        # processor to fail with "Failed to apply prompt replacement for mm_items['image'][0]".
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": GLMOCR_PROMPT},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mm_data = {"image": image}

        config = self.backend_config
        sampling_params = self._sampling_params_class(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            stop=["<|endoftext|>", "<|end_of_text|>"],
        )
        outputs = self._backend.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text

    def _infer_api(self, image: Image.Image) -> str:
        from omnidocs.vlm import VLMAPIConfig, vlm_completion

        config = self.backend_config
        vlm_config = VLMAPIConfig(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            api_version=config.api_version,
            extra_headers=config.extra_headers,
        )
        return vlm_completion(vlm_config, GLMOCR_PROMPT, image)