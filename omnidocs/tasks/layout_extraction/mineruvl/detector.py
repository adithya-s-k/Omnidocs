"""MinerU VL layout detector.

Uses MinerU2.5-2509-1.2B for document layout detection.
Detects 22+ element types including text, titles, tables, equations, figures, code.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy as np
from PIL import Image

# Import utilities from text_extraction module (shared code)
from ...text_extraction.mineruvl.utils import (
    DEFAULT_PROMPTS,
    DEFAULT_SAMPLING_PARAMS,
    SYSTEM_PROMPT,
    BlockType,
    ContentBlock,
    MinerUSamplingParams,
    SamplingParams,
    get_rgb_image,
    parse_layout_output,
    prepare_for_layout,
)
from ..base import BaseLayoutExtractor
from ..models import BoundingBox, LayoutBox, LayoutLabel, LayoutOutput

if TYPE_CHECKING:
    from .api import MinerUVLLayoutAPIConfig
    from .mlx import MinerUVLLayoutMLXConfig
    from .pytorch import MinerUVLLayoutPyTorchConfig
    from .vllm import MinerUVLLayoutVLLMConfig

# Type alias for all backend configs
MinerUVLLayoutBackendConfig = Union[
    "MinerUVLLayoutPyTorchConfig",
    "MinerUVLLayoutVLLMConfig",
    "MinerUVLLayoutMLXConfig",
    "MinerUVLLayoutAPIConfig",
]

# Mapping from MinerU BlockType to Omnidocs LayoutLabel
MINERUVL_LABEL_MAPPING = {
    BlockType.TEXT: LayoutLabel.TEXT,
    BlockType.TITLE: LayoutLabel.TITLE,
    BlockType.TABLE: LayoutLabel.TABLE,
    BlockType.IMAGE: LayoutLabel.FIGURE,
    BlockType.CODE: LayoutLabel.TEXT,
    BlockType.ALGORITHM: LayoutLabel.TEXT,
    BlockType.HEADER: LayoutLabel.PAGE_HEADER,
    BlockType.FOOTER: LayoutLabel.PAGE_FOOTER,
    BlockType.PAGE_NUMBER: LayoutLabel.PAGE_FOOTER,
    BlockType.PAGE_FOOTNOTE: LayoutLabel.FOOTNOTE,
    BlockType.ASIDE_TEXT: LayoutLabel.TEXT,
    BlockType.EQUATION: LayoutLabel.FORMULA,
    BlockType.EQUATION_BLOCK: LayoutLabel.FORMULA,
    BlockType.REF_TEXT: LayoutLabel.TEXT,
    BlockType.LIST: LayoutLabel.LIST,
    BlockType.PHONETIC: LayoutLabel.TEXT,
    BlockType.TABLE_CAPTION: LayoutLabel.CAPTION,
    BlockType.IMAGE_CAPTION: LayoutLabel.CAPTION,
    BlockType.CODE_CAPTION: LayoutLabel.CAPTION,
    BlockType.TABLE_FOOTNOTE: LayoutLabel.FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE: LayoutLabel.FOOTNOTE,
    BlockType.UNKNOWN: LayoutLabel.UNKNOWN,
}


class MinerUVLLayoutDetector(BaseLayoutExtractor):
    """
    MinerU VL layout detector.

    Uses MinerU2.5-2509-1.2B for document layout detection.
    Detects 22+ element types including text, titles, tables,
    equations, figures, code, and more.

    For full document extraction (layout + content), use MinerUVLTextExtractor
    from the text_extraction module instead.

    Example:
        ```python
        from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

        detector = MinerUVLLayoutDetector(
            backend=MinerUVLLayoutPyTorchConfig(device="cuda")
        )
        result = detector.extract(image)

        for box in result.bboxes:
            print(f"{box.label}: {box.confidence:.2f}")
        ```
    """

    def __init__(self, backend: MinerUVLLayoutBackendConfig):
        """
        Initialize MinerU VL layout detector.

        Args:
            backend: Backend configuration (PyTorch, VLLM, MLX, or API)
        """
        self.backend_config = backend
        self._client = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Load VLM client based on backend config."""
        config_type = type(self.backend_config).__name__

        if config_type == "MinerUVLLayoutPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "MinerUVLLayoutVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "MinerUVLLayoutMLXConfig":
            self._load_mlx_backend()
        elif config_type == "MinerUVLLayoutAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(f"Unknown backend config: {config_type}")

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        config = self.backend_config

        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.float16 if device == "cuda" else torch.float32,
        }
        dtype = dtype_map.get(config.torch_dtype, torch.float16)

        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": dtype,
        }
        if device == "cuda":
            if config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                model_kwargs["attn_implementation"] = "sdpa"
        if config.device_map:
            model_kwargs["device_map"] = config.device_map

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model,
            **model_kwargs,
        )
        if not config.device_map:
            model = model.to(device)
        model = model.eval()

        processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
        )

        self._client = _TransformersClient(model, processor, config.max_new_tokens)
        self._layout_size = config.layout_image_size

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend."""
        from vllm import LLM

        config = self.backend_config

        llm = LLM(
            model=config.model,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            download_dir=config.download_dir,
            disable_custom_all_reduce=config.disable_custom_all_reduce,
            enforce_eager=config.enforce_eager,
        )

        self._client = _VLLMClient(llm, config.max_tokens)
        self._layout_size = config.layout_image_size

    def _load_mlx_backend(self) -> None:
        """Load MLX backend (Apple Silicon)."""
        from mlx_vlm import load

        config = self.backend_config
        model, processor = load(config.model)

        self._client = _MLXClient(model, processor, config.max_tokens)
        self._layout_size = config.layout_image_size

    def _load_api_backend(self) -> None:
        """Load API backend (VLLM server)."""
        config = self.backend_config

        self._client = _APIClient(
            server_url=config.server_url,
            model_name=config.model_name,
            timeout=config.timeout,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
        self._layout_size = config.layout_image_size

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
    ) -> LayoutOutput:
        """
        Detect layout elements in the image.

        Args:
            image: Input image (PIL Image, numpy array, or file path)

        Returns:
            LayoutOutput with standardized labels and bounding boxes
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Run layout detection
        blocks = self._detect_layout(pil_image)

        # Convert to LayoutOutput
        bboxes = []
        for block in blocks:
            # Convert normalized [0,1] to pixel coords
            x1, y1, x2, y2 = block.bbox
            pixel_bbox = BoundingBox(
                x1=x1 * width,
                y1=y1 * height,
                x2=x2 * width,
                y2=y2 * height,
            )

            # Map label
            label = MINERUVL_LABEL_MAPPING.get(block.type, LayoutLabel.UNKNOWN)

            bboxes.append(
                LayoutBox(
                    label=label,
                    bbox=pixel_bbox,
                    confidence=1.0,  # MinerU VL doesn't output confidence
                    original_label=block.type.value,
                )
            )

        return LayoutOutput(
            bboxes=bboxes,
            image_width=width,
            image_height=height,
            model_name="MinerU2.5-2509-1.2B",
        )

    def _detect_layout(self, image: Image.Image) -> List[ContentBlock]:
        """Run layout detection with VLM."""
        layout_image = prepare_for_layout(image, self._layout_size)
        prompt = DEFAULT_PROMPTS["[layout]"]
        params = DEFAULT_SAMPLING_PARAMS["[layout]"]

        output = self._client.predict(layout_image, prompt, params)
        return parse_layout_output(output)


# ============= Backend Client Implementations =============
# (Simplified versions for layout-only detection)


class _TransformersClient:
    """HuggingFace Transformers client for layout detection."""

    def __init__(self, model, processor, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

        skip_token_ids = set()
        for field in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if hasattr(model.config, field):
                token_id = getattr(model.config, field)
                if isinstance(token_id, int):
                    skip_token_ids.add(token_id)
            if hasattr(processor.tokenizer, field):
                token_id = getattr(processor.tokenizer, field)
                if isinstance(token_id, int):
                    skip_token_ids.add(token_id)
        self.skip_token_ids = skip_token_ids

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        image = get_rgb_image(image)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        )

        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[chat_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)

        sp = sp or MinerUSamplingParams()
        output_ids = self.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=sp.max_new_tokens or self.max_new_tokens,
            do_sample=False,
        )

        output_ids = output_ids.cpu().tolist()
        output_ids = [ids[len(in_ids) :] for in_ids, ids in zip(inputs.input_ids, output_ids)]
        output_ids = [[id for id in ids if id not in self.skip_token_ids] for ids in output_ids]

        output_texts = self.processor.batch_decode(
            output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return output_texts[0]


class _VLLMClient:
    """VLLM client for layout detection."""

    def __init__(self, llm, max_tokens: int):
        self.llm = llm
        self.max_tokens = max_tokens
        self.tokenizer = llm.get_tokenizer()

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        from vllm import SamplingParams as VllmSamplingParams

        image = get_rgb_image(image)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        )

        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sp = sp or MinerUSamplingParams()
        vllm_sp = VllmSamplingParams(
            max_tokens=sp.max_new_tokens or self.max_tokens,
            temperature=sp.temperature or 0.0,
            skip_special_tokens=False,
        )

        outputs = self.llm.generate(
            prompts=[{"prompt": chat_prompt, "multi_modal_data": {"image": image}}],
            sampling_params=[vllm_sp],
        )
        return outputs[0].outputs[0].text


class _MLXClient:
    """MLX client for layout detection."""

    def __init__(self, model, processor, max_tokens: int):
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens
        from mlx_vlm import generate

        self.generate_fn = generate

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        image = get_rgb_image(image)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        )

        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sp = sp or MinerUSamplingParams()
        response = self.generate_fn(
            model=self.model,
            processor=self.processor,
            prompt=chat_prompt,
            image=image,
            max_tokens=sp.max_new_tokens or self.max_tokens,
            temperature=sp.temperature or 0.0,
        )
        return response.text


class _APIClient:
    """HTTP API client for layout detection."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        timeout: int,
        max_retries: int,
        max_tokens: int,
        api_key: str = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        import base64
        from io import BytesIO

        import httpx

        image = get_rgb_image(image)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_url = f"data:image/png;base64,{b64}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        sp = sp or MinerUSamplingParams()
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": sp.max_new_tokens or self.max_tokens,
            "temperature": sp.temperature or 0.0,
            "skip_special_tokens": False,
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]
