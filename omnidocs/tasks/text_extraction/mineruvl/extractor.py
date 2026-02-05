"""MinerU VL text extractor with layout-aware two-step extraction.

MinerU VL performs document extraction in two steps:
1. Layout Detection: Detect regions with types (text, table, equation, etc.)
2. Content Recognition: Extract text/table/equation content from each region
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import numpy as np
from PIL import Image

from ..base import BaseTextExtractor
from ..models import OutputFormat, TextOutput
from .utils import (
    DEFAULT_PROMPTS,
    DEFAULT_SAMPLING_PARAMS,
    SYSTEM_PROMPT,
    BlockType,
    ContentBlock,
    MinerUSamplingParams,
    SamplingParams,
    get_rgb_image,
    parse_layout_output,
    prepare_for_extract,
    prepare_for_layout,
    simple_post_process,
)

if TYPE_CHECKING:
    from .api import MinerUVLTextAPIConfig
    from .mlx import MinerUVLTextMLXConfig
    from .pytorch import MinerUVLTextPyTorchConfig
    from .vllm import MinerUVLTextVLLMConfig

# Type alias for all backend configs
MinerUVLTextBackendConfig = Union[
    "MinerUVLTextPyTorchConfig",
    "MinerUVLTextVLLMConfig",
    "MinerUVLTextMLXConfig",
    "MinerUVLTextAPIConfig",
]


class MinerUVLTextExtractor(BaseTextExtractor):
    """
    MinerU VL text extractor with layout-aware extraction.

    Performs two-step extraction:
    1. Layout detection (detect regions)
    2. Content recognition (extract text/table/equation from each region)

    Supports multiple backends:
    - PyTorch (HuggingFace Transformers)
    - VLLM (high-throughput GPU)
    - MLX (Apple Silicon)
    - API (VLLM OpenAI-compatible server)

    Example:
        ```python
        from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        extractor = MinerUVLTextExtractor(
            backend=MinerUVLTextPyTorchConfig(device="cuda")
        )
        result = extractor.extract(image)

        print(result.content)  # Combined text + tables + equations
        print(result.blocks)   # List of ContentBlock objects
        ```
    """

    def __init__(self, backend: MinerUVLTextBackendConfig):
        """
        Initialize MinerU VL text extractor.

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

        if config_type == "MinerUVLTextPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "MinerUVLTextVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "MinerUVLTextMLXConfig":
            self._load_mlx_backend()
        elif config_type == "MinerUVLTextAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(f"Unknown backend config: {config_type}")

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        config = self.backend_config

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
            "auto": torch.float16 if device == "cuda" else torch.float32,
        }
        dtype = dtype_map.get(config.torch_dtype, torch.float16)

        # Load model
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
            max_concurrency=config.max_concurrency,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
        )
        self._layout_size = config.layout_image_size

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        """
        Extract text with layout-aware two-step extraction.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            output_format: Output format ('html' or 'markdown')

        Returns:
            TextOutput with extracted content and metadata
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Step 1: Layout detection
        blocks = self._detect_layout(pil_image)

        # Step 2: Content extraction for each block
        blocks = self._extract_content(pil_image, blocks)

        # Post-process (OTSL to HTML for tables)
        blocks = simple_post_process(blocks)

        # Combine content
        content = self._combine_content(blocks, output_format)

        # Build raw output with blocks info
        raw_output = self._build_raw_output(blocks)

        return TextOutput(
            content=content,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            image_width=width,
            image_height=height,
            model_name="MinerU2.5-2509-1.2B",
        )

    def extract_with_blocks(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> tuple[TextOutput, List[ContentBlock]]:
        """
        Extract text and return both TextOutput and ContentBlocks.

        This method provides access to the detailed block information
        including bounding boxes and block types.

        Args:
            image: Input image
            output_format: Output format

        Returns:
            Tuple of (TextOutput, List[ContentBlock])
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Two-step extraction
        blocks = self._detect_layout(pil_image)
        blocks = self._extract_content(pil_image, blocks)
        blocks = simple_post_process(blocks)

        content = self._combine_content(blocks, output_format)
        raw_output = self._build_raw_output(blocks)

        text_output = TextOutput(
            content=content,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            image_width=width,
            image_height=height,
            model_name="MinerU2.5-2509-1.2B",
        )

        return text_output, blocks

    def _detect_layout(self, image: Image.Image) -> List[ContentBlock]:
        """Run layout detection."""
        layout_image = prepare_for_layout(image, self._layout_size)
        prompt = DEFAULT_PROMPTS["[layout]"]
        params = DEFAULT_SAMPLING_PARAMS["[layout]"]

        output = self._client.predict(layout_image, prompt, params)
        return parse_layout_output(output)

    def _extract_content(
        self,
        image: Image.Image,
        blocks: List[ContentBlock],
    ) -> List[ContentBlock]:
        """Extract content from each detected block."""
        block_images, prompts, params, indices = prepare_for_extract(
            image, blocks, DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS
        )

        if block_images:
            outputs = self._client.batch_predict(block_images, prompts, params)
            for idx, output in zip(indices, outputs):
                blocks[idx].content = output

        return blocks

    def _combine_content(
        self,
        blocks: List[ContentBlock],
        output_format: str,
    ) -> str:
        """Combine block contents into final output."""
        parts = []

        for block in blocks:
            if not block.content:
                continue

            if block.type == BlockType.TEXT:
                parts.append(block.content)
            elif block.type == BlockType.TITLE:
                if output_format == "markdown":
                    parts.append(f"# {block.content}")
                else:
                    parts.append(f"<h1>{block.content}</h1>")
            elif block.type == BlockType.TABLE:
                parts.append(block.content)  # Already HTML
            elif block.type == BlockType.EQUATION:
                if output_format == "markdown":
                    parts.append(f"$${block.content}$$")
                else:
                    parts.append(f"<math>{block.content}</math>")
            elif block.type in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
            ]:
                # Skip page elements in main content
                continue
            elif block.type in [
                BlockType.TABLE_CAPTION,
                BlockType.IMAGE_CAPTION,
                BlockType.CODE_CAPTION,
            ]:
                if output_format == "markdown":
                    parts.append(f"*{block.content}*")
                else:
                    parts.append(f"<figcaption>{block.content}</figcaption>")
            elif block.type == BlockType.CODE:
                if output_format == "markdown":
                    parts.append(f"```\n{block.content}\n```")
                else:
                    parts.append(f"<pre><code>{block.content}</code></pre>")
            else:
                # Default: just add the content
                parts.append(block.content)

        return "\n\n".join(parts)

    def _build_raw_output(self, blocks: List[ContentBlock]) -> str:
        """Build raw output string with all block contents."""
        parts = []
        for i, block in enumerate(blocks):
            parts.append(f"[Block {i + 1}] Type: {block.type.value}")
            parts.append(f"  BBox: {block.bbox}")
            if block.angle:
                parts.append(f"  Angle: {block.angle}")
            if block.content:
                preview = block.content[:200] + "..." if len(block.content) > 200 else block.content
                parts.append(f"  Content: {preview}")
            parts.append("")
        return "\n".join(parts)


# ============= Backend Client Implementations =============


class _TransformersClient:
    """HuggingFace Transformers client for MinerU VL."""

    def __init__(self, model, processor, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

        # Get skip token IDs
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

    def _build_messages(self, prompt: str) -> List[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_messages = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def _build_generate_kwargs(self, sp: SamplingParams):
        sp = sp or MinerUSamplingParams()
        do_sample = (sp.temperature or 0.0) > 0.0 and (sp.top_k or 1) > 1

        kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": sp.max_new_tokens or self.max_new_tokens,
        }
        if do_sample:
            if sp.temperature:
                kwargs["temperature"] = sp.temperature
            if sp.top_p:
                kwargs["top_p"] = sp.top_p
            if sp.top_k:
                kwargs["top_k"] = sp.top_k
        if sp.repetition_penalty:
            kwargs["repetition_penalty"] = sp.repetition_penalty
        if sp.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = sp.no_repeat_ngram_size

        return kwargs

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        return self.batch_predict([image], [prompt], [sp])[0]

    def batch_predict(
        self,
        images: List[Image.Image],
        prompts: List[str],
        sampling_params: List[SamplingParams],
    ) -> List[str]:
        outputs = []
        for img, prompt, sp in zip(images, prompts, sampling_params):
            img = get_rgb_image(img)
            chat_prompt = self.processor.apply_chat_template(
                self._build_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[chat_prompt],
                images=[img],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)

            generate_kwargs = self._build_generate_kwargs(sp)
            output_ids = self.model.generate(**inputs, use_cache=True, **generate_kwargs)

            output_ids = output_ids.cpu().tolist()
            output_ids = [ids[len(in_ids) :] for in_ids, ids in zip(inputs.input_ids, output_ids)]
            output_ids = [[id for id in ids if id not in self.skip_token_ids] for ids in output_ids]

            output_texts = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            outputs.append(output_texts[0])

        return outputs


class _VLLMClient:
    """VLLM client for MinerU VL."""

    def __init__(self, llm, max_tokens: int):
        self.llm = llm
        self.max_tokens = max_tokens
        self.tokenizer = llm.get_tokenizer()

    def _build_messages(self, prompt: str) -> List[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_messages = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def _build_sampling_params(self, sp: SamplingParams):
        from vllm import SamplingParams as VllmSamplingParams

        sp = sp or MinerUSamplingParams()
        kwargs = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repetition_penalty": sp.repetition_penalty,
            "max_tokens": sp.max_new_tokens or self.max_tokens,
            "skip_special_tokens": False,
        }
        return VllmSamplingParams(**{k: v for k, v in kwargs.items() if v is not None})

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        return self.batch_predict([image], [prompt], [sp])[0]

    def batch_predict(
        self,
        images: List[Image.Image],
        prompts: List[str],
        sampling_params: List[SamplingParams],
    ) -> List[str]:
        image_objs = [get_rgb_image(img) for img in images]

        chat_prompts = [
            self.tokenizer.apply_chat_template(
                self._build_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        vllm_sp_list = [self._build_sampling_params(sp) for sp in sampling_params]

        vllm_prompts = [
            {"prompt": chat_prompt, "multi_modal_data": {"image": image}}
            for chat_prompt, image in zip(chat_prompts, image_objs)
        ]

        outputs = self.llm.generate(
            prompts=vllm_prompts,
            sampling_params=vllm_sp_list,
            use_tqdm=True,
        )

        return [output.outputs[0].text for output in outputs]


class _MLXClient:
    """MLX client for MinerU VL (Apple Silicon)."""

    def __init__(self, model, processor, max_tokens: int):
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens

        from mlx_vlm import generate

        self.generate_fn = generate

    def _build_messages(self, prompt: str) -> List[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_messages = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def _build_generate_kwargs(self, sp: SamplingParams):
        sp = sp or MinerUSamplingParams()
        return {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "max_tokens": sp.max_new_tokens or self.max_tokens,
        }

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        image = get_rgb_image(image)
        chat_prompt = self.processor.apply_chat_template(
            self._build_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )

        generate_kwargs = self._build_generate_kwargs(sp)
        response = self.generate_fn(
            model=self.model,
            processor=self.processor,
            prompt=chat_prompt,
            image=image,
            **generate_kwargs,
        )
        return response.text

    def batch_predict(
        self,
        images: List[Image.Image],
        prompts: List[str],
        sampling_params: List[SamplingParams],
    ) -> List[str]:
        # MLX doesn't support batching, process sequentially
        results = []
        for image, prompt, sp in zip(images, prompts, sampling_params):
            result = self.predict(image, prompt, sp)
            results.append(result)
        return results


class _APIClient:
    """HTTP API client for MinerU VL (VLLM server)."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        timeout: int,
        max_retries: int,
        max_concurrency: int,
        max_tokens: int,
        api_key: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self.max_tokens = max_tokens
        self.api_key = api_key

    def _get_image_data_url(self, image: Image.Image) -> str:
        import base64
        from io import BytesIO

        image = get_rgb_image(image)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _build_request_body(self, image: Image.Image, prompt: str, sp: SamplingParams):
        sp = sp or MinerUSamplingParams()
        image_url = self._get_image_data_url(image)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_messages = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": user_messages})

        body = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": sp.max_new_tokens or self.max_tokens,
            "skip_special_tokens": False,
        }
        if sp.temperature is not None:
            body["temperature"] = sp.temperature
        if sp.top_p is not None:
            body["top_p"] = sp.top_p

        return body

    def predict(self, image: Image.Image, prompt: str, sp: SamplingParams = None) -> str:
        import httpx

        body = self._build_request_body(image, prompt, sp)
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

    def batch_predict(
        self,
        images: List[Image.Image],
        prompts: List[str],
        sampling_params: List[SamplingParams],
    ) -> List[str]:
        import asyncio

        import httpx

        async def _async_batch():
            semaphore = asyncio.Semaphore(self.max_concurrency)
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient(timeout=self.timeout) as client:

                async def fetch_one(idx, image, prompt, sp):
                    async with semaphore:
                        body = self._build_request_body(image, prompt, sp)
                        response = await client.post(
                            f"{self.server_url}/v1/chat/completions",
                            json=body,
                            headers=headers,
                        )
                        response.raise_for_status()
                        data = response.json()
                        return idx, data["choices"][0]["message"]["content"]

                tasks = [
                    fetch_one(i, img, p, sp) for i, (img, p, sp) in enumerate(zip(images, prompts, sampling_params))
                ]
                results = await asyncio.gather(*tasks)

            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]

        return asyncio.run(_async_batch())
