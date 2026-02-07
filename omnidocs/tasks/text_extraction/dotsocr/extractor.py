"""
Dots OCR text extractor with layout-aware extraction.

A Vision-Language Model optimized for document OCR with structured output
containing layout information, bounding boxes, and multi-format text.

Supports PyTorch, VLLM, and API backends.

Example:
    ```python
    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

    extractor = DotsOCRTextExtractor(
            backend=DotsOCRPyTorchConfig(model="rednote-hilab/dots.ocr")
        )
    result = extractor.extract(image, include_layout=True)
    print(result.content)
    for elem in result.layout:
            print(f"{elem.category}: {elem.bbox}")
    ```
"""

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import numpy as np
from PIL import Image

from ....cache import add_reference, get_cache_key, get_cached, set_cached
from ..base import BaseTextExtractor
from ..models import DotsOCRTextOutput, LayoutElement, OutputFormat

if TYPE_CHECKING:
    from .api import DotsOCRAPIConfig
    from .pytorch import DotsOCRPyTorchConfig
    from .vllm import DotsOCRVLLMConfig

# Union type for all supported backends
DotsOCRBackendConfig = Union[
    "DotsOCRPyTorchConfig",
    "DotsOCRVLLMConfig",
    "DotsOCRAPIConfig",
]

# Dots OCR default prompt
DOTS_OCR_PROMPT = """Please output the layout information from the PDF image, \
including each layout element's bbox, its category, and the corresponding text \
content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', \
'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', \
'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""


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


def _parse_json_output(raw_output: str) -> Optional[list]:
    """
    Parse JSON output from Dots OCR.

    Handles cases where output may have instructions or text before/after JSON.

    Args:
        raw_output: Raw model output

    Returns:
        Parsed JSON list or None if parsing fails
    """
    try:
        # Try direct JSON parse first
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        # Look for array pattern [...] or object pattern {...}
        array_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
                # If it's a single object, wrap in array
                return [parsed] if isinstance(parsed, dict) else parsed
            except json.JSONDecodeError:
                pass

    return None


def _format_layout_as_text(layout: list, output_format: str) -> str:
    """
    Convert layout elements to text in requested format.

    Args:
        layout: List of layout elements with category, bbox, text
        output_format: Desired format (markdown/html/json)

    Returns:
        Formatted text string
    """
    if output_format == "json":
        return json.dumps(layout, indent=2, ensure_ascii=False)

    # Extract text from layout elements in reading order
    text_parts = []
    for elem in layout:
        if "text" in elem and elem["text"]:
            text_parts.append(elem["text"])

    if output_format == "html":
        # Wrap in HTML structure
        html_parts = ['<div class="document">']
        for part in text_parts:
            html_parts.append(f'<div class="element">{part}</div>')
        html_parts.append("</div>")
        return "\n".join(html_parts)
    else:
        # Markdown - just join text parts
        return "\n\n".join(text_parts)


class DotsOCRTextExtractor(BaseTextExtractor):
    """
    Dots OCR Vision-Language Model text extractor with layout detection.

    Extracts text from document images with layout information including:
    - 11 layout categories (Caption, Footnote, Formula, List-item, etc.)
    - Bounding boxes (normalized to 0-1024)
    - Multi-format text (Markdown, LaTeX, HTML)
    - Reading order preservation

    Supports PyTorch, VLLM, and API backends.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        # Initialize with PyTorch backend
        extractor = DotsOCRTextExtractor(
                backend=DotsOCRPyTorchConfig(model="rednote-hilab/dots.ocr")
            )

        # Extract with layout
        result = extractor.extract(image, include_layout=True)
        print(f"Found {result.num_layout_elements} elements")
        print(result.content)
        ```
    """

    def __init__(self, backend: DotsOCRBackendConfig):
        """
        Initialize Dots OCR text extractor.

        Args:
            backend: Backend configuration. One of:
                - DotsOCRPyTorchConfig: PyTorch/HuggingFace backend
                - DotsOCRVLLMConfig: VLLM high-throughput backend
                - DotsOCRAPIConfig: API backend (online VLLM server)
        """
        self.backend_config = backend
        self._backend: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._loaded = False

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load appropriate backend based on config type.

        Uses unified model cache with reference counting to share models.
        """
        config_type = type(self.backend_config).__name__

        # Check cache first (skip API backend which has no model to cache)
        if config_type != "DotsOCRAPIConfig":
            cache_key = get_cache_key(self.backend_config)
            self._cache_key = cache_key
            cached = get_cached(cache_key)
            if cached is not None:
                if config_type == "DotsOCRPyTorchConfig":
                    self._model, self._processor = cached
                elif config_type == "DotsOCRVLLMConfig":
                    (self._backend,) = cached
                    from vllm import SamplingParams

                    self._sampling_params_class = SamplingParams
                add_reference(cache_key, self)
                self._loaded = True
                return

        if config_type == "DotsOCRPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "DotsOCRVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "DotsOCRAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(
                f"Unknown backend config: {config_type}. "
                f"Expected one of: DotsOCRPyTorchConfig, DotsOCRVLLMConfig, DotsOCRAPIConfig"
            )

        # Cache the loaded model (skip API)
        if config_type == "DotsOCRPyTorchConfig":
            set_cached(cache_key, (self._model, self._processor), owner=self)
        elif config_type == "DotsOCRVLLMConfig":
            set_cached(cache_key, (self._backend,), owner=self)

        self._loaded = True

    def _load_pytorch_backend(self) -> None:
        """Load PyTorch/HuggingFace backend."""
        try:
            import torch
            import transformers.processing_utils as pu
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "PyTorch backend requires torch and transformers. Install with: uv add torch transformers accelerate"
            ) from e

        config = self.backend_config
        cache_dir = _get_model_cache_dir()

        print(f"Loading Dots OCR model: {config.model}")
        print(f"Cache directory: {cache_dir}")

        # PATCH: Bypass video_processor validation for Dots OCR compatibility
        # Dots OCR doesn't need video processing but transformers 4.55+ requires it
        _orig_check = pu.ProcessorMixin.check_argument_for_proper_class
        pu.ProcessorMixin.check_argument_for_proper_class = lambda *args, **kwargs: None

        # Load processor with validation disabled
        self._processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            cache_dir=cache_dir,
        )

        # Restore validation
        pu.ProcessorMixin.check_argument_for_proper_class = _orig_check

        # Load model (Dots OCR uses AutoModelForCausalLM)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": cache_dir,
        }

        # Set dtype
        if config.torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif config.torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif config.torch_dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32

        # Set attention implementation
        if config.attn_implementation != "eager":
            model_kwargs["attn_implementation"] = config.attn_implementation

        # Load model with AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model,
            device_map=config.device_map,
            **model_kwargs,
        ).eval()

        print(f"Model loaded on device: {config.device}")

    def _load_vllm_backend(self) -> None:
        """Load VLLM backend."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError("VLLM backend requires vllm. Install with: uv add vllm") from e

        config = self.backend_config

        print(f"Loading Dots OCR with VLLM: {config.model}")

        # Initialize VLLM
        self._backend = LLM(
            model=config.model,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=config.disable_custom_all_reduce,
        )

        # Store sampling params class for later use
        self._sampling_params_class = SamplingParams

        print("VLLM model loaded")

    def _load_api_backend(self) -> None:
        """Load API backend."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError("API backend requires litellm. Install with: uv add litellm openai") from e

        config = self.backend_config
        self._api_config = config
        self._litellm = litellm

        print(f"API backend configured: {config.api_base}")

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["markdown", "html", "json"] = "markdown",
        include_layout: bool = False,
        custom_prompt: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> DotsOCRTextOutput:
        """
        Extract text from image using Dots OCR.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            output_format: Output format ("markdown", "html", or "json")
            include_layout: Include layout bounding boxes in output
            custom_prompt: Override default extraction prompt
            max_tokens: Maximum tokens for generation

        Returns:
            DotsOCRTextOutput with extracted content and optional layout

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Prepare image
        img = self._prepare_image(image)

        # Get prompt
        prompt = custom_prompt or DOTS_OCR_PROMPT

        # Run inference based on backend
        config_type = type(self.backend_config).__name__

        if config_type == "DotsOCRPyTorchConfig":
            raw_output = self._infer_pytorch(img, prompt, max_tokens)
        elif config_type == "DotsOCRVLLMConfig":
            raw_output = self._infer_vllm(img, prompt, max_tokens)
        elif config_type == "DotsOCRAPIConfig":
            raw_output = self._infer_api(img, prompt, max_tokens)
        else:
            raise RuntimeError(f"Unknown backend: {config_type}")

        # Parse output
        return self._parse_output(
            raw_output,
            img.size,
            output_format,
            include_layout,
        )

    def _infer_pytorch(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        """Run inference with PyTorch backend."""
        import torch
        from qwen_vl_utils import process_vision_info

        # Prepare messages in Qwen format - pass PIL Image directly
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # PIL Image object
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with processor
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info - NO video_kwargs
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs - NO video_kwargs
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        # Generate - Remove temperature/do_sample to avoid warnings
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )

        # Trim input tokens and decode
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def _infer_vllm(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        """Run inference with VLLM backend."""
        # Prepare input in VLLM format
        # Format: <|img|><|imgpad|><|endofimg|>{PROMPT}
        vllm_prompt = f"<|img|><|imgpad|><|endofimg|>{prompt}"

        inputs = [{"prompt": vllm_prompt, "multi_modal_data": {"image": image}}]

        # Sampling parameters
        sampling_params = self._sampling_params_class(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.8,
            top_k=20,
            min_tokens=100,
        )

        # Generate
        outputs = self._backend.generate(inputs, sampling_params=sampling_params)
        raw_output = outputs[0].outputs[0].text

        return raw_output

    def _infer_api(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        """Run inference with API backend."""
        import base64
        from io import BytesIO

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Prepare messages in OpenAI format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_base64}",
                    },
                ],
            }
        ]

        # Call API
        response = self._litellm.completion(
            model=f"openai/{self._api_config.model}",
            api_base=self._api_config.api_base,
            api_key=self._api_config.api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=self._api_config.timeout,
        )

        return response.choices[0].message.content

    def _parse_output(
        self,
        raw_output: str,
        image_size: tuple,
        output_format: str,
        include_layout: bool,
    ) -> DotsOCRTextOutput:
        """
        Parse raw model output into DotsOCRTextOutput.

        Args:
            raw_output: Raw model output
            image_size: (width, height) of input image
            output_format: Desired output format
            include_layout: Whether to include layout elements

        Returns:
            DotsOCRTextOutput with parsed content and layout
        """
        # Try to parse JSON
        parsed_json = _parse_json_output(raw_output)

        if parsed_json is None:
            # JSON parsing failed
            return DotsOCRTextOutput(
                content=raw_output,
                format=OutputFormat[output_format.upper()],
                layout=[],
                has_layout=False,
                raw_output=raw_output,
                error="Failed to parse JSON output",
                truncated=len(raw_output) > 50000,
                image_width=image_size[0],
                image_height=image_size[1],
            )

        # Convert to layout elements
        layout_elements = []
        if include_layout:
            for elem in parsed_json:
                if isinstance(elem, dict) and "bbox" in elem and "category" in elem:
                    layout_elements.append(
                        LayoutElement(
                            bbox=elem["bbox"],
                            category=elem["category"],
                            text=elem.get("text"),
                        )
                    )

        # Format content
        content = _format_layout_as_text(parsed_json, output_format)

        return DotsOCRTextOutput(
            content=content,
            format=OutputFormat[output_format.upper()],
            layout=layout_elements,
            has_layout=include_layout,
            raw_output=raw_output,
            error=None,
            truncated=False,
            image_width=image_size[0],
            image_height=image_size[1],
        )
