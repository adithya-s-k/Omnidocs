"""
Qwen3-VL layout detector.

A Vision-Language Model for flexible layout detection with custom label support.
Supports PyTorch, VLLM, MLX, and API backends.

Example:
    >>> from omnidocs.tasks.layout_extraction import QwenLayoutDetector
    >>> from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
    >>>
    >>> detector = QwenLayoutDetector(
    ...     backend=QwenLayoutPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
    ... )
    >>> result = detector.extract(image)
    >>>
    >>> # With custom labels
    >>> result = detector.extract(image, custom_labels=["code_block", "sidebar"])
"""

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ..base import BaseLayoutExtractor
from ..models import (
    BoundingBox,
    CustomLabel,
    LayoutBox,
    LayoutLabel,
    LayoutOutput,
)

if TYPE_CHECKING:
    from .api import QwenLayoutAPIConfig
    from .mlx import QwenLayoutMLXConfig
    from .pytorch import QwenLayoutPyTorchConfig
    from .vllm import QwenLayoutVLLMConfig

# Union type for all supported backends
QwenLayoutBackendConfig = Union[
    "QwenLayoutPyTorchConfig",
    "QwenLayoutVLLMConfig",
    "QwenLayoutMLXConfig",
    "QwenLayoutAPIConfig",
]

# Default labels for layout detection
DEFAULT_LAYOUT_LABELS = [
    "title",
    "text",
    "list",
    "table",
    "figure",
    "caption",
    "formula",
    "footnote",
    "page_header",
    "page_footer",
]

# Coordinate range used by Qwen3-VL (0-999 relative coordinates)
QWEN_COORD_RANGE = 999


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


class QwenLayoutDetector(BaseLayoutExtractor):
    """
    Qwen3-VL Vision-Language Model layout detector.

    A flexible VLM-based layout detector that supports custom labels.
    Unlike fixed-label models (DocLayoutYOLO, RT-DETR), Qwen can detect
    any document elements specified at runtime.

    Supports PyTorch, VLLM, MLX, and API backends.

    Example:
        >>> from omnidocs.tasks.layout_extraction import QwenLayoutDetector, CustomLabel
        >>> from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
        >>>
        >>> # Initialize with PyTorch backend
        >>> detector = QwenLayoutDetector(
        ...     backend=QwenLayoutPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
        ... )
        >>>
        >>> # Basic extraction with default labels
        >>> result = detector.extract(image)
        >>>
        >>> # With custom labels (strings)
        >>> result = detector.extract(image, custom_labels=["code_block", "sidebar"])
        >>>
        >>> # With typed custom labels
        >>> labels = [
        ...     CustomLabel(name="code_block", color="#E74C3C"),
        ...     CustomLabel(name="sidebar", description="Side panel content"),
        ... ]
        >>> result = detector.extract(image, custom_labels=labels)
    """

    def __init__(self, backend: QwenLayoutBackendConfig):
        """
        Initialize Qwen layout detector.

        Args:
            backend: Backend configuration. One of:
                - QwenLayoutPyTorchConfig: PyTorch/HuggingFace backend
                - QwenLayoutVLLMConfig: VLLM high-throughput backend
                - QwenLayoutMLXConfig: MLX backend for Apple Silicon
                - QwenLayoutAPIConfig: API backend (OpenRouter, etc.)
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

        if config_type == "QwenLayoutPyTorchConfig":
            self._load_pytorch_backend()
        elif config_type == "QwenLayoutVLLMConfig":
            self._load_vllm_backend()
        elif config_type == "QwenLayoutMLXConfig":
            self._load_mlx_backend()
        elif config_type == "QwenLayoutAPIConfig":
            self._load_api_backend()
        else:
            raise TypeError(
                f"Unknown backend config: {config_type}. "
                f"Expected one of: QwenLayoutPyTorchConfig, QwenLayoutVLLMConfig, "
                f"QwenLayoutMLXConfig, QwenLayoutAPIConfig"
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
        custom_labels: Optional[List[Union[str, CustomLabel]]] = None,
    ) -> LayoutOutput:
        """
        Run layout detection on an image.

        Args:
            image: Input image as:
                - PIL.Image.Image: PIL image object
                - np.ndarray: Numpy array (HWC format, RGB)
                - str or Path: Path to image file
            custom_labels: Optional custom labels to detect. Can be:
                - None: Use default labels (title, text, table, figure, etc.)
                - List[str]: Simple label names ["code_block", "sidebar"]
                - List[CustomLabel]: Typed labels with metadata

        Returns:
            LayoutOutput with detected layout boxes

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image format is not supported
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        # Normalize labels
        label_names = self._normalize_labels(custom_labels)

        # Build prompt
        prompt = self._build_detection_prompt(label_names)

        # Run inference based on backend
        config_type = type(self.backend_config).__name__
        if config_type == "QwenLayoutPyTorchConfig":
            raw_output = self._infer_pytorch(pil_image, prompt)
        elif config_type == "QwenLayoutVLLMConfig":
            raw_output = self._infer_vllm(pil_image, prompt)
        elif config_type == "QwenLayoutMLXConfig":
            raw_output = self._infer_mlx(pil_image, prompt)
        elif config_type == "QwenLayoutAPIConfig":
            raw_output = self._infer_api(pil_image, prompt)
        else:
            raise RuntimeError(f"Unknown backend: {config_type}")

        # Parse detections
        detections = self._parse_json_output(raw_output)

        # Convert to LayoutOutput
        layout_boxes = self._build_layout_boxes(detections, width, height)

        # Sort by position (reading order)
        layout_boxes.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))

        return LayoutOutput(
            bboxes=layout_boxes,
            image_width=width,
            image_height=height,
            model_name=f"Qwen3-VL ({type(self.backend_config).__name__})",
        )

    def _normalize_labels(self, custom_labels: Optional[List[Union[str, CustomLabel]]]) -> List[str]:
        """Normalize labels to list of strings."""
        if custom_labels is None:
            return DEFAULT_LAYOUT_LABELS

        label_names = []
        for label in custom_labels:
            if isinstance(label, str):
                label_names.append(label)
            elif isinstance(label, CustomLabel):
                label_names.append(label.name)
            else:
                raise TypeError(f"Expected str or CustomLabel, got {type(label).__name__}")

        return label_names

    def _build_detection_prompt(self, labels: List[str]) -> str:
        """Build detection prompt for Qwen3-VL."""
        labels_str = ", ".join(labels)
        return (
            f"Detect all layout elements in this document image. "
            f"Identify elements from these categories: {labels_str}. "
            f"Output as JSON array with format: "
            f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "element_type"}}, ...] '
            f"where coordinates are relative (0-999)."
        )

    def _parse_json_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse JSON output from model, handling markdown fencing and truncation."""
        # Remove markdown fencing if present
        lines = output.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                output = "\n".join(lines[i + 1 :])
                output = output.split("```")[0]
                break

        # Try direct parsing first
        try:
            result = json.loads(output)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: regex extraction for truncated or malformed output
        pattern = r'\{"bbox_2d"\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\}'
        matches = re.findall(pattern, output)

        results = []
        for match in matches:
            x1, y1, x2, y2, label = match
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            # Filter out invalid coordinates (> 999 means hallucination)
            if all(0 <= c <= QWEN_COORD_RANGE for c in bbox):
                results.append({"bbox_2d": bbox, "label": label})

        return results

    def _convert_relative_to_absolute(self, bbox: List[int], width: int, height: int) -> List[float]:
        """Convert relative (0-999) to absolute pixel coordinates."""
        x1 = bbox[0] / QWEN_COORD_RANGE * width
        y1 = bbox[1] / QWEN_COORD_RANGE * height
        x2 = bbox[2] / QWEN_COORD_RANGE * width
        y2 = bbox[3] / QWEN_COORD_RANGE * height

        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

    def _build_layout_boxes(self, detections: List[Dict[str, Any]], width: int, height: int) -> List[LayoutBox]:
        """Convert parsed detections to LayoutBox objects."""
        layout_boxes = []

        for det in detections:
            if "bbox_2d" not in det or "label" not in det:
                continue

            bbox = det["bbox_2d"]

            # Validate bbox structure before accessing coordinates
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            if not all(isinstance(c, (int, float)) for c in bbox):
                continue

            label_str = det["label"].lower()

            # Convert relative to absolute coordinates
            abs_bbox = self._convert_relative_to_absolute(bbox, width, height)

            # Map to standard label if possible
            try:
                standard_label = LayoutLabel(label_str)
            except ValueError:
                standard_label = LayoutLabel.UNKNOWN

            layout_boxes.append(
                LayoutBox(
                    label=standard_label,
                    bbox=BoundingBox.from_list(abs_bbox),
                    confidence=1.0,  # VLM doesn't output confidence
                    original_label=label_str,
                )
            )

        return layout_boxes

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
