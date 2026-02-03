"""
Test Registry for OmniDocs.

Defines all available tests with their specifications including backend type,
task category, and GPU requirements. Used by both Modal and local runners.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Backend(str, Enum):
    """Inference backend types."""

    VLLM = "vllm"
    PYTORCH_GPU = "pytorch_gpu"
    PYTORCH_CPU = "pytorch_cpu"
    MLX = "mlx"
    API = "api"


class Task(str, Enum):
    """Task categories."""

    LAYOUT = "layout_extraction"
    TEXT = "text_extraction"
    OCR = "ocr_extraction"
    TABLE = "table_extraction"
    READING = "reading_order"


@dataclass
class TestSpec:
    """Specification for a single test."""

    name: str
    module: str  # Relative module path, e.g., "text_extraction.qwen_vllm"
    backend: Backend
    task: Task
    gpu_type: Optional[str] = None  # e.g., "L40S:1", "A10G:1", "T4:1", None for CPU
    timeout: int = 600  # seconds
    tags: List[str] = field(default_factory=list)


# Complete test registry
TEST_REGISTRY: List[TestSpec] = [
    # ============================================================
    # TEXT EXTRACTION
    # ============================================================
    # Qwen Text Extractor
    TestSpec(
        name="qwen_text_vllm",
        module="text_extraction.qwen_vllm",
        backend=Backend.VLLM,
        task=Task.TEXT,
        gpu_type="L40S:1",
        tags=["qwen", "vlm"],
    ),
    TestSpec(
        name="qwen_text_pytorch",
        module="text_extraction.qwen_pytorch",
        backend=Backend.PYTORCH_GPU,
        task=Task.TEXT,
        gpu_type="A10G:1",
        tags=["qwen", "vlm"],
    ),
    TestSpec(
        name="qwen_text_mlx",
        module="text_extraction.qwen_mlx",
        backend=Backend.MLX,
        task=Task.TEXT,
        gpu_type=None,
        tags=["qwen", "vlm", "apple_silicon"],
    ),
    TestSpec(
        name="qwen_text_api",
        module="text_extraction.qwen_api",
        backend=Backend.API,
        task=Task.TEXT,
        gpu_type=None,
        tags=["qwen", "vlm", "api"],
    ),
    # Nanonets Text Extractor
    TestSpec(
        name="nanonets_text_vllm",
        module="text_extraction.nanonets_vllm",
        backend=Backend.VLLM,
        task=Task.TEXT,
        gpu_type="L40S:1",
        tags=["nanonets"],
    ),
    TestSpec(
        name="nanonets_text_pytorch",
        module="text_extraction.nanonets_pytorch",
        backend=Backend.PYTORCH_GPU,
        task=Task.TEXT,
        gpu_type="A10G:1",
        tags=["nanonets"],
    ),
    TestSpec(
        name="nanonets_text_mlx",
        module="text_extraction.nanonets_mlx",
        backend=Backend.MLX,
        task=Task.TEXT,
        gpu_type=None,
        tags=["nanonets", "apple_silicon"],
    ),
    # DotsOCR Text Extractor
    TestSpec(
        name="dotsocr_text_vllm",
        module="text_extraction.dotsocr_vllm",
        backend=Backend.VLLM,
        task=Task.TEXT,
        gpu_type="L40S:1",
        tags=["dotsocr"],
    ),
    TestSpec(
        name="dotsocr_text_pytorch",
        module="text_extraction.dotsocr_pytorch",
        backend=Backend.PYTORCH_GPU,
        task=Task.TEXT,
        gpu_type="A10G:1",
        tags=["dotsocr"],
    ),
    TestSpec(
        name="dotsocr_text_api",
        module="text_extraction.dotsocr_api",
        backend=Backend.API,
        task=Task.TEXT,
        gpu_type=None,
        tags=["dotsocr", "api"],
    ),
    # ============================================================
    # LAYOUT EXTRACTION
    # ============================================================
    # Qwen Layout Detector
    TestSpec(
        name="qwen_layout_vllm",
        module="layout_extraction.qwen_vllm",
        backend=Backend.VLLM,
        task=Task.LAYOUT,
        gpu_type="L40S:1",
        tags=["qwen", "vlm"],
    ),
    TestSpec(
        name="qwen_layout_pytorch",
        module="layout_extraction.qwen_pytorch",
        backend=Backend.PYTORCH_GPU,
        task=Task.LAYOUT,
        gpu_type="A10G:1",
        tags=["qwen", "vlm"],
    ),
    TestSpec(
        name="qwen_layout_mlx",
        module="layout_extraction.qwen_mlx",
        backend=Backend.MLX,
        task=Task.LAYOUT,
        gpu_type=None,
        tags=["qwen", "vlm", "apple_silicon"],
    ),
    TestSpec(
        name="qwen_layout_api",
        module="layout_extraction.qwen_api",
        backend=Backend.API,
        task=Task.LAYOUT,
        gpu_type=None,
        tags=["qwen", "vlm", "api"],
    ),
    # DocLayout-YOLO
    TestSpec(
        name="doclayout_yolo_gpu",
        module="layout_extraction.doclayout_yolo_gpu",
        backend=Backend.PYTORCH_GPU,
        task=Task.LAYOUT,
        gpu_type="T4:1",
        tags=["yolo", "fast"],
    ),
    TestSpec(
        name="doclayout_yolo_cpu",
        module="layout_extraction.doclayout_yolo_cpu",
        backend=Backend.PYTORCH_CPU,
        task=Task.LAYOUT,
        gpu_type=None,
        tags=["yolo", "fast"],
    ),
    # RT-DETR
    TestSpec(
        name="rtdetr_gpu",
        module="layout_extraction.rtdetr_gpu",
        backend=Backend.PYTORCH_GPU,
        task=Task.LAYOUT,
        gpu_type="T4:1",
        tags=["rtdetr", "transformer"],
    ),
    TestSpec(
        name="rtdetr_cpu",
        module="layout_extraction.rtdetr_cpu",
        backend=Backend.PYTORCH_CPU,
        task=Task.LAYOUT,
        gpu_type=None,
        tags=["rtdetr", "transformer"],
    ),
    # ============================================================
    # OCR EXTRACTION
    # ============================================================
    # EasyOCR
    TestSpec(
        name="easyocr_gpu",
        module="ocr_extraction.easyocr_gpu",
        backend=Backend.PYTORCH_GPU,
        task=Task.OCR,
        gpu_type="T4:1",
        tags=["easyocr"],
    ),
    TestSpec(
        name="easyocr_cpu",
        module="ocr_extraction.easyocr_cpu",
        backend=Backend.PYTORCH_CPU,
        task=Task.OCR,
        gpu_type=None,
        tags=["easyocr"],
    ),
    # PaddleOCR
    TestSpec(
        name="paddleocr_gpu",
        module="ocr_extraction.paddleocr_gpu",
        backend=Backend.PYTORCH_GPU,
        task=Task.OCR,
        gpu_type="T4:1",
        tags=["paddleocr"],
    ),
    TestSpec(
        name="paddleocr_cpu",
        module="ocr_extraction.paddleocr_cpu",
        backend=Backend.PYTORCH_CPU,
        task=Task.OCR,
        gpu_type=None,
        tags=["paddleocr"],
    ),
    # Tesseract
    TestSpec(
        name="tesseract_gpu",
        module="ocr_extraction.tesseract_gpu",
        backend=Backend.PYTORCH_GPU,
        task=Task.OCR,
        gpu_type="T4:1",
        tags=["tesseract"],
    ),
    TestSpec(
        name="tesseract_cpu",
        module="ocr_extraction.tesseract_cpu",
        backend=Backend.PYTORCH_CPU,
        task=Task.OCR,
        gpu_type=None,
        tags=["tesseract"],
    ),
    # ============================================================
    # TABLE EXTRACTION
    # ============================================================
    TestSpec(
        name="tableformer_gpu",
        module="table_extraction.tableformer_gpu",
        backend=Backend.PYTORCH_GPU,
        task=Task.TABLE,
        gpu_type="T4:1",
        tags=["tableformer", "transformer"],
    ),
    TestSpec(
        name="tableformer_cpu",
        module="table_extraction.tableformer_cpu",
        backend=Backend.PYTORCH_CPU,
        task=Task.TABLE,
        gpu_type=None,
        tags=["tableformer", "transformer"],
    ),
    # ============================================================
    # READING ORDER
    # ============================================================
    TestSpec(
        name="reading_order_rule_based",
        module="reading_order.rule_based",
        backend=Backend.PYTORCH_CPU,
        task=Task.READING,
        gpu_type=None,
        tags=["rule_based"],
    ),
]


def get_tests(
    task: Optional[Task] = None,
    backend: Optional[Backend] = None,
    gpu_only: bool = False,
    cpu_only: bool = False,
    tags: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
) -> List[TestSpec]:
    """
    Filter tests by task, backend, compute type, or tags.

    Args:
        task: Filter by task category (e.g., Task.TEXT)
        backend: Filter by backend type (e.g., Backend.VLLM)
        gpu_only: Only return tests that require GPU
        cpu_only: Only return tests that run on CPU
        tags: Filter by tags (tests must have ALL specified tags)
        names: Filter by specific test names

    Returns:
        Filtered list of TestSpec objects
    """
    tests = TEST_REGISTRY.copy()

    if task:
        tests = [t for t in tests if t.task == task]

    if backend:
        tests = [t for t in tests if t.backend == backend]

    if gpu_only:
        tests = [t for t in tests if t.gpu_type is not None]

    if cpu_only:
        tests = [t for t in tests if t.gpu_type is None]

    if tags:
        tests = [t for t in tests if all(tag in t.tags for tag in tags)]

    if names:
        tests = [t for t in tests if t.name in names]

    return tests


def get_test_by_name(name: str) -> Optional[TestSpec]:
    """Get a single test by name."""
    for test in TEST_REGISTRY:
        if test.name == name:
            return test
    return None


def list_tests() -> None:
    """Print all registered tests in a formatted table."""
    print(f"{'Name':<30} {'Task':<20} {'Backend':<15} {'GPU':<10}")
    print("-" * 75)
    for t in TEST_REGISTRY:
        gpu = t.gpu_type or "CPU"
        print(f"{t.name:<30} {t.task.value:<20} {t.backend.value:<15} {gpu:<10}")
