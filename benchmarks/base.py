"""
benchmarks/base.py
Shared dataclasses and the abstract BenchmarkRunner base used by both
OmniDocBench and olmOCR-bench runners.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# OmniDocBench dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PageSample:
    """One page image from OmniDocBench."""

    image_bytes: bytes
    image_name: str  # e.g. "eastmoney_59c...pdf_11.jpg"


@dataclass
class PageResult:
    """Inference result for one OmniDocBench page."""

    image_name: str
    model: str
    markdown: str
    latency_s: float
    failed: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# olmOCR-bench dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OlmTestCase:
    """One unit-test case from an olmOCR-bench JSONL file."""

    pdf_bytes: bytes  # raw bytes of the single-page PDF
    page_num: int  # 0-indexed page in the original source doc
    check_type: str  # text_present | text_absent | reading_order | table | math
    split: str  # which JSONL split this came from
    case_id: str  # unique identifier for reporting
    payload: dict  # full original JSON record (for scorer)


@dataclass
class OlmResult:
    """Scored result for one olmOCR-bench test case."""

    case_id: str
    split: str
    check_type: str
    model: str
    passed: bool
    latency_s: float
    failed: bool = False  # True if the extractor raised an exception
    error: str = ""
    gt: str = ""  # ground-truth value(s) extracted from payload
    predicted: str = ""  # raw model output (truncated for logging)


# ---------------------------------------------------------------------------
# Abstract runner
# ---------------------------------------------------------------------------


class BenchmarkRunner(ABC):
    """
    Base class for local benchmark runners.

    Subclasses implement `run_model(extractor, samples)` and
    `score_results(results)`.  The top-level `run()` method handles
    model instantiation from the registry, result collection, and
    writing the summary JSON.
    """

    def __init__(
        self,
        model_keys: List[str],
        output_dir: Path,
        max_samples: Optional[int] = None,
    ):
        self.model_keys = model_keys
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples

    @abstractmethod
    def load_dataset(self) -> Any:
        """Load and return the benchmark dataset samples."""

    @abstractmethod
    def run_model(self, extractor, samples: Any) -> Any:
        """Run inference for a single model over all samples."""

    @abstractmethod
    def score_and_report(self, all_results: dict) -> dict:
        """Score collected results and return a metrics dict."""
