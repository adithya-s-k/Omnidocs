"""
Base test class for standalone OmniDocs tests.

Provides common functionality for all standalone test scripts including
timing, result formatting, and error handling.
"""

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image


@dataclass
class TestResult:
    """Result from a standalone test run."""

    success: bool
    test_name: str
    backend: str
    task: str
    load_time: float  # seconds
    inference_time: float  # seconds
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Total execution time."""
        return self.load_time + self.inference_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "test_name": self.test_name,
            "backend": self.backend,
            "task": self.task,
            "load_time": self.load_time,
            "inference_time": self.inference_time,
            "total_time": self.total_time,
            "error": self.error,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        status = "PASS" if self.success else "FAIL"
        lines = [
            f"[{status}] {self.test_name}",
            f"  Backend: {self.backend}",
            f"  Task: {self.task}",
            f"  Load time: {self.load_time:.2f}s",
            f"  Inference time: {self.inference_time:.2f}s",
            f"  Total time: {self.total_time:.2f}s",
        ]
        if self.error:
            lines.append(f"  Error: {self.error}")
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class BaseOmnidocsTest(ABC):
    """
    Base class for standalone OmniDocs tests.

    Subclasses must implement:
        - test_name: Property returning the test name
        - backend_name: Property returning the backend name
        - task_name: Property returning the task category
        - create_extractor(): Create the extractor instance
        - run_extraction(extractor, image): Run extraction and return output
    """

    @property
    @abstractmethod
    def test_name(self) -> str:
        """Unique test name, e.g., 'qwen_text_vllm'."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Backend identifier, e.g., 'vllm', 'pytorch_gpu', 'mlx'."""
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task category, e.g., 'text_extraction', 'layout_extraction'."""
        pass

    @abstractmethod
    def create_extractor(self) -> Any:
        """Create and return the extractor instance."""
        pass

    @abstractmethod
    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        """Run extraction on the image and return the result."""
        pass

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        """
        Extract metadata from the extraction result.

        Override in subclasses to provide task-specific metadata.
        """
        return {}

    def run(self, image: Union[str, Path, Image.Image]) -> TestResult:
        """
        Run the test on the given image.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            TestResult with timing and output information
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        try:
            # Create extractor with timing
            start = time.time()
            extractor = self.create_extractor()
            load_time = time.time() - start

            # Run extraction with timing
            start = time.time()
            output = self.run_extraction(extractor, image)
            inference_time = time.time() - start

            # Get metadata from result
            metadata = self.get_metadata(output)

            return TestResult(
                success=True,
                test_name=self.test_name,
                backend=self.backend_name,
                task=self.task_name,
                load_time=load_time,
                inference_time=inference_time,
                output=output,
                metadata=metadata,
            )

        except Exception as e:
            return TestResult(
                success=False,
                test_name=self.test_name,
                backend=self.backend_name,
                task=self.task_name,
                load_time=0.0,
                inference_time=0.0,
                error=str(e),
            )


def run_standalone_test(test_class: type, default_image: Optional[str] = None) -> None:
    """
    Helper function to run a standalone test from command line.

    Usage in test scripts:
        if __name__ == "__main__":
            run_standalone_test(MyTest)

    Args:
        test_class: The test class to instantiate and run
        default_image: Optional default image path if none provided
    """
    if len(sys.argv) < 2 and default_image is None:
        print(f"Usage: python -m tests.standalone.{test_class.__module__} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    test = test_class()
    result = test.run(image_path)
    print(result)
    sys.exit(0 if result.success else 1)
