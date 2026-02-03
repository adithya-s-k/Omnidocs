"""
Local Test Runner for OmniDocs.

Runs OmniDocs tests locally, primarily for CPU and MLX (Apple Silicon) tests.
Can also run GPU tests if a local GPU is available.

Usage:
    cd Omnidocs
    python -m tests.runners.local_runner --test easyocr_cpu --image test.png
    python -m tests.runners.local_runner --cpu-only --image test.png
    python -m tests.runners.local_runner --mlx --image test.png
    python -m tests.runners.local_runner --list
"""

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from tests.runners.registry import Backend, Task, TestSpec, get_tests


def run_test(spec: TestSpec, image: Image.Image) -> Dict:
    """
    Run a single test and return the result.

    Args:
        spec: Test specification
        image: PIL Image to test with

    Returns:
        Dictionary with test results
    """
    module_path = f"tests.standalone.{spec.module}"

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        return {
            "success": False,
            "test_name": spec.name,
            "error": f"Failed to import module {module_path}: {e}",
            "load_time": 0,
            "inference_time": 0,
        }

    # Find the test class
    test_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and name.endswith("Test")
            and name != "BaseOmnidocsTest"
        ):
            test_class = obj
            break

    if test_class is None:
        return {
            "success": False,
            "test_name": spec.name,
            "error": f"No test class found in {module_path}",
            "load_time": 0,
            "inference_time": 0,
        }

    # Run the test
    test = test_class()
    result = test.run(image)
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Local test runner for OmniDocs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available tests
    python -m tests.runners.local_runner --list

    # Run a specific test
    python -m tests.runners.local_runner --test easyocr_cpu --image test.png

    # Run all CPU tests
    python -m tests.runners.local_runner --cpu-only --image test.png

    # Run MLX tests (Apple Silicon)
    python -m tests.runners.local_runner --backend mlx --image test.png

    # Run tests for a specific task
    python -m tests.runners.local_runner --task ocr_extraction --cpu-only --image test.png
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[t.value for t in Task],
        help="Filter by task category",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=[b.value for b in Backend],
        help="Filter by backend type",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only run CPU tests",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Only run GPU tests (requires local GPU)",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Run MLX tests (Apple Silicon)",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run a specific test by name",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests without running",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Parse filters
    task_enum = Task(args.task) if args.task else None
    backend_enum = Backend(args.backend) if args.backend else None

    # Handle MLX shortcut
    if args.mlx:
        backend_enum = Backend.MLX

    names = [args.test] if args.test else None

    # Get filtered tests
    tests = get_tests(
        task=task_enum,
        backend=backend_enum,
        gpu_only=args.gpu_only,
        cpu_only=args.cpu_only,
        names=names,
    )

    if args.list:
        print(f"\n{'Name':<30} {'Task':<20} {'Backend':<15} {'GPU':<10}")
        print("-" * 75)
        for t in tests:
            gpu = t.gpu_type or "CPU"
            print(f"{t.name:<30} {t.task.value:<20} {t.backend.value:<15} {gpu:<10}")
        print(f"\nTotal: {len(tests)} tests")
        return 0

    if not args.image:
        print("Error: --image is required to run tests")
        print("Usage: python -m tests.runners.local_runner --image test.png --cpu-only")
        return 1

    # Load test image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    image = Image.open(image_path)

    print(f"\nRunning {len(tests)} tests")
    print(f"Image: {image_path}")
    print("-" * 75)

    results = []
    start_time = datetime.now()

    for spec in tests:
        print(f"  Running {spec.name}...", end="", flush=True)
        result = run_test(spec, image)
        results.append(result)

        status = "PASS" if result["success"] else "FAIL"
        time_str = f"{result.get('inference_time', 0):.2f}s"
        print(f" [{status}] ({time_str})")

        if not result["success"] and result.get("error"):
            print(f"    Error: {result['error']}")

    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print("\n" + "=" * 75)
    print(f"SUMMARY: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print("=" * 75)

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['test_name']}: {r.get('error', 'Unknown error')}")

    # Write results to file
    output_file = args.output or f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
