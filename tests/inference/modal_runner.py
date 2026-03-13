"""
Modal Runner for OmniDocs Inference Tests.

Uploads standalone test scripts to Modal containers with omnidocs installed
and executes them via subprocess. Scripts are pure Python (no Modal imports).

Usage:
    cd Omnidocs
    modal run tests/inference/modal_runner.py --list
    modal run tests/inference/modal_runner.py --test qwen_text_vllm
    modal run tests/inference/modal_runner.py --backend vllm
    modal run tests/inference/modal_runner.py --task text_extraction
    modal run tests/inference/modal_runner.py --run-all
"""

import io
import json
import runpy
import sys
import traceback
from pathlib import Path
from typing import Optional

import modal

# ============= Paths =============

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent  # Omnidocs/
SCRIPTS_DIR = SCRIPT_DIR / "scripts"
MODEL_CACHE_DIR = "/data/.cache"

# ============= Modal Images =============

# --- VLLM Image ---
cuda_vllm = "12.8.1"
VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm --system")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[vllm]' --system")
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/opt/test_scripts",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "OMNIDOCS_MODELS_DIR": MODEL_CACHE_DIR,
            "HF_HOME": MODEL_CACHE_DIR,
            "VLLM_USE_V1": "0",
            "VLLM_DISABLE_V1": "1",
        }
    )
)

# --- PyTorch Image ---
cuda_pytorch = "12.8.0"
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
PYTORCH_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .uv_pip_install(flash_attn_wheel)
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/opt/test_scripts",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "OMNIDOCS_MODELS_DIR": MODEL_CACHE_DIR,
            "HF_HOME": MODEL_CACHE_DIR,
        }
    )
)

# --- OCR Image (extends PyTorch with tesseract) ---
OCR_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "libglib2.0-0",
        "libgl1",
        "libglx-mesa0",
        "libgl1-mesa-dri",
        "tesseract-ocr",
        "libtesseract-dev",
    )
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[pytorch,ocr]' --system")
    .uv_pip_install(flash_attn_wheel)
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/opt/test_scripts",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "OMNIDOCS_MODELS_DIR": MODEL_CACHE_DIR,
            "HF_HOME": MODEL_CACHE_DIR,
        }
    )
)

# --- CPU Image (no CUDA) ---
CPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libglib2.0-0",
        "libgl1",
        "libglx-mesa0",
        "libgl1-mesa-dri",
        "tesseract-ocr",
        "libtesseract-dev",
    )
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands("uv pip install '/opt/omnidocs[ocr]' --system")
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/opt/test_scripts",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

GLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .uv_pip_install(flash_attn_wheel)
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/opt/test_scripts",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMNIDOCS_MODELS_DIR": MODEL_CACHE_DIR, "HF_HOME": MODEL_CACHE_DIR})
)

GLM_VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm==0.17.0 --system")
    .add_local_dir(
        str(OMNIDOCS_DIR),
        remote_path="/opt/omnidocs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"],
    )
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[vllm]' --system --override /tmp/overrides.txt"
    )
    .add_local_dir(
        str(SCRIPTS_DIR),
        remote_path="/opt/test_scripts",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMNIDOCS_MODELS_DIR": MODEL_CACHE_DIR, "HF_HOME": MODEL_CACHE_DIR, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
)
# ============= Modal App =============

app = modal.App("omnidocs-inference-tests")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")


# ============= Script Execution =============


def _execute_script(script_module: str) -> dict:
    """Execute a test script in-process via runpy and parse results.

    Runs scripts directly (not via subprocess) so VLLM V1 engine can
    properly initialize CUDA and spawn its worker processes.

    Args:
        script_module: Module path like "text_extraction.qwen_vllm"

    Returns:
        Dict with status, test name, and any parsed results.
    """
    script_path = "/opt/test_scripts/" + script_module.replace(".", "/") + ".py"

    print(f"Executing: {script_path}")
    print("=" * 60)

    # Capture stdout to parse __RESULT_JSON__ line
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = _Tee(old_stdout, captured)

    try:
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as e:
        # Scripts may call sys.exit(0) for SKIP
        if e.code == 0:
            output = captured.getvalue()
            if "SKIP:" in output:
                return {"status": "skipped", "test": script_module}
            # Normal exit
        else:
            sys.stdout = old_stdout
            return {"status": "failed", "test": script_module, "error": f"exit code {e.code}"}
    except Exception:
        sys.stdout = old_stdout
        err = traceback.format_exc()
        print("ERROR:", err[-2000:])
        return {"status": "failed", "test": script_module, "error": err[-500:]}
    finally:
        sys.stdout = old_stdout

    # Parse __RESULT_JSON__ line from captured output
    output = captured.getvalue()
    result = {"status": "success", "test": script_module}

    for line in output.split("\n"):
        if line.startswith("__RESULT_JSON__:"):
            try:
                result = json.loads(line[len("__RESULT_JSON__:") :])
            except json.JSONDecodeError:
                pass
            break

    return result


class _Tee:
    """Write to two streams simultaneously, delegating fileno() to the first real stream."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()


# ============= Modal Functions (one per image+GPU combo) =============


@app.function(
    image=VLLM_IMAGE,
    gpu="L40S:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def run_vllm_test(script_module: str) -> dict:
    """Run a test script on VLLM image with L40S GPU."""
    return _execute_script(script_module)


@app.function(
    image=PYTORCH_IMAGE,
    gpu="A10G:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def run_pytorch_gpu_test(script_module: str) -> dict:
    """Run a test script on PyTorch image with A10G GPU."""
    return _execute_script(script_module)


@app.function(
    image=PYTORCH_IMAGE,
    gpu="T4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def run_pytorch_t4_test(script_module: str) -> dict:
    """Run a test script on PyTorch image with T4 GPU."""
    return _execute_script(script_module)

@app.function(
    image=GLM_IMAGE,
    gpu="A10G:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def run_glm_pytorch_test(script_module: str) -> dict:
    """Run a GLM-OCR PyTorch test (transformers>=5.0.0 image) with A10G GPU."""
    return _execute_script(script_module)


@app.function(
    image=GLM_VLLM_IMAGE,
    gpu="L40S:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=900,
)
def run_glm_vllm_test(script_module: str) -> dict:
    """Run a GLM-OCR VLLM test (vllm==0.17.0 + transformers>=5.0.0) with L40S GPU."""
    return _execute_script(script_module)

@app.function(
    image=OCR_IMAGE,
    gpu="T4:1",
    secrets=[secret],
    volumes={"/data": volume},
    timeout=600,
)
def run_ocr_gpu_test(script_module: str) -> dict:
    """Run a test script on OCR image with T4 GPU."""
    return _execute_script(script_module)


@app.function(
    image=CPU_IMAGE,
    secrets=[secret],
    timeout=600,
)
def run_cpu_test(script_module: str) -> dict:
    """Run a test script on CPU image (no GPU)."""
    return _execute_script(script_module)


# ============= Routing =============

# Map GPU type strings to runner functions
GPU_RUNNERS = {
    "L40S:1": run_vllm_test,
    "A10G:1": run_pytorch_gpu_test,
    "T4:1": run_pytorch_t4_test,
}


def _get_runner(spec):
    """Get the appropriate Modal function for a test spec."""
    import sys

    sys.path.insert(0, str(SCRIPT_DIR))
    from registry import Task

    if spec.gpu_type is None:
        return run_cpu_test

    if "glm" in spec.tags:
        from registry import Backend
        if spec.backend == Backend.VLLM:
            return run_glm_vllm_test
        return run_glm_pytorch_test
    # OCR tasks with GPU go to OCR image
    if spec.task == Task.OCR and spec.gpu_type:
        return run_ocr_gpu_test

    # GPU tests
    runner = GPU_RUNNERS.get(spec.gpu_type)
    if runner:
        return runner

    # Fallback: use T4 for unknown GPU types
    return run_pytorch_t4_test


def _run_test(spec) -> dict:
    """Run a single test and return results."""
    runner = _get_runner(spec)
    return runner.remote(spec.module)


# ============= CLI =============


@app.local_entrypoint()
def main(
    test: str = "",
    list: bool = False,
    run_all: bool = False,
    backend: str = "",
    task: str = "",
):
    """
    Run OmniDocs inference tests on Modal.

    Args:
        test: Run a specific test by name (e.g., "qwen_text_vllm")
        list: List all available tests
        run_all: Run all tests
        backend: Filter by backend (vllm, pytorch_gpu, pytorch_cpu, api)
        task: Filter by task (text_extraction, layout_extraction, ocr_extraction, table_extraction, reading_order)
    """
    # Import registry locally - not available on remote containers
    import sys

    sys.path.insert(0, str(SCRIPT_DIR))
    from registry import Backend, Task, get_test_by_name, get_tests, list_tests

    if list:
        list_tests()
        return

    # Determine which tests to run
    filter_task: Optional[Task] = None
    filter_backend: Optional[Backend] = None

    if task:
        try:
            filter_task = Task(task)
        except ValueError:
            print(f"Unknown task: {task}")
            print(f"Valid tasks: {[t.value for t in Task]}")
            return

    if backend:
        try:
            filter_backend = Backend(backend)
        except ValueError:
            print(f"Unknown backend: {backend}")
            print(f"Valid backends: {[b.value for b in Backend]}")
            return

    if test:
        spec = get_test_by_name(test)
        if not spec:
            print(f"Unknown test: {test}")
            print("\nAvailable tests:")
            list_tests()
            return

        # Skip MLX tests (local only)
        if spec.backend == Backend.MLX:
            print(f"SKIP: {spec.name} is MLX-only (run locally, not on Modal)")
            return

        print(f"Running: {spec.name}")
        result = _run_test(spec)
        _print_single_result(result)
        return

    if run_all or filter_task or filter_backend:
        specs = get_tests(task=filter_task, backend=filter_backend)
        # Filter out MLX tests
        specs = [s for s in specs if s.backend != Backend.MLX]

        if not specs:
            print("No matching tests found.")
            return

        print(f"\nRunning {len(specs)} tests")
        print("=" * 60)

        results = []
        for spec in specs:
            print(f"\n>>> {spec.name} ({spec.backend.value}, {spec.gpu_type or 'CPU'})")
            try:
                result = _run_test(spec)
                results.append(result)
                status = result.get("status", "unknown")
                print(f"    -> {status}")
            except Exception as e:
                results.append({"status": "failed", "test": spec.name, "error": str(e)})
                print(f"    -> FAILED: {e}")

        _print_summary(results)
        return

    # No arguments - show help
    print("Usage:")
    print("  modal run tests/inference/modal_runner.py --list")
    print("  modal run tests/inference/modal_runner.py --test qwen_text_vllm")
    print("  modal run tests/inference/modal_runner.py --backend vllm")
    print("  modal run tests/inference/modal_runner.py --task text_extraction")
    print("  modal run tests/inference/modal_runner.py --run-all")


def _print_single_result(result: dict):
    """Print a single test result."""
    print("\n" + "=" * 60)
    print("TEST RESULT")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key}: {value}")


def _print_summary(results: list):
    """Print summary of all test results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    print(f"Passed:  {passed}/{len(results)}")
    print(f"Failed:  {failed}/{len(results)}")
    print(f"Skipped: {skipped}/{len(results)}")
    print()

    for result in results:
        status = result.get("status", "unknown")
        test_name = result.get("test", "unknown")

        if status == "success":
            load = result.get("load_time", "?")
            infer = result.get("inference_time", "?")
            print(f"  [OK]   {test_name}: load={load}, inference={infer}")
        elif status == "skipped":
            print(f"  [SKIP] {test_name}")
        else:
            error = result.get("error", "unknown")[:80]
            print(f"  [FAIL] {test_name}: {error}")
