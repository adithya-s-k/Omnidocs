"""
Modal Benchmark Runner for OmniDocs.

Runs benchmarks on Modal GPUs using the benchmark logic from benchmarks/.
This file only owns Modal infra (images, app, GPU functions, entrypoint) —
all dataset loading, inference loops, scoring, and reporting are imported
directly from the benchmarks/ package.

Usage:
    cd Omnidocs

    # OmniDocBench (omnidocbench end-to-end eval)
    modal run tests/benchmark/modal_benchmark.py --benchmark omnidocbench
    modal run tests/benchmark/modal_benchmark.py --benchmark omnidocbench --models glmocr,deepseek
    modal run tests/benchmark/modal_benchmark.py --benchmark omnidocbench --max-samples 10 --models qwen
    modal run tests/benchmark/modal_benchmark.py --benchmark omnidocbench --output-dir results/run_01

    # olmOCR-Bench (binary unit-test pass-rates)
    modal run tests/benchmark/modal_benchmark.py --benchmark olmocr
    modal run tests/benchmark/modal_benchmark.py --benchmark olmocr --models glmocr,deepseek
    modal run tests/benchmark/modal_benchmark.py --benchmark olmocr --splits arxiv_math,table_tests
    modal run tests/benchmark/modal_benchmark.py --benchmark olmocr --max-per-split 20
    modal run tests/benchmark/modal_benchmark.py --benchmark olmocr --output results/olmocr_run01.json

    # Multilingual / NayanaOCRBench
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual --models glmocr,nanonets
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual --languages en,hi,kn
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual --max-per-language 10
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual --output-dir results/nayana_run01

    # Skip inference and re-run eval on existing output files
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual --eval-only --output-dir results/nayana_run01

    # Resume an interrupted run (all benchmarks)
    # Re-run the same command — completed models are detected via run_state.json
    # and skipped automatically; only missing or failed models are re-spawned.
    modal run tests/benchmark/modal_benchmark.py --benchmark multilingual --output-dir results/nayana_run01

    # List models / splits / languages
    modal run tests/benchmark/modal_benchmark.py --benchmark olmocr --list-info

Parameters:
    --benchmark         Which benchmark to run: omnidocbench | olmocr | multilingual.
    --models            Comma-separated model keys to run (default: all registered models).
    --output-dir        Directory for .md outputs and results.json
                        (omnidocbench, multilingual; default: results/<benchmark>/<run_id>/).
    --output            JSON output file path (olmocr only;
                        default: results/olmocrbench/<run_id>/results.json).
    --max-samples       Max pages to load from OmniDocBench (omnidocbench only; default: all).
    --splits            Comma-separated olmOCR-Bench split names (olmocr only; default: all 7).
    --max-per-split     Max test cases per split (olmocr only; default: all).
    --languages         Comma-separated language codes (multilingual only; default: all 22).
    --max-per-language  Max pages per language (multilingual only; default: all).
    --list-info         Print available models/splits/languages and exit (olmocr only).
    --eval-only         Skip inference and re-run eval on existing .md files in --output-dir
                        (multilingual only). Requires a prior inference run in that directory.

Resume & failure management:
    All benchmarks persist a run_state.json file inside --output-dir after each
    model is successfully collected.  Re-running the exact same command will
    skip any model already recorded in run_state.json and only process the
    remaining ones.

    For omnidocbench and multilingual local runners, inference is also
    crash-safe at the page level: each page's .md file is written to disk
    immediately after prediction, so a mid-run crash loses at most one page.
    On resume, any page whose .md file already exists and is non-empty is
    skipped automatically.

    Models that fail to spawn or whose Modal call raises an exception are
    recorded with an error in run_state.json but are NOT added to
    completed_models, so they will be retried on the next run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# tests/benchmark/ -> tests/ -> Omnidocs/
OMNIDOCS_DIR = Path(__file__).parent.parent.parent
MODEL_CACHE = "/data/.cache"

# ---------------------------------------------------------------------------
# Shared image-build constants
# ---------------------------------------------------------------------------

_cuda_pytorch = "12.8.0"
_cuda_vllm = "12.8.1"
_flash_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
_ignore = ["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"]
_env_base = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "OMNIDOCS_MODELS_DIR": MODEL_CACHE,
    "HF_HOME": MODEL_CACHE,
}

# ---------------------------------------------------------------------------
# Modal images — one per model family
# ---------------------------------------------------------------------------

PYTORCH_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{_cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "poppler-utils")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .run_commands("uv pip install datasets pdf2image pillow --system")
    .uv_pip_install(_flash_wheel)
    .env(_env_base)
)

VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{_cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0", "poppler-utils")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm --system")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[vllm]' --system")
    .run_commands("uv pip install datasets pdf2image pillow --system")
    .env({**_env_base, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
)

GLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{_cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "poppler-utils")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .run_commands("uv pip install datasets pdf2image pillow --system")
    .uv_pip_install(_flash_wheel)
    .env(_env_base)
)

LIGHTON_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{_cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "poppler-utils")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .run_commands("uv pip install datasets pdf2image pillow --system")
    .uv_pip_install(_flash_wheel)
    .env(_env_base)
)

# ---------------------------------------------------------------------------
# Modal app + shared infrastructure
# ---------------------------------------------------------------------------

app = modal.App("omnidocs-benchmarks")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

# ---------------------------------------------------------------------------
# Helpers — serialise/deserialise dataclasses over the Modal boundary
#
# Modal serialises function args/returns with cloudpickle, so passing
# dataclass instances works fine.  We expose thin dict <-> dataclass
# converters here so the remote functions stay decoupled from local imports.
# ---------------------------------------------------------------------------


def _samples_to_dicts(samples) -> list:
    """PageSample list -> plain dicts for Modal transport."""
    return [{"image_bytes": s.image_bytes, "image_name": s.image_name} for s in samples]


def _cases_to_dicts(cases) -> list:
    """OlmTestCase list -> plain dicts for Modal transport."""
    from dataclasses import asdict

    return [asdict(c) for c in cases]


# ---------------------------------------------------------------------------
# ─── BENCHMARK: OmniDocBench ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _omnidocbench_remote(extractor_factory, sample_dicts: list) -> list:
    """
    Runs inside the Modal container.
    Reconstructs PageSamples, delegates to benchmarks.omnidocbench.runner,
    and returns results as plain dicts.
    """
    import sys

    sys.path.insert(0, "/opt/omnidocs")

    from dataclasses import asdict

    from benchmarks.base import PageSample
    from benchmarks.omnidocbench.runner import run_inference_remote

    samples = [PageSample(**d) for d in sample_dicts]
    extractor = extractor_factory()
    results = run_inference_remote(extractor, samples, model_key="remote")
    return [asdict(r) for r in results]


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_qwen(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_qwen

    return _omnidocbench_remote(_make_qwen, sample_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_deepseek(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_deepseek

    return _omnidocbench_remote(_make_deepseek, sample_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_nanonets(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_nanonets

    return _omnidocbench_remote(_make_nanonets, sample_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_granitedocling(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_granitedocling

    return _omnidocbench_remote(_make_granitedocling, sample_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_mineruvl(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_mineruvl

    return _omnidocbench_remote(_make_mineruvl, sample_dicts)


@app.function(image=GLM_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_glmocr(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_glmocr

    return _omnidocbench_remote(_make_glmocr, sample_dicts)


@app.function(image=LIGHTON_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_lighton(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_lighton

    return _omnidocbench_remote(_make_lighton, sample_dicts)


@app.function(image=VLLM_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=3600)
def omnidocbench_infer_dotsocr(sample_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_dotsocr

    return _omnidocbench_remote(_make_dotsocr, sample_dicts)


OMNIDOCBENCH_REGISTRY: Dict[str, object] = {
    "qwen": omnidocbench_infer_qwen,
    "deepseek": omnidocbench_infer_deepseek,
    "nanonets": omnidocbench_infer_nanonets,
    "granitedocling": omnidocbench_infer_granitedocling,
    "mineruvl": omnidocbench_infer_mineruvl,
    "glmocr": omnidocbench_infer_glmocr,
    "lighton": omnidocbench_infer_lighton,
    "dotsocr": omnidocbench_infer_dotsocr,
}


def _run_omnidocbench_benchmark(
    model_ids: List[str],
    max_samples: int,
    output_dir: str,
    eval_only: bool = False,
) -> None:
    import datetime

    from benchmarks.base import PageResult
    from benchmarks.omnidocbench.dataset import load_omnidocbench
    from benchmarks.omnidocbench.runner import write_md_files

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) if output_dir else Path("results") / "omnidocbench" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_root}")

    if eval_only and not output_dir:
        print("ERROR: --output-dir is required when using --eval-only")
        sys.exit(1)

    if not eval_only:
        print("\nLoading OmniDocBench dataset...")
        samples = load_omnidocbench(max_samples=max_samples if max_samples > 0 else None)
        print(f"  → {len(samples)} pages loaded\n")
        if not samples:
            print("No samples loaded. Exiting.")
            sys.exit(1)

        sample_dicts = _samples_to_dicts(samples)

        print(f"Spawning {len(model_ids)} models in parallel...\n")
        futures = {}
        for mid in model_ids:
            print(f"  ↑ Spawning {mid}")
            try:
                futures[mid] = OMNIDOCBENCH_REGISTRY[mid].spawn(sample_dicts)
            except Exception as e:
                print(f"  [ERROR] Failed to spawn {mid}: {e}")
                futures[mid] = None

        summary = {}
        for mid, future in futures.items():
            print(f"\n{'=' * 60}\n  Collecting: {mid}\n{'=' * 60}")
            model_dir = out_root / mid

            if future is None:
                print("  [SKIPPED] spawn failed")
                summary[mid] = {"written": 0, "failed": len(samples)}
                continue

            try:
                result_dicts: list = future.get()
            except Exception as e:
                print(f"  [ERROR] {mid}: {e}")
                summary[mid] = {"written": 0, "failed": len(samples)}
                continue

            results = [PageResult(**d) for d in result_dicts]
            written, failed = write_md_files(results, model_dir)
            summary[mid] = {"written": written, "failed": failed}
            print(f"  Done: {written} written, {failed} failed → {model_dir}")

    else:
        # eval_only: rebuild summary from existing .md files on disk
        summary = {}
        for mid in model_ids:
            model_dir = out_root / mid
            md_files = list(model_dir.glob("*.md")) if model_dir.exists() else []
            written = sum(1 for f in md_files if f.stat().st_size > 0)
            summary[mid] = {"written": written, "failed": len(md_files) - written}

    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE — results in: {out_root}")
    print(f"{'=' * 60}")
    for mid, s in summary.items():
        print(f"  {mid:<18} {s['written']} pages written, {s['failed']} failed")

    from benchmarks.omnidocbench.evaluator import run_evaluation

    eval_scores = run_evaluation(
        run_output_dir=out_root,
        model_keys=[mid for mid, s in summary.items() if s.get("written", 0) > 0],
    )
    results_json = {
        "run_id": run_id,
        "benchmark": "omnidocbench",
        "execution": "modal",
        "models": model_ids,
        "inference": summary,
        "eval_scores": eval_scores,
    }
    (out_root / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_root / 'results.json'}")


# ---------------------------------------------------------------------------
# ─── BENCHMARK: olmOCR-Bench ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _olmocr_remote(extractor_factory, case_dicts: list) -> list:
    """
    Runs inside the Modal container.
    Reconstructs OlmTestCases, delegates to benchmarks.olmocrbench.runner,
    and returns results as plain dicts.
    """
    import sys

    sys.path.insert(0, "/opt/omnidocs")

    from dataclasses import asdict

    from benchmarks.base import OlmTestCase
    from benchmarks.olmocrbench.runner import run_cases

    cases = [OlmTestCase(**d) for d in case_dicts]
    extractor = extractor_factory()
    results = run_cases(extractor, cases, model_key="remote")
    return [asdict(r) for r in results]


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_qwen(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_qwen

    return _olmocr_remote(_make_qwen, case_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_deepseek(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_deepseek

    return _olmocr_remote(_make_deepseek, case_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_nanonets(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_nanonets

    return _olmocr_remote(_make_nanonets, case_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_granitedocling(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_granitedocling

    return _olmocr_remote(_make_granitedocling, case_dicts)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_mineruvl(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_mineruvl

    return _olmocr_remote(_make_mineruvl, case_dicts)


@app.function(image=GLM_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_glmocr(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_glmocr

    return _olmocr_remote(_make_glmocr, case_dicts)


@app.function(image=LIGHTON_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_lighton(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_lighton

    return _olmocr_remote(_make_lighton, case_dicts)


@app.function(image=VLLM_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def olmocr_bench_dotsocr(case_dicts: list) -> list:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_dotsocr

    return _olmocr_remote(_make_dotsocr, case_dicts)


OLMOCR_REGISTRY: Dict[str, object] = {
    "qwen": olmocr_bench_qwen,
    "deepseek": olmocr_bench_deepseek,
    "nanonets": olmocr_bench_nanonets,
    "granitedocling": olmocr_bench_granitedocling,
    "mineruvl": olmocr_bench_mineruvl,
    "glmocr": olmocr_bench_glmocr,
    "lighton": olmocr_bench_lighton,
    "dotsocr": olmocr_bench_dotsocr,
}


def _run_olmocr_benchmark(
    model_ids: List[str],
    split_names: List[str],
    max_per_split: int,
    output: str,
    list_info: bool,
) -> None:
    import datetime

    from benchmarks.olmocrbench.dataset import OLM_SPLITS, load_olmocr_bench
    from benchmarks.olmocrbench.runner import aggregate, print_report

    if list_info:
        print("\nAvailable models:", list(OLMOCR_REGISTRY.keys()))
        print("\nAvailable splits:")
        for s in OLM_SPLITS:
            print(f"  {s}")
        return

    max_samples = max_per_split if max_per_split > 0 else None

    print("\nLoading olmOCR-Bench from HuggingFace...")
    print(f"  Splits: {split_names}  |  Max per split: {max_samples or 'all'}")
    cases = load_olmocr_bench(split_names, max_per_split=max_samples)
    print(f"\nTotal test cases: {len(cases)}\nModels to run: {model_ids}\n")
    if not cases:
        print("No test cases loaded. Exiting.")
        sys.exit(1)

    case_dicts = _cases_to_dicts(cases)

    print(f"Spawning {len(model_ids)} models in parallel...\n")
    futures = {}
    for mid in model_ids:
        print(f"  ↑ Spawning {mid}")
        try:
            futures[mid] = OLMOCR_REGISTRY[mid].spawn(case_dicts)
        except Exception as e:
            print(f"  [ERROR] Failed to spawn {mid}: {e}")
            futures[mid] = None

    from benchmarks.base import OlmResult

    all_metrics = []
    all_raw = {}

    for mid, future in futures.items():
        print(f"\n{'=' * 60}\n  Waiting: {mid}\n{'=' * 60}")

        if future is None:
            results = [
                OlmResult(
                    case_id=c.case_id,
                    split=c.split,
                    check_type=c.check_type,
                    model=mid,
                    passed=False,
                    latency_s=0.0,
                    failed=True,
                    error="spawn failed",
                )
                for c in cases
            ]
        else:
            try:
                result_dicts = future.get()
                results = [OlmResult(**d) for d in result_dicts]
            except Exception as e:
                print(f"  [ERROR] {mid}: {e}")
                results = [
                    OlmResult(
                        case_id=c.case_id,
                        split=c.split,
                        check_type=c.check_type,
                        model=mid,
                        passed=False,
                        latency_s=0.0,
                        failed=True,
                        error=str(e),
                    )
                    for c in cases
                ]

        metrics = aggregate(results, mid)
        all_metrics.append(metrics)
        from dataclasses import asdict

        all_raw[mid] = [asdict(r) for r in results]

        passed_n = sum(1 for r in results if r.passed and not r.failed)
        print(f"  Done: {passed_n}/{len(results)} passed  overall={metrics['overall'] * 100:.1f}%")

    print_report(all_metrics, split_names)

    results_json = {
        "run_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "olmocrbench",
        "execution": "modal",
        "models": model_ids,
        "splits": split_names,
        "inference": {
            m["model"]: {
                "cases_run": m["samples_run"],
                "cases_failed": m["samples_failed"],
            }
            for m in all_metrics
        },
        "eval_scores": {
            m["model"]: {
                "overall": m["overall"],
                "by_split": m["by_split"],
                "by_check": m["by_check"],
                "latency_p50_s": m["latency_p50_s"],
                "latency_p95_s": m["latency_p95_s"],
            }
            for m in all_metrics
        },
    }
    out_path = Path(output) if output else (Path("results") / "olmocrbench" / results_json["run_id"] / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# ─── BENCHMARK: Multilingual / NayanaOCRBench ────────────────────────────────
# ---------------------------------------------------------------------------


def _multilingual_remote(extractor_factory, languages: list, max_per_language) -> dict:
    """
    Runs inside the Modal container.
    Delegates to benchmarks.multilingual.runner and returns plain dicts.
    """
    import sys

    sys.path.insert(0, "/opt/omnidocs")

    from dataclasses import asdict

    from benchmarks.multilingual.dataset import load_multilingual
    from benchmarks.multilingual.runner import run_inference_remote

    samples_by_lang, gt_by_lang = load_multilingual(
        languages=languages,
        max_per_language=max_per_language,
    )

    extractor = extractor_factory()
    all_result_dicts = []

    for lang, samples in samples_by_lang.items():
        results = run_inference_remote(extractor, samples, model_key="remote", lang=lang)
        for r in results:
            d = asdict(r)
            d["language"] = lang  # needed by _run_multilingual_benchmark
            all_result_dicts.append(d)

    # Serialise GT records (plain dicts, already JSON-safe)
    gt_serialised = {lang: records for lang, records in gt_by_lang.items()}

    return {"results": all_result_dicts, "gt_by_language": gt_serialised}


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_qwen(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_qwen

    return _multilingual_remote(_make_qwen, languages, max_per_language)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_deepseek(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_deepseek

    return _multilingual_remote(_make_deepseek, languages, max_per_language)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_nanonets(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_nanonets

    return _multilingual_remote(_make_nanonets, languages, max_per_language)


@app.function(image=PYTORCH_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_granitedocling(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_granitedocling

    return _multilingual_remote(_make_granitedocling, languages, max_per_language)


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_mineruvl(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_mineruvl

    return _multilingual_remote(_make_mineruvl, languages, max_per_language)


@app.function(image=GLM_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_glmocr(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_glmocr

    return _multilingual_remote(_make_glmocr, languages, max_per_language)


@app.function(image=LIGHTON_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_lighton(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_lighton

    return _multilingual_remote(_make_lighton, languages, max_per_language)


@app.function(image=VLLM_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def multilingual_infer_dotsocr(languages: list, max_per_language) -> dict:
    import sys

    sys.path.insert(0, "/opt/omnidocs")
    from benchmarks.registry import _make_dotsocr

    return _multilingual_remote(_make_dotsocr, languages, max_per_language)


MULTILINGUAL_REGISTRY: Dict[str, object] = {
    "qwen": multilingual_infer_qwen,
    "deepseek": multilingual_infer_deepseek,
    "nanonets": multilingual_infer_nanonets,
    "granitedocling": multilingual_infer_granitedocling,
    "mineruvl": multilingual_infer_mineruvl,
    "glmocr": multilingual_infer_glmocr,
    "lighton": multilingual_infer_lighton,
    "dotsocr": multilingual_infer_dotsocr,
}


def _run_multilingual_benchmark(
    model_ids: List[str],
    target_langs: List[str],
    max_per_language: int,
    output_dir: str,
    eval_only: bool = False,
) -> None:
    import datetime

    from benchmarks.base import PageResult
    from benchmarks.multilingual.runner import (
        _load_state,
        _save_state,
        write_md_files,
    )
    from benchmarks.multilingual.runner import run_multilingual as _local_eval_only

    max_pl = max_per_language if max_per_language > 0 else None
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) if output_dir else Path("results") / "nayana" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    if eval_only and not output_dir:
        print("ERROR: --output-dir is required when using --eval-only")
        sys.exit(1)
    if eval_only:
        # Delegate entirely to the local runner's eval-only path
        _local_eval_only(
            model_keys=model_ids,
            languages=target_langs,
            output_dir=out_root,
            run_eval=True,
            eval_only=True,
        )
        return

    print(f"\nOutput directory: {out_root}")
    print(f"Models:           {model_ids}")
    print(f"Languages:        {target_langs}")
    print(f"Max per language: {max_pl or 'all'}")

    # ── Load state so we can skip already-collected models ───────────────────
    state = _load_state(out_root)
    completed_models: list = state.get("completed_models", [])
    summary: dict = state.get("inference", {})

    gt_written = (out_root / "gt").exists() and any((out_root / "gt").glob("*.json"))
    gt_dir = out_root / "gt"

    # ── Spawn only models that haven't been collected yet ────────────────────
    models_to_run = [mid for mid in model_ids if mid not in completed_models]
    if not models_to_run:
        print("\nAll models already collected. Proceeding to eval.")
    else:
        print(f"\nSpawning {len(models_to_run)} models in parallel ({len(completed_models)} already done)...\n")
        futures = {}
        for mid in models_to_run:
            print(f"  ↑ Spawning {mid}")
            try:
                futures[mid] = MULTILINGUAL_REGISTRY[mid].spawn(target_langs, max_pl)
            except Exception as e:
                print(f"  [ERROR] Failed to spawn {mid}: {e}")
                futures[mid] = None

        for mid, future in futures.items():
            print(f"\n{'=' * 60}\n  Collecting: {mid}\n{'=' * 60}")
            if future is None:
                print("  [SKIPPED] spawn failed")
                summary[mid] = {"error": "spawn failed"}
                _save_state(out_root, {"completed_models": completed_models, "inference": summary})
                continue
            try:
                output = future.get()
            except Exception as e:
                print(f"  [ERROR] {mid}: {e}")
                summary[mid] = {"error": str(e)}
                # Don't add to completed_models — it can be retried
                _save_state(out_root, {"completed_models": completed_models, "inference": summary})
                continue

            # Write GT once
            if not gt_written:
                gt_dir.mkdir(exist_ok=True)
                for lang, records in output["gt_by_language"].items():
                    (gt_dir / f"{lang}.json").write_text(
                        json.dumps(records, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                print(f"  GT written: {gt_dir}  ({len(output['gt_by_language'])} languages)")
                gt_written = True

            # Write .md files and count
            lang_stats: Dict[str, dict] = {}
            results_by_lang: Dict[str, list] = {}
            for r_dict in output["results"]:
                lang = r_dict["language"]
                results_by_lang.setdefault(lang, []).append(
                    PageResult(**{k: v for k, v in r_dict.items() if k != "language"})
                )

            for lang, results in results_by_lang.items():
                model_lang_dir = out_root / mid / lang
                written, failed = write_md_files(results, model_lang_dir)
                lang_stats[lang] = {"written": written, "failed": failed}

            summary[mid] = lang_stats
            completed_models.append(mid)

            # ── Persist state after each model is collected ───────────────
            _save_state(out_root, {"completed_models": completed_models, "inference": summary})

            total_w = sum(s["written"] for s in lang_stats.values())
            total_f = sum(s["failed"] for s in lang_stats.values())
            print(f"  Done: {total_w} written, {total_f} failed → {out_root / mid}")

    print(f"\n{'=' * 60}\nINFERENCE COMPLETE — results in: {out_root}\n{'=' * 60}")
    for mid, stats in summary.items():
        if "error" in stats:
            print(f"  {mid:<18} ERROR: {stats['error']}")
        else:
            total_w = sum(s.get("written", 0) for s in stats.values())
            total_f = sum(s.get("failed", 0) for s in stats.values())
            print(f"  {mid:<18} {total_w} written, {total_f} failed")

    # ── Eval ─────────────────────────────────────────────────────────────────
    if gt_written:
        gt_by_lang = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in gt_dir.glob("*.json")}
        from benchmarks.multilingual.evaluator import run_evaluation

        eval_scores = run_evaluation(
            run_output_dir=out_root,
            model_keys=[mid for mid, s in summary.items() if "error" not in s],
            languages=target_langs,
            gt_by_lang=gt_by_lang,
        )
        results_json = {
            "run_id": run_id,
            "benchmark": "multilingual",
            "execution": "modal",
            "models": model_ids,
            "languages": target_langs,
            "inference": summary,
            "eval_scores": eval_scores,
        }
        (out_root / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_root / 'results.json'}")


# ---------------------------------------------------------------------------
# ─── Local entrypoint ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    # ── benchmark selector ──────────────────────────────────────────────────
    benchmark: str = "",  # omnidocbench | olmocr | multilingual
    # ── shared ──────────────────────────────────────────────────────────────
    models: str = "",  # comma-separated model keys (default: all)
    # ── omnidocbench ─────────────────────────────────────────────────────────
    max_samples: int = 0,  # 0 = full dataset
    output_dir: str = "",  # default: results/omnidocbench/<run_id>/
    # ── olmocr ───────────────────────────────────────────────────────────────
    splits: str = "",  # comma-separated split names (default: all 7)
    max_per_split: int = 0,  # 0 = all test cases per split
    output: str = "",  # JSON output path for full results
    list_info: bool = False,  # print models/splits and exit
    # ── multilingual ─────────────────────────────────────────────────────────
    languages: str = "",  # comma-separated language codes (default: all 22)
    max_per_language: int = 0,  # 0 = all pages per language
    eval_only: bool = False,
):
    """
    Unified Modal benchmark runner for OmniDocs.

    Select which benchmark to run with --benchmark:
      omnidocbench  — OmniDocBench end-to-end eval (NED/BLEU/TEDS)
      olmocr        — olmOCR-Bench binary unit-test pass rates
      multilingual  — NayanaOCRBench across 22 languages
    """
    from benchmarks.multilingual.dataset import ALL_LANGUAGES
    from benchmarks.olmocrbench.dataset import OLM_SPLITS

    benchmarks = {"omnidocbench", "olmocr", "multilingual"}

    if not benchmark:
        print(__doc__)
        sys.exit(0)

    if benchmark not in benchmarks:
        print(f"Unknown benchmark: {benchmark!r}. Choose from: {', '.join(sorted(benchmarks))}")
        sys.exit(1)

    # Resolve registry for the chosen benchmark
    registry = {
        "omnidocbench": OMNIDOCBENCH_REGISTRY,
        "olmocr": OLMOCR_REGISTRY,
        "multilingual": MULTILINGUAL_REGISTRY,
    }[benchmark]

    model_ids = [m.strip() for m in models.split(",") if m.strip()] if models else list(registry.keys())
    unknown = [m for m in model_ids if m not in registry]
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list(registry.keys())}")
        sys.exit(1)

    # Dispatch
    if benchmark == "omnidocbench":
        _run_omnidocbench_benchmark(
            model_ids=model_ids,
            max_samples=max_samples,
            output_dir=output_dir,
            eval_only=eval_only,
        )

    elif benchmark == "olmocr":
        split_names = [s.strip() for s in splits.split(",") if s.strip()] if splits else OLM_SPLITS
        bad_splits = [s for s in split_names if s not in OLM_SPLITS]
        if bad_splits:
            print(f"Unknown splits: {bad_splits}. Available: {OLM_SPLITS}")
            sys.exit(1)
        _run_olmocr_benchmark(
            model_ids=model_ids,
            split_names=split_names,
            max_per_split=max_per_split,
            output=output,
            list_info=list_info,
        )

    elif benchmark == "multilingual":
        target_langs = [lang.strip() for lang in languages.split(",") if lang.strip()] if languages else ALL_LANGUAGES
        bad_langs = [lang for lang in target_langs if lang not in ALL_LANGUAGES]
        if bad_langs:
            print(f"Unknown languages: {bad_langs}. Available: {ALL_LANGUAGES}")
            sys.exit(1)
        _run_multilingual_benchmark(
            model_ids=model_ids,
            target_langs=target_langs,
            max_per_language=max_per_language,
            output_dir=output_dir,
            eval_only=eval_only,
        )
