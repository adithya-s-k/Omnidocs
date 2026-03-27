"""
OmniDocBench Inference Runner for Modal.

Runs each model on all OmniDocBench page images and downloads one .md file
per page per model — ready to feed directly into the official OmniDocBench
eval pipeline (pdf_validation.py).

Usage:
    # Run all models
    modal run tests/inference/benchmark_omnidocbench.py

    # Run specific models
    modal run tests/inference/benchmark_omnidocbench.py --models glmocr,dotsocr,nanonets

    # Limit pages (fast iteration)
    modal run tests/inference/benchmark_omnidocbench.py --max-samples 10 --models glmocr

    # Custom output dir (default: results/omnidocbench/<run_id>/)
    modal run tests/inference/benchmark_omnidocbench.py --output-dir results/run_01

Output structure (downloaded locally):
    results/omnidocbench/<run_id>/
    ├── glmocr/
    │   ├── eastmoney_59c...pdf_11.md
    │   ├── eastmoney_7ca...pdf_16.md
    │   └── ...
    ├── dotsocr/
    │   └── ...
    └── nanonets/
        └── ...

Then run official eval (see bottom of this file for instructions).
"""

from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent
MODEL_CACHE  = "/data/.cache"

# ---------------------------------------------------------------------------
# Images — identical to existing benchmark_omnidocbench.py
# ---------------------------------------------------------------------------

cuda_vllm        = "12.8.1"
cuda_pytorch     = "12.8.0"
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
_ignore   = ["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"]
_env_base = {"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMNIDOCS_MODELS_DIR": MODEL_CACHE, "HF_HOME": MODEL_CACHE}

PYTORCH_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm --system")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[vllm]' --system")
    .env({**_env_base, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
)

GLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

GLM_VLLM_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_vllm}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev", "libgl1", "libglib2.0-0")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm==0.17.0 --system")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[vllm]' --system --override /tmp/overrides.txt"
    )
    .env({**_env_base, "VLLM_USE_V1": "0", "VLLM_DISABLE_V1": "1"})
)

LIGHTON_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands(
        "echo 'transformers>=5.0.0' > /tmp/overrides.txt && "
        "uv pip install '/opt/omnidocs[pytorch]' --system --override /tmp/overrides.txt"
    )
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app    = modal.App("omnidocs-inference")
secret = modal.Secret.from_name("adithya-hf-wandb")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageSample:
    image_bytes: bytes
    image_name: str   # e.g. "eastmoney_59c...pdf_11.jpg"


@dataclass
class PageResult:
    image_name: str   # stem used to name the .md file
    model: str
    markdown: str     # full model output
    latency_s: float
    failed: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Dataset loader — OmniDocBench only
# ---------------------------------------------------------------------------

def load_omnidocbench(max_samples: int = None) -> List[PageSample]:
    import huggingface_hub
    from PIL import Image as PILImage
    PILImage.MAX_IMAGE_PIXELS = None

    json_path = huggingface_hub.hf_hub_download(
        repo_id="opendatalab/OmniDocBench",
        filename="OmniDocBench.json",
        repo_type="dataset",
    )
    annotations = json.loads(Path(json_path).read_text(encoding="utf-8"))

    samples: List[PageSample] = []
    for ann in annotations:
        if max_samples and len(samples) >= max_samples:
            break

        img_name = Path(ann["page_info"]["image_path"]).name
        try:
            img_path = huggingface_hub.hf_hub_download(
                repo_id="opendatalab/OmniDocBench",
                filename=f"images/{img_name}",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  [skip] {img_name}: {e}")
            continue

        img = PILImage.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        samples.append(PageSample(image_bytes=buf.getvalue(), image_name=img_name))
        print(f"  [{len(samples)}] loaded {img_name}")

    return samples


# ---------------------------------------------------------------------------
# Core inference loop (runs inside Modal container)
# Returns list of PageResult — markdown content travels back over RPC
# ---------------------------------------------------------------------------

def _run_inference(extractor_factory, samples: List[PageSample]) -> List[PageResult]:
    import io as _io
    from PIL import Image

    extractor = extractor_factory()
    results = []

    for s in samples:
        img = Image.open(_io.BytesIO(s.image_bytes)).convert("RGB")
        t0 = time.perf_counter()
        try:
            out = extractor.extract(img, output_format="markdown")
            markdown  = (getattr(out, "plain_text", None) or out.content or "").strip()
            latency   = time.perf_counter() - t0
            results.append(PageResult(
                image_name=s.image_name,
                model=getattr(out, "model_name", None) or "unknown",
                markdown=markdown,
                latency_s=latency,
            ))
            print(f"  ✓ {s.image_name:<60} {latency:.2f}s  {len(markdown)} chars")
        except Exception as exc:
            import traceback
            latency = time.perf_counter() - t0
            results.append(PageResult(
                image_name=s.image_name, model="unknown",
                markdown="", latency_s=latency,
                failed=True, error=str(exc),
            ))
            print(f"  ✗ {s.image_name:<60} FAILED: {exc}")
            traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# One Modal function per model — same image/GPU assignments as before
# ---------------------------------------------------------------------------

@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_qwen(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    return _run_inference(
        lambda: QwenTextExtractor(backend=QwenTextPyTorchConfig(
            model="Qwen/Qwen3-VL-2B-Instruct", device="cuda")),
        samples,
    )

@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_deepseek(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
    from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig
    return _run_inference(
        lambda: DeepSeekOCRTextExtractor(backend=DeepSeekOCRTextPyTorchConfig(
            model="unsloth/DeepSeek-OCR-2", device="cuda",
            torch_dtype="bfloat16", use_flash_attention=False, crop_mode=True)),
        samples,
    )

@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_nanonets(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig
    return _run_inference(
        lambda: NanonetsTextExtractor(backend=NanonetsTextPyTorchConfig(device="cuda")),
        samples,
    )

@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_granitedocling(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
    from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextPyTorchConfig
    return _run_inference(
        lambda: GraniteDoclingTextExtractor(backend=GraniteDoclingTextPyTorchConfig(
            device="cuda", torch_dtype="bfloat16", use_flash_attention=False)),
        samples,
    )

@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_mineruvl(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig
    return _run_inference(
        lambda: MinerUVLTextExtractor(backend=MinerUVLTextPyTorchConfig(device="cuda")),
        samples,
    )

@app.function(image=GLM_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_glmocr(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
    from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig
    return _run_inference(
        lambda: GLMOCRTextExtractor(backend=GLMOCRPyTorchConfig(device="cuda")),
        samples,
    )

@app.function(image=LIGHTON_IMAGE, gpu="A10G:1", secrets=[secret], timeout=3600)
def infer_lighton(samples: List[PageSample]) -> List[PageResult]:
    from omnidocs.tasks.text_extraction import LightOnTextExtractor
    from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig
    return _run_inference(
        lambda: LightOnTextExtractor(backend=LightOnTextPyTorchConfig(device="cuda")),
        samples,
    )

@app.function(image=VLLM_IMAGE, gpu="L40S:1", secrets=[secret], timeout=3600)
def infer_dotsocr(samples: List[PageSample]) -> List[PageResult]:
    import os
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch; torch.cuda.init()
    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig
    return _run_inference(
        lambda: DotsOCRTextExtractor(backend=DotsOCRVLLMConfig(
            model="rednote-hilab/dots.ocr",
            gpu_memory_utilization=0.90, max_model_len=32768, enforce_eager=True)),
        samples,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen":           infer_qwen,
    "deepseek":       infer_deepseek,
    "nanonets":       infer_nanonets,
    "granitedocling": infer_granitedocling,
    "mineruvl":       infer_mineruvl,
    "glmocr":         infer_glmocr,
    "lighton":        infer_lighton,
    "dotsocr":        infer_dotsocr,
}


# ---------------------------------------------------------------------------
# Local entrypoint — runs inference, downloads .md files directly
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    models:      str = "",
    max_samples: int = 0,       # 0 = full dataset (all 1355 pages)
    output_dir:  str = "",
):
    import datetime

    # Resolve models
    model_ids = [m.strip() for m in models.split(",") if m.strip()] \
                if models else list(MODEL_REGISTRY.keys())
    unknown = [m for m in model_ids if m not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Output directory
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) if output_dir else Path("results") / "omnidocbench" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_root}")

    # Load dataset locally
    print("\nLoading OmniDocBench dataset...")
    samples = load_omnidocbench(max_samples=max_samples if max_samples > 0 else None)
    print(f"  → {len(samples)} pages loaded\n")

    if not samples:
        print("No samples loaded. Exiting.")
        sys.exit(1)

    # Spawn all models in parallel
    print(f"Spawning {len(model_ids)} models in parallel...\n")
    futures = {}
    for model_id in model_ids:
        print(f"  ↑ Spawning {model_id}")
        try:
            futures[model_id] = MODEL_REGISTRY[model_id].spawn(samples)
        except Exception as e:
            print(f"  [ERROR] Failed to spawn {model_id}: {e}")
            futures[model_id] = None

    # Collect results and write .md files directly to local disk
    summary = {}
    for model_id, future in futures.items():
        print(f"\n{'='*60}\n  Collecting: {model_id}\n{'='*60}")

        model_dir = out_root / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        if future is None:
            print(f"  [SKIPPED] spawn failed")
            summary[model_id] = {"written": 0, "failed": len(samples)}
            continue

        try:
            results: List[PageResult] = future.get()
        except Exception as e:
            print(f"  [ERROR] {model_id}: {e}")
            summary[model_id] = {"written": 0, "failed": len(samples)}
            continue

        written = 0
        failed  = 0
        for r in results:
            # Filename: same stem as image, .md extension
            # e.g. "eastmoney_59c...pdf_11.jpg" → "eastmoney_59c...pdf_11.md"
            stem    = Path(r.image_name).stem
            md_path = model_dir / f"{stem}.md"

            if r.failed or not r.markdown:
                # Write empty file so eval pipeline knows this page was attempted
                md_path.write_text("", encoding="utf-8")
                failed += 1
                print(f"  ✗ {r.image_name} — {r.error[:80]}")
            else:
                md_path.write_text(r.markdown, encoding="utf-8")
                written += 1

        summary[model_id] = {"written": written, "failed": failed}
        print(f"  Done: {written} written, {failed} failed → {model_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE — results in: {out_root}")
    print(f"{'='*60}")
    for model_id, s in summary.items():
        print(f"  {model_id:<18} {s['written']} pages written, {s['failed']} failed")

    # Print eval instructions
    print(f"""
{'='*60}
HOW TO RUN OFFICIAL OMNIDOCBENCH EVAL
{'='*60}

1. Clone the official eval repo (once):
   git clone https://github.com/opendatalab/OmniDocBench
   cd OmniDocBench
   pip install -r requirements.txt

2. Download OmniDocBench.json (if not already cached by HF):
   Already at: ~/.cache/huggingface/hub/.../OmniDocBench.json
   Or download from: https://huggingface.co/datasets/opendatalab/OmniDocBench

3. For each model, create a config file and run eval:

   Example for glmocr — create configs/my_glmocr.yaml:

       pred:
         data_path: {(out_root / 'glmocr').resolve()}
         data_type: md

       gt:
         data_path: /path/to/OmniDocBench.json
         data_type: json

       tasks:
         - end2end

   Then run:
   python pdf_validation.py --config configs/my_glmocr.yaml

4. Results will be printed to stdout and saved as JSON next to the config.

   Repeat step 3 for each model directory under:
   {out_root.resolve()}
""")