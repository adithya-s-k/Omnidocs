"""
NayanaOCRBench Inference Runner for Modal.

Downloads the dataset INSIDE the Modal container, runs inference, and returns
both the .md predictions and the per-language GT JSON back to the local
entrypoint for writing to disk.

Dataset: v1v1d1/NayanaOCRBench_Natural_final_transformed
  - 22 language subsets: ar, bn, de, en, es, fr, gu, hi, it, ja, kn, ko,
                          ml, mr, or, pa, ru, sa, ta, te, th, zh
  - Each row: image + id + omnidocbench (GT JSON, same schema as OmniDocBench)

Usage:
    # Run all models, all languages
    modal run tests/benchmark/benchmark_multilingual.py

    # Run specific models
    modal run tests/benchmark/benchmark_multilingual.py --models glmocr,nanonets

    # Run specific languages
    modal run tests/benchmark/benchmark_multilingual.py --languages en,hi,kn

    # Limit pages per language (fast iteration)
    modal run tests/benchmark/benchmark_multilingual.py --max-per-language 10 --models glmocr

    # Custom output dir (default: results/nayana/<run_id>/)
    modal run tests/benchmark/benchmark_multilingual.py --output-dir results/nayana_run01

Output structure (downloaded locally):
    results/nayana/<run_id>/
    ├── gt/
    │   ├── en.json          <- per-language GT JSON (OmniDocBench format)
    │   ├── hi.json
    │   └── ...
    ├── glmocr/
    │   ├── en/
    │   │   ├── en_pdf_0000_page_00001.md
    │   │   └── ...
    │   ├── hi/
    │   └── ...
    ├── nanonets/
    │   └── ...
    └── summary.json

Then run official eval per model per language:
    python pdf_validation.py --config configs/<run_id>_glmocr_en.yaml
    (see eval instructions printed at the end)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OMNIDOCS_DIR = SCRIPT_DIR.parent.parent
MODEL_CACHE = "/data/.cache"

# ---------------------------------------------------------------------------
# Images — identical to benchmark_official.py, with `datasets` added
# ---------------------------------------------------------------------------

cuda_vllm = "12.8.1"
cuda_pytorch = "12.8.0"
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)
_ignore = ["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/.*"]
_env_base = {"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMNIDOCS_MODELS_DIR": MODEL_CACHE, "HF_HOME": MODEL_CACHE}

PYTORCH_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_pytorch}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri")
    .run_commands("pip install uv")
    .add_local_dir(str(OMNIDOCS_DIR), remote_path="/opt/omnidocs", copy=True, ignore=_ignore)
    .run_commands("uv pip install '/opt/omnidocs[pytorch]' --system")
    .run_commands("uv pip install datasets --system")
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
    .run_commands("uv pip install datasets --system")
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
    .run_commands("uv pip install datasets --system")
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
    .run_commands("uv pip install datasets --system")
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
    .run_commands("uv pip install datasets --system")
    .uv_pip_install(flash_attn_wheel)
    .env(_env_base)
)

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("omnidocs-nayana-bench")
volume = modal.Volume.from_name("omnidocs", create_if_missing=True)
secret = modal.Secret.from_name("adithya-hf-wandb")

# ---------------------------------------------------------------------------
# All 22 language subsets in NayanaOCRBench_Natural_final_transformed
# ---------------------------------------------------------------------------

ALL_LANGUAGES = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "gu",
    "hi",
    "it",
    "ja",
    "kn",
    "ko",
    "ml",
    "mr",
    "or",
    "pa",
    "ru",
    "sa",
    "ta",
    "te",
    "th",
    "zh",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    image_name: str  # e.g. "en_pdf_0000_page_00001"
    language: str  # e.g. "en"
    model: str
    markdown: str
    latency_s: float
    failed: bool = False
    error: str = ""


@dataclass
class InferenceOutput:
    """
    Bundles predictions + GT in a single RPC return value.
    GT records are in OmniDocBench JSON format so pdf_validation.py
    can consume them without any changes — just a different gt_data_path.
    """

    results: List[PageResult]
    gt_by_language: Dict[str, List[dict]]  # lang -> list of omnidocbench GT records


# ---------------------------------------------------------------------------
# Core: dataset download + inference — runs INSIDE the Modal container
# ---------------------------------------------------------------------------


def _load_and_infer(
    extractor_factory,
    languages: List[str],
    max_per_language: Optional[int],
) -> InferenceOutput:
    """
    1. Downloads NayanaOCRBench from HF (cached in /data/.cache via volume).
    2. Runs extractor.extract() on each page image.
    3. Returns results + GT together — no data sent from local machine.
    """
    from datasets import load_dataset
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = None

    extractor = extractor_factory()

    all_results: List[PageResult] = []
    gt_by_language: Dict[str, List[dict]] = {}

    for lang in languages:
        print(f"\n  [{lang}] Downloading dataset subset...")
        try:
            ds = load_dataset(
                "v1v1d1/NayanaOCRBench_Natural_final_transformed",
                name=lang,
                split="train",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [{lang}] SKIP — could not load: {e}")
            continue

        gt_records = []
        loaded = 0

        for row in ds:
            if max_per_language is not None and loaded >= max_per_language:
                break

            image_name = row["id"]
            pil_image = row["image"].convert("RGB")

            # GT record — OmniDocBench JSON schema, may be pre-parsed or a string
            gt_record = row["omnidocbench"]
            if isinstance(gt_record, str):
                gt_record = json.loads(gt_record)
            for element in gt_record.get("layout_dets", []):
                if element.get("category_type") == "table" and element.get("html") is None:
                    text_val = element.get("text", "")
                    if text_val and text_val.strip().startswith("<table"):
                        element["html"] = text_val

            gt_records.append(gt_record)

            # Run inference directly on the PIL image — no byte serialisation needed
            t0 = time.perf_counter()
            try:
                out = extractor.extract(pil_image, output_format="markdown")
                markdown = (getattr(out, "plain_text", None) or out.content or "").strip()
                latency = time.perf_counter() - t0
                all_results.append(
                    PageResult(
                        image_name=image_name,
                        language=lang,
                        model=getattr(out, "model_name", None) or "unknown",
                        markdown=markdown,
                        latency_s=latency,
                    )
                )
                print(f"  ✓ [{lang}] {image_name:<55} {latency:.2f}s  {len(markdown)} chars")
            except Exception as exc:
                import traceback

                latency = time.perf_counter() - t0
                all_results.append(
                    PageResult(
                        image_name=image_name,
                        language=lang,
                        model="unknown",
                        markdown="",
                        latency_s=latency,
                        failed=True,
                        error=str(exc),
                    )
                )
                print(f"  ✗ [{lang}] {image_name:<55} FAILED: {exc}")
                traceback.print_exc()

            loaded += 1

        gt_by_language[lang] = gt_records
        print(f"  [{lang}] Done — {loaded} pages processed")

    return InferenceOutput(results=all_results, gt_by_language=gt_by_language)


# ---------------------------------------------------------------------------
# One Modal function per model — same image/GPU as benchmark_official.py
# ---------------------------------------------------------------------------


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_qwen(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

    return _load_and_infer(
        lambda: QwenTextExtractor(backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-2B-Instruct", device="cuda")),
        languages,
        max_per_language,
    )


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_deepseek(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
    from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

    return _load_and_infer(
        lambda: DeepSeekOCRTextExtractor(
            backend=DeepSeekOCRTextPyTorchConfig(
                model="unsloth/DeepSeek-OCR-2",
                device="cuda",
                torch_dtype="bfloat16",
                use_flash_attention=False,
                crop_mode=True,
            )
        ),
        languages,
        max_per_language,
    )


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_nanonets(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

    return _load_and_infer(
        lambda: NanonetsTextExtractor(backend=NanonetsTextPyTorchConfig(device="cuda")),
        languages,
        max_per_language,
    )


@app.function(image=PYTORCH_IMAGE, gpu="L40s:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_granitedocling(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
    from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextPyTorchConfig

    return _load_and_infer(
        lambda: GraniteDoclingTextExtractor(
            backend=GraniteDoclingTextPyTorchConfig(device="cuda", torch_dtype="bfloat16", use_flash_attention=False)
        ),
        languages,
        max_per_language,
    )


@app.function(image=PYTORCH_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_mineruvl(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

    return _load_and_infer(
        lambda: MinerUVLTextExtractor(backend=MinerUVLTextPyTorchConfig(device="cuda")),
        languages,
        max_per_language,
    )


@app.function(image=GLM_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_glmocr(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
    from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

    return _load_and_infer(
        lambda: GLMOCRTextExtractor(backend=GLMOCRPyTorchConfig(device="cuda")),
        languages,
        max_per_language,
    )


@app.function(image=LIGHTON_IMAGE, gpu="A10G:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_lighton(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    from omnidocs.tasks.text_extraction import LightOnTextExtractor
    from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

    return _load_and_infer(
        lambda: LightOnTextExtractor(backend=LightOnTextPyTorchConfig(device="cuda")),
        languages,
        max_per_language,
    )


@app.function(image=VLLM_IMAGE, gpu="L40S:1", secrets=[secret], volumes={"/data": volume}, timeout=7200)
def infer_dotsocr(languages: List[str], max_per_language: Optional[int]) -> InferenceOutput:
    import os

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch

    torch.cuda.init()
    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

    return _load_and_infer(
        lambda: DotsOCRTextExtractor(
            backend=DotsOCRVLLMConfig(
                model="rednote-hilab/dots.ocr", gpu_memory_utilization=0.90, max_model_len=32768, enforce_eager=True
            )
        ),
        languages,
        max_per_language,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen": infer_qwen,
    "deepseek": infer_deepseek,
    "nanonets": infer_nanonets,
    "granitedocling": infer_granitedocling,
    "mineruvl": infer_mineruvl,
    "glmocr": infer_glmocr,
    "lighton": infer_lighton,
    "dotsocr": infer_dotsocr,
}


# ---------------------------------------------------------------------------
# Local entrypoint — only writes files, does zero downloading
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    models: str = "",
    languages: str = "",
    max_per_language: int = 0,  # 0 = all pages
    output_dir: str = "",
):
    import datetime

    # Resolve models
    model_ids = [m.strip() for m in models.split(",") if m.strip()] if models else list(MODEL_REGISTRY.keys())
    unknown = [m for m in model_ids if m not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Resolve languages
    target_langs = [lang.strip() for lang in languages.split(",") if lang.strip()] if languages else ALL_LANGUAGES
    unknown_langs = [lang for lang in target_langs if lang not in ALL_LANGUAGES]
    if unknown_langs:
        print(f"Unknown languages: {unknown_langs}. Available: {ALL_LANGUAGES}")
        sys.exit(1)

    max_pl = max_per_language if max_per_language > 0 else None

    # Output directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) if output_dir else Path("results") / "nayana" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_root}")
    print(f"Models:           {model_ids}")
    print(f"Languages:        {target_langs}")
    print(f"Max per language: {max_pl or 'all'}")

    # Spawn all models in parallel — each container downloads the dataset itself
    print(f"\nSpawning {len(model_ids)} models in parallel...\n")
    futures = {}
    for model_id in model_ids:
        print(f"  ↑ Spawning {model_id}")
        try:
            futures[model_id] = MODEL_REGISTRY[model_id].spawn(target_langs, max_pl)
        except Exception as e:
            print(f"  [ERROR] Failed to spawn {model_id}: {e}")
            futures[model_id] = None

    # GT written once from the first successful result (same for all models)
    gt_written = False
    gt_dir = out_root / "gt"

    summary = {}
    for model_id, future in futures.items():
        print(f"\n{'=' * 60}\n  Collecting: {model_id}\n{'=' * 60}")

        if future is None:
            print("  [SKIPPED] spawn failed")
            summary[model_id] = {"error": "spawn failed"}
            continue

        try:
            output: InferenceOutput = future.get()
        except Exception as e:
            print(f"  [ERROR] {model_id}: {e}")
            summary[model_id] = {"error": str(e)}
            continue

        # Write GT JSON files once — identical across all models
        if not gt_written:
            gt_dir.mkdir(exist_ok=True)
            for lang, records in output.gt_by_language.items():
                gt_path = gt_dir / f"{lang}.json"
                gt_path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(f"  GT written: {gt_dir}  ({len(output.gt_by_language)} languages)")
            gt_written = True

        # Write .md files: <model>/<language>/<image_name>.md
        lang_stats: Dict[str, Dict] = {}
        for r in output.results:
            lang_dir = out_root / model_id / r.language
            lang_dir.mkdir(parents=True, exist_ok=True)
            md_path = lang_dir / f"{r.image_name}.md"

            if r.failed or not r.markdown:
                md_path.write_text("", encoding="utf-8")
                lang_stats.setdefault(r.language, {"written": 0, "failed": 0})["failed"] += 1
                print(f"  ✗ [{r.language}] {r.image_name} — {r.error[:80]}")
            else:
                md_path.write_text(r.markdown, encoding="utf-8")
                lang_stats.setdefault(r.language, {"written": 0, "failed": 0})["written"] += 1

        summary[model_id] = lang_stats
        total_w = sum(s["written"] for s in lang_stats.values())
        total_f = sum(s["failed"] for s in lang_stats.values())
        print(f"  Done: {total_w} written, {total_f} failed → {out_root / model_id}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE — results in: {out_root}")
    print(f"{'=' * 60}")
    for model_id, stats in summary.items():
        if "error" in stats:
            print(f"  {model_id:<18} ERROR: {stats['error']}")
        else:
            total_w = sum(s.get("written", 0) for s in stats.values())
            total_f = sum(s.get("failed", 0) for s in stats.values())
            print(f"  {model_id:<18} {total_w} written, {total_f} failed")

    # Write summary.json
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "benchmark": "NayanaOCRBench",
                "languages": target_langs,
                "models": model_ids,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Eval instructions
    if gt_written:
        example_model = model_ids[0]
        example_lang = target_langs[0]
        print(f"""
{"=" * 60}
HOW TO RUN OFFICIAL OMNIDOCBENCH EVAL ON NAYANA RESULTS
{"=" * 60}

1. Clone the official eval repo (once):
   git clone https://github.com/opendatalab/OmniDocBench
   cd OmniDocBench
   pip install -r requirements.txt

2. For each model + language, create a config and run eval.

   Example for {example_model} / {example_lang}:

     end2end_eval:
       metrics:
         text_block:
           metric: [Edit_dist]
         display_formula:
           metric: [Edit_dist, CDM_plain]
         table:
           metric: [TEDS, Edit_dist]
         reading_order:
           metric: [Edit_dist]
       dataset:
         dataset_name: end2end_dataset
         ground_truth:
           data_path: {(gt_dir / f"{example_lang}.json").resolve()}
         prediction:
           data_path: {(out_root / example_model / example_lang).resolve()}
         match_method: quick_match

   Then run:
   python pdf_validation.py --config configs/nayana_{example_model}_{example_lang}.yaml

3. GT files:   {gt_dir.resolve()}/<lang>.json
   Pred dirs:  {out_root.resolve()}/<model>/<lang>/
""")

    import yaml

    omnidocbench_dir = OMNIDOCS_DIR.parent / "OmniDocBench"
    configs_dir = omnidocbench_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    for model_id in model_ids:
        for lang in target_langs:
            pred_path = out_root / model_id / lang
            gt_path = gt_dir / f"{lang}.json"
            if not pred_path.exists() or not gt_path.exists():
                continue

            config = {
                "end2end_eval": {
                    "metrics": {
                        "text_block": {"metric": ["Edit_dist"]},
                        "display_formula": {"metric": ["Edit_dist", "CDM_plain"]},
                        "table": {"metric": ["TEDS", "Edit_dist"]},
                        "reading_order": {"metric": ["Edit_dist"]},
                    },
                    "dataset": {
                        "dataset_name": "end2end_dataset",
                        "ground_truth": {"data_path": str(gt_path.resolve())},
                        "prediction": {"data_path": str(pred_path.resolve())},
                        "match_method": "quick_match",
                    },
                }
            }

            config_path = configs_dir / f"nayana_{model_id}_{lang}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"  Config written: {config_path.name}")
