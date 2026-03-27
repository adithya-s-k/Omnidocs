"""
benchmarks/registry.py

Central model registry.  Maps a string key → a zero-argument factory
function that returns an instantiated TextExtractor ready to call
`extractor.extract(image)`.

When a new model is added to omnidocs/tasks/text_extraction/, add one
entry here.  No other benchmark file needs to change.

Each entry is a callable (factory) so the extractor (and its model
weights) are only loaded when actually needed — not on import.
"""

from __future__ import annotations

from typing import Callable, Dict


def _make_qwen():
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    return QwenTextExtractor(
        backend=QwenTextPyTorchConfig(
            model="Qwen/Qwen3-VL-2B-Instruct",
            device="cuda",
        )
    )


def _make_deepseek():
    from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
    from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig
    return DeepSeekOCRTextExtractor(
        backend=DeepSeekOCRTextPyTorchConfig(
            model="unsloth/DeepSeek-OCR-2",
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
            crop_mode=True,
        )
    )


def _make_nanonets():
    from omnidocs.tasks.text_extraction import NanonetsTextExtractor
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig
    return NanonetsTextExtractor(
        backend=NanonetsTextPyTorchConfig(device="cuda")
    )


def _make_granitedocling():
    from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
    from omnidocs.tasks.text_extraction.granitedocling import GraniteDoclingTextPyTorchConfig
    return GraniteDoclingTextExtractor(
        backend=GraniteDoclingTextPyTorchConfig(
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
        )
    )


def _make_mineruvl():
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig
    return MinerUVLTextExtractor(
        backend=MinerUVLTextPyTorchConfig(device="cuda")
    )


def _make_glmocr():
    from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
    from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig
    return GLMOCRTextExtractor(
        backend=GLMOCRPyTorchConfig(device="cuda")
    )


def _make_lighton():
    from omnidocs.tasks.text_extraction import LightOnTextExtractor
    from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig
    return LightOnTextExtractor(
        backend=LightOnTextPyTorchConfig(device="cuda")
    )


def _make_dotsocr():
    import os
    # DotsOCR requires vLLM v0 engine — must be set before vllm is imported
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        import torch
        torch.cuda.init()
    except Exception:
        pass
    from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
    from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig
    return DotsOCRTextExtractor(
        backend=DotsOCRVLLMConfig(
            model="rednote-hilab/dots.ocr",
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            enforce_eager=True,
        )
    )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

# model_key → zero-argument factory that returns a ready extractor
MODEL_REGISTRY: Dict[str, Callable] = {
    "qwen":           _make_qwen,
    "deepseek":       _make_deepseek,
    "nanonets":       _make_nanonets,
    "granitedocling": _make_granitedocling,
    "mineruvl":       _make_mineruvl,
    "glmocr":         _make_glmocr,
    "lighton":        _make_lighton,
    "dotsocr":        _make_dotsocr,
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_extractor(model_key: str):
    """Instantiate and return the extractor for the given model key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key]()
