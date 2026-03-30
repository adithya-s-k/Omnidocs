"""
benchmarks/multilingual/runner.py

Local inference loop for the NayanaOCRBench multilingual benchmark.

For each model:
  1. Instantiate the extractor via the registry.
  2. Iterate over all PageSamples per language, call extractor.extract(image).
  3. Write one .md file per page to results/multilingual/<run_id>/<model>/<lang>/.
  4. After all models, delegate to evaluator.py to run the official scoring.

Mirrors benchmarks/omnidocbench/runner.py — same inference loop, same
write_md_files helper — extended with a per-language outer loop and GT
handling from the Nayana dataset.

Usage:
    python -m benchmarks.multilingual --models glmocr,qwen --languages en,hi,kn
    python -m benchmarks.multilingual --max-per-language 5 --no-eval
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.base import PageResult, PageSample
from benchmarks.registry import MODEL_REGISTRY, get_extractor

# ---------------------------------------------------------------------------
# Core inference loop (identical to omnidocbench/runner.py)
# ---------------------------------------------------------------------------


def run_inference(
    extractor,
    samples: List[PageSample],
    model_key: str,
    lang: str,
) -> List[PageResult]:
    """
    Run a single extractor over all samples for one language.
    Returns a list of PageResult objects.
    """
    from PIL import Image

    results: List[PageResult] = []

    for s in samples:
        img = Image.open(io.BytesIO(s.image_bytes)).convert("RGB")
        t0 = time.perf_counter()
        try:
            out = extractor.extract(img, output_format="markdown")
            markdown = (getattr(out, "plain_text", None) or out.content or "").strip()
            latency = time.perf_counter() - t0
            results.append(
                PageResult(
                    image_name=s.image_name,
                    model=getattr(out, "model_name", None) or model_key,
                    markdown=markdown,
                    latency_s=latency,
                )
            )
            print(f"  ✓ [{lang}] {s.image_name:<55} {latency:.2f}s  {len(markdown)} chars")
        except Exception as exc:
            latency = time.perf_counter() - t0
            results.append(
                PageResult(
                    image_name=s.image_name,
                    model=model_key,
                    markdown="",
                    latency_s=latency,
                    failed=True,
                    error=str(exc),
                )
            )
            print(f"  ✗ [{lang}] {s.image_name:<55} FAILED: {exc}")
            traceback.print_exc()

    return results


def write_md_files(
    results: List[PageResult],
    model_lang_dir: Path,
) -> tuple[int, int]:
    """
    Write one .md file per page result.
    Returns (written_count, failed_count).
    """
    model_lang_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0

    for r in results:
        md_path = model_lang_dir / f"{r.image_name}.md"
        if r.failed or not r.markdown:
            md_path.write_text("", encoding="utf-8")
            failed += 1
        else:
            md_path.write_text(r.markdown, encoding="utf-8")
            written += 1

    return written, failed


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_multilingual(
    model_keys: List[str],
    languages: Optional[List[str]] = None,
    max_per_language: Optional[int] = None,
    output_dir: Optional[Path] = None,
    run_eval: bool = True,
    eval_repo_path: Optional[Path] = None,
) -> dict:
    """
    Run the NayanaOCRBench multilingual benchmark locally.

    Args:
        model_keys:        List of model keys from the registry.
        languages:         Language codes to evaluate (None = all 22).
        max_per_language:  Limit pages per language (None = all).
        output_dir:        Where to write results. Defaults to
                           results/multilingual/<run_id>/.
        run_eval:          Whether to run the official eval after inference.
        eval_repo_path:    Path to the cloned OmniDocBench eval repo.

    Returns:
        Summary dict with per-model per-language inference counts and
        eval scores (if run_eval=True).
    """
    import datetime

    from benchmarks.multilingual.dataset import ALL_LANGUAGES, load_multilingual
    from benchmarks.multilingual.evaluator import run_evaluation

    # Validate models
    unknown = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")

    # Resolve languages
    if languages is None:
        languages = ALL_LANGUAGES
    unknown_langs = [lang for lang in languages if lang not in ALL_LANGUAGES]
    if unknown_langs:
        raise ValueError(f"Unknown languages: {unknown_langs}. Available: {ALL_LANGUAGES}")

    # Resolve output directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "multilingual" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Models:           {model_keys}")
    print(f"Languages:        {languages}")
    print(f"Max per language: {max_per_language or 'all'}")

    # Load dataset once — shared across all models
    print("\nLoading NayanaOCRBench dataset...")
    samples_by_lang, gt_by_lang = load_multilingual(
        languages=languages,
        max_per_language=max_per_language,
    )

    if not samples_by_lang:
        print("No samples loaded. Exiting.")
        sys.exit(1)

    # Run inference for each model sequentially, all languages per model
    inference_summary: Dict[str, Dict[str, dict]] = {}

    for model_key in model_keys:
        print(f"\n{'=' * 60}")
        print(f"  Running inference: {model_key}")
        print(f"{'=' * 60}")

        try:
            extractor = get_extractor(model_key)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {model_key}: {exc}")
            inference_summary[model_key] = {
                lang: {"written": 0, "failed": len(samples), "load_error": str(exc)}
                for lang, samples in samples_by_lang.items()
            }
            continue

        lang_stats: Dict[str, dict] = {}

        for lang, samples in samples_by_lang.items():
            results = run_inference(extractor, samples, model_key, lang)
            model_lang_dir = output_dir / model_key / lang
            written, failed = write_md_files(results, model_lang_dir)
            lang_stats[lang] = {"written": written, "failed": failed}
            print(f"  [{lang}] {written} written, {failed} failed → {model_lang_dir}")

        inference_summary[model_key] = lang_stats

        # Free GPU memory before loading the next model
        del extractor
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # Print inference summary
    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE — results in: {output_dir}")
    print(f"{'=' * 60}")
    for model_key, lang_stats in inference_summary.items():
        total_w = sum(s.get("written", 0) for s in lang_stats.values())
        total_f = sum(s.get("failed", 0) for s in lang_stats.values())
        print(f"  {model_key:<18} {total_w} written, {total_f} failed")

    # Run official evaluation
    eval_scores: Dict[str, Dict[str, dict]] = {}
    if run_eval:
        eval_scores = run_evaluation(
            run_output_dir=output_dir,
            model_keys=list(inference_summary.keys()),
            languages=languages,
            gt_by_lang=gt_by_lang,
            eval_repo_path=eval_repo_path,
        )

    # Write summary.json
    summary = {
        "run_id": run_id,
        "benchmark": "NayanaOCRBench",
        "languages": languages,
        "models": model_keys,
        "inference": inference_summary,
        "eval_scores": eval_scores,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")

    return summary
