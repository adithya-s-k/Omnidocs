"""
benchmarks/multilingual/runner.py
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

_STATE_FILE = "run_state.json"


def _load_state(output_dir: Path) -> dict:
    """Load run_state.json if it exists, otherwise return empty state."""
    p = output_dir / _STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(output_dir: Path, state: dict) -> None:
    """Atomically persist run_state.json."""
    p = output_dir / _STATE_FILE
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_inference(
    extractor,
    samples: List[PageSample],
    model_key: str,
    lang: str,
    model_lang_dir: Path,  # ← needed so we can skip existing pages
) -> List[PageResult]:
    """
    Run a single extractor over samples for one language.
    Pages whose .md output file already exists and is non-empty are skipped.
    Returns a list of PageResult objects (skipped pages are included as
    already-written successes so counts stay accurate).
    """
    from PIL import Image

    results: List[PageResult] = []

    for s in samples:
        md_path = model_lang_dir / f"{s.image_name}.md"

        # ── Resume: skip pages that were already successfully predicted ──
        if md_path.exists() and md_path.stat().st_size > 0:
            print(f"  ↷ [{lang}] {s.image_name:<55} (already done, skipping)")
            results.append(
                PageResult(
                    image_name=s.image_name,
                    model=model_key,
                    markdown=md_path.read_text(encoding="utf-8"),
                    latency_s=0.0,
                )
            )
            continue

        img = Image.open(io.BytesIO(s.image_bytes)).convert("RGB")
        t0 = time.perf_counter()
        try:
            out = extractor.extract(img, output_format="markdown")
            markdown = (getattr(out, "plain_text", None) or out.content or "").strip()
            latency = time.perf_counter() - t0

            # ── Write immediately so a crash later doesn't lose this page ──
            model_lang_dir.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")

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
            # Write empty file so we know this page was attempted
            model_lang_dir.mkdir(parents=True, exist_ok=True)
            md_path.write_text("", encoding="utf-8")
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


def run_inference_remote(
    extractor,
    samples: List[PageSample],
    model_key: str,
    lang: str,
) -> List[PageResult]:
    """
    Inference-only variant for Modal remote execution — no disk I/O.
    Results are returned in memory and written locally by the entrypoint.
    Use run_inference() for local execution (writes pages to disk as it goes).
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
    Pages that were already written by run_inference (resume path) are skipped.
    Returns (written_count, failed_count).
    """
    model_lang_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0

    for r in results:
        md_path = model_lang_dir / f"{r.image_name}.md"
        if md_path.exists():
            # Already written by run_inference directly — just count it
            if r.failed or not r.markdown:
                failed += 1
            else:
                written += 1
            continue
        if r.failed or not r.markdown:
            md_path.write_text("", encoding="utf-8")
            failed += 1
        else:
            md_path.write_text(r.markdown, encoding="utf-8")
            written += 1

    return written, failed


def run_multilingual(
    model_keys: List[str],
    languages: Optional[List[str]] = None,
    max_per_language: Optional[int] = None,
    output_dir: Optional[Path] = None,
    run_eval: bool = True,
    eval_only: bool = False,  # ← NEW: skip inference, run eval on existing files
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
        eval_only:         Skip inference entirely; run eval on existing .md files.
        eval_repo_path:    Path to the cloned OmniDocBench eval repo.

    Returns:
        Summary dict with per-model per-language inference counts and
        eval scores (if run_eval=True).
    """
    import datetime

    from benchmarks.multilingual.dataset import ALL_LANGUAGES, load_multilingual
    from benchmarks.multilingual.evaluator import run_evaluation

    # ── Validate models ──────────────────────────────────────────────────────
    unknown = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")

    # ── Resolve languages ────────────────────────────────────────────────────
    if languages is None:
        languages = ALL_LANGUAGES
    unknown_langs = [lang for lang in languages if lang not in ALL_LANGUAGES]
    if unknown_langs:
        raise ValueError(f"Unknown languages: {unknown_langs}. Available: {ALL_LANGUAGES}")

    # ── Resolve output directory ─────────────────────────────────────────────
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "multilingual" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Models:           {model_keys}")
    print(f"Languages:        {languages}")
    print(f"Max per language: {max_per_language or 'all'}")
    if eval_only:
        print("Mode:             EVAL ONLY (skipping inference)")

    # ── Load run state (for resume) ──────────────────────────────────────────
    state = _load_state(output_dir)
    # state schema: { "completed_models": ["glmocr", ...], "inference": { ... } }
    completed_models: list = state.get("completed_models", [])
    inference_summary: Dict[str, Dict[str, dict]] = state.get("inference", {})

    if not eval_only:
        # ── Load dataset once — shared across all models ─────────────────────
        print("\nLoading NayanaOCRBench dataset...")
        samples_by_lang, gt_by_lang = load_multilingual(
            languages=languages,
            max_per_language=max_per_language,
        )
        if not samples_by_lang:
            print("No samples loaded. Exiting.")
            sys.exit(1)

        # ── Write GT to disk (needed for eval-only re-runs too) ──────────────
        gt_dir = output_dir / "gt"
        gt_dir.mkdir(exist_ok=True)
        for lang, records in gt_by_lang.items():
            gt_path = gt_dir / f"{lang}.json"
            if not gt_path.exists():
                gt_path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        # ── Inference loop ───────────────────────────────────────────────────
        for model_key in model_keys:
            if model_key in completed_models:
                print(f"\n  ↷ {model_key}: already completed (from run_state.json), skipping")
                continue

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
                _save_state(output_dir, {"completed_models": completed_models, "inference": inference_summary})
                continue

            lang_stats: Dict[str, dict] = {}

            for lang, samples in samples_by_lang.items():
                model_lang_dir = output_dir / model_key / lang
                results = run_inference(extractor, samples, model_key, lang, model_lang_dir)
                written, failed = write_md_files(results, model_lang_dir)
                lang_stats[lang] = {"written": written, "failed": failed}
                print(f"  [{lang}] {written} written, {failed} failed → {model_lang_dir}")

            inference_summary[model_key] = lang_stats
            completed_models.append(model_key)

            # ── Persist state after each model completes ──────────────────
            _save_state(output_dir, {"completed_models": completed_models, "inference": inference_summary})

            # Free GPU memory before loading the next model
            del extractor
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        print(f"\n{'=' * 60}")
        print(f"INFERENCE COMPLETE — results in: {output_dir}")
        print(f"{'=' * 60}")
        for model_key, lang_stats in inference_summary.items():
            total_w = sum(s.get("written", 0) for s in lang_stats.values())
            total_f = sum(s.get("failed", 0) for s in lang_stats.values())
            print(f"  {model_key:<18} {total_w} written, {total_f} failed")

    else:
        # ── eval_only: read GT from disk ─────────────────────────────────────
        gt_dir = output_dir / "gt"
        if not gt_dir.exists():
            raise FileNotFoundError(
                f"GT directory not found at {gt_dir}. Run inference first before using --eval-only."
            )
        gt_by_lang = {}
        for lang in languages:
            gt_path = gt_dir / f"{lang}.json"
            if gt_path.exists():
                gt_by_lang[lang] = json.loads(gt_path.read_text(encoding="utf-8"))
            else:
                print(f"  [WARN] No GT file for language '{lang}' — skipping")

        # Build inference_summary from existing .md files on disk
        for model_key in model_keys:
            lang_stats = {}
            for lang in languages:
                model_lang_dir = output_dir / model_key / lang
                md_files = list(model_lang_dir.glob("*.md")) if model_lang_dir.exists() else []
                written = sum(1 for f in md_files if f.stat().st_size > 0)
                failed = len(md_files) - written
                lang_stats[lang] = {"written": written, "failed": failed}
            inference_summary[model_key] = lang_stats

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_scores: Dict[str, Dict[str, dict]] = {}
    if run_eval or eval_only:
        eval_scores = run_evaluation(
            run_output_dir=output_dir,
            model_keys=list(inference_summary.keys()),
            languages=languages,
            gt_by_lang=gt_by_lang,
            eval_repo_path=eval_repo_path,
        )

    # ── Write results.json ────────────────────────────────────────────────────
    results_json = {
        "run_id": run_id if not eval_only else state.get("run_id", output_dir.name),
        "benchmark": "multilingual",
        "execution": "local",
        "models": model_keys,
        "languages": languages,
        "inference": inference_summary,
        "eval_scores": eval_scores,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")

    return results_json
