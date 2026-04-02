"""
benchmarks/omnidocbench/runner.py
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
    p = output_dir / _STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(output_dir: Path, state: dict) -> None:
    (output_dir / _STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_inference(
    extractor,
    samples: List[PageSample],
    model_key: str,
    model_dir: Path,
) -> List[PageResult]:
    """
    Run extractor over all samples. Pages whose .md already exists and is
    non-empty are skipped (resume support). Each page is written to disk
    immediately after inference so a crash loses at most one page.
    """
    from PIL import Image

    results: List[PageResult] = []

    for s in samples:
        stem = Path(s.image_name).stem
        md_path = model_dir / f"{stem}.md"

        # Resume: skip already-completed pages
        if md_path.exists() and md_path.stat().st_size > 0:
            print(f"  ↷ {s.image_name:<60} (already done, skipping)")
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

            # Write immediately — crash-safe
            model_dir.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")

            results.append(
                PageResult(
                    image_name=s.image_name,
                    model=getattr(out, "model_name", None) or model_key,
                    markdown=markdown,
                    latency_s=latency,
                )
            )
            print(f"  ✓ {s.image_name:<60} {latency:.2f}s  {len(markdown)} chars")
        except Exception as exc:
            latency = time.perf_counter() - t0
            model_dir.mkdir(parents=True, exist_ok=True)
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
            print(f"  ✗ {s.image_name:<60} FAILED: {exc}")
            traceback.print_exc()

    return results


def run_inference_remote(
    extractor,
    samples: List[PageSample],
    model_key: str,
) -> List[PageResult]:
    """
    Inference-only variant for Modal remote execution — no disk I/O.
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
            print(f"  ✓ {s.image_name:<60} {latency:.2f}s  {len(markdown)} chars")
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
            print(f"  ✗ {s.image_name:<60} FAILED: {exc}")
            traceback.print_exc()

    return results


def write_md_files(
    results: List[PageResult],
    model_dir: Path,
) -> tuple[int, int]:
    """
    Write .md files for any results not already on disk (run_inference
    writes them inline; this handles the Modal collection path).
    Returns (written_count, failed_count).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0

    for r in results:
        stem = Path(r.image_name).stem
        md_path = model_dir / f"{stem}.md"

        if md_path.exists():
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


def run_omnidocbench(
    model_keys: List[str],
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
    run_eval: bool = True,
    eval_only: bool = False,
    eval_repo_path: Optional[Path] = None,
    omnidocbench_json: Optional[Path] = None,
) -> dict:
    """
    Run OmniDocBench locally for the given models.

    Args:
        model_keys:        List of model keys from the registry.
        max_samples:       Limit pages loaded (None = full dataset).
        output_dir:        Where to write .md files and results.json.
        run_eval:          Whether to run eval after inference.
        eval_only:         Skip inference; run eval on existing .md files.
        eval_repo_path:    Path to the OmniDocBench eval repo.
        omnidocbench_json: Path to OmniDocBench.json GT file.
    """
    import datetime

    unknown = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "omnidocbench" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    if eval_only:
        print("Mode: EVAL ONLY (skipping inference)")

    # Load state for resume
    state = _load_state(output_dir)
    completed_models: list = state.get("completed_models", [])
    inference_summary: Dict[str, dict] = state.get("inference", {})

    if not eval_only:
        from benchmarks.omnidocbench.dataset import load_omnidocbench

        print("\nLoading OmniDocBench dataset...")
        samples = load_omnidocbench(max_samples=max_samples)
        if not samples:
            print("No samples loaded. Exiting.")
            sys.exit(1)

        for model_key in model_keys:
            if model_key in completed_models:
                print(f"\n  ↷ {model_key}: already completed (run_state.json), skipping")
                continue

            print(f"\n{'=' * 60}")
            print(f"  Running inference: {model_key}")
            print(f"{'=' * 60}")

            model_dir = output_dir / model_key

            try:
                extractor = get_extractor(model_key)
            except Exception as exc:
                print(f"  [ERROR] Failed to load {model_key}: {exc}")
                inference_summary[model_key] = {"written": 0, "failed": len(samples), "load_error": str(exc)}
                _save_state(output_dir, {"completed_models": completed_models, "inference": inference_summary})
                continue

            results = run_inference(extractor, samples, model_key, model_dir)
            written, failed = write_md_files(results, model_dir)
            inference_summary[model_key] = {"written": written, "failed": failed}
            completed_models.append(model_key)
            _save_state(output_dir, {"completed_models": completed_models, "inference": inference_summary})

            print(f"\n  Done: {written} written, {failed} failed → {model_dir}")

            del extractor
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        print(f"\n{'=' * 60}")
        print(f"INFERENCE COMPLETE — results in: {output_dir}")
        print(f"{'=' * 60}")
        for model_key, s in inference_summary.items():
            print(f"  {model_key:<18} {s.get('written', 0)} written, {s.get('failed', 0)} failed")

    else:
        # eval_only: reconstruct inference_summary from disk
        for model_key in model_keys:
            model_dir = output_dir / model_key
            md_files = list(model_dir.glob("*.md")) if model_dir.exists() else []
            written = sum(1 for f in md_files if f.stat().st_size > 0)
            inference_summary[model_key] = {"written": written, "failed": len(md_files) - written}

    eval_scores: Dict[str, dict] = {}
    if run_eval or eval_only:
        from benchmarks.omnidocbench.evaluator import run_evaluation

        eval_scores = run_evaluation(
            run_output_dir=output_dir,
            model_keys=list(inference_summary.keys()),
            eval_repo_path=eval_repo_path,
            omnidocbench_json=omnidocbench_json,
        )

    results_json = {
        "run_id": run_id if not eval_only else state.get("run_id", output_dir.name),
        "benchmark": "omnidocbench",
        "execution": "local",
        "models": model_keys,
        "inference": inference_summary,
        "eval_scores": eval_scores,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")
    return results_json
