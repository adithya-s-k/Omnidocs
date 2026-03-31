"""
benchmarks/multilingual/evaluator.py
Runs the OmniDocBench end-to-end eval in-process for each model+language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def run_evaluation(
    run_output_dir: Path,
    model_keys: List[str],
    languages: List[str],
    gt_by_lang: Dict[str, List[dict]],
    eval_repo_path: Optional[Path] = None,  # kept for API compat, unused
) -> Dict[str, Dict[str, dict]]:
    from benchmarks.omnidocbench_eval.run_eval import run_eval

    run_output_dir = Path(run_output_dir)
    result_dir = run_output_dir / "result"
    gt_dir = run_output_dir / "gt"

    # Write GT JSON files (may already exist from inference step, but write anyway)
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_paths: Dict[str, Path] = {}
    for lang, records in gt_by_lang.items():
        p = gt_dir / f"{lang}.json"
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        gt_paths[lang] = p

    print(f"\n{'=' * 60}")
    print("RUNNING OMNIDOCBENCH EVALUATION (multilingual)")
    print(f"{'=' * 60}")

    eval_scores: Dict[str, Dict[str, dict]] = {}

    for model_key in model_keys:
        eval_scores[model_key] = {}
        for lang in languages:
            if lang not in gt_paths:
                eval_scores[model_key][lang] = {"error": "no GT available"}
                continue

            pred_dir = run_output_dir / model_key / lang
            if not pred_dir.exists() or not list(pred_dir.glob("*.md")):
                print(f"  [SKIP] {model_key}/{lang}: no prediction .md files")
                eval_scores[model_key][lang] = {"error": "no prediction files"}
                continue

            print(f"  Evaluating: {model_key}/{lang}  ({len(list(pred_dir.glob('*.md')))} pages)")
            save_name = f"{run_output_dir.name}_{model_key}_{lang}"
            try:
                scores = run_eval(
                    gt_data_path=gt_paths[lang],
                    pred_data_path=pred_dir,
                    save_name=save_name,
                    result_dir=result_dir,
                )
                eval_scores[model_key][lang] = scores
            except Exception as exc:
                print(f"  [ERROR] {model_key}/{lang}: {exc}")
                eval_scores[model_key][lang] = {"error": str(exc)}

    _print_eval_summary(eval_scores, languages)
    return eval_scores


def _print_eval_summary(eval_scores: Dict[str, Dict[str, dict]], languages: List[str]) -> None:
    print(f"\n{'=' * 60}\nEVAL SUMMARY (multilingual)\n{'=' * 60}")
    for model_key, lang_scores in eval_scores.items():
        print(f"\n  {model_key}")
        for lang in languages:
            scores = lang_scores.get(lang, {})
            if "error" in scores:
                print(f"    [{lang}]  ERROR: {scores['error']}")
            else:
                numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
                parts = "  ".join(f"{k}={v:.4f}" for k, v in numeric.items()) if numeric else str(scores)[:100]
                print(f"    [{lang}]  {parts}")
    print()
