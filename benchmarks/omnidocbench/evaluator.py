"""
benchmarks/omnidocbench/evaluator.py
Runs the OmniDocBench end-to-end eval in-process (no subprocess, no git clone).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def run_evaluation(
    run_output_dir: Path,
    model_keys: List[str],
    omnidocbench_json: Optional[Path] = None,
) -> Dict[str, dict]:
    from benchmarks.omnidocbench_eval.run_eval import run_eval

    run_output_dir = Path(run_output_dir)

    if omnidocbench_json is None:
        import huggingface_hub
        omnidocbench_json = Path(huggingface_hub.hf_hub_download(
            repo_id="opendatalab/OmniDocBench",
            filename="OmniDocBench.json",
            repo_type="dataset",
        ))

    result_dir = run_output_dir / "result"
    eval_scores: Dict[str, dict] = {}

    print(f"\n{'=' * 60}")
    print("RUNNING OMNIDOCBENCH EVALUATION")
    print(f"{'=' * 60}")

    for model_key in model_keys:
        pred_dir = run_output_dir / model_key
        if not pred_dir.exists() or not list(pred_dir.glob("*.md")):
            print(f"  [SKIP] {model_key}: no prediction .md files in {pred_dir}")
            eval_scores[model_key] = {"error": "no prediction files"}
            continue

        print(f"  Evaluating: {model_key}  ({len(list(pred_dir.glob('*.md')))} pages)")
        save_name = f"{run_output_dir.name}_{model_key}"
        try:
            scores = run_eval(
                gt_data_path=omnidocbench_json,
                pred_data_path=pred_dir,
                save_name=save_name,
                result_dir=result_dir,
            )
            eval_scores[model_key] = scores
        except Exception as exc:
            print(f"  [ERROR] {model_key}: {exc}")
            eval_scores[model_key] = {"error": str(exc)}

    _print_eval_summary(eval_scores)
    return eval_scores


def _print_eval_summary(eval_scores: Dict[str, dict]) -> None:
    print(f"\n{'=' * 60}\nEVAL SUMMARY\n{'=' * 60}")
    for model_key, scores in eval_scores.items():
        if "error" in scores:
            print(f"  {model_key:<18}  ERROR: {scores['error']}")
        else:
            numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
            parts = "  ".join(f"{k}={v:.4f}" for k, v in numeric.items()) if numeric else json.dumps(scores)[:120]
            print(f"  {model_key:<18}  {parts}")
    print()