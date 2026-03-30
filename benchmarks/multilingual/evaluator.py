"""
benchmarks/multilingual/evaluator.py

Handles everything after inference for the multilingual benchmark:
  1. Writes per-language GT JSON files (with html fix applied).
  2. Clones the official OmniDocBench eval repo (once, cached).
  3. Generates a YAML config per model per language inside the eval repo's
     configs/ directory.
  4. Runs pdf_validation.py as a subprocess for each model+language pair.
  5. Collects the score output and returns an aggregated dict.

Mirrors the structure of benchmarks/omnidocbench/evaluator.py exactly,
with the addition of per-language loops and GT writing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

EVAL_REPO_URL = "https://github.com/opendatalab/OmniDocBench"
DEFAULT_EVAL_REPO_PATH = Path(__file__).parent / "eval_repo"

_CONFIG_TEMPLATE = """\
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
    display_formula:
      metric:
        - Edit_dist
        - CDM_plain
    table:
      metric:
        - TEDS
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {gt_data_path}
    prediction:
      data_path: {pred_data_path}
    match_method: quick_match
"""


# ---------------------------------------------------------------------------
# Helpers (identical to omnidocbench/evaluator.py)
# ---------------------------------------------------------------------------


def _ensure_eval_repo(eval_repo_path: Path) -> Path:
    """Clone the OmniDocBench eval repo if not already present."""
    if eval_repo_path.exists() and (eval_repo_path / "pdf_validation.py").exists():
        print(f"  Eval repo found at: {eval_repo_path}")
        return eval_repo_path

    print(f"  Cloning OmniDocBench eval repo → {eval_repo_path} ...")
    eval_repo_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", EVAL_REPO_URL, str(eval_repo_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone eval repo:\n{result.stderr}")

    req_file = eval_repo_path / "requirements.txt"
    if req_file.exists():
        print("  Installing eval repo requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            check=True,
        )

    print("  Eval repo ready.")
    return eval_repo_path


def _write_gt_files(
    gt_by_lang: Dict[str, List[dict]],
    gt_dir: Path,
) -> Dict[str, Path]:
    """
    Write one GT JSON file per language to gt_dir.
    Returns a dict of lang -> path.
    """
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_paths: Dict[str, Path] = {}
    for lang, records in gt_by_lang.items():
        gt_path = gt_dir / f"{lang}.json"
        gt_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        gt_paths[lang] = gt_path
    print(f"  GT files written to: {gt_dir}  ({len(gt_paths)} languages)")
    return gt_paths


def _generate_config(
    model_key: str,
    lang: str,
    run_id: str,
    gt_json_path: Path,
    pred_dir: Path,
    configs_dir: Path,
) -> Path:
    """Write a YAML config for one model+language pair."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_name = f"nayana_{model_key}_{lang}.yaml"
    config_path = configs_dir / config_name
    content = _CONFIG_TEMPLATE.format(
        gt_data_path=str(gt_json_path.resolve()),
        pred_data_path=str(pred_dir.resolve()),
    )
    config_path.write_text(content, encoding="utf-8")
    print(f"  Config written: {config_path.name}")
    return config_path


def _run_pdf_validation(
    config_path: Path,
    eval_repo_path: Path,
    model_key: str,
    lang: str,
) -> Optional[dict]:
    """Run pdf_validation.py for one config and return parsed scores."""
    print(f"  Running eval: {model_key} / {lang} ...")
    result = subprocess.run(
        [sys.executable, "pdf_validation.py", "--config", str(config_path)],
        cwd=str(eval_repo_path),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"  [ERROR] pdf_validation.py exited with code {result.returncode}")
        return None

    # Official script writes results to result/<model_key>_quick_match_metric_result.json
    result_json_path = eval_repo_path / "result" / f"{model_key}_quick_match_metric_result.json"
    if result_json_path.exists():
        try:
            return json.loads(result_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [WARN] Could not parse result JSON: {exc}")

    # Fallback: scan stdout for a JSON line
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass

    print("  [WARN] Could not locate evaluation results JSON.")
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    run_output_dir: Path,
    model_keys: List[str],
    languages: List[str],
    gt_by_lang: Dict[str, List[dict]],
    eval_repo_path: Optional[Path] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Write GT files, generate configs, and run pdf_validation.py for every
    model+language combination.

    Args:
        run_output_dir: Root output dir (contains <model>/<lang>/ subdirs).
        model_keys:     Models to evaluate.
        languages:      Languages to evaluate.
        gt_by_lang:     GT records keyed by language (from dataset.py).
        eval_repo_path: Path to the cloned eval repo (auto-cloned if None).

    Returns:
        Nested dict: model_key -> lang -> raw scores dict.
    """
    run_output_dir = Path(run_output_dir)
    run_id = run_output_dir.name

    if eval_repo_path is None:
        eval_repo_path = DEFAULT_EVAL_REPO_PATH
    eval_repo_path = Path(eval_repo_path)

    print(f"\n{'=' * 60}")
    print("RUNNING OFFICIAL OMNIDOCBENCH EVALUATION (multilingual)")
    print(f"{'=' * 60}")

    eval_repo_path = _ensure_eval_repo(eval_repo_path)

    # Write GT JSON files once
    gt_dir = run_output_dir / "gt"
    gt_paths = _write_gt_files(gt_by_lang, gt_dir)

    configs_dir = eval_repo_path / "configs"
    eval_scores: Dict[str, Dict[str, dict]] = {}

    for model_key in model_keys:
        eval_scores[model_key] = {}
        for lang in languages:
            if lang not in gt_paths:
                print(f"  [SKIP] {model_key}/{lang}: no GT available")
                eval_scores[model_key][lang] = {"error": "no GT available"}
                continue

            pred_dir = run_output_dir / model_key / lang
            if not pred_dir.exists() or not list(pred_dir.glob("*.md")):
                print(f"  [SKIP] {model_key}/{lang}: no prediction .md files")
                eval_scores[model_key][lang] = {"error": "no prediction files"}
                continue

            config_path = _generate_config(
                model_key=model_key,
                lang=lang,
                run_id=run_id,
                gt_json_path=gt_paths[lang],
                pred_dir=pred_dir,
                configs_dir=configs_dir,
            )

            scores = _run_pdf_validation(config_path, eval_repo_path, model_key, lang)
            eval_scores[model_key][lang] = scores if scores is not None else {"error": "evaluation failed"}

    _print_eval_summary(eval_scores, languages)
    return eval_scores


def _print_eval_summary(
    eval_scores: Dict[str, Dict[str, dict]],
    languages: List[str],
) -> None:
    print(f"\n{'=' * 60}")
    print("EVAL SUMMARY (multilingual)")
    print(f"{'=' * 60}")
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
