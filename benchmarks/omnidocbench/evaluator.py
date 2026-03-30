"""
benchmarks/omnidocbench/evaluator.py

Handles everything after inference:
  1. Clones the official OmniDocBench eval repo (once, cached).
  2. Resolves the OmniDocBench.json ground-truth path from HF cache.
  3. Dynamically generates a YAML config per model inside the eval repo's
     configs/ directory — the user never touches a config file.
  4. Runs pdf_validation.py as a subprocess for each model.
  5. Collects the JSON score output and returns an aggregated dict.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_REPO_URL = "https://github.com/opendatalab/OmniDocBench"

# Default location to clone the eval repo (relative to this file's package dir)
DEFAULT_EVAL_REPO_PATH = Path(__file__).parent / "eval_repo"

# Template for the YAML config — only data_path values are substituted
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
# Helpers
# ---------------------------------------------------------------------------


def _ensure_eval_repo(eval_repo_path: Path) -> Path:
    """
    Clone the OmniDocBench eval repo if it doesn't exist yet.
    Installs its requirements on first clone.
    Returns the path to the cloned repo.
    """
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

    # Install requirements
    req_file = eval_repo_path / "requirements.txt"
    if req_file.exists():
        print("  Installing eval repo requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            check=True,
        )

    print("  Eval repo ready.")
    return eval_repo_path


def _resolve_omnidocbench_json(omnidocbench_json: Optional[Path]) -> Path:
    """
    If a path is provided, use it directly.
    Otherwise, locate OmniDocBench.json from the HuggingFace Hub cache
    (it will already be there if load_omnidocbench() ran first).
    """
    if omnidocbench_json is not None:
        p = Path(omnidocbench_json)
        if not p.exists():
            raise FileNotFoundError(f"OmniDocBench.json not found at: {p}")
        return p

    # Try to resolve from HF hub cache without re-downloading
    try:
        import huggingface_hub

        cached = huggingface_hub.hf_hub_download(
            repo_id="opendatalab/OmniDocBench",
            filename="OmniDocBench.json",
            repo_type="dataset",
        )
        return Path(cached)
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve OmniDocBench.json from HF cache. "
            "Pass --omnidocbench-json explicitly, or run inference first "
            "(which downloads the file)."
        ) from exc


def _generate_config(
    model_key: str,
    run_id: str,
    gt_json_path: Path,
    pred_dir: Path,
    configs_dir: Path,
) -> Path:
    """
    Write a YAML config for pdf_validation.py and return its path.
    Config is written inside the eval repo's configs/ directory.
    """
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_name = f"{run_id}_{model_key}.yaml"
    config_path = configs_dir / config_name

    content = _CONFIG_TEMPLATE.format(
        gt_data_path=str(gt_json_path.resolve()),
        pred_data_path=str(pred_dir.resolve()),
    )
    config_path.write_text(content, encoding="utf-8")
    print(f"  Config written: {config_path}")
    return config_path


def _run_pdf_validation(config_path: Path, eval_repo_path: Path, model_key: str) -> Optional[dict]:
    """
    Run pdf_validation.py with the given config.
    Returns the parsed JSON results dict, or None if evaluation failed.
    The official script writes a JSON file alongside the config.
    """
    print(f"  Running pdf_validation.py for config: {config_path.name} ...")
    result = subprocess.run(
        [sys.executable, "pdf_validation.py", "--config", str(config_path)],
        cwd=str(eval_repo_path),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"  [ERROR] pdf_validation.py exited with code {result.returncode}")
        return None

    # The official script writes results as a JSON file next to the config
    # (same stem, .json extension)
    result_json_path = eval_repo_path / "result" / f"{model_key}_quick_match_metric_result.json"
    if result_json_path.exists():
        try:
            return json.loads(result_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [WARN] Could not parse result JSON: {exc}")

    # Fallback: try to parse JSON from stdout
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
    eval_repo_path: Optional[Path] = None,
    omnidocbench_json: Optional[Path] = None,
) -> Dict[str, dict]:
    """
    For each model_key, generate a YAML config and run the official
    OmniDocBench evaluation (pdf_validation.py).

    Args:
        run_output_dir:    Root output dir for this run (contains <model>/ subdirs).
        model_keys:        Which models to evaluate (must have .md files written).
        eval_repo_path:    Path to the cloned eval repo (cloned automatically if absent).
        omnidocbench_json: Path to OmniDocBench.json (resolved from HF cache if None).

    Returns:
        Dict mapping model_key → raw scores dict from pdf_validation.py.
    """
    run_output_dir = Path(run_output_dir)
    # Derive a run_id from the output directory name for config file naming
    run_id = run_output_dir.name

    if eval_repo_path is None:
        eval_repo_path = DEFAULT_EVAL_REPO_PATH
    eval_repo_path = Path(eval_repo_path)

    print(f"\n{'=' * 60}")
    print("RUNNING OFFICIAL OMNIDOCBENCH EVALUATION")
    print(f"{'=' * 60}")

    # Step 1 — ensure eval repo is present
    eval_repo_path = _ensure_eval_repo(eval_repo_path)

    # Step 2 — resolve ground-truth JSON
    gt_json = _resolve_omnidocbench_json(omnidocbench_json)
    print(f"  Ground-truth JSON: {gt_json}")

    configs_dir = eval_repo_path / "configs"
    eval_scores: Dict[str, dict] = {}

    # Step 3 — for each model: generate config → run eval → collect scores
    for model_key in model_keys:
        pred_dir = run_output_dir / model_key
        if not pred_dir.exists():
            print(f"  [SKIP] {model_key}: prediction directory not found at {pred_dir}")
            eval_scores[model_key] = {"error": "prediction directory not found"}
            continue

        md_files = list(pred_dir.glob("*.md"))
        if not md_files:
            print(f"  [SKIP] {model_key}: no .md files found in {pred_dir}")
            eval_scores[model_key] = {"error": "no .md files found"}
            continue

        print(f"\n  Evaluating: {model_key}  ({len(md_files)} pages)")

        config_path = _generate_config(
            model_key=model_key,
            run_id=run_id,
            gt_json_path=gt_json,
            pred_dir=pred_dir,
            configs_dir=configs_dir,
        )

        scores = _run_pdf_validation(config_path, eval_repo_path, model_key)
        eval_scores[model_key] = scores if scores is not None else {"error": "evaluation failed"}

    # Step 4 — print a quick comparison table
    _print_eval_summary(eval_scores)

    return eval_scores


def _print_eval_summary(eval_scores: Dict[str, dict]) -> None:
    """Print a concise side-by-side summary of eval scores."""
    print(f"\n{'=' * 60}")
    print("EVAL SUMMARY")
    print(f"{'=' * 60}")
    for model_key, scores in eval_scores.items():
        if "error" in scores:
            print(f"  {model_key:<18}  ERROR: {scores['error']}")
        else:
            # Best-effort: print the top-level numeric values
            numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
            if numeric:
                parts = "  ".join(f"{k}={v:.4f}" for k, v in numeric.items())
                print(f"  {model_key:<18}  {parts}")
            else:
                print(f"  {model_key:<18}  {json.dumps(scores)[:120]}")
    print()
