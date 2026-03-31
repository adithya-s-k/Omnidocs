"""
benchmarks/omnidocbench_eval/run_eval.py

In-process replacement for OmniDocBench's pdf_validation.py.
Call run_eval() directly instead of shelling out to a subprocess.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Make the eval subpackage importable with its own relative imports.
# OmniDocBench modules use bare imports like `from registry.registry import ...`
# so we inject this package's directory onto sys.path once.
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

# Now safe to import — triggers registry population via __init__ files
import dataset   # noqa: F401  registers all dataset classes
import task      # noqa: F401  registers all task classes
import metrics   # noqa: F401  registers all metric classes

from registry.registry import EVAL_TASK_REGISTRY, DATASET_REGISTRY


def run_eval(
    gt_data_path: str | Path,
    pred_data_path: str | Path,
    save_name: str,
    result_dir: str | Path = "./result",
    match_method: str = "quick_match",
) -> dict:
    """
    Run the OmniDocBench end-to-end evaluation in-process.

    Args:
        gt_data_path:   Path to OmniDocBench.json (or per-language GT JSON).
        pred_data_path: Path to directory of prediction .md files.
        save_name:      Stem used for result filenames (e.g. "glmocr_en").
        result_dir:     Directory where per-element JSON results are written.
        match_method:   "quick_match" | "simple_match" | "no_split".

    Returns:
        The result_all dict (same content as the *_metric_result.json file).
    """
    gt_data_path   = str(Path(gt_data_path).resolve())
    pred_data_path = str(Path(pred_data_path).resolve())
    result_dir     = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    # Patch the hardcoded './result/' writes inside End2EndEval to use result_dir
    _orig_cwd = Path.cwd()
    os.chdir(result_dir.parent)   # End2EndEval writes to ./result/<save_name>_*
    # Ensure ./result/ exists relative to cwd
    (result_dir.parent / "result").mkdir(exist_ok=True)

    cfg_task = {
        "dataset": {
            "dataset_name": "end2end_dataset",
            "ground_truth": {"data_path": gt_data_path},
            "prediction":   {"data_path": pred_data_path},
            "match_method": match_method,
        },
        "metrics": {
            "text_block":     {"metric": ["Edit_dist"]},
            "display_formula": {"metric": ["Edit_dist", "CDM_plain"]},
            "table":          {"metric": ["TEDS", "Edit_dist"]},
            "reading_order":  {"metric": ["Edit_dist"]},
        },
    }

    try:
        val_dataset = DATASET_REGISTRY.get("end2end_dataset")(cfg_task)
        End2EndEval = EVAL_TASK_REGISTRY.get("end2end_eval")
        End2EndEval(val_dataset, cfg_task["metrics"], gt_data_path, save_name)
    finally:
        os.chdir(_orig_cwd)

    # Read back the metric result JSON that End2EndEval wrote
    result_json = result_dir.parent / "result" / f"{save_name}_metric_result.json"
    if result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    return {}