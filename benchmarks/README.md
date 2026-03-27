# OmniDocs Local Benchmarks

Local (non-Modal) benchmark runners for OmniDocs text extraction models.
Runs directly on your machine using `extractor.extract(image)` — no cloud
infrastructure required.

---

## Supported Benchmarks

| Benchmark | Source | Scoring |
|---|---|---|
| **OmniDocBench** | `opendatalab/OmniDocBench` | Official `pdf_validation.py` (Edit Distance, TEDS, CDM) |
| **olmOCR-bench** | `allenai/olmOCR-bench` | Built-in binary pass/fail (text_present, text_absent, reading_order, table, math) |

---

## Supported Models

Run `python -m benchmarks.olmocr --list-info` or
`python -m benchmarks.omnidocbench --list-models` to see the current list.

Default models: `qwen`, `deepseek`, `nanonets`, `granitedocling`,
`mineruvl`, `glmocr`, `lighton`, `dotsocr`

---

## Quick Start

### OmniDocBench

```bash
# Run all models — full dataset + official eval
python -m benchmarks.omnidocbench

# Run specific models
python -m benchmarks.omnidocbench --models glmocr,deepseek

# Quick iteration (10 pages, skip eval)
python -m benchmarks.omnidocbench --models qwen --max-samples 10 --no-eval

# Custom output directory
python -m benchmarks.omnidocbench --models nanonets --output-dir results/run_01
```

**Output layout:**
```
results/omnidocbench/<run_id>/
├── glmocr/
│   ├── eastmoney_59c...pdf_11.md
│   └── ...
├── deepseek/
│   └── ...
└── summary.json          ← aggregated eval scores
```

The official OmniDocBench eval repo is cloned automatically to
`benchmarks/omnidocbench/eval_repo/` on first run.  YAML configs are
generated per-model at `eval_repo/configs/<run_id>_<model>.yaml` —
you never need to write config files manually.

---

### olmOCR-bench

```bash
# Run all models, all splits
python -m benchmarks.olmocr

# Run specific models and splits
python -m benchmarks.olmocr --models glmocr,qwen --splits arxiv_math,table_tests

# Quick iteration (20 cases per split)
python -m benchmarks.olmocr --models deepseek --max-per-split 20

# Custom output directory
python -m benchmarks.olmocr --output-dir results/olmocr_run01

# List models and splits
python -m benchmarks.olmocr --list-info
```

**Available splits:** `arxiv_math`, `headers_footers`, `long_tiny_text`,
`multi_column`, `old_scans`, `old_scans_math`, `table_tests`

**Output layout:**
```
results/olmocr/<run_id>/
├── raw_results.json      ← per-case pass/fail with latency
└── summary.json          ← leaderboard-style table
```

---

## Adding a New Model

Edit `benchmarks/registry.py` only — add one factory function and one
entry to `MODEL_REGISTRY`.  No other file needs to change.

```python
def _make_mymodel():
    from omnidocs.tasks.text_extraction import MyModelExtractor
    from omnidocs.tasks.text_extraction.mymodel import MyModelConfig
    return MyModelExtractor(backend=MyModelConfig(device="cuda"))

MODEL_REGISTRY["mymodel"] = _make_mymodel
```

---

## Prerequisites

```bash
# For olmOCR-bench PDF rendering
pip install pdf2image pillow
# Also requires poppler:
#   Ubuntu: apt install poppler-utils
#   macOS:  brew install poppler

# For OmniDocBench dataset download
pip install huggingface_hub

# For OmniDocBench official eval (auto-installed on first run)
# git clone https://github.com/opendatalab/OmniDocBench
```