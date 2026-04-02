"# Benchmarking

OmniDocs includes a built-in benchmarking suite that evaluates text extraction models across three standard benchmarks. Inference runs on Modal (remote GPU) or locally, and evaluation always runs locally in-process — no external repos or subprocesses required.

---

## Benchmarks

| Benchmark | Dataset | What It Measures | Output |
|-----------|---------|-----------------|--------|
| **OmniDocBench** | 1,355 pages across document types | End-to-end text quality (Edit Distance, TEDS, CDM) | `results/omnidocbench/<run_id>/` |
| **NayanaOCRBench** | Multi-language natural document scans | Text quality per language | `results/nayana/<run_id>/` |
| **olmOCR-Bench** | Binary unit tests across 7 splits | Pass/fail rates for specific OCR capabilities | `results/olmocrbench/<run_id>/` |

---

## Supported Models

| Key | Model |
|-----|-------|
| `glmocr` | GLM-OCR |
| `qwen` | Qwen3-VL |
| `deepseek` | DeepSeek-OCR |
| `nanonets` | Nanonets OCR2 |
| `granitedocling` | Granite Docling |
| `mineruvl` | MinerU VL |
| `lighton` | LightOn OCR-2 |
| `dotsocr` | DotsOCR |

---

## Running on Modal (Remote GPU)

Modal runs inference in parallel across models on cloud GPUs. Evaluation runs locally after results are collected.

### Prerequisites

```bash
pip install modal
modal setup
modal secret create adithya-hf-wandb HF_TOKEN=your_hf_token
```

### OmniDocBench

```bash
# All models, full dataset
modal run tests/benchmark/test_benchmark.py --benchmark omnidocbench

# Specific models
modal run tests/benchmark/test_benchmark.py --benchmark omnidocbench --models glmocr,qwen

# Limit pages (faster iteration)
modal run tests/benchmark/test_benchmark.py --benchmark omnidocbench --models glmocr --max-samples 10

# Custom output directory
modal run tests/benchmark/test_benchmark.py --benchmark omnidocbench --output-dir results/run_01
```

### NayanaOCRBench (Multilingual)

```bash
# All models, all 22 languages
modal run tests/benchmark/test_benchmark.py --benchmark multilingual

# Specific languages
modal run tests/benchmark/test_benchmark.py --benchmark multilingual --languages en,hi,kn

# Limit pages per language
modal run tests/benchmark/test_benchmark.py --benchmark multilingual --models glmocr --languages en --max-per-language 10
```

### olmOCR-Bench

```bash
# All models, all splits
modal run tests/benchmark/test_benchmark.py --benchmark olmocr

# Specific splits
modal run tests/benchmark/test_benchmark.py --benchmark olmocr --splits arxiv_math,table_tests

# Limit cases per split
modal run tests/benchmark/test_benchmark.py --benchmark olmocr --models glmocr --max-per-split 20

# List available models and splits
modal run tests/benchmark/test_benchmark.py --benchmark olmocr --list-info
```

---

## Running Locally (No Modal)

Local runs use the same dataset loading, inference loop, and evaluation as Modal — the only difference is models run sequentially on your local GPU instead of in parallel on cloud GPUs.

### Prerequisites

```bash
cd Omnidocs
uv sync --group pytorch
```

Install eval dependencies (required for OmniDocBench and NayanaOCRBench evaluation):

```bash
pip install pylatexenc beautifulsoup4 func-timeout loguru apted Levenshtein mmeval tabulate rapidfuzz evaluate
```

### OmniDocBench

```bash
# All models
python -m benchmarks.omnidocbench

# Specific models, limit pages
python -m benchmarks.omnidocbench --models glmocr --max-samples 10

# Custom output directory
python -m benchmarks.omnidocbench --models glmocr --output-dir results/my_run

# Inference only, skip eval
python -m benchmarks.omnidocbench --models glmocr --no-eval

# List available models
python -m benchmarks.omnidocbench --list-models
```

### NayanaOCRBench (Multilingual)

```bash
# All models, all 22 languages
python -m benchmarks.multilingual

# Specific models and languages
python -m benchmarks.multilingual --models glmocr --languages en,hi,kn

# Limit pages per language
python -m benchmarks.multilingual --models glmocr --max-per-language 5

# List available languages
python -m benchmarks.multilingual --list-languages
```

### olmOCR-Bench

```bash
# All models, all splits
python -m benchmarks.olmocrbench

# Specific models and splits
python -m benchmarks.olmocrbench --models glmocr --splits arxiv_math,table_tests

# Limit cases per split
python -m benchmarks.olmocrbench --models glmocr --max-per-split 20

# List available models and splits
python -m benchmarks.olmocrbench --list-info
```

---

## Output Format

Every run produces a `results.json` in the output directory. All three benchmarks use the same top-level structure, with `\"execution\"` distinguishing Modal from local runs.

```json
{
  \"run_id\": \"20260331_102745\",
  \"benchmark\": \"omnidocbench\",
  \"execution\": \"modal\",
  \"models\": [\"glmocr\"],
  \"inference\": {
    \"glmocr\": {
      \"pages_written\": 10,
      \"pages_failed\": 0
    }
  },
  \"eval_scores\": {
    \"glmocr\": { ... }
  }
}
```

### benchmark-specific fields

| Field | omnidocbench | multilingual | olmocrbench |
|-------|-------------|--------------|-------------|
| `languages` | — | `[\"en\", \"hi\", \"kn\"]` | — |
| `splits` | — | — | `[\"arxiv_math\", ...]` |

### eval_scores structure

**OmniDocBench / NayanaOCRBench** — scores from the official OmniDocBench evaluator:

```json
\"eval_scores\": {
  \"glmocr\": {
    \"text_block\": {
      \"all\": { \"Edit_dist\": { \"ALL_page_avg\": 0.035, \"edit_whole\": 0.085 } }
    },
    \"table\": {
      \"all\": { \"TEDS\": { \"all\": 0.25 }, \"Edit_dist\": { \"ALL_page_avg\": 0.89 } }
    },
    \"display_formula\": { ... },
    \"reading_order\": { ... }
  }
}
```

**NayanaOCRBench** adds a language nesting level:

```json
\"eval_scores\": {
  \"glmocr\": {
    \"en\": { \"text_block\": { ... }, \"table\": { ... } },
    \"hi\": { \"text_block\": { ... }, \"table\": { ... } }
  }
}
```

**olmOCR-Bench** — binary pass rates:

```json
\"eval_scores\": {
  \"glmocr\": {
    \"overall\": 0.82,
    \"by_split\": { \"arxiv_math\": 0.91, \"table_tests\": 0.74 },
    \"by_check\": { \"text_present\": 0.88, \"table\": 0.71 },
    \"latency_p50_s\": 12.4,
    \"latency_p95_s\": 28.1
  }
}
```

---

## Key Metrics

### OmniDocBench / NayanaOCRBench

| Metric | What It Measures | Better |
|--------|-----------------|--------|
| `Edit_dist ALL_page_avg` | Character-level error rate across pages | Lower |
| `TEDS` | Table structure + content similarity | Higher |
| `TEDS_structure_only` | Table structure only (ignores cell content) | Higher |
| `CDM_plain` | Formula similarity | Higher |

### olmOCR-Bench

| Metric | What It Measures | Better |
|--------|-----------------|--------|
| `overall` | % of unit tests passed across all splits | Higher |
| `by_split` | Pass rate per dataset split | Higher |
| `by_check` | Pass rate per check type (text_present, table, math, etc.) | Higher |

---

## Directory Structure

```
benchmarks/
├── base.py                    # Shared dataclasses (PageSample, PageResult, OlmTestCase, OlmResult)
├── registry.py                # Model registry and factory functions
├── omnidocbench/
│   ├── dataset.py             # Loads OmniDocBench from HuggingFace
│   ├── runner.py              # Inference loop + run_omnidocbench()
│   ├── evaluator.py           # Calls omnidocbench_eval in-process
│   └── __main__.py            # CLI: python -m benchmarks.omnidocbench
├── multilingual/
│   ├── dataset.py             # Loads NayanaOCRBench from HuggingFace
│   ├── runner.py              # Inference loop + run_multilingual()
│   ├── evaluator.py           # Calls omnidocbench_eval per language
│   └── __main__.py            # CLI: python -m benchmarks.multilingual
├── olmocrbench/
│   ├── dataset.py             # Loads olmOCR-Bench from HuggingFace
│   ├── runner.py              # Inference + scoring + run_olmocrbench_bench()
│   ├── scorer.py              # Binary pass/fail scoring per check type
│   └── __main__.py            # CLI: python -m benchmarks.olmocrbench
└── omnidocbench_eval/         # OmniDocBench eval source (copied, runs in-process)
    ├── run_eval.py            # In-process entry point (replaces pdf_validation.py)
    ├── dataset/
    ├── task/
    ├── metrics/
    ├── registry/
    └── utils/

tests/benchmark/
└── test_benchmark.py          # Modal runner (inference on GPU, eval locally)
```

---

## How Evaluation Works

For OmniDocBench and NayanaOCRBench, evaluation runs in-process using source copied from the official [OmniDocBench repo](https://github.com/opendatalab/OmniDocBench) into `benchmarks/omnidocbench_eval/`. This means:

- No git clone at runtime
- No subprocess calls
- No external pip installs during the run
- Works identically locally and on Modal

For olmOCR-Bench, scoring is binary (pass/fail) and runs inline during inference — no separate eval step needed.

---

## Adding a New Model to Benchmarks

1. Add the model to `benchmarks/registry.py` with a `_make_<model>` factory function.
2. Add a Modal function for each benchmark in `tests/benchmark/test_benchmark.py`.
3. Add the model key to all three `*_REGISTRY` dicts in `test_benchmark.py`.

The local runners pick up new models automatically from the registry.
"
