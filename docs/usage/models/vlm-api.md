# VLM API (Any Cloud Provider)

Use **any vision-language model** through a single, provider-agnostic API. Powered by [litellm](https://docs.litellm.ai/), VLM API works with OpenRouter, Google Gemini, Azure OpenAI, OpenAI, and any OpenAI-compatible endpoint (including self-hosted VLLM servers).

---

## Why VLM API?

- **No GPU required** -- use cloud models directly
- **Provider-agnostic** -- switch between providers by changing one string
- **Custom prompts** -- tailor extraction to your domain
- **Structured output** -- extract data into Pydantic schemas
- **Works with any VLM** -- Gemini, GPT, Qwen, Llama, and more

---

## Quick Start

```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

# Just set your env var: GOOGLE_API_KEY, OPENROUTER_API_KEY, etc.
config = VLMAPIConfig(model="gemini/gemini-2.5-flash")

extractor = VLMTextExtractor(config=config)
result = extractor.extract("document.png", output_format="markdown")
print(result.content)
```

---

## Supported Providers

| Provider | Model Format | Env Variable |
|----------|-------------|-------------|
| **Google Gemini** | `gemini/gemini-2.5-flash` | `GOOGLE_API_KEY` |
| **OpenRouter** | `openrouter/qwen/qwen3-vl-32b` | `OPENROUTER_API_KEY` |
| **Azure OpenAI** | `azure/gpt-5-mini` | `AZURE_API_KEY` |
| **OpenAI** | `openai/gpt-4o` | `OPENAI_API_KEY` |
| **Self-hosted VLLM** | `openai/model-name` | -- |

API keys can be passed explicitly or read automatically from environment variables.

```python
# Explicit API key
config = VLMAPIConfig(
    model="openrouter/qwen/qwen3-vl-32b",
    api_key="sk-or-...",
)

# From environment (recommended)
# export OPENROUTER_API_KEY=sk-or-...
config = VLMAPIConfig(model="openrouter/qwen/qwen3-vl-32b")

# Self-hosted VLLM server
config = VLMAPIConfig(
    model="openai/my-model",
    api_base="http://localhost:8000/v1",
)

# Azure (requires api_version)
config = VLMAPIConfig(
    model="azure/gpt-5-mini",
    api_version="2024-12-01-preview",
)
```

---

## VLMAPIConfig

```python
from omnidocs.vlm import VLMAPIConfig

config = VLMAPIConfig(
    model="gemini/gemini-2.5-flash",  # Required: litellm model string
    api_key=None,          # Optional: auto-reads from env
    api_base=None,         # Optional: override endpoint URL
    max_tokens=8192,       # Max tokens to generate
    temperature=0.1,       # Sampling temperature
    timeout=180,           # Request timeout (seconds)
    api_version=None,      # Required for Azure
    extra_headers=None,    # Additional HTTP headers
)
```

---

## Tasks

### Text Extraction

```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
extractor = VLMTextExtractor(config=config)

# Default prompt
result = extractor.extract("document.png", output_format="markdown")

# Custom prompt
result = extractor.extract(
    "document.png",
    prompt="Extract only the table data as a markdown table",
)
```

### Layout Detection

```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.layout_extraction import VLMLayoutDetector

config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
detector = VLMLayoutDetector(config=config)

# Default labels
result = detector.extract("document.png")
for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")

# Custom labels
result = detector.extract(
    "document.png",
    custom_labels=["code_block", "sidebar", "diagram"],
)
```

### Structured Extraction

Extract structured data into Pydantic schemas. The model returns validated, typed objects.

```python
from pydantic import BaseModel
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

class Invoice(BaseModel):
    vendor: str
    total: float
    items: list[str]
    date: str

config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
extractor = VLMStructuredExtractor(config=config)

result = extractor.extract(
    image="invoice.png",
    schema=Invoice,
    prompt="Extract invoice details from this document.",
)

# result.data is a validated Invoice instance
print(result.data.vendor)
print(result.data.total)
print(result.data.items)
```

---

## Switching Providers

One of the key benefits is being able to swap providers without changing your code:

```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

# Try different providers -- same extractor code
configs = {
    "gemini": VLMAPIConfig(model="gemini/gemini-2.5-flash"),
    "openrouter": VLMAPIConfig(model="openrouter/qwen/qwen3-vl-32b"),
    "azure": VLMAPIConfig(model="azure/gpt-5-mini", api_version="2024-12-01-preview"),
}

for name, config in configs.items():
    extractor = VLMTextExtractor(config=config)
    result = extractor.extract("document.png")
    print(f"{name}: {len(result.content)} chars")
```

---

## Self-Hosted VLLM

VLLM serves an OpenAI-compatible API. Use VLM API to connect:

```python
config = VLMAPIConfig(
    model="openai/my-model-name",
    api_base="https://my-server.modal.run/v1",
    temperature=0.0,
)
extractor = VLMTextExtractor(config=config)
result = extractor.extract("document.png")
```

---

## Troubleshooting

**Authentication error**

Set the correct environment variable for your provider:
```bash
export GOOGLE_API_KEY=...       # Gemini
export OPENROUTER_API_KEY=...   # OpenRouter
export AZURE_API_KEY=...        # Azure
export OPENAI_API_KEY=...       # OpenAI
```

**Azure errors**

Azure requires `api_version`:
```python
config = VLMAPIConfig(
    model="azure/gpt-5-mini",
    api_version="2024-12-01-preview",
)
```

**Structured output fails**

Some providers don't support native JSON schema output. The extractor automatically falls back to prompt-based extraction. If results are still poor, try a more capable model.
