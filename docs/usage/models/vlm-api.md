# VLM API (Any Cloud Provider)

Use **any vision-language model** through a single, provider-agnostic API. Powered by [litellm](https://docs.litellm.ai/), VLM API works with any provider that supports the OpenAI chat completions spec (also known as the Open Responses API). This includes OpenRouter, ANANNAS AI, Google Gemini, Azure OpenAI, OpenAI, and any self-hosted VLLM server.

---

## Why VLM API?

- **No GPU required** -- use cloud models directly
- **Provider-agnostic** -- switch between providers by changing one string
- **Custom prompts** -- tailor extraction to your domain
- **Structured output** -- extract data into Pydantic schemas
- **Works with any VLM** -- Gemini, GPT, Qwen, Claude, Llama, Grok, and more
- **Any litellm-compatible or OpenAI-spec provider** -- if it speaks the OpenAI API, it works

---

## Quick Start

```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

# Just set your env var: OPENROUTER_API_KEY, GOOGLE_API_KEY, etc.
config = VLMAPIConfig(model="openrouter/qwen/qwen3-vl-8b-instruct")

extractor = VLMTextExtractor(config=config)
result = extractor.extract("document.png", output_format="markdown")
print(result.content)
```

---

## Supported Providers

OmniDocs works with **any provider** that is either:

1. **Natively supported by litellm** (use the litellm model prefix)
2. **OpenAI API-compatible** (use `openai/` prefix + `api_base`)

| Provider | Model Format | Env Variable | Notes |
|----------|-------------|-------------|-------|
| **[OpenRouter](https://openrouter.ai/)** | `openrouter/org/model` | `OPENROUTER_API_KEY` | 100+ vision models, pay-per-token |
| **[ANANNAS AI](https://anannas.ai/)** | `openai/model-name` | `ANANNAS_API_KEY` | OpenAI-compatible, wide model selection |
| **[Google Gemini](https://ai.google.dev/)** | `gemini/model-name` | `GOOGLE_API_KEY` | Native litellm support |
| **[Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)** | `azure/deployment-name` | `AZURE_API_KEY` | Requires `api_version` |
| **[OpenAI](https://openai.com/)** | `openai/model-name` | `OPENAI_API_KEY` | Native litellm support |
| **Self-hosted VLLM** | `openai/model-name` | -- | Use `api_base` to point to your server |

---

## Provider Setup Examples

### OpenRouter

Access 100+ vision models through a single API key.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

```python
from omnidocs.vlm import VLMAPIConfig

# Qwen models (great for document extraction)
config = VLMAPIConfig(model="openrouter/qwen/qwen3-vl-8b-instruct")
config = VLMAPIConfig(model="openrouter/qwen/qwen3-vl-32b-instruct")

# Google via OpenRouter
config = VLMAPIConfig(model="openrouter/google/gemini-2.5-flash-image")

# Anthropic via OpenRouter
config = VLMAPIConfig(model="openrouter/anthropic/claude-opus-4.6")

# OpenAI via OpenRouter
config = VLMAPIConfig(model="openrouter/openai/gpt-5.2")
```

??? note "Available vision models on OpenRouter"

    | Provider | Models |
    |----------|--------|
    | **Qwen** | `qwen/qwen3-vl-8b-instruct`, `qwen/qwen3-vl-32b-instruct`, `qwen/qwen3-vl-30b-a3b-instruct`, `qwen/qwen3-vl-8b-thinking`, `qwen/qwen3-vl-30b-a3b-thinking` |
    | **Google** | `google/gemini-3-flash-preview`, `google/gemini-3-pro-preview`, `google/gemini-2.5-flash-image` |
    | **OpenAI** | `openai/gpt-5.2`, `openai/gpt-5.1`, `openai/gpt-5-image-mini` |
    | **Anthropic** | `anthropic/claude-opus-4.6`, `anthropic/claude-opus-4.5`, `anthropic/claude-haiku-4.5` |
    | **Mistral** | `mistralai/mistral-large-3-2512`, `mistralai/ministral-3-14b-2512`, `mistralai/ministral-3-8b-2512`, `mistralai/ministral-3-3b-2512` |
    | **NVIDIA** | `nvidia/nemotron-nano-12b-2-vl` |
    | **AllenAI** | `allenai/molmo2-8b` |
    | **ByteDance** | `bytedance-seed/seed-1.6-flash`, `bytedance-seed/seed-1.6` |
    | **xAI** | `x-ai/grok-4-1-fast` |
    | **Z.AI** | `z-ai/glm-4.6v` |
    | **Amazon** | `amazon/nova-2-lite` |

    All model names above should be prefixed with `openrouter/` in OmniDocs.

### ANANNAS AI

OpenAI-compatible API with access to models from multiple providers.

```bash
export ANANNAS_API_KEY=...
export ANANNAS_BASE_URL=https://api.anannas.ai/v1  # or your ANANNAS endpoint
```

```python
from omnidocs.vlm import VLMAPIConfig

# ANANNAS uses OpenAI-compatible API, so use openai/ prefix + api_base
config = VLMAPIConfig(
    model="openai/qwen3-vl-8b-instruct",
    api_key=os.environ["ANANNAS_API_KEY"],
    api_base=os.environ["ANANNAS_BASE_URL"],
)

# Claude on ANANNAS
config = VLMAPIConfig(
    model="openai/claude-opus-4.6",
    api_key=os.environ["ANANNAS_API_KEY"],
    api_base=os.environ["ANANNAS_BASE_URL"],
)

# GPT-5 on ANANNAS
config = VLMAPIConfig(
    model="openai/gpt-5-mini",
    api_key=os.environ["ANANNAS_API_KEY"],
    api_base=os.environ["ANANNAS_BASE_URL"],
)
```

??? note "Available vision models on ANANNAS AI"

    | Provider | Models |
    |----------|--------|
    | **Anthropic** | `claude-3-haiku`, `claude-haiku-4-5`, `claude-opus-4`, `claude-opus-4-1`, `claude-opus-4-6`, `claude-sonnet-4`, `claude-sonnet-4-5` |
    | **OpenAI** | `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-pro`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini` |
    | **OpenAI o-series** | `o1`, `o1-pro`, `o3`, `o3-pro`, `o4-mini` |
    | **Google** | `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, `google-gemma-3-27b-it` |
    | **Qwen** | `qwen-qwen3-vl-235b-a22b`, `qwen2.5-vl-72b-instruct` |
    | **Meta** | `meta-llama4-maverick-17b-instruct-v1-0`, `meta-llama4-scout-17b-instruct-v1-0`, `meta-llama3-2-90b-instruct-v1-0` |
    | **Amazon** | `amazon-nova-lite-v1-0`, `amazon-nova-premier-v1-0`, `amazon-nova-pro-v1-0` |
    | **Mistral** | `mistral-voxtral-small-24b-2507` |
    | **xAI** | `grok-2-vision`, `grok-4-1-fast` |
    | **Z.AI** | `glm-4.5v`, `glm-4.6v` |
    | **MoonshotAI** | `kimi-k2-5` |

    All model names above should be used without prefix but with `openai/` prefix and `api_base` set to your ANANNAS endpoint.

### Google Gemini (Direct)

```bash
export GOOGLE_API_KEY=...
```

```python
from omnidocs.vlm import VLMAPIConfig

config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
config = VLMAPIConfig(model="gemini/gemini-2.5-pro")
```

### Azure OpenAI

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-resource.openai.azure.com/
```

```python
from omnidocs.vlm import VLMAPIConfig

config = VLMAPIConfig(
    model="azure/gpt-5-mini",
    api_version="2024-12-01-preview",
)
```

### OpenAI (Direct)

```bash
export OPENAI_API_KEY=sk-...
```

```python
from omnidocs.vlm import VLMAPIConfig

config = VLMAPIConfig(model="openai/gpt-4o")
config = VLMAPIConfig(model="openai/gpt-5-mini")
```

### Self-Hosted VLLM

Any VLLM server exposes an OpenAI-compatible endpoint. Use `openai/` prefix with `api_base`:

```python
from omnidocs.vlm import VLMAPIConfig

# Local VLLM server
config = VLMAPIConfig(
    model="openai/Qwen/Qwen3-VL-8B-Instruct",
    api_base="http://localhost:8000/v1",
    temperature=0.0,
)

# Modal-deployed VLLM server
config = VLMAPIConfig(
    model="openai/mineru-vl",
    api_base="https://your-app--server-serve.modal.run/v1",
    temperature=0.0,
)
```

### Any OpenAI-Compatible Provider

Any API that follows the OpenAI chat completions spec works. Use `openai/` prefix and set `api_base`:

```python
from omnidocs.vlm import VLMAPIConfig

config = VLMAPIConfig(
    model="openai/model-name",
    api_key="your-api-key",
    api_base="https://your-provider.com/v1",
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

### Model String Format

The `model` parameter follows litellm conventions:

| Pattern | Example | When to Use |
|---------|---------|-------------|
| `provider/model` | `gemini/gemini-2.5-flash` | Litellm-native providers |
| `openrouter/org/model` | `openrouter/qwen/qwen3-vl-32b-instruct` | OpenRouter |
| `azure/deployment` | `azure/gpt-5-mini` | Azure OpenAI |
| `openai/model` + `api_base` | `openai/qwen3-vl-8b-instruct` | OpenAI-compatible APIs (ANANNAS, VLLM, etc.) |

---

## Tasks

### Text Extraction

```python
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

config = VLMAPIConfig(model="openrouter/qwen/qwen3-vl-8b-instruct")
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

One of the key benefits is being able to swap providers without changing your extraction code:

```python
import os
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.text_extraction import VLMTextExtractor

configs = {
    # Native litellm providers
    "gemini": VLMAPIConfig(model="gemini/gemini-2.5-flash"),
    "openrouter": VLMAPIConfig(model="openrouter/qwen/qwen3-vl-32b-instruct"),
    "azure": VLMAPIConfig(model="azure/gpt-5-mini", api_version="2024-12-01-preview"),

    # OpenAI-compatible providers
    "anannas": VLMAPIConfig(
        model="openai/claude-opus-4.6",
        api_key=os.environ.get("ANANNAS_API_KEY"),
        api_base=os.environ.get("ANANNAS_BASE_URL"),
    ),
    "vllm": VLMAPIConfig(
        model="openai/mineru-vl",
        api_base="https://my-server.modal.run/v1",
    ),
}

for name, config in configs.items():
    extractor = VLMTextExtractor(config=config)
    result = extractor.extract("document.png")
    print(f"{name}: {len(result.content)} chars")
```

---

## Recommended Models for Document Processing

| Use Case | Recommended Model | Provider |
|----------|-------------------|----------|
| **General text extraction** | `qwen/qwen3-vl-32b-instruct` | OpenRouter |
| **Fast + cheap extraction** | `qwen/qwen3-vl-8b-instruct` | OpenRouter |
| **Best quality** | `gemini/gemini-2.5-pro` | Google |
| **Structured output** | `gemini/gemini-2.5-flash` | Google |
| **Layout detection** | `qwen/qwen3-vl-32b-instruct` | OpenRouter |
| **Self-hosted** | Any Qwen3-VL or MinerU VL | VLLM |

---

## Troubleshooting

**Authentication error**

Set the correct environment variable for your provider:
```bash
export GOOGLE_API_KEY=...       # Gemini
export OPENROUTER_API_KEY=...   # OpenRouter
export AZURE_API_KEY=...        # Azure
export OPENAI_API_KEY=...       # OpenAI
export ANANNAS_API_KEY=...      # ANANNAS AI
```

**Azure errors**

Azure requires `api_version`:
```python
config = VLMAPIConfig(
    model="azure/gpt-5-mini",
    api_version="2024-12-01-preview",
)
```

**OpenAI-compatible provider not working**

Make sure you use `openai/` prefix and set `api_base`:
```python
config = VLMAPIConfig(
    model="openai/model-name",      # Must have openai/ prefix
    api_base="https://provider.com/v1",  # Must set api_base
    api_key="your-key",
)
```

**Structured output fails**

Some providers don't support native JSON schema output. The extractor automatically falls back to prompt-based extraction. If results are still poor, try a more capable model (Gemini 2.5 Flash/Pro work well).

**Using a provider not listed here**

OmniDocs works with any provider that is either litellm-supported or follows the OpenAI API spec. Check the [litellm providers list](https://docs.litellm.ai/docs/providers) for native support, or use `openai/` prefix with `api_base` for any OpenAI-compatible endpoint.
