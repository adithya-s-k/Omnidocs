# Structured Extraction

Extract structured data from documents into typed Pydantic schemas.

---

## Input / Output

**Input:** Document image + Pydantic schema + prompt

**Output:** Validated Pydantic model instance

```python
result = extractor.extract(image, schema=Invoice, prompt="Extract invoice fields")
print(result.data.vendor)   # "Acme Corp"
print(result.data.total)    # 1250.00
```

---

## Quick Start

```python
from pydantic import BaseModel
from omnidocs.vlm import VLMAPIConfig
from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

# Define your schema
class Invoice(BaseModel):
    vendor: str
    total: float
    items: list[str]
    date: str

# Initialize
config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
extractor = VLMStructuredExtractor(config=config)

# Extract
result = extractor.extract(
    image="invoice.png",
    schema=Invoice,
    prompt="Extract the invoice details from this document.",
)

# Use typed data
print(f"Vendor: {result.data.vendor}")
print(f"Total: ${result.data.total:.2f}")
for item in result.data.items:
    print(f"  - {item}")
```

---

## Available Models

| Model | Type | Best For |
|-------|------|----------|
| [VLM API](../models/vlm-api.md) | Cloud API | Any provider (Gemini, OpenRouter, Azure) |

---

## Schema Examples

### Resume

```python
class Education(BaseModel):
    institution: str
    degree: str
    year: str

class Resume(BaseModel):
    name: str
    email: str
    skills: list[str]
    education: list[Education]
    experience_years: int
```

### Receipt

```python
class LineItem(BaseModel):
    description: str
    quantity: int
    price: float

class Receipt(BaseModel):
    store: str
    date: str
    items: list[LineItem]
    subtotal: float
    tax: float
    total: float
```

### Scientific Paper

```python
class Paper(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
```

---

## When to Use

- Extracting fields from invoices, receipts, forms
- Parsing resumes or business cards
- Converting documents to database records
- Building document processing pipelines with typed outputs

---

## Tips

- Use specific, detailed prompts for better accuracy
- Keep schemas focused -- extract one logical group at a time
- More capable models (GPT-4o, Gemini Pro) produce better structured output
- The extractor validates output against your schema automatically
