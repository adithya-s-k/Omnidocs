# Getting Started with OmniDocs

Welcome! OmniDocs is a unified Python toolkit for visual document processing. This section will get you up and running in minutes.

## 📖 Choose Your Path

### Path 1: "Just Show Me the Code" (5 minutes)

Go straight to **[Quickstart](quickstart.md)** for a copy-paste working example:
- Load a PDF
- Extract text
- Get the output

**Best for:** Experienced developers, quick prototyping.

### Path 2: "Let Me Understand This First" (30 minutes)

Follow this learning path in order:

1. **[Installation Guide](installation.md)** (5-10 min)
   - Install OmniDocs for your system
   - Choose between PyTorch, VLLM, MLX, or API backend
   - Verify installation works

2. **[Quickstart](quickstart.md)** (5 min)
   - Run a minimal working example
   - Understand the basic API

3. **[First Document](first-document.md)** (10-15 min)
   - Load documents (PDF, URL, images)
   - Access pages and metadata
   - Learn memory management

4. **[Choosing Backends](choosing-backends.md)** (5-10 min)
   - Understand tradeoffs between backends
   - Make informed backend choice
   - See performance comparisons

**Best for:** New users, learning the system deeply.

### Path 3: "I Have a Specific Use Case" (Targeted)

**Need to process PDFs on your Mac?**
→ [Installation: MLX Backend](installation.md#option-3-mlx-apple-silicon) + [Quickstart](quickstart.md)

**Need fast batch processing?**
→ [Choosing Backends: VLLM](choosing-backends.md#vllm-production-throughput) + [First Document: Batch Processing](first-document.md#batch-processing-multiple-files)

**Need to process without GPU?**
→ [Installation: API Backend](installation.md#option-4-api-cloud-based-no-setup) + [Choosing Backends: API](choosing-backends.md#api-backend-cloud-based)

**Need simple one-off extraction?**
→ [Quickstart](quickstart.md)

## 🎯 5-Minute Quick Reference

```python
# 1. Load document (3 seconds)
from omnidocs import Document
doc = Document.from_pdf("document.pdf")

# 2. Initialize extractor (30 seconds on first run)
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# 3. Extract text (1-2 seconds per page)
result = extractor.extract(doc.get_page(0), output_format="markdown")

# 4. Use the output
print(result.content)  # Extracted markdown text
```

## 📋 What Each Guide Covers

### Installation Guide
- **System requirements** - What you need to get started
- **Backend selection** - PyTorch vs VLLM vs MLX vs API
- **Step-by-step installation** - Install commands for each backend
- **Verification steps** - Make sure everything works
- **Troubleshooting** - Common issues and fixes

**Read this if:** You haven't installed OmniDocs yet, or need to switch backends.

### Quickstart
- **5-minute tutorial** - Get working code immediately
- **Load a PDF** - `Document.from_pdf()`
- **Extract text** - `QwenTextExtractor.extract()`
- **Get output** - `result.content`
- **Complete examples** - Copy-paste ready code
- **Common tasks** - Batch processing, different formats

**Read this if:** You want to see working code right away, or get started quickly.

### First Document
- **Document class deep dive** - How documents work
- **Loading methods** - PDF, URL, images, bytes
- **Page access** - Get single pages, iterate, batch process
- **Metadata** - Access document information
- **Memory management** - Cache behavior, clearing cache
- **Performance tips** - Optimize for speed
- **Real-world examples** - Complete use case implementations

**Read this if:** You want to understand document loading deeply, or need to optimize for performance.

### Choosing Backends
- **Decision tree** - Which backend for my use case?
- **Detailed profiles** - Strengths, limitations, performance of each backend
- **Comparison tables** - Speed, cost, setup time
- **Migration guide** - How to switch between backends
- **Troubleshooting** - Backend selection issues

**Read this if:** You're unsure which backend to use, or want to understand the tradeoffs.

## ⚡ Common Questions Answered

**Q: Which backend should I use?**
A: Start with **PyTorch** for development (simplest). Switch to **VLLM** if you need 100+ docs/day. Use **MLX** on Mac, **API** if you don't have GPU. See [Choosing Backends](choosing-backends.md) for details.

**Q: How long does extraction take?**
A: ~1 second per page on modern GPU (PyTorch), ~0.1 second with VLLM, ~2-3 seconds on Mac (MLX), ~5 seconds with API (includes network latency).

**Q: What file formats are supported?**
A: PDF, PNG, JPG, GIF, BMP, TIFF, WebP. Can load from files, URLs, or raw bytes.

**Q: Can I use this on a Mac?**
A: Yes! Use [MLX backend](installation.md#option-3-mlx-apple-silicon) for native performance optimization.

**Q: Can I use this without a GPU?**
A: Yes, three options:
1. Use [API backend](installation.md#option-4-api-cloud-based-no-setup) (costs money, easiest)
2. Use [PyTorch on CPU](quickstart.md#choose-your-backend) (free, very slow, ~20-30 sec per page)
3. Use [MLX on Mac](installation.md#option-3-mlx-apple-silicon) (good Mac performance, free)

**Q: What's the difference between backends?**
A: See [Choosing Backends](choosing-backends.md#backend-comparison) for full comparison. TL;DR: PyTorch is easiest, VLLM is fastest, MLX is for Mac, API is simplest.

**Q: How much GPU VRAM do I need?**
A: ~4-8 GB for 2B models, ~8-16 GB for 8B models. VLLM needs 24+ GB.

**Q: Can I process multiple documents in parallel?**
A: With PyTorch, one at a time. With VLLM, can batch multiple images. See [First Document: Batch Processing](first-document.md#batch-processing-multiple-files).

## 🚀 Get Started Now

### Fastest Path (Right Now)
```bash
pip install omnidocs[pytorch]
python -c "from omnidocs import Document; print('Ready to go!')"
```
Then go to [Quickstart](quickstart.md) for working code.

### Recommended Path (Takes 30 min)
1. [Installation](installation.md) - Install for your system
2. [Quickstart](quickstart.md) - See a working example
3. [First Document](first-document.md) - Understand the API
4. [Choosing Backends](choosing-backends.md) - Pick the best backend

### Learning Path (Deep Dive)
1. [Installation](installation.md) - Understand all backend options
2. [First Document](first-document.md) - Master document loading
3. [Quickstart](quickstart.md) - Learn extraction
4. [Choosing Backends](choosing-backends.md) - Understand performance tradeoffs

## 📚 Next Steps After Getting Started

- **[Guides](../guides/README.md)** - Task-oriented tutorials
- **[Concepts](../concepts/README.md)** - Architecture deep-dives
- **[Models](../models/README.md)** - Model reference
- **[API Reference](../reference/)** - Complete API documentation
- **[Contributing](../CONTRIBUTING.md)** - Contribute to OmniDocs

## 💡 Tips for Success

1. **Start simple** - Use 2B Qwen model for fast feedback, switch to 8B for production
2. **Use PyTorch first** - Easiest to learn, then optimize later
3. **Process one page at a time** - Avoid memory issues with `iter_pages()`
4. **Save results immediately** - Don't accumulate in memory, save to disk
5. **Clear cache regularly** - Free GPU memory with `doc.clear_cache()`
6. **Use page ranges** - `page_range=(0, 100)` for faster loading of partial docs

## 🆘 Getting Help

- **Documentation** - This guide + full API docs
- **GitHub Issues** - [Report bugs or request features](https://github.com/adithya-s-k/OmniDocs/issues)
- **GitHub Discussions** - [Ask questions and chat](https://github.com/adithya-s-k/OmniDocs/discussions)
- **Stack Overflow** - Tag questions with `omnidocs`

## 📊 Learning Paths at a Glance

| Your Background | Recommended Path | Time |
|-----------------|-----------------|------|
| New to Python | [Path 2: Full Learning](#path-2-let-me-understand-this-first-30-minutes) | 30 min |
| Experienced Python | [Path 2: Full Learning](#path-2-let-me-understand-this-first-30-minutes) | 20 min |
| Experienced ML engineer | [Quickstart](quickstart.md) + [Choosing Backends](choosing-backends.md) | 10 min |
| Just want code | [Quickstart](quickstart.md) | 5 min |

---

**Ready? Pick a guide above and start building!**

For the impatient: [Go to Quickstart](quickstart.md)
For the thorough: [Start with Installation](installation.md)
For the strategic: [Jump to Choosing Backends](choosing-backends.md)
