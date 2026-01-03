# Omnidocs Development Roadmap

> A comprehensive roadmap for evolving Omnidocs from its current extraction-focused architecture to a complete, production-ready document understanding library with dual interfaces: Direct Model Inference and Pipeline Builder.

---

## Table of Contents

- [Current State](#current-state)
- [Target Architecture](#target-architecture)
- [Implementation Phases](#implementation-phases)
- [Detailed Phase Breakdown](#detailed-phase-breakdown)
- [Dependencies & Prerequisites](#dependencies--prerequisites)
- [Success Metrics](#success-metrics)
- [Timeline Estimate](#timeline-estimate)

---

## Current State

### What We Have ✅

**Strong Foundation (27+ Task Extractors Implemented)**

| Task Category | Extractors Available | Status |
|---------------|---------------------|--------|
| **Layout Analysis** | YOLOv8, RT-DETR, Surya, Florence, Paddle | ✅ Fully Functional |
| **OCR Extraction** | Surya OCR, PaddleOCR, Tesseract, EasyOCR | ✅ Multi-language Support |
| **Text Extraction** | PyMuPDF, PyPDF2, pdfplumber, Docling, Surya | ✅ Rich Text Features |
| **Table Extraction** | TableTransformer, TableFormer, PPStructure, Camelot, Tabula, Surya | ✅ Multiple Formats |
| **Math/LaTeX** | UniMERNet, Nougat, Donut, Surya | ✅ Formula Recognition |

**Well-Designed Infrastructure:**
- ✅ Pydantic-based output models (JSON, CSV, Markdown export)
- ✅ Consistent base class patterns across all tasks
- ✅ Label/language standardization via mapper pattern
- ✅ Model downloading & caching from HuggingFace
- ✅ Logging & device management utilities
- ✅ Visualization methods for all tasks

**Current Usage Pattern:**
```python
# Low-level extractor usage
from omnidocs.tasks.text_extraction.extractors.pymupdf import PyMuPDFTextExtractor

extractor = PyMuPDFTextExtractor()
result = extractor.extract("document.pdf")
result.save_json("output.json")
```

### What's Missing ❌

**Critical Gaps for Production Readiness:**

1. **Document Abstraction Layer** - No unified document container
2. **Model Wrapper Layer** - No high-level model interfaces
3. **Backend Abstraction** - No VLLM/MLX/API support
4. **Pipeline Builder** - No flexible workflow composition
5. **VLM Support** - No multi-task vision-language models
6. **Configuration System** - Minimal global configuration

**Current Limitations:**
- ❌ Users must work with raw file paths, not document objects
- ❌ No unified interface across different model types
- ❌ Cannot switch backends (VLLM vs Transformers vs MLX)
- ❌ Workflows are hardcoded, not composable
- ❌ No batch processing abstractions
- ❌ No conditional/parallel pipeline execution

---

## Target Architecture

### Two-Interface Design

Based on `Design/FINAL_DESIGN.md`, Omnidocs will provide:

#### **Interface 1: Direct Model Inference** 🎯
*Simple, explicit, single-task focused*

```python
from omnidocs import Document
from omnidocs.models import SuryaOCR
from omnidocs.backends import Backend

# Load document once
doc = Document.from_pdf("paper.pdf")

# Load model with backend selection
model = SuryaOCR(backend=Backend.VLLM, device="cuda")

# Process
result = model.process(doc)

# Export
result.save_json("output.json")
```

**Use Cases:** Quick tasks, scripts, experimentation, one-off processing

---

#### **Interface 2: Pipeline Builder** 🔧
*Complex, multi-step, reusable workflows*

```python
from omnidocs import Pipeline, Step, Document
from omnidocs.models import DocLayoutYOLO, SuryaOCR, TableTransformer
from omnidocs.backends import Backend

# Load models once
layout_model = DocLayoutYOLO(backend=Backend.VLLM)
ocr_model = SuryaOCR(backend=Backend.VLLM)
table_model = TableTransformer(backend=Backend.TRANSFORMERS)

# Build reusable pipeline
pipeline = Pipeline([
    Step.layout(model=layout_model),
    Step.ocr(model=ocr_model),
    Step.tables(model=table_model)
])

# Process single document
doc = Document.from_pdf("invoice.pdf")
result = pipeline.run(doc)

# Batch process
docs = Document.from_folder("pdfs/")
results = pipeline.run_batch(docs, batch_size=16, num_workers=4)

# Save pipeline for reuse
pipeline.save("invoice_pipeline.yaml")
```

**Use Cases:** Production workflows, batch processing, reusable logic, multi-step tasks

---

## Implementation Phases

### Phase 0: Foundation (Current State)
**Status:** ✅ Complete
- Task extractors implemented
- Output models designed
- Infrastructure in place

### Phase 1: Document Abstraction Layer
**Priority:** 🔴 Critical (Required for everything else)
- Implement `Document` class
- Multi-source loading (PDF, URL, folder, bytes, images)
- Lazy text extraction with PyMuPDF → pdfplumber fallback
- Metadata management
- Page image caching

### Phase 2: Model Wrapper Layer
**Priority:** 🔴 Critical
- Create `omnidocs.models` module
- Implement `BaseModel` and `BaseVLM` abstractions
- Wrap existing extractors in high-level model interfaces
- Add configuration management

### Phase 3: Backend Abstraction
**Priority:** 🟠 High
- Create `omnidocs.backends` module
- Implement `Backend` enum and config classes
- Add VLLM backend support
- Add MLX backend (Apple Silicon)
- Add API client backend
- Add Hosted backend integration

### Phase 4: Pipeline Builder
**Priority:** 🟠 High
- Create `omnidocs.pipeline` module
- Implement `Pipeline` class with step composition
- Add `Step` builders for all task types
- Add conditional execution (`Step.when()`)
- Add parallel execution (`Step.parallel()`)
- Add custom steps and transformations
- Add pipeline serialization (YAML save/load)

### Phase 5: VLM Support & Multi-Task
**Priority:** 🟡 Medium
- Implement Vision-Language Model wrappers
- Add multi-task processing capabilities
- Create `Task` enum
- Implement `MultiTaskResult` composition
- Add batch processing optimizations

### Phase 6: Advanced Features
**Priority:** 🟢 Enhancement
- Advanced error handling strategies
- Configuration system with environment variables
- Performance monitoring & telemetry
- Caching & optimization layer
- Documentation & examples

---

## Detailed Phase Breakdown

### Phase 1: Document Abstraction Layer
**Estimated Effort:** 2-3 days | **Priority:** 🔴 Critical

#### Deliverables

**1.1 Core Directory Structure**
```
omnidocs/
├── core/
│   ├── __init__.py
│   ├── document.py       # Main Document class
│   └── exceptions.py     # Custom exceptions
```

**1.2 Document Class Implementation**
- `Document` class with lazy loading
- `DocumentMetadata` Pydantic model
- Constructors:
  - `from_pdf(path, page_range=None, dpi=150)`
  - `from_url(url, timeout=30)`
  - `from_folder(path, pattern="*.pdf", recursive=False)`
  - `from_bytes(data, filename=None)`
  - `from_image(path)` / `from_images(paths)`
- Properties:
  - `page_count`, `pages`, `metadata`
  - `text` (lazy extraction with PyMuPDF → pdfplumber fallback)
- Methods:
  - `get_page(num)`, `get_pages(range)`
  - `get_page_text(num)`
  - `save_images(output_dir)`
  - `to_dict()`

**1.3 Custom Exceptions**
- `DocumentLoadError`
- `PDFConversionError`
- `URLDownloadError`
- `PageRangeError`
- `UnsupportedFormatError`
- `TextExtractionError`

**1.4 Dependencies**
- Add `requests>=2.26.0` to `pyproject.toml`

**1.5 Tests**
- Unit tests for all constructors
- Error handling tests
- Lazy loading tests
- Integration tests with real PDFs

**1.6 Documentation**
- API documentation for Document class
- Usage examples
- Migration guide from current file-based approach

#### Success Criteria
- [ ] All constructors working (pdf, url, folder, bytes, images)
- [ ] Lazy text extraction with fallback strategy
- [ ] All properties and methods functional
- [ ] Test coverage >80%
- [ ] Documentation complete

---

### Phase 2: Model Wrapper Layer
**Estimated Effort:** 3-4 days | **Priority:** 🔴 Critical

#### Deliverables

**2.1 Module Structure**
```
omnidocs/
├── models/
│   ├── __init__.py
│   ├── base.py           # BaseModel, BaseVLM
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── surya.py      # SuryaOCR wrapper
│   │   ├── paddle.py     # PaddleOCR wrapper
│   │   └── tesseract.py  # Tesseract wrapper
│   ├── layout/
│   │   ├── __init__.py
│   │   ├── yolo.py       # DocLayoutYOLO wrapper
│   │   └── rtdetr.py     # RTDETR wrapper
│   ├── table/
│   │   ├── __init__.py
│   │   └── transformer.py # TableTransformer wrapper
│   ├── math/
│   │   ├── __init__.py
│   │   └── unimernet.py   # UniMERNet wrapper
│   └── vlm/
│       ├── __init__.py
│       ├── qwen.py        # QwenVL wrappers
│       ├── chandra.py     # Chandra wrapper
│       └── gemma.py       # Gemma wrapper
```

**2.2 Base Classes**

```python
# omnidocs/models/base.py
class BaseModel(ABC):
    """Base class for all models"""

    def __init__(
        self,
        backend: Backend,
        device: str = "cuda",
        batch_size: int = 8,
        **backend_config
    ):
        pass

    @abstractmethod
    def process(self, document: Document, **kwargs) -> Result:
        """Process single document"""
        pass

    def process_batch(
        self,
        documents: List[Document],
        batch_size: int = None,
        **kwargs
    ) -> List[Result]:
        """Batch process documents"""
        pass

class BaseVLM(BaseModel):
    """Base class for Vision-Language Models"""

    def process(
        self,
        document: Document,
        task: Task,
        **kwargs
    ) -> Result:
        """Process for specific task"""
        pass

    def process_all(
        self,
        document: Document,
        tasks: List[Task],
        **kwargs
    ) -> MultiTaskResult:
        """Process all tasks efficiently"""
        pass
```

**2.3 Model Implementations**

Each model wrapper:
1. Inherits from `BaseModel` or `BaseVLM`
2. Wraps existing task extractors
3. Handles backend selection
4. Provides unified `.process()` interface
5. Supports batch processing

**2.4 Task Enum**
```python
# omnidocs/tasks/__init__.py
from enum import Enum

class Task(str, Enum):
    LAYOUT = "layout"
    OCR = "ocr"
    TEXT = "text"
    TABLE = "table"
    MATH = "math"
```

**2.5 Integration with Existing Extractors**

Update base extractors to accept `Document` objects:
```python
# In tasks/*/base.py
def extract(
    self,
    input: Union[str, Path, Document],
    **kwargs
) -> Output:
    if isinstance(input, Document):
        return self._extract_from_document(input, **kwargs)
    else:
        # Legacy path-based extraction
        ...
```

#### Success Criteria
- [ ] All model wrappers implemented
- [ ] BaseModel/BaseVLM abstractions complete
- [ ] Task enum defined
- [ ] Backwards compatible with existing extractors
- [ ] Examples for each model type

---

### Phase 3: Backend Abstraction
**Estimated Effort:** 2-3 days | **Priority:** 🟠 High

#### Deliverables

**3.1 Module Structure**
```
omnidocs/
├── backends/
│   ├── __init__.py
│   ├── base.py           # Backend enum, base config
│   ├── vllm.py           # VLLM backend
│   ├── transformers.py   # HuggingFace backend
│   ├── mlx.py            # Apple MLX backend
│   ├── api.py            # API client backend
│   └── hosted.py         # Omnidocs Cloud backend
```

**3.2 Backend Enum & Configs**

```python
from enum import Enum
from pydantic import BaseModel

class Backend(str, Enum):
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    MLX = "mlx"
    API = "api"
    HOSTED = "hosted"

class VLLMConfig(BaseModel):
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    trust_remote_code: bool = True

class TransformersConfig(BaseModel):
    device_map: str = "auto"
    torch_dtype: str = "float16"
    use_flash_attention: bool = False

class MLXConfig(BaseModel):
    quantize: bool = False
    max_tokens: int = 4096

class APIConfig(BaseModel):
    api_url: str
    api_key: str
    timeout: int = 300
    max_retries: int = 3
```

**3.3 Backend Implementations**

Each backend provides:
1. Model loading strategy
2. Inference execution
3. Resource management
4. Error handling

**3.4 Model Integration**

Update model wrappers to use backends:
```python
model = SuryaOCR(
    backend=Backend.VLLM,
    config=VLLMConfig(tensor_parallel_size=2)
)
```

#### Success Criteria
- [ ] Backend enum and configs defined
- [ ] VLLM backend implemented
- [ ] Transformers backend implemented
- [ ] MLX backend implemented (macOS)
- [ ] API client backend functional
- [ ] Backend switching works seamlessly

---

### Phase 4: Pipeline Builder
**Estimated Effort:** 4-5 days | **Priority:** 🟠 High

#### Deliverables

**4.1 Module Structure**
```
omnidocs/
├── pipeline/
│   ├── __init__.py
│   ├── builder.py        # Pipeline class
│   ├── step.py           # Step builders
│   ├── condition.py      # Conditional logic
│   ├── transform.py      # Transformations
│   └── result.py         # PipelineResult
```

**4.2 Pipeline Class**

```python
class Pipeline:
    """Multi-step processing pipeline"""

    def __init__(
        self,
        steps: List[Step],
        name: str = None,
        error_handling: str = "stop"  # "stop", "continue", "retry"
    ):
        pass

    def run(self, document: Document) -> PipelineResult:
        """Run pipeline on single document"""
        pass

    def run_batch(
        self,
        documents: List[Document],
        batch_size: int = 8,
        num_workers: int = 4,
        show_progress: bool = True
    ) -> List[PipelineResult]:
        """Run pipeline on multiple documents"""
        pass

    def save(self, path: str) -> None:
        """Save pipeline configuration"""
        pass

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load pipeline from file"""
        pass
```

**4.3 Step Builders**

```python
class Step:
    """Pipeline step builders"""

    @staticmethod
    def layout(model: BaseModel, **kwargs) -> "Step":
        """Layout detection step"""
        pass

    @staticmethod
    def ocr(model: BaseModel, **kwargs) -> "Step":
        """OCR step"""
        pass

    @staticmethod
    def tables(model: BaseModel, **kwargs) -> "Step":
        """Table extraction step"""
        pass

    @staticmethod
    def math(model: BaseModel, **kwargs) -> "Step":
        """Math extraction step"""
        pass

    @staticmethod
    def custom(func: Callable) -> "Step":
        """Custom processing step"""
        pass

    @staticmethod
    def transform(transform: Transform) -> "Step":
        """Transformation step"""
        pass

    @staticmethod
    def when(
        condition: Condition,
        then_step: Step,
        else_step: Step = None
    ) -> "Step":
        """Conditional step"""
        pass

    @staticmethod
    def parallel(*steps: Step) -> "Step":
        """Run steps in parallel"""
        pass
```

**4.4 Conditional Logic**

```python
class Condition:
    """Conditional execution logic"""

    @staticmethod
    def has_layout_type(layout_type: str) -> "Condition":
        """Check if layout contains specific type"""
        pass

    @staticmethod
    def avg_confidence_below(threshold: float) -> "Condition":
        """Check if average confidence is below threshold"""
        pass

    @staticmethod
    def page_count_above(count: int) -> "Condition":
        """Check if page count exceeds threshold"""
        pass

    @staticmethod
    def custom(func: Callable) -> "Condition":
        """Custom condition function"""
        pass
```

**4.5 Transformations**

```python
class Transform(ABC):
    """Base transformation class"""

    @abstractmethod
    def apply(self, result: Any) -> Any:
        """Apply transformation to result"""
        pass

# Built-in transformations
class MergeTextBlocks(Transform):
    def __init__(self, max_distance: int = 10):
        self.max_distance = max_distance

class FilterLowConfidence(Transform):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
```

**4.6 Pipeline Result**

```python
class PipelineResult(BaseModel):
    """Result from pipeline execution"""

    steps_completed: List[str]
    layout: Optional[LayoutResult]
    ocr: Optional[OCRResult]
    tables: Optional[TableResult]
    math: Optional[MathResult]
    custom_data: Optional[dict]
    total_time: float

    def save_json(self, path: str) -> None:
        pass

    def save_markdown(self, path: str) -> None:
        pass
```

**4.7 Serialization**

```python
# Save pipeline to YAML
pipeline.save("invoice_pipeline.yaml")

# Load pipeline
pipeline = Pipeline.load("invoice_pipeline.yaml")
```

#### Success Criteria
- [ ] Pipeline class with step composition
- [ ] All step builders implemented
- [ ] Conditional execution working
- [ ] Parallel execution working
- [ ] Custom steps supported
- [ ] Transformations working
- [ ] Serialization/deserialization complete
- [ ] Batch processing with progress bars
- [ ] Error handling strategies implemented

---

### Phase 5: VLM Support & Multi-Task
**Estimated Effort:** 2-3 days | **Priority:** 🟡 Medium

#### Deliverables

**5.1 VLM Model Wrappers**
- `QwenVL2B` / `QwenVL32B`
- `Chandra`
- `DotsOCR`
- `Gemma`

**5.2 Multi-Task Processing**

```python
# Single task
vlm = QwenVL32B(backend=Backend.VLLM)
result = vlm.process(doc, task=Task.OCR)

# Multi-task (efficient - one pass)
result = vlm.process_all(
    doc,
    tasks=[Task.LAYOUT, Task.OCR, Task.TABLE]
)

print(result.layout.boxes)
print(result.ocr.full_text)
print(result.tables)
```

**5.3 MultiTaskResult**

```python
class MultiTaskResult(BaseModel):
    layout: Optional[LayoutResult]
    ocr: Optional[OCRResult]
    tables: Optional[TableResult]
    math: Optional[MathResult]

    def save_json(self, path: str) -> None:
        pass

    def save_markdown(self, path: str) -> None:
        pass
```

**5.4 Batch Optimizations**
- Efficient batching for VLMs
- Shared context across tasks
- Memory management for large batches

#### Success Criteria
- [ ] VLM wrappers implemented
- [ ] Multi-task processing working
- [ ] MultiTaskResult composition
- [ ] Batch processing optimized
- [ ] Examples for all VLMs

---

### Phase 6: Advanced Features
**Estimated Effort:** 3-4 days | **Priority:** 🟢 Enhancement

#### Deliverables

**6.1 Configuration System**

```python
# omnidocs/config.py
from omnidocs import config

# Set global defaults
config.default_backend = "vllm"
config.default_device = "cuda"
config.model_cache_dir = "/data/models"
config.batch_size = 16
config.log_level = "INFO"
```

**Environment Variables:**
```bash
export OMNIDOCS_MODEL_CACHE=/data/models
export OMNIDOCS_DEVICE=cuda
export OMNIDOCS_BACKEND=vllm
export OMNIDOCS_API_KEY=your-key
export OMNIDOCS_LOG_LEVEL=DEBUG
```

**6.2 Advanced Error Handling**
- Retry strategies with exponential backoff
- Graceful degradation
- Detailed error reporting
- Recovery mechanisms

**6.3 Performance Monitoring**
- Execution time tracking
- Memory usage monitoring
- Throughput metrics
- Bottleneck detection

**6.4 Caching Layer**
- Document cache (avoid re-loading)
- Model cache (singleton patterns)
- Result cache (memoization)

**6.5 Telemetry (Optional)**
- Usage analytics
- Error tracking
- Performance metrics
- Model usage statistics

#### Success Criteria
- [ ] Configuration system complete
- [ ] Environment variable support
- [ ] Advanced error handling
- [ ] Performance monitoring
- [ ] Caching layer functional

---

## Dependencies & Prerequisites

### Development Dependencies

**Phase 1: Document Abstraction**
- `requests>=2.26.0` (URL downloads)
- `PyMuPDF>=1.26.3` (already installed)
- `pdfplumber>=0.11.7` (already installed)
- `Pillow>=10.4.0` (already installed)
- `pydantic>=2.8` (already installed)

**Phase 3: Backend Abstraction**
- `vllm>=0.3.0` (for VLLM backend)
- `mlx>=0.5.0` (for Apple Silicon support)
- `httpx>=0.24.0` (for async API client)

**Phase 4: Pipeline Builder**
- `pyyaml>=6.0` (pipeline serialization)
- `tqdm>=4.65.0` (progress bars - may already be installed)
- `joblib>=1.3.0` (parallel processing)

### External Prerequisites

**For VLM Support (Phase 5):**
- GPU with sufficient VRAM (12GB+ recommended for 2B models, 24GB+ for 32B)
- CUDA 11.8+ or ROCm support
- vLLM installation and configuration

**For MLX Backend (Phase 3):**
- Apple Silicon Mac (M1/M2/M3)
- macOS 13.3+

---

## Success Metrics

### Phase Completion Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | Document class test coverage | >80% |
| Phase 1 | All constructors functional | 100% |
| Phase 2 | Model wrappers implemented | 100% existing extractors |
| Phase 2 | Backwards compatibility | No breaking changes |
| Phase 3 | Backend implementations | 5 backends (VLLM, Transformers, MLX, API, Hosted) |
| Phase 4 | Pipeline features | All step types, conditionals, parallel |
| Phase 4 | Serialization tests | Save/load roundtrip success |
| Phase 5 | VLM support | 4+ VLM models wrapped |
| Phase 6 | Configuration coverage | All settings configurable |

### Quality Metrics

- **Test Coverage:** >80% for all new modules
- **Documentation:** 100% public API documented
- **Examples:** 3+ examples per major feature
- **Performance:** No regression vs current extractors
- **API Stability:** Semantic versioning, deprecation warnings

### User Experience Metrics

- **Ease of Use:** Can complete common tasks in <10 lines of code
- **Flexibility:** Support both simple and complex workflows
- **Performance:** Batch processing >10x faster than sequential
- **Reliability:** Graceful error handling, no silent failures

---

## Timeline Estimate

### Conservative Estimate (Sequential Development)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Document Abstraction | 2-3 days | None |
| Phase 2: Model Wrapper Layer | 3-4 days | Phase 1 |
| Phase 3: Backend Abstraction | 2-3 days | Phase 2 |
| Phase 4: Pipeline Builder | 4-5 days | Phases 1-3 |
| Phase 5: VLM Support | 2-3 days | Phases 1-4 |
| Phase 6: Advanced Features | 3-4 days | Phases 1-5 |

**Total Sequential:** 16-22 days

### Optimistic Estimate (Parallel Development)

With 2-3 developers working in parallel:

| Week | Focus Areas |
|------|-------------|
| Week 1 | Phase 1 (Document) + Phase 3 (Backend design) |
| Week 2 | Phase 2 (Model Wrappers) + Phase 3 (Backend impl) |
| Week 3 | Phase 4 (Pipeline Builder) |
| Week 4 | Phase 5 (VLM) + Phase 6 (Advanced Features) |

**Total Parallel:** 4 weeks

---

## Migration Path

### For Existing Users

**Current Code:**
```python
from omnidocs.tasks.ocr_extraction.extractors.surya import SuryaOCRExtractor

extractor = SuryaOCRExtractor()
result = extractor.extract("document.pdf")
```

**After Phase 1-2:**
```python
from omnidocs import Document
from omnidocs.models import SuryaOCR

doc = Document.from_pdf("document.pdf")
model = SuryaOCR()
result = model.process(doc)
```

**After Phase 4:**
```python
from omnidocs import Document, Pipeline, Step
from omnidocs.models import SuryaOCR

pipeline = Pipeline([
    Step.ocr(model=SuryaOCR())
])

doc = Document.from_pdf("document.pdf")
result = pipeline.run(doc)
```

### Backwards Compatibility

- **Phases 1-3:** No breaking changes, additive only
- **Phase 4+:** Legacy extractors remain functional
- **Deprecation:** Announce 1 version ahead, remove after 2 versions

---

## Open Questions & Decisions Needed

1. **VLM Model Hosting:**
   - Self-hosted vLLM setup vs cloud API?
   - Which VLM models to prioritize?

2. **Pipeline Serialization Format:**
   - YAML vs JSON for pipeline storage?
   - Include model weights or just configs?

3. **API Backend:**
   - OpenAI-compatible API format?
   - Custom protocol for better performance?

4. **Performance Targets:**
   - What's acceptable latency for single doc?
   - Throughput targets for batch processing?

5. **Hosted Backend:**
   - Build Omnidocs Cloud service?
   - Pricing model?

---

## CI/CD & Automation

### GitHub Actions Workflows

Omnidocs uses comprehensive GitHub Actions for continuous integration, testing, and deployment.

#### 1. **Test Workflow** (`.github/workflows/test.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual dispatch

**Jobs:**
- **Unit Tests:** Python 3.9, 3.10, 3.11 on Ubuntu, macOS, Windows
- **GPU Tests:** CUDA-enabled tests on push to main
- **Integration Tests:** End-to-end workflow tests
- **Coverage:** Upload to Codecov (>80% target)

**Matrix Strategy:**
```yaml
os: [ubuntu-latest, macos-latest, windows-latest]
python-version: ["3.9", "3.10", "3.11"]
```

#### 2. **Lint & Code Quality** (`.github/workflows/lint.yml`)

**Triggers:**
- Push to `main` or `develop`
- Pull requests
- Manual dispatch

**Jobs:**
- **Ruff:** Linting and formatting checks
- **mypy:** Strict type checking
- **Bandit:** Security vulnerability scanning
- **Safety:** Dependency vulnerability checks
- **Radon:** Code complexity analysis

**Quality Gates:**
- All linting must pass
- Type checking must pass (strict mode)
- No high-severity security issues

#### 3. **Documentation** (`.github/workflows/docs.yml` & `deploy-mkdocs.yml`)

**Triggers:**
- Push to `main` (for deployment)
- Pull requests (for validation)
- Changes to `docs/**`, `mkdocs.yml`
- Manual dispatch

**Jobs:**
- **Build Docs:** MkDocs Material build with strict mode
- **Deploy:** Deploy to GitHub Pages on main branch
- **Link Validation:** Check for broken links

**Features:**
- Auto-generated API docs with mkdocstrings
- Git revision dates
- Jupyter notebook support
- Material theme with imaging

#### 4. **Release & Publish** (`.github/workflows/release.yml`)

**Triggers:**
- GitHub release published
- Push to version tags (`v*`)
- Manual dispatch (for testing)

**Jobs:**
- **Build:** Create wheel and source distributions
- **Test Install:** Verify installation on multiple platforms
- **Publish TestPyPI:** For testing (manual workflow only)
- **Publish PyPI:** Production release
- **Create Release:** Generate GitHub release with changelog

**Release Process:**
```bash
# 1. Tag version
git tag v1.0.0
git push origin v1.0.0

# 2. Workflow automatically:
#    - Builds package
#    - Tests installation
#    - Publishes to PyPI
#    - Creates GitHub release
```

#### 5. **Pre-commit Checks** (`.github/workflows/pre-commit.yml`)

**Triggers:**
- Pull requests
- Push to main/develop

**Jobs:**
- Run pre-commit hooks on all files
- Auto-fix formatting issues
- Commit fixes automatically (if needed)

**Hooks:**
- Code formatting (ruff)
- Import sorting
- Trailing whitespace
- YAML/JSON validation

---

### Workflow Dependencies

```
┌─────────────┐
│   PR/Push   │
└──────┬──────┘
       │
       ├──────────────┬──────────────┬─────────────┐
       │              │              │             │
       ▼              ▼              ▼             ▼
┌──────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
│   Test   │   │   Lint   │  │Pre-commit│  │  Docs    │
└──────────┘   └──────────┘  └──────────┘  └──────────┘
       │              │              │             │
       └──────────────┴──────────────┴─────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ All Checks OK │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  Ready to     │
              │  Merge/Deploy │
              └───────────────┘

On Release:
┌─────────────┐
│  Tag v1.0.0 │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│Build Package │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Test Install  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Publish PyPI  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Create Release│
└──────────────┘
```

---

### Local Development Setup

**Install pre-commit hooks:**
```bash
pip install pre-commit
pre-commit install
```

**Run checks locally:**
```bash
# Run all tests
uv run pytest tests/ -v

# Run linting
ruff check omnidocs/
ruff format omnidocs/

# Type checking
mypy omnidocs/ --strict

# Build docs locally
mkdocs serve
```

**Test installation:**
```bash
# Build package
python -m build

# Install locally
pip install dist/*.whl
```

---

### Continuous Deployment Strategy

**Branch Protection:**
- `main` branch requires:
  - All status checks passing
  - At least 1 approval
  - Up-to-date with base branch

**Release Strategy:**
1. **Development:** Work on feature branches
2. **Integration:** Merge to `develop` branch for testing
3. **Staging:** Tag release candidates (`v1.0.0-rc1`)
4. **Production:** Tag stable releases (`v1.0.0`)

**Version Numbering:**
- Follow Semantic Versioning (SemVer)
- `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)
- Pre-releases: `1.0.0-rc1`, `1.0.0-alpha1`

---

## Contributing

This roadmap is a living document. Contributions and feedback are welcome:

1. **Feature Requests:** Open an issue with `[Feature Request]` tag
2. **Phase Feedback:** Comment on specific phase implementations
3. **Timeline Adjustments:** Suggest optimizations or dependencies
4. **Pull Requests:** Follow the phase order for coherent development

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2026-01-03 | Added comprehensive CI/CD & GitHub Actions documentation |
| 1.0 | 2026-01-02 | Initial roadmap based on FINAL_DESIGN.md analysis |

---

## References

- [FINAL_DESIGN.md](../Design/FINAL_DESIGN.md) - Target architecture specification
- [Current Extractors](omnidocs/tasks/) - Existing implementations
- [PyProject](pyproject.toml) - Current dependencies
- [GitHub Actions](.github/workflows/) - CI/CD workflows

---

**Last Updated:** 2026-01-03
**Status:** Phase 0 Complete, CI/CD Infrastructure Ready, Phase 1 Ready to Start
