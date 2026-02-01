"""
Smart auto-generating API reference with clean hierarchy.

Auto-discovers package structure and creates organized documentation:
- Converts snake_case to Title Case
- Groups by directory structure
- __init__.py becomes "Overview" for each section
- Handles any depth of nesting
- Future-proof: just add modules, docs auto-generate

Example output structure:
  API Reference/
  ├── Core/
  │   └── Document
  ├── Tasks/
  │   ├── Layout Analysis/
  │   │   ├── Overview
  │   │   ├── Doc Layout Yolo
  │   │   └── Qwen/
  │   │       ├── Overview
  │   │       └── Pytorch
  │   └── Text Extraction/
  │       ├── Overview
  │       └── Dotsocr/
  │           ├── Overview
  │           ├── Extractor
  │           └── Pytorch
  └── Inference/
      ├── Overview
      ├── Pytorch
      └── Vllm
"""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Config
ROOT = Path(__file__).parent.parent
SRC = ROOT / "omnidocs"
PACKAGE_NAME = "omnidocs"

# Smart naming overrides (optional - for special cases)
NAME_OVERRIDES = {
    "ocr": "OCR",
    "vllm": "VLLM",
    "mlx": "MLX",
    "api": "API",
    "pytorch": "PyTorch",
    "dotsocr": "Dots OCR",
    "easyocr": "EasyOCR",
    "paddleocr": "PaddleOCR",
    "yolo": "YOLO",
}


def smart_title(name: str) -> str:
    """
    Convert snake_case to Smart Title Case.

    Examples:
        text_extraction -> Text Extraction
        doc_layout_yolo -> Doc Layout YOLO
        dotsocr -> Dots OCR
    """
    # Check overrides first (exact match)
    if name.lower() in NAME_OVERRIDES:
        return NAME_OVERRIDES[name.lower()]

    # Split by underscore
    parts = name.split("_")

    # Apply overrides to each part, or title case
    result = []
    for part in parts:
        if part.lower() in NAME_OVERRIDES:
            result.append(NAME_OVERRIDES[part.lower()])
        else:
            result.append(part.title())

    return " ".join(result)


def get_module_identifier(path: Path) -> str:
    """Get Python module identifier from file path."""
    rel_path = path.relative_to(ROOT)
    parts = list(rel_path.with_suffix("").parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts) if parts else ""


def get_nav_hierarchy(path: Path) -> tuple:
    """
    Generate navigation hierarchy from file path.

    omnidocs/tasks/text_extraction/dotsocr/pytorch.py
    -> ("Tasks", "Text Extraction", "Dots OCR", "PyTorch")

    omnidocs/tasks/text_extraction/__init__.py
    -> ("Tasks", "Text Extraction", "Overview")
    """
    rel_path = path.relative_to(SRC)
    parts = list(rel_path.with_suffix("").parts)

    # Handle __init__.py -> Overview
    is_init = parts[-1] == "__init__"
    if is_init:
        parts = parts[:-1]

    # Convert to titles
    nav_parts = [smart_title(p) for p in parts]

    # Add "Overview" for __init__ files (package index)
    if is_init and nav_parts:
        nav_parts.append("Overview")
    elif not nav_parts:
        # Root __init__ -> just "Overview"
        nav_parts = ["Overview"]

    return tuple(nav_parts)


def get_doc_path(nav_hierarchy: tuple) -> Path:
    """Generate documentation file path from nav hierarchy."""
    parts = [p.lower().replace(" ", "_") for p in nav_hierarchy]
    filename = parts[-1] + ".md"

    if len(parts) > 1:
        return Path("reference") / "/".join(parts[:-1]) / filename
    return Path("reference") / filename


def should_include(path: Path) -> bool:
    """Filter out files that shouldn't be documented."""
    # Skip __pycache__
    if "__pycache__" in path.parts:
        return False

    # Skip private modules (but keep __init__)
    if path.stem.startswith("_") and path.stem != "__init__":
        return False

    # Skip empty __init__ files (optional)
    # if path.stem == "__init__" and path.stat().st_size < 50:
    #     return False

    return True


# Collect all modules
modules = []
for path in sorted(SRC.rglob("*.py")):
    if should_include(path):
        modules.append(path)

# Generate docs for each module
for path in modules:
    module_id = get_module_identifier(path)
    nav_hierarchy = get_nav_hierarchy(path)
    doc_path = get_doc_path(nav_hierarchy)

    # Skip root package (we'll create a custom index)
    if not module_id or module_id == PACKAGE_NAME:
        continue

    # Register in navigation
    nav[nav_hierarchy] = doc_path.as_posix().replace("reference/", "")

    # Generate the markdown file
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        title = nav_hierarchy[-1]
        fd.write(f"# {title}\n\n")
        fd.write(f"::: {module_id}\n")
        fd.write("    options:\n")
        fd.write("      show_root_heading: false\n")
        fd.write("      show_source: true\n")
        fd.write("      members_order: source\n")

    mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(ROOT))


# Create main index page
with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.write("""# API Reference

Auto-generated documentation for the OmniDocs package.

## Package Structure

```
omnidocs/
├── document.py          # Core document handling
├── tasks/               # Document processing tasks
│   ├── layout_analysis/ # Detect document structure
│   ├── text_extraction/ # Extract text (Markdown/HTML)
│   └── ocr_extraction/  # Extract text with bboxes
├── inference/           # Backend implementations
│   ├── pytorch.py       # PyTorch/HuggingFace
│   ├── vllm.py          # High-throughput VLLM
│   ├── mlx.py           # Apple Silicon
│   └── api.py           # API-based inference
└── utils/               # Utility functions
```

## Quick Start

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

# Load document
doc = Document.from_pdf("document.pdf")

# Initialize extractor
extractor = DotsOCRTextExtractor(
    backend=DotsOCRPyTorchConfig(model="rednote-hilab/dots.ocr")
)

# Extract text
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

Browse the sections in the sidebar to explore the full API.
""")


# Generate navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
