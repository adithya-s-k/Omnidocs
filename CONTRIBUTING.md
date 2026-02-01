# Contributing to OmniDocs

Thank you for your interest in contributing to OmniDocs! 🎉

## Development Setup

1. **Clone the repository**:
```bash
git clone https://github.com/adithya-s-k/OmniDocs.git
cd OmniDocs/Omnidocs
```

2. **Install dependencies with uv**:
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

3. **Run tests**:
```bash
uv run pytest tests/ -v
```

## Project Structure

```
Omnidocs/
├── omnidocs/          # Main package
│   ├── document.py    # Document loading (✅ complete)
│   ├── tasks/         # Task extractors (🚧 in progress)
│   ├── inference/     # Backend implementations (planned)
│   └── utils/         # Utilities
├── tests/             # Test suite
│   ├── fixtures/      # Test data (PDFs, images)
│   └── tasks/         # Future task tests
└── docs/              # Documentation
```

## Design Documents

**🔴 IMPORTANT**: Before implementing any new features, read the design documents:
- `docs/architecture.md` - Backend and config system
- `docs/developer-guide.md` - API design and usage patterns

These documents define the architecture for v0.2+.

## Development Workflow

### 1. Testing Phase (modal_scripts/)
- Test models in isolation using Modal scripts
- Validate inference and outputs
- Benchmark performance

### 2. Integration Phase (omnidocs/)
- Follow the config pattern (single-backend vs multi-backend)
- Use Pydantic for all configs and outputs
- Maintain consistent `.extract()` API
- Add comprehensive tests

### 3. Documentation
- Add docstrings (Google style)
- Update relevant docs
- Add usage examples

## Code Standards

### ✅ Required
- Type hints for all public APIs
- Pydantic models for configs (`extra="forbid"`)
- Docstrings (Google style) for classes and methods
- Tests with >80% coverage

### ❌ Avoid
- String-based factories (use class imports)
- Storing task results in Document
- Breaking changes without deprecation
- Adding features beyond requirements
- Over-engineering

## Version Management

OmniDocs follows [Semantic Versioning](https://semver.org/) with **automated releases**.

### Semantic Versioning Guide

| Version Bump | When to Use | Examples |
|--------------|-------------|----------|
| **PATCH** (0.0.X) | Bug fixes, no API changes | Fix typo in docstring, fix edge case bug, update dependencies |
| **MINOR** (0.X.0) | New features, backward compatible | Add new extractor (DotsOCR), add new backend (MLX), add new output format |
| **MAJOR** (X.0.0) | Breaking API changes | Rename `.extract()` method, change config class structure, remove deprecated features |

#### Detailed Examples

**PATCH bump** (0.2.4 → 0.2.5):
- Fix a bug in `DotsOCRTextExtractor` that caused crashes on certain images
- Fix incorrect bounding box coordinates
- Update a dependency version for security
- Fix typos in documentation

**MINOR bump** (0.2.5 → 0.3.0):
- Add new `DotsOCRTextExtractor` with PyTorch/VLLM/API backends
- Add new `TableExtractor` task
- Add MLX backend support for existing extractors
- Add new output format (e.g., JSON)
- Add new configuration options (backward compatible)

**MAJOR bump** (0.3.0 → 1.0.0):
- Rename `BaseTextExtractor` to `TextExtractor`
- Change `config=` parameter to `backend=` across all extractors
- Remove deprecated `Document.extract_text()` method
- Change output model structure (breaking existing code)

### Automated Release Workflow

**No manual git tags needed!** Releases are automatically triggered when you bump the version.

#### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. Bump version in pyproject.toml                          │
│     version = "0.2.4" → version = "0.2.5"                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Commit and push to main                                 │
│     git commit -m "chore: bump version to 0.2.5"            │
│     git push                                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. CI/CD automatically:                                    │
│     • Detects version change in pyproject.toml              │
│     • Creates git tag (v0.2.5)                              │
│     • Builds and publishes to PyPI                          │
│     • Deploys versioned documentation                       │
└─────────────────────────────────────────────────────────────┘
```

#### Version Bump Process

1. **Update version in `pyproject.toml`**:
```toml
[project]
name = "omnidocs"
version = "0.2.5"  # Bump this
```

2. **Commit and push**:
```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.2.5"
git push origin main
```

3. **Done!** The CI/CD pipeline will automatically:
   - Detect the version change
   - Create git tag `v0.2.5`
   - Build and publish to PyPI
   - Deploy versioned docs at `/0.2.5/` and update `/latest/`

#### Verify Release

```bash
# Check PyPI (after a few minutes)
pip index versions omnidocs

# Check GitHub releases
gh release list

# Check docs versions
# Visit: https://adithya-s-k.github.io/OmniDocs/
```

### Version in Code

Version is defined in `pyproject.toml` and accessible in code:

```python
from omnidocs import __version__
print(__version__)  # "0.2.5"
```

## Documentation

OmniDocs uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) with **versioned documentation** and **auto-generated API reference**.

### Building Docs Locally

```bash
# Install dev dependencies (includes docs)
uv sync --group dev

# Serve docs with live reload
uv run mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Versioned Documentation

We use [mike](https://github.com/jimporter/mike) for multi-version documentation.

```bash
# Deploy a version locally (for testing)
uv run mike deploy 0.2.5 latest --update-aliases

# List deployed versions
uv run mike list

# Serve versioned docs locally
uv run mike serve
```

### Auto-Generated API Reference

API documentation is **automatically generated** from source code docstrings using `mkdocstrings`.

The `scripts/gen_ref_pages.py` script:
- Auto-discovers all modules in `omnidocs/`
- Creates clean navigation hierarchy
- Converts `snake_case` to "Title Case"
- Future-proof: just add modules, docs auto-generate

**No manual API docs needed!** Just write good docstrings:

```python
class MyExtractor(BaseExtractor):
    """
    Brief description of the extractor.

    Detailed description with usage information.

    Example:
        >>> from omnidocs import MyExtractor
        >>> extractor = MyExtractor(config=MyConfig())
        >>> result = extractor.extract(image)

    Attributes:
        config: Configuration object for the extractor.
    """
```

### Documentation Structure

```
docs/
├── README.md              # Homepage
├── architecture.md        # Backend and config system design
├── developer-guide.md     # API design and usage patterns
└── reference/             # Auto-generated API docs (don't edit!)
```

### Automatic Deployment

Documentation is automatically deployed when a new version is released:

| Trigger | Action |
|---------|--------|
| Version bump in `pyproject.toml` | Deploys versioned docs (`/0.2.5/`, `/latest/`) |
| Manual workflow dispatch | Deploy specific version |

The docs are published at: https://adithya-s-k.github.io/OmniDocs/

Users can switch between versions using the version selector in the docs header.

## Pull Request Process

1. **Create a branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes**:
- Follow the design patterns in docs/
- Add tests for new functionality
- Update relevant documentation

3. **Run tests**:
```bash
# Run all tests
uv run pytest tests/ -v

# Run fast tests only
uv run pytest tests/ -v -m "not slow"
```

4. **Submit PR**:
- Provide clear description
- Reference any related issues
- Ensure tests pass

## Commit Guidelines

Follow conventional commits:
```
feat: add DocLayoutYOLO extractor
fix: resolve page range validation
docs: update architecture guide
test: add fixture-based PDF tests
```

## Reference: End-to-End Contribution Example

For a complete example of how to contribute a new feature to OmniDocs, see:

- **Issue**: [#18 - Layout Extraction Module](https://github.com/adithya-s-k/Omnidocs/issues/18)
- **Pull Request**: [#19 - feat: Add layout extraction module](https://github.com/adithya-s-k/Omnidocs/pull/19)

This contribution demonstrates:
1. Creating a feature request issue with proper description
2. Implementing a new task module (`layout_extraction`)
3. Following the config pattern with Pydantic models
4. Adding comprehensive tests (71 tests)
5. Creating an end-to-end example script
6. Proper commit messages and PR description
7. Version bump workflow

## Need Help?

- 📖 Read the [design documents](docs/)
- 🐛 [Open an issue](https://github.com/adithya-s-k/OmniDocs/issues)
- 💬 Ask questions in discussions

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
