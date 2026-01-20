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

OmniDocs follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

### Incrementing Version

Version is managed in **one place**: `omnidocs/_version.py`

```python
# omnidocs/_version.py
__version__ = "0.2.0"
```

The `pyproject.toml` reads from this file dynamically, so you only need to update one file.

### Version Bump Process

1. **Update version**:
```bash
# Edit omnidocs/_version.py
# Change __version__ = "0.2.0" to __version__ = "0.2.1"
```

2. **Verify** (version should be accessible everywhere):
```python
from omnidocs import __version__
print(__version__)  # Should show new version
```

3. **Commit**:
```bash
git add omnidocs/_version.py
git commit -m "chore: bump version to 0.2.1"
```

4. **Tag and release** (maintainers only):
```bash
git tag v0.2.1
git push origin v0.2.1
```

This will trigger the GitHub Actions workflow to:
- Build the package
- Create a GitHub release
- Publish to PyPI

### When to Bump

- **Patch** (0.2.X): Bug fixes, typos, documentation
- **Minor** (0.X.0): New features, new task extractors, new models
- **Major** (X.0.0): Breaking API changes (rare, requires discussion)

## Documentation

OmniDocs uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) for documentation.

### Building Docs Locally

```bash
# Install docs dependencies
uv sync --group docs

# Serve docs with live reload
uv run mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Building Static Site

```bash
# Build static site to site/ directory
uv run mkdocs build

# Build with strict mode (warnings become errors)
uv run mkdocs build --strict
```

### Documentation Structure

```
docs/
├── README.md              # Homepage (also serves as package README)
├── architecture.md        # Backend and config system design
└── developer-guide.md     # API design and usage patterns
```

### Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when:
- Code is pushed to the `master` branch
- A maintainer manually triggers the workflow

The docs are published at: https://adithya-s-k.github.io/OmniDocs/

### Adding New Documentation

1. Create markdown files in `docs/`
2. Update `mkdocs.yml` nav section to include new pages
3. Use Google-style docstrings in code (automatically extracted)
4. Preview changes locally with `uv run mkdocs serve`

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
