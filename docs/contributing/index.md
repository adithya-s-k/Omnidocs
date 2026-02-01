# Contributing to OmniDocs

Thank you for your interest in contributing to OmniDocs!

## Quick Links

- [Development Workflow](workflow.md) - 6-phase implementation process
- [Adding Models](adding-models.md) - How to add new model support
- [Style Guide](style-guide.md) - Code standards and conventions

---

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/adithya-s-k/OmniDocs.git
cd OmniDocs/Omnidocs
```

### 2. Install dependencies with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 3. Run tests

```bash
uv run pytest tests/ -v
```

---

## Project Structure

```
Omnidocs/
├── omnidocs/          # Main package
│   ├── document.py    # Document loading
│   ├── tasks/         # Task extractors
│   ├── inference/     # Backend implementations
│   └── utils/         # Utilities
├── tests/             # Test suite
│   ├── fixtures/      # Test data (PDFs, images)
│   └── tasks/         # Task tests
└── docs/              # Documentation
```

---

## Design Documents

!!! warning "Read Before Implementing"
    Before implementing new features, read the architecture docs:

    - [Architecture Overview](../concepts/architecture-overview.md) - System design
    - [Backend System](../concepts/backend-system.md) - Multi-backend support
    - [Config Pattern](../concepts/config-pattern.md) - Configuration design

---

## Building Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Serve docs with live reload
uv run mkdocs serve

# Open http://127.0.0.1:8000
```

---

## Need Help?

- [Open an issue](https://github.com/adithya-s-k/OmniDocs/issues)
- Check the [Roadmap](../ROADMAP.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
