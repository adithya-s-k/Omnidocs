# Installation

OmniDocs can be installed in several ways depending on your environment and preferences. Choose the method that works best for you.

## 1. Install from PyPI (Recommended)

```bash
pip install omnidocs
```


## 2. Install with uv (fast, modern Python package manager)

If you have [uv](https://github.com/astral-sh/uv) installed 
(`pip install uv`) you can:

```bash
uv pip install omnidocs
```

Or, to create a new virtual environment and install:

```bash
uv venv .venv
uv pip install omnidocs
```

Or, to sync all dependencies from a lock file (if provided):

```bash
uv sync
```

## 3. Install from Source (Latest Development Version)

```bash
git clone https://github.com/adithya-s-k/OmniDocs.git
cd OmniDocs
# Option 1: pip install (classic)
pip install .
# Option 2: uv sync (fast, reproducible)
uv venv .venv
uv sync
```

## 4. (Optional) Create a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
---

For more details, see the [GitHub repository](https://github.com/adithya-s-k/OmniDocs) or the [Getting Started guide](getting_started.md).
