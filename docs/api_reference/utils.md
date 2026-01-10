# Utilities

This page documents utility modules in the `omnidocs.utils` package used across the library.

## Overview

`omnidocs.utils` provides small helper modules for language handling, logging configuration, and model configuration management. These utilities are lightweight and intended to be used by other components and extractors.

## Modules

### `omnidocs.utils.language`

- Purpose: language detection and normalization utilities.
- Key functions:
  - `detect_language(text: str) -> str`: Return a language code for `text`.
  - `normalize_language_code(code: str) -> str`: Normalize variants to a canonical code.

Usage example

```python
from omnidocs.utils.language import detect_language

lang = detect_language("This is a test")
print(lang)
```

### `omnidocs.utils.logging`

- Purpose: small helpers to configure and retrieve library loggers.
- Key functions:
  - `get_logger(name: str) -> logging.Logger`: Returns a configured logger.
  - `configure_logging(level: str = "INFO")`: Configure global logging settings for the library.

Usage example

```python
from omnidocs.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("OmniDocs utilities ready")
```

### `omnidocs.utils.model_config`

- Purpose: load and manage model configuration for extractors and models.
- Key helpers/classes:
  - `load_model_config(path: str) -> dict`: Load a model configuration from file.
  - `ModelConfig`: Lightweight dataclass wrapping model config values.

Usage example

```python
from omnidocs.utils.model_config import load_model_config

config = load_model_config("configs/default_model.yaml")
print(config.get("model_name"))
```

## See also

- [API Reference Index](index.md)
- [Core API](core.md)
