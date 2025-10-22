"""
Core module for Omnidocs model management.

This module provides centralized model registry, configuration, and loading utilities.
"""

from omnidocs.core.model_registry import (
    ModelConfig,
    HuggingFaceModelConfig,
    YOLOModelConfig,
    LibraryManagedModelConfig,
    VLMModelConfig,
    ModelType,
    TaskType,
    ModelRegistry,
)

from omnidocs.core.model_loader import (
    ModelLoader,
    get_model_config,
    list_models_for_task,
)

# Import models catalog to auto-register all models
from omnidocs.core import models_catalog

__all__ = [
    # Model configurations
    "ModelConfig",
    "HuggingFaceModelConfig",
    "YOLOModelConfig",
    "LibraryManagedModelConfig",
    "VLMModelConfig",
    # Enums
    "ModelType",
    "TaskType",
    # Registry
    "ModelRegistry",
    # Loader utilities
    "ModelLoader",
    "get_model_config",
    "list_models_for_task",
]
