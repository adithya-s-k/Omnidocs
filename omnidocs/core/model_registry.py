"""
Centralized model registry and configuration system for Omnidocs.

This module provides a unified interface for managing all model configurations
across different tasks (layout analysis, OCR, table extraction, VLMs, etc.).
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Type of model architecture/framework."""
    HUGGINGFACE_TRANSFORMERS = "huggingface_transformers"  # Standard HF models
    YOLO = "yolo"  # YOLO-based models
    RTDETR = "rtdetr"  # RT-DETR models
    LIBRARY_MANAGED = "library_managed"  # Models managed by external libraries (Surya, EasyOCR, etc.)
    VLM = "vlm"  # Vision Language Models
    CUSTOM = "custom"  # Custom model implementations


class TaskType(str, Enum):
    """Type of task the model performs."""
    LAYOUT_ANALYSIS = "layout_analysis"
    TABLE_EXTRACTION = "table_extraction"
    OCR = "ocr"
    TEXT_EXTRACTION = "text_extraction"
    MATH_EXPRESSION = "math_expression"
    VLM_PROCESSING = "vlm_processing"  # For future VLM tasks
    DOCUMENT_UNDERSTANDING = "document_understanding"  # For future multi-modal tasks


class ModelConfig(BaseModel):
    """Base configuration for all models."""

    # Core identification
    id: str = Field(..., description="Unique identifier for this model configuration")
    name: str = Field(..., description="Human-readable model name")
    model_type: ModelType = Field(..., description="Type of model architecture")
    task_type: TaskType = Field(..., description="Type of task this model performs")

    # Model source and versioning
    source: str = Field(..., description="Source identifier (HF repo, local path, library name)")
    version: Optional[str] = Field(None, description="Model version or variant")

    # Download and storage
    requires_download: bool = Field(True, description="Whether model needs to be downloaded")
    local_dir: Optional[str] = Field(None, description="Local directory name for model storage")
    model_files: Optional[List[str]] = Field(None, description="Specific model files to download")

    # Model parameters
    confidence_threshold: float = Field(0.5, description="Default confidence threshold for predictions")
    device: Optional[str] = Field(None, description="Device to run model on (cuda/cpu)")

    # Task-specific configurations
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Additional task-specific configuration")

    # Model components (for multi-component models)
    sub_models: Optional[Dict[str, 'ModelConfig']] = Field(None, description="Sub-models for multi-stage pipelines")

    # Metadata
    description: Optional[str] = Field(None, description="Description of model capabilities")
    paper_url: Optional[str] = Field(None, description="Link to research paper")
    license: Optional[str] = Field(None, description="Model license")

    class Config:
        use_enum_values = True


class HuggingFaceModelConfig(ModelConfig):
    """Configuration for HuggingFace Transformers models."""

    model_type: ModelType = ModelType.HUGGINGFACE_TRANSFORMERS

    # HuggingFace-specific fields
    hf_model_id: str = Field(..., description="HuggingFace model identifier (e.g., 'facebook/nougat-base')")
    requires_processor: bool = Field(False, description="Whether model needs a processor")
    processor_class: Optional[str] = Field(None, description="Processor class name if needed")
    model_class: Optional[str] = Field(None, description="Model class name")
    trust_remote_code: bool = Field(False, description="Whether to trust remote code")

    def __init__(self, **data):
        if 'source' not in data and 'hf_model_id' in data:
            data['source'] = data['hf_model_id']
        super().__init__(**data)


class YOLOModelConfig(ModelConfig):
    """Configuration for YOLO-based models."""

    model_type: ModelType = ModelType.YOLO

    # YOLO-specific fields
    model_repo: str = Field(..., description="HuggingFace repository for model weights")
    model_filename: str = Field(..., description="Model weight file name")
    yolo_version: Optional[str] = Field("v10", description="YOLO version (v8, v10, etc.)")
    imgsz: int = Field(1024, description="Input image size")

    def __init__(self, **data):
        if 'source' not in data and 'model_repo' in data:
            data['source'] = data['model_repo']
        if 'model_files' not in data and 'model_filename' in data:
            data['model_files'] = [data['model_filename']]
        super().__init__(**data)


class LibraryManagedModelConfig(ModelConfig):
    """Configuration for models managed by external libraries (Surya, EasyOCR, etc.)."""

    model_type: ModelType = ModelType.LIBRARY_MANAGED
    requires_download: bool = False  # Library handles downloads

    # Library-specific fields
    library_name: str = Field(..., description="Name of the library managing the model")
    predictor_class: Optional[str] = Field(None, description="Predictor/model class to instantiate")
    init_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Kwargs for model initialization")

    def __init__(self, **data):
        if 'source' not in data and 'library_name' in data:
            data['source'] = data['library_name']
        super().__init__(**data)


class VLMModelConfig(ModelConfig):
    """Configuration for Vision Language Models."""

    model_type: ModelType = ModelType.VLM

    # VLM-specific fields
    hf_model_id: Optional[str] = Field(None, description="HuggingFace model identifier if applicable")
    supports_multimodal: bool = Field(True, description="Supports text+image input")
    max_tokens: int = Field(2048, description="Maximum output tokens")
    supports_batch: bool = Field(False, description="Supports batch processing")

    # Vision encoder settings
    vision_encoder: Optional[str] = Field(None, description="Vision encoder model/config")
    image_size: Optional[int] = Field(None, description="Expected input image size")

    # Language model settings
    language_model: Optional[str] = Field(None, description="Language model component")

    def __init__(self, **data):
        if 'source' not in data and 'hf_model_id' in data:
            data['source'] = data['hf_model_id']
        super().__init__(**data)


class ModelRegistry:
    """Centralized registry for all model configurations."""

    _registry: Dict[str, ModelConfig] = {}
    _task_index: Dict[TaskType, List[str]] = {}
    _type_index: Dict[ModelType, List[str]] = {}

    @classmethod
    def register(cls, config: ModelConfig) -> None:
        """Register a model configuration."""
        cls._registry[config.id] = config

        # Update task index
        if config.task_type not in cls._task_index:
            cls._task_index[config.task_type] = []
        cls._task_index[config.task_type].append(config.id)

        # Update type index
        if config.model_type not in cls._type_index:
            cls._type_index[config.model_type] = []
        cls._type_index[config.model_type].append(config.id)

    @classmethod
    def get(cls, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID."""
        return cls._registry.get(model_id)

    @classmethod
    def get_by_task(cls, task_type: TaskType) -> List[ModelConfig]:
        """Get all models for a specific task type."""
        model_ids = cls._task_index.get(task_type, [])
        return [cls._registry[mid] for mid in model_ids]

    @classmethod
    def get_by_type(cls, model_type: ModelType) -> List[ModelConfig]:
        """Get all models of a specific type."""
        model_ids = cls._type_index.get(model_type, [])
        return [cls._registry[mid] for mid in model_ids]

    @classmethod
    def list_all(cls) -> List[ModelConfig]:
        """List all registered models."""
        return list(cls._registry.values())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
        cls._task_index.clear()
        cls._type_index.clear()


# Allow ModelConfig to reference itself for sub_models
ModelConfig.model_rebuild()
