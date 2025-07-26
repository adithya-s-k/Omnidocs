"""
Global model configuration for OmniDocs.

This module provides centralized model directory setup and environment variable management
to avoid code duplication across extractors and prevent race conditions.
"""

import os
from pathlib import Path
from typing import Optional

def setup_model_environment() -> Path:
    """
    Setup model environment variables once for the entire application.
    
    This function:
    1. Calculates the omnidocs models directory dynamically
    2. Creates the directory if it doesn't exist
    3. Sets HuggingFace environment variables to use our models directory
    4. Uses a flag to prevent multiple setups
    
    Returns:
        Path: The models directory path
    """
    # Check if already setup to prevent multiple calls
    if 'OMNIDOCS_MODELS_SETUP' in os.environ:
        # Return the already configured models directory
        return Path(os.environ["HF_HOME"])
    
    # Calculate omnidocs root dynamically
    current_file = Path(__file__)
    omnidocs_root = current_file.parent.parent  # Go up to omnidocs/ root
    models_dir = omnidocs_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variables for HuggingFace to use our models directory
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    
    # Set flag to prevent re-setup
    os.environ["OMNIDOCS_MODELS_SETUP"] = "true"
    
    return models_dir

def get_models_directory() -> Path:
    """
    Get the models directory, setting up environment if needed.
    
    Returns:
        Path: The models directory path
    """
    return setup_model_environment()

def get_model_path(extractor_name: str, model_name: str) -> Path:
    """
    Get standardized model path for a specific extractor and model.
    
    Args:
        extractor_name: Name of the extractor (e.g., 'donut', 'nougat')
        model_name: Name/ID of the model (e.g., 'naver-clova-ix/donut-base')
        
    Returns:
        Path: Full path where the model should be stored
    """
    models_dir = get_models_directory()
    # Replace slashes in model names to create valid directory names
    safe_model_name = model_name.replace("/", "_")
    return models_dir / extractor_name / safe_model_name
