"""
LightOnOCR-2-1B Inference using Transformers
Usage: python mlx_lighton_text.py

Note: MLX is only available for Apple Silicon (M1/M2/M3).
This script uses transformers library for cross-platform compatibility.
For GPU acceleration on Windows/Linux, ensure torch with CUDA is installed.
"""

import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies if not already installed."""
    required_packages = [
        "transformers>=4.57.6",
        "pillow",
        "numpy",
        "torch",
        "huggingface_hub",
        "accelerate"
    ]
    
    try:
        import transformers
        print("Dependencies check passed.")
    except ImportError:
        print("Installing required dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *required_packages
        ])
        print("Dependencies installed successfully.")

# Install dependencies first
install_dependencies()

from PIL import Image, ImageDraw
import numpy as np
import torch

try:
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
except ImportError as e:
    print(f"Error: Could not import transformers. {e}")
    print("Install with: pip install transformers>=4.57.6 torch accelerate")
    sys.exit(1)

MODEL_NAME = "lightonai/LightOnOCR-2-1B"

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"\nLoading model: {MODEL_NAME}...")
try:
    processor = LightOnOcrProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print("\nCreating test document image...")
# Create a more realistic test document image
image = Image.new("RGB", (800, 1000), color="white")
draw = ImageDraw.Draw(image)

# Add document-like content
draw.text((50, 50), "Sample Document", fill="black")
draw.line([(50, 100), (750, 100)], fill="black", width=2)
draw.rectangle([(50, 150), (750, 300)], outline="black", width=1)
draw.text((60, 160), "Section 1: Introduction", fill="black")
draw.text((60, 200), "This is sample text content.", fill="black")
draw.rectangle([(50, 350), (750, 500)], outline="black", width=1)
draw.text((60, 360), "Section 2: Content", fill="black")
draw.text((60, 400), "More document content goes here.", fill="black")

print("Running inference...")
print("-" * 60)

try:
    # Prepare inputs
    inputs = processor(image, return_tensors="pt").to(device)
    
    print("Processing image...")
    # Generate output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False
    )
    
    # Decode result
    result_text = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
    
    print("\n" + "=" * 60)
    print("EXTRACTION RESULT:")
    print("=" * 60)
    print(result_text[:800] if len(result_text) > 800 else result_text)
    print("\n" + "=" * 60)
    print(f"Total output length: {len(result_text)} characters")
    print("Inference completed successfully!")
    
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)