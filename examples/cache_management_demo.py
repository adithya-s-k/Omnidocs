"""
Example demonstrating unified cache directory management.

This script shows how to use the OMNIDOCS_MODEL_CACHE environment variable
and the cache utility functions.
"""

import os
from omnidocs.utils.cache import get_cache_info, get_model_cache_dir

print("=" * 60)
print("OmniDocs Cache Management Demo")
print("=" * 60)

# 1. Check default cache configuration
print("\n1. Default Configuration:")
print("-" * 60)
info = get_cache_info()
for key, value in info.items():
    print(f"  {key}: {value}")

# 2. Get cache directory
print("\n2. Cache Directory:")
print("-" * 60)
cache_dir = get_model_cache_dir()
print(f"  Model cache: {cache_dir}")
print(f"  Cache exists: {cache_dir.exists()}")

# 3. Demonstrate custom cache directory
print("\n3. Custom Cache Directory:")
print("-" * 60)
custom_cache = get_model_cache_dir("/tmp/custom-omnidocs-cache")
print(f"  Custom cache: {custom_cache}")
print(f"  Custom cache exists: {custom_cache.exists()}")

# 4. Show environment variable override
print("\n4. Environment Variable Override:")
print("-" * 60)
os.environ["OMNIDOCS_MODEL_CACHE"] = "/tmp/env-cache"
env_cache = get_model_cache_dir()
print(f"  Set OMNIDOCS_MODEL_CACHE=/tmp/env-cache")
print(f"  Resolved cache: {env_cache}")

# 5. Show backend-specific usage
print("\n5. Backend Configuration Example:")
print("-" * 60)
print("""
  from omnidocs.tasks.text_extraction import QwenTextExtractor
  from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

  # Uses OMNIDOCS_MODEL_CACHE environment variable
  extractor = QwenTextExtractor(
      backend=QwenTextPyTorchConfig(
          model="Qwen/Qwen3-VL-8B-Instruct"
      )
  )

  # Override with custom cache directory
  extractor = QwenTextExtractor(
      backend=QwenTextPyTorchConfig(
          model="Qwen/Qwen3-VL-8B-Instruct",
          cache_dir="/custom/cache"  # Override
      )
  )
""")

print("\n" + "=" * 60)
print("Cache management configured successfully!")
print("=" * 60)
print("\nNext steps:")
print("  1. Set OMNIDOCS_MODEL_CACHE environment variable")
print("  2. Use cache_dir parameter for per-backend customization")
print("  3. See docs/guides/cache-management.md for more details")
print("=" * 60)
