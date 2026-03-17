#!/usr/bin/env python3
"""GLM-OCR text extraction - API backend (self-hosted VLLM server).

Starts a local VLLM server inside the container, tests the GLMOCRAPIConfig
against it, then shuts it down. Validates both vLLM serving and the
omnidocs API backend.

Requires: GPU container with vllm>=0.17.0 and transformers>=5.3.0.
"""

import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

# ============= Config =============
MODEL_NAME = "zai-org/GLM-OCR"
VLLM_PORT = 8192
CACHE_DIR = os.environ.get("HF_HOME", "/data/.cache")
SERVER_URL = f"http://localhost:{VLLM_PORT}"

# ============= Start VLLM server =============
print("Starting VLLM server for GLM-OCR...")
cmd = [
    "vllm",
    "serve",
    MODEL_NAME,
    "--served-model-name",
    "glm-ocr",
    "--host",
    "0.0.0.0",
    "--port",
    str(VLLM_PORT),
    "--gpu-memory-utilization",
    "0.85",
    "--tensor-parallel-size",
    "1",
    "--max-model-len",
    "8192",
    "--trust-remote-code",
    "--download-dir",
    CACHE_DIR,
    "--dtype",
    "bfloat16",
    "--enforce-eager",
]
print(f"Command: {' '.join(cmd)}")

vllm_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# ============= Wait for server =============
start_time = time.time()
timeout = 600
ready = False

while time.time() - start_time < timeout:
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code == 200:
            print(f"VLLM server ready after {time.time() - start_time:.1f}s")
            ready = True
            break
    except requests.exceptions.RequestException:
        pass

    if vllm_proc.poll() is not None:
        output = vllm_proc.stdout.read().decode() if vllm_proc.stdout else ""
        print(f"VLLM server exited with code {vllm_proc.returncode}")
        print(output[-2000:])
        sys.exit(1)

    elapsed = time.time() - start_time
    if int(elapsed) % 30 == 0:
        print(f"  Waiting for server... ({elapsed:.0f}s)")
    time.sleep(5)

if not ready:
    vllm_proc.terminate()
    print(f"VLLM server not ready after {timeout}s")
    sys.exit(1)

# ============= Run omnidocs API extraction =============
try:
    img = create_test_image()

    from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
    from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

    with Timer("Client init") as t_load:
        extractor = GLMOCRTextExtractor(
            backend=GLMOCRAPIConfig(
                model="openai/glm-ocr",
                api_key="dummy",
                api_base=f"{SERVER_URL}/v1",
                timeout=300,
            )
        )

    with Timer("Inference") as t_infer:
        result = extractor.extract(img)

    verify_text_result(result)
    print_result(
        "glm_ocr_text_api",
        {
            "model": result.model_name,
            "content_length": len(result.content),
            "load_time": f"{t_load.elapsed:.2f}s",
            "inference_time": f"{t_infer.elapsed:.2f}s",
        },
    )
finally:
    vllm_proc.terminate()
    try:
        vllm_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        vllm_proc.kill()
