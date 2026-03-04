#!/usr/bin/env python3
"""MinerU VL text extraction - API backend (VLLM online server).

Starts a local VLLM server, then tests the omnidocs MinerUVLTextAPIConfig
against it. Validates both VLLM online serving and the omnidocs API backend.

Requires: GPU container with vllm installed (runs on Modal via modal_runner).
"""
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import Timer, create_test_image, print_result, verify_text_result

# ============= Config =============
MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"
VLLM_PORT = 8192
CACHE_DIR = os.environ.get("HF_HOME", "/data/.cache")

# ============= Start VLLM server =============
print("Starting VLLM server...")
cmd = [
    "vllm", "serve", MODEL_NAME,
    "--served-model-name", "mineru-vl",
    "--host", "0.0.0.0",
    "--port", str(VLLM_PORT),
    "--gpu-memory-utilization", "0.85",
    "--tensor-parallel-size", "1",
    "--max-model-len", "16384",
    "--trust-remote-code",
    "--download-dir", CACHE_DIR,
    "--dtype", "bfloat16",
    "--enforce-eager",
    "--limit-mm-per-prompt", '{"image": 4}',
]
print(f"Command: {' '.join(cmd)}")

vllm_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# ============= Wait for server =============
import requests

SERVER_URL = f"http://localhost:{VLLM_PORT}"
start_time = time.time()
timeout = 600  # 10 minutes for cold start + model download
ready = False

while time.time() - start_time < timeout:
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code == 200:
            elapsed = time.time() - start_time
            print(f"VLLM server ready after {elapsed:.1f}s")
            ready = True
            break
    except requests.exceptions.RequestException:
        pass

    # Check if process died
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

    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextAPIConfig

    with Timer("Client init") as t_load:
        extractor = MinerUVLTextExtractor(
            backend=MinerUVLTextAPIConfig(
                server_url=SERVER_URL,
                model_name="mineru-vl",
                timeout=300,
                max_tokens=4096,
            )
        )

    with Timer("Inference") as t_infer:
        result = extractor.extract(img)

    verify_text_result(result)
    print_result("mineruvl_text_api", {
        "model": result.model_name,
        "content_length": len(result.content),
        "load_time": f"{t_load.elapsed:.2f}s",
        "inference_time": f"{t_infer.elapsed:.2f}s",
    })
finally:
    # Clean shutdown
    vllm_proc.terminate()
    try:
        vllm_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        vllm_proc.kill()
