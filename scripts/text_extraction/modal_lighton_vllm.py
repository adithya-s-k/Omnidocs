import modal

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libglib2.0-0", "libgl1", "libglx-mesa0")
    .run_commands(
        "pip install uv",
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "uv pip install --system vllm",
        "uv pip install --system pillow pypdfium2 requests",
    )
)

app = modal.App("lighton-ocr-vllm")


@app.function(
    image=IMAGE,
    gpu="A10G:1",
    timeout=600,
)
def run_lighton_ocr(image_url: str):
    import base64
    import socket
    import subprocess
    import time
    from io import BytesIO

    import requests
    from PIL import Image

    model = "lightonai/LightOnOCR-2-1B"

    #  Start vLLM server
    server = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--mm-processor-cache-gb",
            "0",
            "--no-enable-prefix-caching",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    #  Wait for server to be ready
    def wait_for_server(host="localhost", port=8000, timeout=240):
        start = time.time()
        while time.time() - start < timeout:
            # If server crashed, exit early
            if server.poll() is not None:
                raise RuntimeError("vLLM server failed to start.")

            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError:
                time.sleep(2)

        raise RuntimeError("Timed out waiting for vLLM server.")

    wait_for_server()

    #  Prepare image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Call OpenAI-style endpoint
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": "Transcribe this document."},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
    }

    result = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        timeout=300,
    )

    result.raise_for_status()

    output_text = result.json()["choices"][0]["message"]["content"]

    server.terminate()
    server.wait()

    return output_text


@app.local_entrypoint()
def main():
    target_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
    result = run_lighton_ocr.remote(target_url)

    print("\n" + "=" * 50)
    print("OCR RESULT:")
    print("=" * 50)
    print(result)
