"""
LightOnOCR-2-1B Inference on Modal
Usage: modal run lighton_ocr_modal.py
"""

import modal

# CUDA 12.8 is the latest, but for compatibility with current torch builds
# on Modal, 12.4.0 is often more stable.
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libglib2.0-0", "libgl1", "libglx-mesa0")
    .run_commands("pip install uv")
    # Install the latest transformers (v5 candidate) + dependencies
    .run_commands(
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "uv pip install --system pillow pypdfium2",
        "uv pip install --system requests",
        "uv pip install --system transformers>=5.0.0",  # Ensures v5 support
        "uv pip install --system huggingface_hub[hf_transfer] accelerate",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("lighton-ocr-experiment")
volume = modal.Volume.from_name("ocr-cache", create_if_missing=True)


@app.function(
    image=IMAGE,
    gpu="A10G:1",
    volumes={"/data": volume},
    timeout=600,
)
def run_lighton_ocr(image_url: str):
    from io import BytesIO

    import requests
    import torch
    from PIL import Image
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    model_id = "lightonai/LightOnOCR-2-1B"
    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading {model_id}...")
    processor = LightOnOcrProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = LightOnOcrForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(
        device
    )

    # Load image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # The LightOn prompt usually expects a transcription instruction.  For image
    # inputs we need a single content item where type is "image" and the actual
    # PIL image is provided in the same dict.  Previously the type and image were
    # split across separate dicts which meant the model never saw the image.
    # conversation = [{
    #     "role": "user",
    #     "content": [
    #         {"type": "url", "url": image_url}
    #     ]
    # }]
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "Transcribe this document."}],
        }
    ]

    print("Processing inputs...")
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move to GPU and ensure correct dtypes
    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

    print("Generating text...")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    # Slice output to remove the prompt tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    return output_text


@app.local_entrypoint()
def main():
    # target_url = "https://target.scene7.com/is/image/Target/ScreenShot2022-03-22at125438PM-220322-1647972511350?scl=1&qlt=80&fmt=png"
    # target_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
    # target_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQt3PLwQx_EyfrzI8D-dOCLdqPkCo-F3hDGTg&s"
    target_url = "https://towardsdatascience.com/wp-content/uploads/2024/05/16Vy32tUuqtAwz24CfS83wg-1.png"
    # target_url = "https://images.examples.com/wp-content/uploads/2017/06/Short-Essay-for-high-school-Students-Edit-Download-10-16-2024_07_23_PM.png"
    result = run_lighton_ocr.remote(target_url)
    print("\n" + "=" * 50)
    print("OCR RESULT:")
    print("=" * 50)
    print(result)


"""Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable
higher rate limits and faster downloads. You are using a model of type mistral3 to instantiate a
model of type lighton_ocr. This may be expected if you are loading a checkpoint that shares a subset
of the architecture (e.g., loading a sam2_video checkpoint into Sam2Model), but is otherwise not
supported and can yield errors. Please verify that the checkpoint is compatible with the model you
are instantiating."""
