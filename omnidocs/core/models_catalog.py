"""
Catalog of all available models in Omnidocs.

This module registers all supported models with the ModelRegistry.
Add new models here to make them available throughout the system.
"""

from omnidocs.core.model_registry import (
    ModelRegistry,
    HuggingFaceModelConfig,
    YOLOModelConfig,
    LibraryManagedModelConfig,
    VLMModelConfig,
    TaskType,
    ModelType,
)


# =============================================================================
# LAYOUT ANALYSIS MODELS
# =============================================================================

# Surya Layout Analysis
SURYA_LAYOUT = LibraryManagedModelConfig(
    id="surya-layout",
    name="Surya Layout Analysis",
    task_type=TaskType.LAYOUT_ANALYSIS,
    library_name="surya",
    predictor_class="LayoutPredictor",
    description="Fast and accurate layout detection using Surya",
    local_dir="surya",
)

# DocLayout-YOLO
DOCLAYOUT_YOLO = YOLOModelConfig(
    id="doclayout-yolo-docstructbench",
    name="DocLayout-YOLO DocStructBench",
    task_type=TaskType.LAYOUT_ANALYSIS,
    model_repo="juliozhao/DocLayout-YOLO-DocStructBench",
    model_filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    yolo_version="v10",
    imgsz=1024,
    confidence_threshold=0.25,
    local_dir="DocLayout-YOLO-DocStructBench",
    description="YOLO-based document layout analysis trained on DocStructBench",
)

# RT-DETR Layout
RTDETR_LAYOUT = YOLOModelConfig(
    id="rtdetr-publaynet",
    name="RT-DETR PubLayNet",
    task_type=TaskType.LAYOUT_ANALYSIS,
    model_type=ModelType.RTDETR,
    model_repo="amd/rtdetr-publaynet",
    model_filename="rtdetr_r50vd_6x_coco_publaynet.safetensors",
    confidence_threshold=0.5,
    local_dir="rtdetr-publaynet",
    description="Real-time DETR model for document layout analysis",
)

# YOLOv10 Layout
YOLOV10_LAYOUT = YOLOModelConfig(
    id="yolov10-doclaynet",
    name="YOLOv10 DocLayNet",
    task_type=TaskType.LAYOUT_ANALYSIS,
    model_repo="omoured/YOLOv10-Document-Layout-Analysis",
    model_filename="yolov10_doclaynet.pt",
    yolo_version="v10",
    confidence_threshold=0.2,
    local_dir="YOLOv10-Document-Layout-Analysis",
    description="YOLOv10 trained on DocLayNet dataset",
)


# =============================================================================
# TABLE EXTRACTION MODELS
# =============================================================================

# Table Transformer Detection
TABLE_TRANSFORMER_DETECTION = HuggingFaceModelConfig(
    id="table-transformer-detection",
    name="Table Transformer Detection",
    task_type=TaskType.TABLE_EXTRACTION,
    hf_model_id="microsoft/table-transformer-detection",
    confidence_threshold=0.7,
    local_dir="table-transformer-detection",
    description="Microsoft's transformer model for table detection",
    extra_config={
        "classes": ["table"],
    },
)

# Table Transformer Structure Recognition
TABLE_TRANSFORMER_STRUCTURE = HuggingFaceModelConfig(
    id="table-transformer-structure",
    name="Table Transformer Structure Recognition",
    task_type=TaskType.TABLE_EXTRACTION,
    hf_model_id="microsoft/table-structure-recognition-v1.1-all",
    confidence_threshold=0.7,
    local_dir="table-structure-recognition",
    description="Microsoft's transformer model for table structure recognition",
    extra_config={
        "classes": [
            "table", "table column", "table row", "table column header",
            "table projected row header", "table spanning cell"
        ],
    },
)

# TableFormer Models (same as Table Transformer but different interface)
TABLEFORMER_DETECTION = HuggingFaceModelConfig(
    id="tableformer-detection",
    name="TableFormer Detection",
    task_type=TaskType.TABLE_EXTRACTION,
    hf_model_id="microsoft/table-transformer-detection",
    confidence_threshold=0.7,
    local_dir="tableformer-detection",
    description="TableFormer detection model",
    extra_config={
        "classes": ["table"],
    },
)

TABLEFORMER_STRUCTURE = HuggingFaceModelConfig(
    id="tableformer-structure",
    name="TableFormer Structure",
    task_type=TaskType.TABLE_EXTRACTION,
    hf_model_id="microsoft/table-structure-recognition-v1.1-all",
    confidence_threshold=0.7,
    local_dir="tableformer-structure",
    description="TableFormer structure recognition model",
    extra_config={
        "classes": [
            "table", "table column", "table row", "table column header",
            "table projected row header", "table spanning cell"
        ],
    },
)

# Surya Table Extraction
SURYA_TABLE = LibraryManagedModelConfig(
    id="surya-table",
    name="Surya Table Extraction",
    task_type=TaskType.TABLE_EXTRACTION,
    library_name="surya",
    predictor_class="TableRecognitionPredictor",
    description="Table extraction using Surya",
    local_dir="surya",
)


# =============================================================================
# MATH EXPRESSION EXTRACTION MODELS
# =============================================================================

# Nougat Models
NOUGAT_BASE = HuggingFaceModelConfig(
    id="nougat-base",
    name="Nougat Base",
    task_type=TaskType.MATH_EXPRESSION,
    hf_model_id="facebook/nougat-base",
    requires_processor=True,
    local_dir="nougat_ckpt",
    description="Nougat base model for LaTeX extraction from images",
)

NOUGAT_SMALL = HuggingFaceModelConfig(
    id="nougat-small",
    name="Nougat Small",
    task_type=TaskType.MATH_EXPRESSION,
    hf_model_id="facebook/nougat-small",
    requires_processor=True,
    local_dir="nougat_small_ckpt",
    description="Nougat small model for LaTeX extraction from images",
)

# Donut Model
DONUT_CORD = HuggingFaceModelConfig(
    id="donut-cord-v2",
    name="Donut CORD v2",
    task_type=TaskType.MATH_EXPRESSION,
    hf_model_id="naver-clova-ix/donut-base-finetuned-cord-v2",
    requires_processor=True,
    processor_class="DonutProcessor",
    model_class="VisionEncoderDecoderModel",
    local_dir="donut-cord",
    description="Donut model fine-tuned on CORD dataset",
)

# Surya Math
SURYA_MATH = LibraryManagedModelConfig(
    id="surya-math",
    name="Surya Math Expression",
    task_type=TaskType.MATH_EXPRESSION,
    library_name="surya",
    description="Math expression extraction using Surya",
    local_dir="surya",
)


# =============================================================================
# OCR MODELS
# =============================================================================

# Surya OCR
SURYA_OCR = LibraryManagedModelConfig(
    id="surya-ocr",
    name="Surya OCR",
    task_type=TaskType.OCR,
    library_name="surya",
    predictor_class="DetectionPredictor",
    description="Fast and accurate OCR using Surya",
    local_dir="surya",
    extra_config={
        "supports_languages": True,
        "recognition_predictor": "RecognitionPredictor",
    },
)

# EasyOCR
EASYOCR = LibraryManagedModelConfig(
    id="easyocr",
    name="EasyOCR",
    task_type=TaskType.OCR,
    library_name="easyocr",
    local_dir="easyocr",
    description="EasyOCR with support for 80+ languages",
    extra_config={
        "supports_languages": True,
        "default_languages": ["en"],
    },
)

# PaddleOCR
PADDLE_OCR = LibraryManagedModelConfig(
    id="paddleocr",
    name="PaddleOCR",
    task_type=TaskType.OCR,
    library_name="paddleocr",
    local_dir="paddleocr",
    description="PaddlePaddle OCR with high accuracy",
    extra_config={
        "supports_languages": True,
        "default_lang": "en",
    },
)

# Tesseract OCR
TESSERACT = LibraryManagedModelConfig(
    id="tesseract",
    name="Tesseract OCR",
    task_type=TaskType.OCR,
    library_name="tesseract",
    requires_download=False,
    description="Traditional Tesseract OCR engine",
    extra_config={
        "supports_languages": True,
    },
)


# =============================================================================
# TEXT EXTRACTION MODELS
# =============================================================================

# Surya Text Detection
SURYA_TEXT = LibraryManagedModelConfig(
    id="surya-text",
    name="Surya Text Detection",
    task_type=TaskType.TEXT_EXTRACTION,
    library_name="surya",
    predictor_class="DetectionPredictor",
    description="Text detection and ordering using Surya",
    local_dir="surya",
    extra_config={
        "ordering_predictor": "OrderingPredictor",
    },
)


# =============================================================================
# VLM MODELS (for future expansion)
# =============================================================================

# Example VLM configurations (commented out, uncomment when implementing)
"""
# GPT-4 Vision
GPT4_VISION = VLMModelConfig(
    id="gpt4-vision",
    name="GPT-4 Vision",
    task_type=TaskType.VLM_PROCESSING,
    source="openai",
    supports_multimodal=True,
    max_tokens=4096,
    description="OpenAI GPT-4 with vision capabilities",
    extra_config={
        "api_based": True,
        "requires_api_key": True,
    },
)

# Claude 3 Vision
CLAUDE3_OPUS = VLMModelConfig(
    id="claude-3-opus",
    name="Claude 3 Opus",
    task_type=TaskType.VLM_PROCESSING,
    source="anthropic",
    supports_multimodal=True,
    max_tokens=4096,
    description="Anthropic Claude 3 Opus with vision",
    extra_config={
        "api_based": True,
        "requires_api_key": True,
    },
)

# LLaVA
LLAVA_1_6 = VLMModelConfig(
    id="llava-1.6-vicuna-7b",
    name="LLaVA 1.6 Vicuna 7B",
    task_type=TaskType.VLM_PROCESSING,
    hf_model_id="liuhaotian/llava-v1.6-vicuna-7b",
    supports_multimodal=True,
    max_tokens=2048,
    image_size=336,
    description="LLaVA 1.6 vision-language model",
)

# Qwen-VL
QWEN_VL = VLMModelConfig(
    id="qwen-vl-chat",
    name="Qwen-VL Chat",
    task_type=TaskType.VLM_PROCESSING,
    hf_model_id="Qwen/Qwen-VL-Chat",
    supports_multimodal=True,
    max_tokens=2048,
    description="Qwen Vision-Language model",
)
"""


# =============================================================================
# REGISTRATION
# =============================================================================

def register_all_models():
    """Register all models with the ModelRegistry."""
    models = [
        # Layout Analysis
        SURYA_LAYOUT,
        DOCLAYOUT_YOLO,
        RTDETR_LAYOUT,
        YOLOV10_LAYOUT,
        # Table Extraction
        TABLE_TRANSFORMER_DETECTION,
        TABLE_TRANSFORMER_STRUCTURE,
        TABLEFORMER_DETECTION,
        TABLEFORMER_STRUCTURE,
        SURYA_TABLE,
        # Math Expression
        NOUGAT_BASE,
        NOUGAT_SMALL,
        DONUT_CORD,
        SURYA_MATH,
        # OCR
        SURYA_OCR,
        EASYOCR,
        PADDLE_OCR,
        TESSERACT,
        # Text Extraction
        SURYA_TEXT,
        # VLMs (uncomment when ready)
        # GPT4_VISION,
        # CLAUDE3_OPUS,
        # LLAVA_1_6,
        # QWEN_VL,
    ]

    for model in models:
        ModelRegistry.register(model)


# Auto-register on import
register_all_models()
