import time
import copy
import base64
import cv2
import numpy as np
from io import BytesIO
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor, BaseOCRMapper, OCROutput, OCRText

logger = get_logger(__name__)

# Custom implementations
def alpha_to_color(img, alpha_color=(255, 255, 255)):
    """Convert transparent pixels to specified color."""
    if len(img.shape) == 4:  # RGBA
        alpha_channel = img[:, :, 3]
        rgb_channels = img[:, :, :3]
        
        # Create a mask for transparent pixels
        transparent_mask = alpha_channel == 0
        
        # Replace transparent pixels with alpha_color
        for i in range(3):
            rgb_channels[:, :, i][transparent_mask] = alpha_color[i]
        
        return rgb_channels
    return img

def binarize_img(img):
    """Convert image to binary (black and white)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def check_and_read(image_file):
    """Simple implementation to check and read image files."""
    try:
        img = cv2.imread(image_file)
        return img, False, False  # img, flag_gif, flag_pdf
    except:
        return None, False, False

def sorted_boxes(dt_boxes):
    """Sort text boxes in order from top to bottom, left to right."""
    if len(dt_boxes) == 0:
        return []
    
    num_boxes = len(dt_boxes)
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def __is_overlaps_y_exceeds_threshold(bbox1, bbox2, overlap_ratio_threshold=0.8):
    """Check if two bounding boxes overlap on the y-axis."""
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    min_height = min(height1, height2)

    return (overlap / min_height) > overlap_ratio_threshold

def bbox_to_points(bbox):
    """Change bbox(shape: N * 4) to polygon(shape: N * 8)."""
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype('float32')

def points_to_bbox(points):
    """Change polygon(shape: N * 8) to bbox(shape: N * 4)."""
    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return [x0, y0, x1, y1]

def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def remove_intervals(original, masks):
    """Remove masked intervals from original range."""
    merged_masks = merge_intervals(masks)
    result = []
    original_start, original_end = original

    for mask in merged_masks:
        mask_start, mask_end = mask
        if mask_start > original_end or mask_end < original_start:
            continue
        if original_start < mask_start:
            result.append([original_start, mask_start - 1])
        original_start = max(mask_end + 1, original_start)

    if original_start <= original_end:
        result.append([original_start, original_end])
    return result

def update_det_boxes(dt_boxes, mfd_res):
    """Update detection boxes by removing formula regions."""
    new_dt_boxes = []
    for text_box in dt_boxes:
        text_bbox = points_to_bbox(text_box)
        masks_list = []
        for mf_box in mfd_res:
            mf_bbox = mf_box['bbox']
            if __is_overlaps_y_exceeds_threshold(text_bbox, mf_bbox):
                masks_list.append([mf_bbox[0], mf_bbox[2]])
        text_x_range = [text_bbox[0], text_bbox[2]]
        text_remove_mask_range = remove_intervals(text_x_range, masks_list)
        temp_dt_box = []
        for text_remove_mask in text_remove_mask_range:
            temp_dt_box.append(bbox_to_points([text_remove_mask[0], text_bbox[1], text_remove_mask[1], text_bbox[3]]))
        if len(temp_dt_box) > 0:
            new_dt_boxes.extend(temp_dt_box)
    return new_dt_boxes

def merge_spans_to_line(spans):
    """Merge spans into lines based on Y-axis overlap."""
    if len(spans) == 0:
        return []
    
    spans.sort(key=lambda span: span['bbox'][1])
    lines = []
    current_line = [spans[0]]
    
    for span in spans[1:]:
        if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox']):
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    
    if current_line:
        lines.append(current_line)
    return lines

def merge_overlapping_spans(spans):
    """Merge overlapping spans on the same line."""
    if not spans:
        return []

    spans.sort(key=lambda x: x[0])
    merged = []
    
    for span in spans:
        x1, y1, x2, y2 = span
        if not merged or merged[-1][2] < x1:
            merged.append(span)
        else:
            last_span = merged.pop()
            x1 = min(last_span[0], x1)
            y1 = min(last_span[1], y1)
            x2 = max(last_span[2], x2)
            y2 = max(last_span[3], y2)
            merged.append((x1, y1, x2, y2))
    return merged

def merge_det_boxes(dt_boxes):
    """Merge detection boxes into larger text regions."""
    dt_boxes_dict_list = []
    for text_box in dt_boxes:
        text_bbox = points_to_bbox(text_box)
        text_box_dict = {'bbox': text_bbox}
        dt_boxes_dict_list.append(text_box_dict)

    lines = merge_spans_to_line(dt_boxes_dict_list)
    new_dt_boxes = []
    
    for line in lines:
        line_bbox_list = [span['bbox'] for span in line]
        merged_spans = merge_overlapping_spans(line_bbox_list)
        for span in merged_spans:
            new_dt_boxes.append(bbox_to_points(span))
    return new_dt_boxes

class PaddleOCRMapper(BaseOCRMapper):
    """Label mapper for PaddleOCR model output."""
    
    def __init__(self):
        super().__init__('paddleocr')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup language mappings for PaddleOCR."""
        mapping = {
            'en': 'en',
            'ch': 'ch',
            'chinese_cht': 'chinese_cht',
            'ta': 'ta',
            'te': 'te',
            'ka': 'ka',
            'ja': 'japan',
            'ko': 'korean',
            'hi': 'hi',
            'ar': 'ar',
            'cyrillic': 'cyrillic',
            'devanagari': 'devanagari',
            'fr': 'fr',
            'de': 'german',
            'es': 'es',
            'pt': 'pt',
            'ru': 'ru',
            'it': 'it',
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class PaddleOCRExtractor(BaseOCRExtractor):
    """PaddleOCR based text extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        drop_score: float = 0.5,
        **kwargs
    ):
        """Initialize PaddleOCR Extractor."""
        super().__init__(
            device=device, 
            show_log=show_log, 
            languages=languages or ['en'],
            engine_name='paddleocr'
        )
        
        self._label_mapper = PaddleOCRMapper()
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.drop_score = drop_score
        
        try:
            from paddleocr import PaddleOCR
            self.PaddleOCR = PaddleOCR
            
        except ImportError as e:
            logger.error("Failed to import PaddleOCR")
            raise ImportError(
                "PaddleOCR is not available. Please install it with: pip install paddlepaddle paddleocr"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        Model download is handled automatically by PaddleOCR library.
        This method is required by the abstract base class but PaddleOCR
        handles model downloading internally when first initialized.
        """
        if self.show_log:
            logger.info("Model downloading is handled automatically by PaddleOCR library")
        return None
    
    def _load_model(self) -> None:
        """Load PaddleOCR models."""
        try:
            # Map languages to PaddleOCR format
            paddle_languages = []
            for lang in self.languages:
                mapped_lang = self._label_mapper.from_standard_language(lang)
                paddle_languages.append(mapped_lang)
            
            # Initialize PaddleOCR with inference package
            self.paddle_ocr = self.PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=paddle_languages[0] if paddle_languages else 'en',
                use_gpu=self.use_gpu,
                show_log=self.show_log
            )
            
            if self.show_log:
                logger.info(f"PaddleOCR models loaded")
                
        except Exception as e:
            logger.error("Failed to load PaddleOCR models", exc_info=True)
            raise
    
    def preprocess_image(self, image, alpha_color=(255, 255, 255), inv=False, bin=False):
        """Preprocess image for OCR."""
        image = alpha_to_color(image, alpha_color)
        if inv:
            image = cv2.bitwise_not(image)
        if bin:
            image = binarize_img(image)
        return image
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        mfd_res: Optional[List] = None,
        **kwargs
    ) -> OCROutput:
        """Extract text using PaddleOCR."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            img = images[0]
            
            # Convert PIL to cv2 format if needed
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Perform OCR
            result = self.paddle_ocr.ocr(img, cls=self.use_angle_cls)
            
            # Convert to standardized format
            texts = []
            full_text_parts = []
            
            if result and result[0]:
                for i, detection in enumerate(result[0]):
                    bbox_points, (text, confidence) = detection
                    
                    if confidence < self.drop_score:
                        continue
                    
                    text = text.strip()
                    if not text:
                        continue
                    
                    # Convert points to bbox format
                    points = np.array(bbox_points)
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    bbox = [float(min(x_coords)), float(min(y_coords)), 
                           float(max(x_coords)), float(max(y_coords))]
                    
                    # Detect language
                    detected_lang = self.detect_text_language(text)
                    
                    ocr_text = OCRText(
                        text=text,
                        confidence=float(confidence),
                        bbox=bbox,
                        language=detected_lang,
                        reading_order=i
                    )
                    
                    texts.append(ocr_text)
                    full_text_parts.append(text)
            
            img_size = img.shape[:2][::-1]  # (width, height)
            
            ocr_output = OCROutput(
                texts=texts,
                full_text=' '.join(full_text_parts),
                source_img_size=img_size
            )
            
            if self.show_log:
                logger.info(f"Extracted {len(texts)} text regions")
            
            return ocr_output
            
        except Exception as e:
            logger.error("Error during PaddleOCR extraction", exc_info=True)
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )
    
    def predict(self, img, **kwargs):
        """Predict method for compatibility with original interface."""
        try:
            result = self.extract(img, **kwargs)
            
            # Convert to original format
            ocr_res = []
            for text_obj in result.texts:
                # Convert bbox back to points format
                x0, y0, x1, y1 = text_obj.bbox
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                poly = [coord for point in points for coord in point]
                
                ocr_res.append({
                    "category_type": "text",
                    'poly': poly,
                    'score': text_obj.confidence,
                    'text': text_obj.text,
                })
            
            return ocr_res
            
        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            return []
