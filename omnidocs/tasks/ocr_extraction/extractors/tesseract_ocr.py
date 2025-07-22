import sys
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor, BaseOCRMapper, OCROutput, OCRText

logger = get_logger(__name__)

class TesseractOCRMapper(BaseOCRMapper):
    """Label mapper for Tesseract OCR model output."""
    
    def __init__(self):
        super().__init__('tesseract')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup language mappings for Tesseract OCR."""
        # Tesseract language codes
        mapping = {
            'afr': 'af',
            'amh': 'am',
            'ara': 'ar',
            'asm': 'as',
            'aze': 'az',
            'aze_cyrl': 'az',
            'bel': 'be',
            'ben': 'bn',
            'bod': 'bo',
            'bos': 'bs',
            'bul': 'bg',
            'cat': 'ca',
            'ceb': 'ceb',
            'ces': 'cs',
            'chi_sim': 'zh',
            'chi_tra': 'zh-TW',
            'chr': 'chr',
            'cym': 'cy',
            'dan': 'da',
            'deu': 'de',
            'dzo': 'dz',
            'ell': 'el',
            'eng': 'en',
            'enm': 'enm',
            'epo': 'eo',
            'est': 'et',
            'eus': 'eu',
            'fas': 'fa',
            'fin': 'fi',
            'fra': 'fr',
            'frk': 'frk',
            'frm': 'frm',
            'gle': 'ga',
            'glg': 'gl',
            'grc': 'grc',
            'guj': 'gu',
            'hat': 'ht',
            'heb': 'he',
            'hin': 'hi',
            'hrv': 'hr',
            'hun': 'hu',
            'iku': 'iu',
            'ind': 'id',
            'isl': 'is',
            'ita': 'it',
            'ita_old': 'it',
            'jav': 'jv',
            'jpn': 'ja',
            'kan': 'kn',
            'kat': 'ka',
            'kat_old': 'ka',
            'kaz': 'kk',
            'khm': 'km',
            'kir': 'ky',
            'kor': 'ko',
            'kur': 'ku',
            'lao': 'lo',
            'lat': 'la',
            'lav': 'lv',
            'lit': 'lt',
            'mal': 'ml',
            'mar': 'mr',
            'mkd': 'mk',
            'mlt': 'mt',
            'mon': 'mn',
            'mri': 'mi',
            'msa': 'ms',
            'mya': 'my',
            'nep': 'ne',
            'nld': 'nl',
            'nor': 'no',
            'ori': 'or',
            'pan': 'pa',
            'pol': 'pl',
            'por': 'pt',
            'pus': 'ps',
            'ron': 'ro',
            'rus': 'ru',
            'san': 'sa',
            'sin': 'si',
            'slk': 'sk',
            'slv': 'sl',
            'spa': 'es',
            'spa_old': 'es',
            'sqi': 'sq',
            'srp': 'sr',
            'srp_latn': 'sr',
            'swa': 'sw',
            'swe': 'sv',
            'syr': 'syr',
            'tam': 'ta',
            'tel': 'te',
            'tgk': 'tg',
            'tgl': 'tl',
            'tha': 'th',
            'tir': 'ti',
            'tur': 'tr',
            'uig': 'ug',
            'ukr': 'uk',
            'urd': 'ur',
            'uzb': 'uz',
            'uzb_cyrl': 'uz',
            'vie': 'vi',
            'yid': 'yi',
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class TesseractOCRExtractor(BaseOCRExtractor):
    """Tesseract OCR based text extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        psm: int = 6,
        oem: int = 3,
        config: str = "",
        **kwargs
    ):
        """Initialize Tesseract OCR Extractor."""
        super().__init__(
            device=device, 
            show_log=show_log, 
            languages=languages or ['en'],
            engine_name='tesseract'
        )
        
        self.psm = psm  # Page segmentation mode
        self.oem = oem  # OCR engine mode
        self.config = config
        self._label_mapper = TesseractOCRMapper()
        
        try:
            import pytesseract
            from pytesseract import Output
            self.pytesseract = pytesseract
            self.Output = Output
            
            # Set Tesseract executable path for Windows
            import os
            tesseract_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            ]
            
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    if self.show_log:
                        logger.info(f"Found Tesseract at: {path}")
                    break
            else:
                # Try to find in PATH
                import shutil
                tesseract_cmd = shutil.which('tesseract')
                if tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                    if self.show_log:
                        logger.info(f"Found Tesseract in PATH: {tesseract_cmd}")
                
        except ImportError as e:
            logger.error("Failed to import pytesseract")
            raise ImportError(
                "pytesseract is not available. Please install it with: pip install pytesseract"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Path:
        """Tesseract uses system installation."""
        if self.show_log:
            logger.info("Tesseract uses system installation")
        return None
    
    def _load_model(self) -> None:
        """Initialize Tesseract configuration."""
        try:
            # Map languages to Tesseract format
            tesseract_languages = []
            for lang in self.languages:
                mapped_lang = self._label_mapper.from_standard_language(lang)
                tesseract_languages.append(mapped_lang)
            
            self.lang_string = '+'.join(tesseract_languages)
            
            # Build config string
            config_parts = [f'--psm {self.psm}', f'--oem {self.oem}']
            if self.config:
                config_parts.append(self.config)
            self.tesseract_config = ' '.join(config_parts)
            
            if self.show_log:
                logger.info(f"Tesseract configured with languages: {self.lang_string}")
                logger.info(f"Tesseract config: {self.tesseract_config}")
                
        except Exception as e:
            logger.error("Failed to configure Tesseract", exc_info=True)
            raise
    
    def postprocess_output(self, raw_output: dict, img_size: Tuple[int, int]) -> OCROutput:
        """Convert Tesseract output to standardized OCROutput format."""
        texts = []
        full_text_parts = []
        
        n_boxes = len(raw_output['text'])
        
        for i in range(n_boxes):
            text = raw_output['text'][i].strip()
            
            if not text:
                continue
            
            confidence = float(raw_output['conf'][i])
            
            if confidence < 0:
                continue
            
            x = int(raw_output['left'][i])
            y = int(raw_output['top'][i])
            w = int(raw_output['width'][i])
            h = int(raw_output['height'][i])
            bbox = [float(x), float(y), float(x + w), float(y + h)]

            # Create polygon from bbox
            polygon = [[float(x), float(y)], [float(x + w), float(y)], 
                       [float(x + w), float(y + h)], [float(x), float(y + h)]]

            detected_lang = self.detect_text_language(text)

            ocr_text = OCRText(
                text=text,
                confidence=confidence / 100.0,
                bbox=bbox,
                polygon=polygon,
                language=detected_lang,
                reading_order=i
            )
            
            texts.append(ocr_text)
            full_text_parts.append(text)
        
        return OCROutput(
            texts=texts,
            full_text=' '.join(full_text_parts),
            source_img_size=img_size
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> OCROutput:
        """Extract text using Tesseract OCR."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            img = images[0]
            
            # Convert PIL to numpy array
            img_array = np.array(img)
            
            # Run OCR with detailed output
            raw_output = self.pytesseract.image_to_data(
                img_array,
                lang=self.lang_string,
                config=self.tesseract_config,
                output_type=self.Output.DICT
            )
            
            # Convert to standardized format
            result = self.postprocess_output(raw_output, img.size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.texts)} text regions")
            
            return result
            
        except Exception as e:
            logger.error("Error during Tesseract extraction", exc_info=True)
            return OCROutput(
                texts=[],
                full_text="",
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )