"""
Language utilities for OmniDocs.

This module provides standardized language code mappings and utilities
that can be used across different tasks (OCR, layout analysis, etc.).
"""

from typing import Dict, List, Optional
from enum import Enum


class LanguageCode(Enum):
    """Standard ISO 639-1 language codes supported by OmniDocs."""
    
    # Major languages
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    CHINESE = "zh"  # Generic Chinese
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    
    # European languages
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    POLISH = "pl"
    CZECH = "cs"
    HUNGARIAN = "hu"
    ROMANIAN = "ro"
    BULGARIAN = "bg"
    GREEK = "el"
    TURKISH = "tr"
    
    # Other languages
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    TAGALOG = "tl"
    BENGALI = "bn"
    GUJARATI = "gu"
    PUNJABI = "pa"
    TELUGU = "te"
    TAMIL = "ta"
    KANNADA = "kn"
    MALAYALAM = "ml"
    MARATHI = "mr"
    URDU = "ur"
    PERSIAN = "fa"
    HEBREW = "he"
    
    @classmethod
    def get_all_codes(cls) -> List[str]:
        """Get all supported language codes."""
        return [lang.value for lang in cls]
    
    @classmethod
    def is_valid_code(cls, code: str) -> bool:
        """Check if a language code is valid."""
        return code.lower() in [lang.value.lower() for lang in cls]


class GlobalLanguageMapper:
    """Global language mapper that handles different OCR engine formats."""
    
    def __init__(self):
        self._engine_mappings: Dict[str, Dict[str, str]] = {}
        self._setup_default_mappings()
    
    def _setup_default_mappings(self):
        """Setup default language mappings for common OCR engines."""
        
        # Tesseract language mappings
        self._engine_mappings['tesseract'] = {
            'eng': LanguageCode.ENGLISH.value,
            'chi_sim': LanguageCode.CHINESE_SIMPLIFIED.value,
            'chi_tra': LanguageCode.CHINESE_TRADITIONAL.value,
            'jpn': LanguageCode.JAPANESE.value,
            'kor': LanguageCode.KOREAN.value,
            'ara': LanguageCode.ARABIC.value,
            'hin': LanguageCode.HINDI.value,
            'fra': LanguageCode.FRENCH.value,
            'deu': LanguageCode.GERMAN.value,
            'spa': LanguageCode.SPANISH.value,
            'ita': LanguageCode.ITALIAN.value,
            'por': LanguageCode.PORTUGUESE.value,
            'rus': LanguageCode.RUSSIAN.value,
            'nld': LanguageCode.DUTCH.value,
            'swe': LanguageCode.SWEDISH.value,
            'nor': LanguageCode.NORWEGIAN.value,
            'dan': LanguageCode.DANISH.value,
            'fin': LanguageCode.FINNISH.value,
            'pol': LanguageCode.POLISH.value,
            'ces': LanguageCode.CZECH.value,
            'hun': LanguageCode.HUNGARIAN.value,
            'ron': LanguageCode.ROMANIAN.value,
            'bul': LanguageCode.BULGARIAN.value,
            'ell': LanguageCode.GREEK.value,
            'tur': LanguageCode.TURKISH.value,
            'tha': LanguageCode.THAI.value,
            'vie': LanguageCode.VIETNAMESE.value,
            'ben': LanguageCode.BENGALI.value,
            'guj': LanguageCode.GUJARATI.value,
            'pan': LanguageCode.PUNJABI.value,
            'tel': LanguageCode.TELUGU.value,
            'tam': LanguageCode.TAMIL.value,
            'kan': LanguageCode.KANNADA.value,
            'mal': LanguageCode.MALAYALAM.value,
            'mar': LanguageCode.MARATHI.value,
            'urd': LanguageCode.URDU.value,
            'fas': LanguageCode.PERSIAN.value,
            'heb': LanguageCode.HEBREW.value,
        }
        
        # PaddleOCR language mappings
        self._engine_mappings['paddle'] = {
            'en': LanguageCode.ENGLISH.value,
            'ch': LanguageCode.CHINESE_SIMPLIFIED.value,
            'chinese_cht': LanguageCode.CHINESE_TRADITIONAL.value,
            'japan': LanguageCode.JAPANESE.value,
            'korean': LanguageCode.KOREAN.value,
            'arabic': LanguageCode.ARABIC.value,
            'hi': LanguageCode.HINDI.value,
            'french': LanguageCode.FRENCH.value,
            'german': LanguageCode.GERMAN.value,
            'spanish': LanguageCode.SPANISH.value,
            'portuguese': LanguageCode.PORTUGUESE.value,
            'russian': LanguageCode.RUSSIAN.value,
            'italian': LanguageCode.ITALIAN.value,
            'thai': LanguageCode.THAI.value,
            'vietnamese': LanguageCode.VIETNAMESE.value,
            'bengali': LanguageCode.BENGALI.value,
            'gujarati': LanguageCode.GUJARATI.value,
            'punjabi': LanguageCode.PUNJABI.value,
            'telugu': LanguageCode.TELUGU.value,
            'tamil': LanguageCode.TAMIL.value,
            'kannada': LanguageCode.KANNADA.value,
            'malayalam': LanguageCode.MALAYALAM.value,
            'marathi': LanguageCode.MARATHI.value,
            'urdu': LanguageCode.URDU.value,
        }
        
        # EasyOCR language mappings
        self._engine_mappings['easyocr'] = {
            'en': LanguageCode.ENGLISH.value,
            'ch_sim': LanguageCode.CHINESE_SIMPLIFIED.value,
            'ch_tra': LanguageCode.CHINESE_TRADITIONAL.value,
            'ja': LanguageCode.JAPANESE.value,
            'ko': LanguageCode.KOREAN.value,
            'ar': LanguageCode.ARABIC.value,
            'hi': LanguageCode.HINDI.value,
            'fr': LanguageCode.FRENCH.value,
            'de': LanguageCode.GERMAN.value,
            'es': LanguageCode.SPANISH.value,
            'pt': LanguageCode.PORTUGUESE.value,
            'ru': LanguageCode.RUSSIAN.value,
            'it': LanguageCode.ITALIAN.value,
            'nl': LanguageCode.DUTCH.value,
            'sv': LanguageCode.SWEDISH.value,
            'no': LanguageCode.NORWEGIAN.value,
            'da': LanguageCode.DANISH.value,
            'fi': LanguageCode.FINNISH.value,
            'pl': LanguageCode.POLISH.value,
            'cs': LanguageCode.CZECH.value,
            'hu': LanguageCode.HUNGARIAN.value,
            'ro': LanguageCode.ROMANIAN.value,
            'bg': LanguageCode.BULGARIAN.value,
            'el': LanguageCode.GREEK.value,
            'tr': LanguageCode.TURKISH.value,
            'th': LanguageCode.THAI.value,
            'vi': LanguageCode.VIETNAMESE.value,
            'bn': LanguageCode.BENGALI.value,
            'gu': LanguageCode.GUJARATI.value,
            'pa': LanguageCode.PUNJABI.value,
            'te': LanguageCode.TELUGU.value,
            'ta': LanguageCode.TAMIL.value,
            'kn': LanguageCode.KANNADA.value,
            'ml': LanguageCode.MALAYALAM.value,
            'mr': LanguageCode.MARATHI.value,
            'ur': LanguageCode.URDU.value,
            'fa': LanguageCode.PERSIAN.value,
            'he': LanguageCode.HEBREW.value,
        }
        
        # Surya OCR language mappings
        self._engine_mappings['surya'] = {
            'en': LanguageCode.ENGLISH.value,
            'zh': LanguageCode.CHINESE.value,
            'ja': LanguageCode.JAPANESE.value,
            'ko': LanguageCode.KOREAN.value,
            'ar': LanguageCode.ARABIC.value,
            'hi': LanguageCode.HINDI.value,
            'fr': LanguageCode.FRENCH.value,
            'de': LanguageCode.GERMAN.value,
            'es': LanguageCode.SPANISH.value,
            'pt': LanguageCode.PORTUGUESE.value,
            'ru': LanguageCode.RUSSIAN.value,
            'it': LanguageCode.ITALIAN.value,
            'th': LanguageCode.THAI.value,
            'vi': LanguageCode.VIETNAMESE.value,
            'bn': LanguageCode.BENGALI.value,
            'ta': LanguageCode.TAMIL.value,
            'te': LanguageCode.TELUGU.value,
            'kn': LanguageCode.KANNADA.value,
            'ml': LanguageCode.MALAYALAM.value,
            'mr': LanguageCode.MARATHI.value,
            'gu': LanguageCode.GUJARATI.value,
            'pa': LanguageCode.PUNJABI.value,
            'ur': LanguageCode.URDU.value,
            'fa': LanguageCode.PERSIAN.value,
            'he': LanguageCode.HEBREW.value,
        }
    
    def register_engine_mapping(self, engine_name: str, mapping: Dict[str, str]) -> None:
        """Register a new engine's language mapping.
        
        Args:
            engine_name: Name of the OCR engine
            mapping: Dictionary mapping engine codes to standard codes
        """
        self._engine_mappings[engine_name] = mapping
    
    def to_standard(self, engine_name: str, engine_code: str) -> str:
        """Convert engine-specific language code to standard format.
        
        Args:
            engine_name: Name of the OCR engine
            engine_code: Engine-specific language code
            
        Returns:
            Standard ISO 639-1 language code
        """
        if engine_name not in self._engine_mappings:
            return engine_code
        
        mapping = self._engine_mappings[engine_name]
        return mapping.get(engine_code.lower(), engine_code)
    
    def from_standard(self, engine_name: str, standard_code: str) -> str:
        """Convert standard language code to engine-specific format.
        
        Args:
            engine_name: Name of the OCR engine
            standard_code: Standard ISO 639-1 language code
            
        Returns:
            Engine-specific language code
        """
        if engine_name not in self._engine_mappings:
            return standard_code
        
        mapping = self._engine_mappings[engine_name]
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping.get(standard_code.lower(), standard_code)
    
    def get_supported_languages(self, engine_name: str) -> List[str]:
        """Get list of supported languages for a specific engine.
        
        Args:
            engine_name: Name of the OCR engine
            
        Returns:
            List of standard language codes supported by the engine
        """
        if engine_name not in self._engine_mappings:
            return []
        
        return list(self._engine_mappings[engine_name].values())
    
    def get_engine_codes(self, engine_name: str) -> List[str]:
        """Get list of engine-specific language codes.
        
        Args:
            engine_name: Name of the OCR engine
            
        Returns:
            List of engine-specific language codes
        """
        if engine_name not in self._engine_mappings:
            return []
        
        return list(self._engine_mappings[engine_name].keys())
    
    def get_supported_engines(self) -> List[str]:
        """Get list of supported OCR engines."""
        return list(self._engine_mappings.keys())


class LanguageDetector:
    """Simple language detection utilities."""
    
    # Character ranges for different writing systems
    LANGUAGE_RANGES = {
        LanguageCode.CHINESE.value: [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Extension A
            (0x20000, 0x2A6DF),  # CJK Extension B
        ],
        LanguageCode.JAPANESE.value: [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FFF),  # Kanji (shared with Chinese)
        ],
        LanguageCode.KOREAN.value: [
            (0xAC00, 0xD7AF),  # Hangul Syllables
            (0x1100, 0x11FF),  # Hangul Jamo
        ],
        LanguageCode.ARABIC.value: [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
        ],
        LanguageCode.HINDI.value: [
            (0x0900, 0x097F),  # Devanagari
        ],
        LanguageCode.THAI.value: [
            (0x0E00, 0x0E7F),  # Thai
        ],
        LanguageCode.BENGALI.value: [
            (0x0980, 0x09FF),  # Bengali
        ],
        LanguageCode.TAMIL.value: [
            (0x0B80, 0x0BFF),  # Tamil
        ],
        LanguageCode.TELUGU.value: [
            (0x0C00, 0x0C7F),  # Telugu
        ],
        LanguageCode.KANNADA.value: [
            (0x0C80, 0x0CFF),  # Kannada
        ],
        LanguageCode.MALAYALAM.value: [
            (0x0D00, 0x0D7F),  # Malayalam
        ],
        LanguageCode.GUJARATI.value: [
            (0x0A80, 0x0AFF),  # Gujarati
        ],
        LanguageCode.PUNJABI.value: [
            (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
        ],
        LanguageCode.HEBREW.value: [
            (0x0590, 0x05FF),  # Hebrew
        ],
        LanguageCode.GREEK.value: [
            (0x0370, 0x03FF),  # Greek and Coptic
        ],
        LanguageCode.RUSSIAN.value: [
            (0x0400, 0x04FF),  # Cyrillic
        ],
    }
    
    @classmethod
    def detect_script(cls, text: str) -> Optional[str]:
        """Detect the primary script/language of the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language code or None if unable to detect
        """
        if not text:
            return None
        
        # Count characters for each language
        language_scores = {}
        
        for char in text:
            char_code = ord(char)
            for language, ranges in cls.LANGUAGE_RANGES.items():
                for start, end in ranges:
                    if start <= char_code <= end:
                        language_scores[language] = language_scores.get(language, 0) + 1
                        break
        
        if not language_scores:
            # Default to English for Latin script
            return LanguageCode.ENGLISH.value
        
        # Return language with highest score
        return max(language_scores, key=language_scores.get)
    
    @classmethod
    def is_mixed_script(cls, text: str, threshold: float = 0.1) -> bool:
        """Check if text contains mixed scripts.
        
        Args:
            text: Input text to analyze
            threshold: Minimum ratio for considering a script significant
            
        Returns:
            True if text contains multiple scripts above threshold
        """
        if not text:
            return False
        
        language_scores = {}
        total_chars = 0
        
        for char in text:
            if char.isalnum():  # Only count alphanumeric characters
                total_chars += 1
                char_code = ord(char)
                for language, ranges in cls.LANGUAGE_RANGES.items():
                    for start, end in ranges:
                        if start <= char_code <= end:
                            language_scores[language] = language_scores.get(language, 0) + 1
                            break
        
        if total_chars == 0:
            return False
        
        # Check how many languages exceed the threshold
        significant_languages = sum(
            1 for score in language_scores.values()
            if score / total_chars >= threshold
        )
        
        return significant_languages > 1


# Global instance for easy access
global_language_mapper = GlobalLanguageMapper()


def get_language_mapper() -> GlobalLanguageMapper:
    """Get the global language mapper instance."""
    return global_language_mapper


def detect_language(text: str) -> Optional[str]:
    """Convenience function to detect language from text."""
    return LanguageDetector.detect_script(text)


def is_supported_language(code: str) -> bool:
    """Check if a language code is supported by OmniDocs."""
    return LanguageCode.is_valid_code(code)


def get_all_supported_languages() -> List[str]:
    """Get all language codes supported by OmniDocs."""
    return LanguageCode.get_all_codes()
