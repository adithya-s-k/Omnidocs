"""
Tests for OCR extraction configuration classes.
"""

import pytest
from pydantic import ValidationError

from omnidocs.tasks.ocr_extraction import (
    EasyOCRConfig,
    PaddleOCRConfig,
    TesseractOCRConfig,
)


class TestTesseractOCRConfig:
    """Tests for TesseractOCRConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TesseractOCRConfig()
        assert config.languages == ["eng"]
        assert config.tessdata_dir is None
        assert config.oem == 3
        assert config.psm == 3
        assert config.config_params is None

    def test_custom_languages(self):
        """Test setting custom languages."""
        config = TesseractOCRConfig(languages=["eng", "fra", "deu"])
        assert config.languages == ["eng", "fra", "deu"]

    def test_custom_oem(self):
        """Test setting OCR Engine Mode."""
        config = TesseractOCRConfig(oem=1)  # LSTM only
        assert config.oem == 1

    def test_custom_psm(self):
        """Test setting Page Segmentation Mode."""
        config = TesseractOCRConfig(psm=6)  # Uniform block
        assert config.psm == 6

    def test_custom_tessdata_dir(self):
        """Test setting tessdata directory."""
        config = TesseractOCRConfig(tessdata_dir="/custom/tessdata")
        assert config.tessdata_dir == "/custom/tessdata"

    def test_config_params(self):
        """Test setting additional config parameters."""
        config = TesseractOCRConfig(config_params={"tessedit_char_whitelist": "0123456789"})
        assert config.config_params == {"tessedit_char_whitelist": "0123456789"}

    def test_oem_validation(self):
        """Test OEM range validation."""
        with pytest.raises(ValidationError):
            TesseractOCRConfig(oem=-1)
        with pytest.raises(ValidationError):
            TesseractOCRConfig(oem=4)

    def test_psm_validation(self):
        """Test PSM range validation."""
        with pytest.raises(ValidationError):
            TesseractOCRConfig(psm=-1)
        with pytest.raises(ValidationError):
            TesseractOCRConfig(psm=14)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            TesseractOCRConfig(unknown_field="value")


class TestEasyOCRConfig:
    """Tests for EasyOCRConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EasyOCRConfig()
        assert config.languages == ["en"]
        assert config.gpu is True
        assert config.model_storage_directory is None
        assert config.download_enabled is True
        assert config.detector is True
        assert config.recognizer is True
        assert config.quantize is True

    def test_custom_languages(self):
        """Test setting custom languages."""
        config = EasyOCRConfig(languages=["en", "ch_sim", "ja"])
        assert config.languages == ["en", "ch_sim", "ja"]

    def test_gpu_disabled(self):
        """Test disabling GPU."""
        config = EasyOCRConfig(gpu=False)
        assert config.gpu is False

    def test_custom_model_directory(self):
        """Test setting custom model directory."""
        config = EasyOCRConfig(model_storage_directory="/custom/models")
        assert config.model_storage_directory == "/custom/models"

    def test_download_disabled(self):
        """Test disabling automatic downloads."""
        config = EasyOCRConfig(download_enabled=False)
        assert config.download_enabled is False

    def test_detector_disabled(self):
        """Test disabling text detection."""
        config = EasyOCRConfig(detector=False)
        assert config.detector is False

    def test_quantize_disabled(self):
        """Test disabling quantization."""
        config = EasyOCRConfig(quantize=False)
        assert config.quantize is False

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            EasyOCRConfig(unknown_field="value")


class TestPaddleOCRConfig:
    """Tests for PaddleOCRConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PaddleOCRConfig()
        assert config.lang == "en"
        assert config.device == "cpu"

    def test_custom_language(self):
        """Test setting custom language."""
        config = PaddleOCRConfig(lang="ch")
        assert config.lang == "ch"

    def test_gpu_device(self):
        """Test setting GPU device."""
        config = PaddleOCRConfig(device="gpu")
        assert config.device == "gpu"

    def test_japanese_language(self):
        """Test Japanese language setting."""
        config = PaddleOCRConfig(lang="japan")
        assert config.lang == "japan"

    def test_korean_language(self):
        """Test Korean language setting."""
        config = PaddleOCRConfig(lang="korean")
        assert config.lang == "korean"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            PaddleOCRConfig(unknown_field="value")


class TestConfigImmutability:
    """Tests for config immutability (extra=forbid)."""

    def test_tesseract_config_forbids_extra(self):
        """Test TesseractOCRConfig forbids extra fields."""
        with pytest.raises(ValidationError):
            TesseractOCRConfig(invalid_param=True)

    def test_easyocr_config_forbids_extra(self):
        """Test EasyOCRConfig forbids extra fields."""
        with pytest.raises(ValidationError):
            EasyOCRConfig(invalid_param=True)

    def test_paddleocr_config_forbids_extra(self):
        """Test PaddleOCRConfig forbids extra fields."""
        with pytest.raises(ValidationError):
            PaddleOCRConfig(invalid_param=True)
