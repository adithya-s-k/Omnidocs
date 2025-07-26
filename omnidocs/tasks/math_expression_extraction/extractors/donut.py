from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import torch
from PIL import Image
import json
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput
from omnidocs.utils.model_config import setup_model_environment

logger = get_logger(__name__)


# Setup model environment
_MODELS_DIR = setup_model_environment()

class DonutMapper(BaseLatexMapper):
    """Label mapper for Donut model output."""
    
    def _setup_mapping(self):
        # Donut outputs JSON, extract math content
        mapping = {
            r"\n": " ",     # Remove newlines
            r"  ": " ",     # Remove double spaces
            r"\\": r"\\",    # Fix escaped backslashes
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class DonutExtractor(BaseLatexExtractor):
    """Donut (NAVER CLOVA) based expression extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        model_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """Initialize Donut Extractor."""
        super().__init__(device=device, show_log=show_log)
        
        self._label_mapper = DonutMapper()
        self.model_name = model_name
        
        # Set default paths
        if model_path is None:
            model_path = _MODELS_DIR / "donut_models" / model_name.replace("/", "_")
        
        self.model_path = Path(model_path)
        
        # Check dependencies
        self._check_dependencies()
        
        # Download model if needed
        if not self._model_exists():
            if self.show_log:
                logger.info(f"Model not found at {self.model_path}, will download from HuggingFace")
            self._download_model()
        
        try:
            self._load_model()
            if self.show_log:
                logger.success("Donut model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Donut model", exc_info=True)
            raise
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import transformers
            import torch
            import json
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install transformers torch torchvision"
            ) from e
    
    def _model_exists(self) -> bool:
        """Check if model files exist locally."""
        # Check for key model files that indicate successful download
        required_files = [
            "config.json",
            "pytorch_model.bin",  # or model.safetensors
            "preprocessor_config.json"
        ]
        
        # Also check for safetensors format
        safetensors_file = self.model_path / "model.safetensors"
        pytorch_bin_file = self.model_path / "pytorch_model.bin"
        
        if self.model_path.exists():
            has_config = (self.model_path / "config.json").exists()
            has_preprocessor = (self.model_path / "preprocessor_config.json").exists()
            has_model_file = safetensors_file.exists() or pytorch_bin_file.exists()
            
            return has_config and has_preprocessor and has_model_file
        
        return False
    
    def _download_model(self) -> Path:
        """Download model from HuggingFace if it doesn't exist locally."""
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            
            logger.info(f"Downloading Donut model: {self.model_name}")
            logger.info(f"Saving to: {self.model_path}")
            
            # Create model directory
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Download and save processor
            if self.show_log:
                logger.info("Downloading processor...")
            processor = DonutProcessor.from_pretrained(self.model_name)
            processor.save_pretrained(self.model_path)
            
            # Download and save model
            if self.show_log:
                logger.info("Downloading model...")
            model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            model.save_pretrained(self.model_path)
            
            if self.show_log:
                logger.success(f"Model downloaded successfully to {self.model_path}")
            
            return self.model_path
            
        except Exception as e:
            logger.error("Error downloading Donut model", exc_info=True)
            # Clean up partial download
            if self.model_path.exists():
                import shutil
                shutil.rmtree(self.model_path)
            raise
    
    def _load_model(self) -> None:
        """Load Donut model and processor."""
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try to load from local path first, fallback to HuggingFace
            try:
                if self._model_exists():
                    logger.info(f"Loading Donut model from local path: {self.model_path}")
                    model_source = str(self.model_path)
                else:
                    logger.info(f"Loading Donut model from HuggingFace: {self.model_name}")
                    model_source = self.model_name
                
                # Load processor and model
                self.processor = DonutProcessor.from_pretrained(model_source)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_source)
                self.model.to(self.device)
                self.model.eval()
                
                if self.show_log:
                    logger.info(f"Loaded Donut model on {self.device}")
                    
            except Exception as local_error:
                logger.warning(f"Failed to load from local path, trying HuggingFace: {local_error}")
                # Fallback to direct HuggingFace loading
                self.processor = DonutProcessor.from_pretrained(self.model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            logger.error("Error loading Donut model", exc_info=True)
            raise
    
    def _extract_math_from_json(self, json_str: str) -> str:
        """Extract mathematical content from Donut's JSON output."""
        try:
            # Try to parse as JSON
            data = json.loads(json_str)
            
            # Look for math-related fields
            math_content = ""
            
            if isinstance(data, dict):
                # Common field names that might contain math
                math_fields = ['text', 'content', 'formula', 'equation', 'math']
                for field in math_fields:
                    if field in data:
                        math_content += str(data[field]) + " "
                
                # If no specific fields, concatenate all string values
                if not math_content.strip():
                    for value in data.values():
                        if isinstance(value, str):
                            math_content += value + " "
            
            return math_content.strip()
            
        except json.JSONDecodeError:
            # If not valid JSON, return as is
            return json_str
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using Donut."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            
            expressions = []
            for img in images:
                # Prepare image for Donut
                pixel_values = self.processor(img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Prepare task prompt (adjust based on your specific task)
                task_prompt = "<s_cord-v2>"  # Default CORD v2 task(this is used for receipt/invoice parsing)
                decoder_input_ids = self.processor.tokenizer(
                    task_prompt, 
                    add_special_tokens=False, 
                    return_tensors="pt" #returns pytorch tensor 
                ).input_ids
                decoder_input_ids = decoder_input_ids.to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.model.decoder.config.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )
                
                # Decode output
                #converts the generated token IDs back into a string
                sequence = self.processor.batch_decode(outputs.sequences)[0]
                #removes any pos and eos 
                sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
                #removes task prompt 
                sequence = sequence.replace(task_prompt, "")
                
                # Extract math content from JSON-like output
                math_content = self._extract_math_from_json(sequence)
                
                # Map to standard format
                mapped_expr = self.map_expression(math_content)
                expressions.append(mapped_expr)
            
            return LatexOutput(
                expressions=expressions,
                source_img_size=images[0].size if images else None
            )
            
        except Exception as e:
            logger.error("Error during Donut extraction", exc_info=True)
            raise