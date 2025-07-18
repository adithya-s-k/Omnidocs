#!/usr/bin/env python3
# Copyright (c) Facebook Research, Adapted by omnidocs
#repo: https://github.com/facebookresearch/nougat
#license: MIT License

import os
import sys
import json
import math
import time
import torch

import requests
import shutil
import tarfile
import zipfile
import hashlib
import argparse
import tempfile
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer
import re

import os
from pathlib import Path

# Set up model directory for HuggingFace downloads
def _setup_hf_model_dir():
    """Set up the model directory for HuggingFace to use omnidocs/models."""
    current_file = Path(__file__)
    omnidocs_root = current_file.parent.parent.parent.parent  # Go up to omnidocs root
    models_dir = omnidocs_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variables BEFORE any imports
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    
    return models_dir

_MODELS_DIR = _setup_hf_model_dir()

# Now do the other imports
import sys
import logging

# Import omnidocs modules
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput

logger = get_logger(__name__)

# Configuration - Using Hugging Face models
NOUGAT_CHECKPOINTS = {
    "base": {
        "hf_model": "facebook/nougat-base",
        "extract_dir": "nougat_ckpt"
    },
    "small": {
        "hf_model": "facebook/nougat-small",
        "extract_dir": "nougat_small_ckpt"
    }
}

# Model Constants
CONTEXT_SIZE = 4096
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
IMAGE_TOKEN = "<image>"

# ===================== Utility Functions =====================
def md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, target_path, expected_md5=None):
    """Download a file with progress bar and MD5 check"""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if target_path.exists():
        if expected_md5 and md5(target_path) == expected_md5:
            logger.info(f"File already exists and MD5 matches: {target_path}")
            return target_path
        logger.info(f"File exists but MD5 doesn't match. Re-downloading: {target_path}")
    
    logger.info(f"Downloading {url} to {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(target_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    
    if expected_md5:
        actual_md5 = md5(target_path)
        if actual_md5 != expected_md5:
            raise ValueError(f"MD5 mismatch for {target_path}: expected {expected_md5}, got {actual_md5}")
    
    return target_path

def extract_archive(archive_path, extract_dir):
    """Extract tar.gz or zip archive"""
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    if str(archive_path).endswith('.tar.gz'):
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=extract_path)
    elif str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    return extract_path

# ===================== Model Components =====================
@dataclass
class ModelConfig:
    """Configuration for the Nougat model"""
    vocab_size: int = 50280
    hidden_size: int = 1024
    encoder_hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 4098
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    encoder_layers: int = 12
    decoder_layers: int = 12
    patch_size: int = 16
    max_length: int = 4096
    max_patches: int = 4096
    width: int = 2560
    height: int = 2560
    encoder_name: str = "swin"
    embed_dim: int = 192
    depths: List[int] = None
    num_heads: List[int] = None
    window_size: int = 12
    patch_norm: bool = True
    

class NougatDecoder(nn.Module):
    """The Nougat decoder module"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize position embeddings
        self.max_positions = config.max_position_embeddings - 2  # Subtract 2 for BOS and EOS tokens
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
    def get_position_ids(self, input_ids):
        """Get position IDs for the decoder"""
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        return position_ids
        
    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Get position IDs and embeddings
        position_ids = self.get_position_ids(input_ids)
        inputs_embeds = self.embed_tokens(input_ids) 
        pos_embeds = self.position_embedding(position_ids)
        
        hidden_states = self.dropout(inputs_embeds + pos_embeds)
        
        # Stack output states and attentions if needed
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        
        next_decoder_cache = () if use_cache else None
        
        # Decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=None if past_key_values is None else past_key_values[idx],
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the final layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
            
        return Seq2SeqLMOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            decoder_hidden_states=all_hidden_states,
            decoder_attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
        )

class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention"""
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_probs_dropout_prob)
        self.cross_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_probs_dropout_prob)
        
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        # Self-Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Convert shapes for MultiheadAttention (seq_len, batch, dim)
        hidden_states_transformed = hidden_states.transpose(0, 1)
        self_attn_output, self_attn_weights = self.self_attn(
            hidden_states_transformed, 
            hidden_states_transformed, 
            hidden_states_transformed,
            attn_mask=None,  # attention_mask, 
            need_weights=output_attentions
        )
        self_attn_output = self_attn_output.transpose(0, 1)
        hidden_states = residual + self.dropout(self_attn_output)
        
        # Cross-Attention
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            
            # Convert shapes for MultiheadAttention
            hidden_states_transformed = hidden_states.transpose(0, 1)
            encoder_hidden_states_transformed = encoder_hidden_states.transpose(0, 1)
            
            cross_attn_output, cross_attn_weights = self.cross_attn(
                query=hidden_states_transformed,
                key=encoder_hidden_states_transformed,
                value=encoder_hidden_states_transformed,
                attn_mask=None,  # encoder_attention_mask
                need_weights=output_attentions
            )
            cross_attn_output = cross_attn_output.transpose(0, 1)
            hidden_states = residual + self.dropout(cross_attn_output)
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        
        if use_cache:
            outputs += (None,)  # Cache is not implemented 
        return outputs


class NougatModel(nn.Module):
    """The complete Nougat model with encoder and decoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize encoder based on configuration
        if config.encoder_name == "swin":
            self.encoder = SwinTransformer(
                img_size=(config.height, config.width),
                patch_size=config.patch_size,
                in_chans=3,
                embed_dim=config.embed_dim,
                depths=config.depths or [2, 2, 18, 2],
                num_heads=config.num_heads or [6, 12, 24, 48],
                window_size=config.window_size,
                patch_norm=config.patch_norm,
            )
        else:
            self.encoder = VisionTransformer(
                img_size=(config.height, config.width),
                patch_size=config.patch_size,
                in_chans=3,
                embed_dim=config.encoder_hidden_size,
                depth=config.encoder_layers,
                num_heads=config.num_attention_heads,
            )
        
        # Initialize decoder
        self.decoder = NougatDecoder(config)
        
        # Project encoder output to decoder input dimension if needed
        if config.encoder_name == "swin":
            self.encoder_proj = nn.Linear(self.encoder.num_features, config.hidden_size)
        else:
            if config.encoder_hidden_size != config.hidden_size:
                self.encoder_proj = nn.Linear(config.encoder_hidden_size, config.hidden_size)
            else:
                self.encoder_proj = nn.Identity()
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def get_encoder(self):
        return self.encoder
        
    def get_decoder(self):
        return self.decoder
        
    def get_output_embeddings(self):
        return self.lm_head
        
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Encode if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder.forward_features(pixel_values)
            
        # Project encoder outputs to decoder dimension
        if self.config.encoder_name == "swin":
            # For Swin, reshape from (B, H*W, C) to (B, H*W, hidden_size)
            encoder_hidden_states = self.encoder_proj(encoder_outputs)
        else:
            # For Vision Transformer
            encoder_hidden_states = self.encoder_proj(encoder_outputs)
            
        # Decode
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=None,  # No encoder attention mask for images
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs[0]
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            # Calculate loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shifted_logits.view(-1, self.config.vocab_size), 
                shifted_labels.view(-1)
            )
            
        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return (loss,) + output if loss is not None else output
            
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
        )


# ===================== Model API =====================
class Nougat:
    """Main Nougat API for document understanding"""
    def __init__(
        self, 
        model_dir="omnidocs/models", 
        model_type="base",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.device = device
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download and set up the model
        self.setup_model()
        
    def setup_model(self):
        """Download and set up the model and tokenizer using Hugging Face"""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel

            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]

            logger.info(f"Loading Nougat model from Hugging Face: {hf_model_name}")

            # Load processor and model from Hugging Face
            self.processor = NougatProcessor.from_pretrained(hf_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model from Hugging Face: {e}")
            raise
        
    def preprocess_image(self, image_path):
        """Preprocess the input image"""
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
            
        # Resize to model's expected dimensions
        transform = transforms.Compose([
            transforms.Resize((self.model.config.height, self.model.config.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        pixel_values = transform(image).unsqueeze(0)
        return pixel_values.to(self.device)
    
    @torch.no_grad()
    @log_execution_time
    def generate(
        self,
        image_path,
        max_length=512,
        num_beams=4,
        temperature=1.0,
        no_repeat_ngram_size=3
    ):
        """Generate text from document image using Hugging Face model"""
        try:
            # Load and preprocess image
            from PIL import Image, ImageOps
            image = Image.open(image_path).convert('RGB')

            # Add padding to make it look more like a document page (helps with math recognition)
            padded_image = ImageOps.expand(image, border=100, fill='white')

            # Process image with Hugging Face processor
            pixel_values = self.processor(padded_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text using the model with optimized parameters for math
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=max(1, num_beams),  # Ensure at least 1 beam
                do_sample=False,
                early_stopping=False if num_beams == 1 else True  # Only use early_stopping with beam search
            )

            # Decode the generated text
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            return generated_text

        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise

        
    def __call__(self, image_path, **kwargs):
        """Convenience method to process an image"""
        return self.generate(image_path, **kwargs)


# ===================== Main Application =====================
@log_execution_time
def process_document(image_path, output_path=None, model_type="small"):
    """Process a document image and generate text"""
    logger.info(f"Processing document: {image_path}")
    
    # Initialize Nougat model
    nougat = Nougat(model_type=model_type)
    
    # Generate text from document
    generated_text = nougat(image_path)
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_text)
        logger.info(f"Output saved to: {output_path}")
    
    return generated_text


def process_pdf(pdf_path, output_dir=None, model_type="base"):
    """Process a PDF file and extract text from each page"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
        return None
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Initialize Nougat model
    nougat = Nougat(model_type=model_type)
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    document = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(document)):
        logger.info(f"Processing page {page_num+1}/{len(document)}")
        
        # Get page as image
        page = document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Process image
        text = nougat(img)
        results.append(text)
        
        # Save individual page if output_dir is provided
        if output_dir:
            output_path = os.path.join(output_dir, f"page_{page_num+1:03d}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
    
    # Combine all results
    combined_text = "\n\n".join(results)
    
    # Save combined text if output_dir is provided
    if output_dir:
        combined_output_path = os.path.join(output_dir, "combined_output.txt")
        with open(combined_output_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
    
    return combined_text


# ===================== BaseLatexExtractor Implementation =====================
class NougatMapper(BaseLatexMapper):
    """Label mapper for Nougat model output."""

    def _setup_mapping(self):
        # Nougat outputs markdown/LaTeX, minimal mapping needed
        mapping = {
            r"\\": r"\\",    # Keep LaTeX backslashes
            r"\n": " ",      # Remove newlines for single expressions
            r"  ": " ",      # Remove double spaces
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class NougatExtractor(BaseLatexExtractor):
    """Nougat (Neural Optical Understanding for Academic Documents) based expression extraction."""

    def __init__(
        self,
        model_type: str = "small",
        device: Optional[str] = None,
        show_log: bool = False,
        **kwargs
    ):
        """Initialize Nougat Extractor."""
        super().__init__(device=device, show_log=show_log)

        self._label_mapper = NougatMapper()
        self.model_type = model_type

        # Check dependencies
        self._check_dependencies()

        try:
            self._load_model()
            if self.show_log:
                logger.success("Nougat model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Nougat model", exc_info=True)
            raise

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            from PIL import Image
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please install with: "
                "pip install transformers torch torchvision"
            ) from e

    def _download_model(self) -> Path:
        """Model download handled by transformers library."""
        logger.info("Model downloading handled by transformers library")
        return None

    def _load_model(self) -> None:
        """Load Nougat model and processor."""
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel

            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Get model name from checkpoint config
            checkpoint_info = NOUGAT_CHECKPOINTS[self.model_type]
            hf_model_name = checkpoint_info["hf_model"]

            logger.info(f"Loading Nougat model from Hugging Face: {hf_model_name}")
            logger.info(f"Models will be downloaded in: {_MODELS_DIR}")

            # Load processor and model
            self.processor = NougatProcessor.from_pretrained(hf_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
            self.model.to(self.device)
            self.model.eval()

            if self.show_log:
                logger.info(f"Loaded Nougat model on {self.device}")

        except Exception as e:
            logger.error("Error loading Nougat model", exc_info=True)
            raise

    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from Nougat's markdown output."""
        import re

        expressions = []

        # Find inline math expressions (between $ ... $)
        inline_math = re.findall(r'\$([^$]+)\$', text)
        expressions.extend(inline_math)

        # Find display math expressions (between $$ ... $$)
        display_math = re.findall(r'\$\$([^$]+)\$\$', text)
        expressions.extend(display_math)

        # Find LaTeX environments
        latex_envs = re.findall(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', text, re.DOTALL)
        for env_name, content in latex_envs:
            if env_name in ['equation', 'align', 'gather', 'multline', 'eqnarray']:
                expressions.append(content.strip())

        # If no specific math found, return the whole text (might contain math)
        if not expressions:
            expressions = [text.strip()]

        return expressions

    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using Nougat."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)

            all_expressions = []
            for img in images:
                # Add padding to make it look more like a document page
                from PIL import ImageOps
                padded_image = ImageOps.expand(img, border=100, fill='white')

                # Process image with Nougat processor
                pixel_values = self.processor(padded_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)

                # Generate text using the model
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        max_length=512,
                        num_beams=1,  # Use greedy decoding for faster inference
                        do_sample=False,
                        early_stopping=False
                    )

                # Decode the generated text
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

                # Extract mathematical expressions from the text
                expressions = self._extract_math_expressions(generated_text)

                # Map expressions to standard format
                mapped_expressions = [self.map_expression(expr) for expr in expressions]
                all_expressions.extend(mapped_expressions)

            return LatexOutput(
                expressions=all_expressions,
                source_img_size=images[0].size if images else None
            )

        except Exception as e:
            logger.error("Error during Nougat extraction", exc_info=True)
            raise
