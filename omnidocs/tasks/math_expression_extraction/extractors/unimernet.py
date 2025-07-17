#!/usr/bin/env python3
"""
UniMERNet (Universal Mathematical Expression Recognition Network) extractor for LaTeX expressions.
"""

# Copyright (c) OpenDataLab (https://github.com/opendatalab/UniMERNet)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import argparse
import logging
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import torch
from PIL import Image
import requests
import shutil

from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.math_expression_extraction.base import BaseLatexExtractor, BaseLatexMapper, LatexOutput

logger = get_logger(__name__)

class UniMERNetMapper(BaseLatexMapper):
    """Label mapper for UniMERNet model output."""
    
    def _setup_mapping(self):
        # UniMERNet outputs LaTeX directly, minimal mapping needed
        mapping = {
            r"\\": r"\\",    # Keep LaTeX backslashes
            r"\n": " ",      # Remove newlines
            r"  ": " ",      # Remove double spaces
        }
        self._mapping = mapping
        self._reverse_mapping = {v: k for k, v in mapping.items()}

class UniMERNetExtractor(BaseLatexExtractor):
    """UniMERNet (Universal Mathematical Expression Recognition Network) based expression extraction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        cfg_path: Optional[str] = None,
        device: Optional[str] = None,
        show_log: bool = False,
        **kwargs
    ):
        """Initialize UniMERNet Extractor."""
        super().__init__(device=device, show_log=show_log)
        
        self._label_mapper = UniMERNetMapper()
        
        # Set default paths
        if model_path is None:
            model_path = "omnidocs/models/unimernet_base"
        if cfg_path is None:
            cfg_path = str(Path(__file__).parent / "UniMERNet" / "configs" / "demo.yaml")
            
        self.model_path = Path(model_path)
        self.cfg_path = Path(cfg_path)
        
        # Check dependencies
        self._check_dependencies()
        
        # Download model if needed
        if not self.model_path.exists():
            self._download_model()
        
        try:
            self._load_model()
            if self.show_log:
                logger.success("UniMERNet model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize UniMERNet model", exc_info=True)
            raise
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            from PIL import Image
            # Check for UniMERNet modules
            unimernet_path = Path(__file__).parent / "UniMERNet" / "unimernet"
            if not unimernet_path.exists():
                raise ImportError("UniMERNet modules not found")
        except ImportError as e:
            logger.error("Failed to import required dependencies")
            raise ImportError(
                "Required dependencies not available. Please ensure UniMERNet modules are available."
            ) from e
    
    def _download_model(self) -> Path:
        """Download UniMERNet model from HuggingFace."""
        try:
            try:
                from huggingface_hub import snapshot_download

                logger.info(f"Downloading UniMERNet model to {self.model_path}")

                # Create model directory
                self.model_path.mkdir(parents=True, exist_ok=True)

                # Download from HuggingFace
                snapshot_download(
                    repo_id="wanderkid/unimernet_base",
                    local_dir=str(self.model_path),
                    local_dir_use_symlinks=False
                )

                logger.info("Model downloaded successfully")
                return self.model_path

            except ImportError:
                logger.warning("huggingface_hub not available, falling back to manual download")
                self._manual_download()
                return self.model_path

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Fallback: try manual download
            self._manual_download()
            return self.model_path
    
    def _manual_download(self):
        """Manual download fallback."""
        logger.info("Attempting manual download...")
        
        # HuggingFace file URLs
        base_url = "https://huggingface.co/wanderkid/unimernet_base/resolve/main"
        files_to_download = [
            "pytorch_model.pth",
            "config.json", 
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json"
        ]
        
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        for filename in files_to_download:
            url = f"{base_url}/{filename}"
            file_path = self.model_path / filename
            
            try:
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                    
                logger.info(f"Downloaded {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise
    
    def _load_model(self) -> None:
        """Load UniMERNet model and processor."""
        try:
            # Add UniMERNet path to sys.path
            unimernet_path = Path(__file__).parent / "UniMERNet"
            if str(unimernet_path) not in sys.path:
                sys.path.insert(0, str(unimernet_path))

            from unimernet.common.config import Config
            import unimernet.datasets.builders
            import unimernet.models
            import unimernet.processors
            import unimernet.tasks as tasks
            from unimernet.processors import load_processor
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load configuration
            args = argparse.Namespace(cfg_path=str(self.cfg_path), options=None)
            cfg = Config(args)
            
            # Update config with actual model path
            cfg.config.model.pretrained = str(self.model_path / "pytorch_model.pth")
            cfg.config.model.model_config.model_name = str(self.model_path)
            cfg.config.model.tokenizer_config.path = str(self.model_path)
            
            # Setup task and model
            task = tasks.setup_task(cfg)
            self.model = task.build_model(cfg).to(self.device)
            self.vis_processor = load_processor(
                'formula_image_eval', 
                cfg.config.datasets.formula_rec_eval.vis_processor.eval
            )
            
            if self.show_log:
                logger.info(f"Loaded UniMERNet model on {self.device}")
                
        except Exception as e:
            logger.error("Error loading UniMERNet model", exc_info=True)
            raise
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> LatexOutput:
        """Extract LaTeX expressions using UniMERNet."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            
            expressions = []
            for img in images:
                # Process image with UniMERNet
                image_tensor = self.vis_processor(img).unsqueeze(0).to(self.device)
                
                # Generate LaTeX
                with torch.no_grad():
                    output = self.model.generate({"image": image_tensor})
                    pred = output["pred_str"][0]
                
                # Map to standard format
                mapped_expr = self.map_expression(pred)
                expressions.append(mapped_expr)
            
            return LatexOutput(
                expressions=expressions,
                source_img_size=images[0].size if images else None
            )
            
        except Exception as e:
            logger.error("Error during UniMERNet extraction", exc_info=True)
            raise
