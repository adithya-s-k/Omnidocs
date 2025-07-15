import time
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import cv2
import torch
from omnidocs.utils.logging import get_logger, log_execution_time
from omnidocs.tasks.table_extraction.base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell

logger = get_logger(__name__)

class TableFormerMapper(BaseTableMapper):
    """Label mapper for TableFormer model output."""
    
    def __init__(self):
        super().__init__('tableformer')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup model and class mappings for TableFormer."""
        self._model_configs = {
            'detection': {
                'model_name': 'microsoft/table-transformer-detection',
                'confidence_threshold': 0.7,
                'classes': ['table']
            },
            'structure': {
                'model_name': 'microsoft/table-structure-recognition-v1.1-all',
                'confidence_threshold': 0.7,
                'classes': ['table', 'table column', 'table row', 'table column header', 
                          'table projected row header', 'table spanning cell']
            }
        }
        
        self._class_mapping = {
            'table': 'table',
            'table column': 'column',
            'table row': 'row',
            'table column header': 'column_header',
            'table projected row header': 'row_header',
            'table spanning cell': 'spanning_cell'
        }

class TableFormerExtractor(BaseTableExtractor):
    """TableFormer based table extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.7,
        max_size: int = 1000,
        **kwargs
    ):
        """Initialize TableFormer Extractor."""
        super().__init__(
            device=device,
            show_log=show_log,
            engine_name='tableformer'
        )
        
        self._label_mapper = TableFormerMapper()
        self.model_name = model_name or "microsoft/table-structure-recognition-v1.1-all"
        self.confidence_threshold = confidence_threshold
        self.max_size = max_size
        
        try:
            from transformers import DetrImageProcessor, TableTransformerForObjectDetection
            import torch
            self.processor_class = DetrImageProcessor
            self.model_class = TableTransformerForObjectDetection
            self.torch = torch
            
        except ImportError as e:
            logger.error("Failed to import TableFormer dependencies")
            raise ImportError(
                "TableFormer dependencies not available. Please install with: pip install transformers torch"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        TableFormer models are downloaded automatically by transformers library.
        This method is required by the abstract base class.
        """
        if self.show_log:
            logger.info("TableFormer models will be downloaded automatically by transformers library")
        return None
    
    def _load_model(self) -> None:
        """Load TableFormer model."""
        try:
            if self.show_log:
                logger.info(f"Loading TableFormer model: {self.model_name}")
            
            # Load processor and model with proper size configuration
            self.processor = self.processor_class.from_pretrained(
                self.model_name,
                size={"shortest_edge": 800, "longest_edge": 1333}  # Fix the size configuration
            )
            self.model = self.model_class.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.show_log:
                logger.info("TableFormer model loaded successfully")
                
        except Exception as e:
            logger.error("Failed to load TableFormer model", exc_info=True)
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for TableFormer."""
        # Resize image if too large
        width, height = image.size
        if max(width, height) > self.max_size:
            if width > height:
                new_width = self.max_size
                new_height = int(height * self.max_size / width)
            else:
                new_height = self.max_size
                new_width = int(width * self.max_size / height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _detect_table_structure(self, image: Image.Image) -> List[Dict]:
        """Detect table structure using TableFormer."""
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Prepare inputs
        inputs = self.processor(images=processed_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([processed_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
        )[0]
        
        # Convert to detection format
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > self.confidence_threshold:
                # Scale back to original image size if resized
                if processed_image.size != image.size:
                    scale_x = image.size[0] / processed_image.size[0]
                    scale_y = image.size[1] / processed_image.size[1]
                    box = [
                        box[0] * scale_x,
                        box[1] * scale_y,
                        box[2] * scale_x,
                        box[3] * scale_y
                    ]
                
                detections.append({
                    "bbox": box.cpu().numpy().tolist() if hasattr(box, 'cpu') else box,
                    "confidence": score.cpu().item() if hasattr(score, 'cpu') else score,
                    "label": self.model.config.id2label[label.item()] if hasattr(label, 'item') else label
                })
        
        return detections
    
    def _create_table_from_detections(self, detections: List[Dict], img_size: Tuple[int, int]) -> Table:
        """Create table structure from detections."""
        # Group detections by type
        rows = [d for d in detections if 'row' in d['label']]
        columns = [d for d in detections if 'column' in d['label']]
        headers = [d for d in detections if 'header' in d['label']]
        
        # Sort rows and columns by position
        rows.sort(key=lambda x: x['bbox'][1])  # Sort by y-coordinate
        columns.sort(key=lambda x: x['bbox'][0])  # Sort by x-coordinate
        
        # Create cells based on row-column intersections
        cells = []
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(columns):
                # Check if row and column intersect
                if self._boxes_intersect(row['bbox'], col['bbox']):
                    # Get intersection bbox
                    cell_bbox = self._get_intersection_bbox(row['bbox'], col['bbox'])
                    
                    # Determine if cell is header
                    is_header = any(self._boxes_intersect(cell_bbox, h['bbox']) for h in headers)
                    
                    # Create cell (text would be extracted by OCR in real implementation)
                    cell_text = f"Cell_{row_idx}_{col_idx}"
                    
                    cell = TableCell(
                        text=cell_text,
                        row=row_idx,
                        col=col_idx,
                        rowspan=1,
                        colspan=1,
                        bbox=cell_bbox,
                        confidence=min(row['confidence'], col['confidence']),
                        is_header=is_header
                    )
                    cells.append(cell)
        
        # Calculate table dimensions
        num_rows = len(rows)
        num_cols = len(columns)
        
        # Get table bbox (encompassing all elements)
        if detections:
            all_bboxes = [d['bbox'] for d in detections]
            min_x = min(bbox[0] for bbox in all_bboxes)
            min_y = min(bbox[1] for bbox in all_bboxes)
            max_x = max(bbox[2] for bbox in all_bboxes)
            max_y = max(bbox[3] for bbox in all_bboxes)
            table_bbox = [min_x, min_y, max_x, max_y]
        else:
            table_bbox = None
        
        # Calculate overall confidence
        overall_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.0
        
        return Table(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            bbox=table_bbox,
            confidence=overall_confidence,
            table_id="table_0",
            structure_confidence=overall_confidence
        )
    
    def _boxes_intersect(self, box1: List[float], box2: List[float]) -> bool:
        """Check if two bounding boxes intersect."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _get_intersection_bbox(self, box1: List[float], box2: List[float]) -> List[float]:
        """Get intersection bounding box of two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)
        
        return [x1, y1, x2, y2]
    
    def postprocess_output(self, raw_output: Dict, img_size: Tuple[int, int]) -> TableOutput:
        """Convert TableFormer output to standardized TableOutput format."""
        tables = []
        
        # Extract table from detections
        detections = raw_output.get('detections', [])
        if detections:
            table = self._create_table_from_detections(detections, img_size)
            tables.append(table)
        
        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={
                'engine': 'tableformer',
                'model_name': self.model_name,
                'confidence_threshold': self.confidence_threshold,
                'max_size': self.max_size
            }
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using TableFormer."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            image = images[0]
            img_size = image.size
            
            # Detect table structure
            detections = self._detect_table_structure(image)
            
            if not detections:
                if self.show_log:
                    logger.info("No table structure detected in the image")
                return TableOutput(
                    tables=[],
                    source_img_size=img_size,
                    metadata={'engine': 'tableformer', 'message': 'No table structure detected'}
                )
            
            # Convert to standardized format
            result = self.postprocess_output({'detections': detections}, img_size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.tables)} tables using TableFormer")
            
            return result
            
        except Exception as e:
            logger.error("Error during TableFormer extraction", exc_info=True)
            return TableOutput(
                tables=[],
                source_img_size=None,
                processing_time=None,
                metadata={"error": str(e)}
            )
    
    def predict(self, input_path: Union[str, Path, Image.Image], **kwargs):
        """Predict method for compatibility with original interface."""
        try:
            result = self.extract(input_path, **kwargs)
            
            # Convert to original format
            table_res = []
            for table in result.tables:
                table_data = {
                    "table_id": table.table_id,
                    "bbox": table.bbox,
                    "confidence": table.confidence,
                    "cells": [cell.to_dict() for cell in table.cells],
                    "num_rows": table.num_rows,
                    "num_cols": table.num_cols
                }
                table_res.append(table_data)
            
            return table_res
            
        except Exception as e:
            logger.error("Error during TableFormer prediction", exc_info=True)
            return []