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

class TableTransformerMapper(BaseTableMapper):
    """Label mapper for Table Transformer model output."""
    
    def __init__(self):
        super().__init__('table_transformer')
        self._setup_mapping()
    
    def _setup_mapping(self):
        """Setup model and class mappings for Table Transformer."""
        self._model_configs = {
            'detection': {
                'model_name': 'microsoft/table-transformer-detection',
                'confidence_threshold': 0.7,
                'classes': ['table']
            },
            'structure': {
                'model_name': 'microsoft/table-transformer-structure-recognition',
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

class TableTransformerExtractor(BaseTableExtractor):
    """Table Transformer based table extraction implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        detection_model: Optional[str] = None,
        structure_model: Optional[str] = None,
        detection_threshold: float = 0.7,
        structure_threshold: float = 0.7,
        **kwargs
    ):
        """Initialize Table Transformer Extractor."""
        super().__init__(
            device=device,
            show_log=show_log,
            engine_name='table_transformer'
        )
        
        self._label_mapper = TableTransformerMapper()
        self.detection_model_name = detection_model or self._label_mapper._model_configs['detection']['model_name']
        self.structure_model_name = structure_model or self._label_mapper._model_configs['structure']['model_name']
        self.detection_threshold = detection_threshold
        self.structure_threshold = structure_threshold
        
        try:
            from transformers import DetrImageProcessor, TableTransformerForObjectDetection
            self.processor_class = DetrImageProcessor
            self.model_class = TableTransformerForObjectDetection
            
        except ImportError as e:
            logger.error("Failed to import Table Transformer dependencies")
            raise ImportError(
                "Table Transformer dependencies not available. Please install with: pip install transformers torch"
            ) from e
        
        self._load_model()
    
    def _download_model(self) -> Optional[Path]:
        """
        Table Transformer models are downloaded automatically by transformers library.
        This method is required by the abstract base class.
        """
        if self.show_log:
            logger.info("Table Transformer models will be downloaded automatically by transformers library")
        return None
    
    def _load_model(self) -> None:
        """Load Table Transformer models."""
        try:
            if self.show_log:
                logger.info("Loading Table Transformer detection model...")
            
            # Load detection model
            self.detection_processor = self.processor_class.from_pretrained(self.detection_model_name)
            self.detection_model = self.model_class.from_pretrained(self.detection_model_name)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            
            if self.show_log:
                logger.info("Loading Table Transformer structure model...")
            
            # Load structure model
            self.structure_processor = self.processor_class.from_pretrained(self.structure_model_name)
            self.structure_model = self.model_class.from_pretrained(self.structure_model_name)
            self.structure_model.to(self.device)
            self.structure_model.eval()
            
            if self.show_log:
                logger.info("Table Transformer models loaded successfully")
                
        except Exception as e:
            logger.error("Failed to load Table Transformer models", exc_info=True)
            raise
    
    def _detect_tables(self, image: Image.Image) -> List[Dict]:
        """Detect tables in the image."""
        inputs = self.detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Convert outputs to COCO format
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detection_processor.post_process_object_detection(
            outputs, threshold=self.detection_threshold, target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > self.detection_threshold:
                tables.append({
                    "bbox": box.cpu().numpy().tolist(),
                    "confidence": score.cpu().item(),
                    "label": "table"
                })
        
        return tables
    
    def _analyze_table_structure(self, image: Image.Image, table_bbox: List[float]) -> Dict:
        """Analyze table structure within detected table region."""
        # Crop table region
        x1, y1, x2, y2 = table_bbox
        cropped_image = image.crop((x1, y1, x2, y2))
        
        inputs = self.structure_processor(images=cropped_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        
        # Convert outputs to COCO format
        target_sizes = torch.tensor([cropped_image.size[::-1]])
        results = self.structure_processor.post_process_object_detection(
            outputs, threshold=self.structure_threshold, target_sizes=target_sizes
        )[0]
        
        structure_elements = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > self.structure_threshold:
                # Convert box coordinates back to original image coordinates
                rel_box = box.cpu().numpy()
                abs_box = [
                    rel_box[0] + x1,
                    rel_box[1] + y1,
                    rel_box[2] + x1,
                    rel_box[3] + y1
                ]
                
                structure_elements.append({
                    "bbox": abs_box,
                    "confidence": score.cpu().item(),
                    "label": self.structure_model.config.id2label[label.item()]
                })
        
        return {
            "elements": structure_elements,
            "table_bbox": table_bbox
        }
    
    def _create_table_cells(self, structure_data: Dict) -> List[TableCell]:
        """Create table cells from structure analysis."""
        elements = structure_data["elements"]
        cells = []
        
        # Group elements by type
        rows = [e for e in elements if 'row' in e['label']]
        columns = [e for e in elements if 'column' in e['label']]
        headers = [e for e in elements if 'header' in e['label']]
        
        # Sort rows and columns by position
        rows.sort(key=lambda x: x['bbox'][1])  # Sort by y-coordinate
        columns.sort(key=lambda x: x['bbox'][0])  # Sort by x-coordinate
        
        # Create cells based on row-column intersections
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(columns):
                # Check if row and column intersect
                if self._boxes_intersect(row['bbox'], col['bbox']):
                    # Determine cell text (simplified - in real implementation, use OCR)
                    cell_text = f"Cell_{row_idx}_{col_idx}"
                    
                    # Check if cell is header
                    is_header = any(self._boxes_intersect(row['bbox'], h['bbox']) or 
                                  self._boxes_intersect(col['bbox'], h['bbox']) for h in headers)
                    
                    cell = TableCell(
                        text=cell_text,
                        row=row_idx,
                        col=col_idx,
                        rowspan=1,
                        colspan=1,
                        bbox=self._get_intersection_bbox(row['bbox'], col['bbox']),
                        confidence=min(row['confidence'], col['confidence']),
                        is_header=is_header
                    )
                    cells.append(cell)
        
        return cells
    
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
        """Convert Table Transformer output to standardized TableOutput format."""
        tables = []
        
        for i, table_data in enumerate(raw_output.get('tables', [])):
            cells = table_data.get('cells', [])
            
            # Convert cells to TableCell objects if they aren't already
            if cells and not isinstance(cells[0], TableCell):
                cells = [TableCell(**cell) if isinstance(cell, dict) else cell for cell in cells]
            
            # Calculate table dimensions
            num_rows = max([cell.row for cell in cells]) + 1 if cells else 0
            num_cols = max([cell.col for cell in cells]) + 1 if cells else 0
            
            table = Table(
                cells=cells,
                num_rows=num_rows,
                num_cols=num_cols,
                bbox=table_data.get('bbox'),
                confidence=table_data.get('confidence', 0.0),
                table_id=f"table_{i}",
                structure_confidence=table_data.get('structure_confidence', 0.0)
            )
            
            tables.append(table)
        
        return TableOutput(
            tables=tables,
            source_img_size=img_size,
            metadata={
                'engine': 'table_transformer',
                'detection_model': self.detection_model_name,
                'structure_model': self.structure_model_name,
                'detection_threshold': self.detection_threshold,
                'structure_threshold': self.structure_threshold
            }
        )
    
    @log_execution_time
    def extract(
        self,
        input_path: Union[str, Path, Image.Image],
        **kwargs
    ) -> TableOutput:
        """Extract tables using Table Transformer."""
        try:
            # Preprocess input
            images = self.preprocess_input(input_path)
            image = images[0]
            img_size = image.size
            
            # Detect tables
            detected_tables = self._detect_tables(image)
            
            if not detected_tables:
                if self.show_log:
                    logger.info("No tables detected in the image")
                return TableOutput(
                    tables=[],
                    source_img_size=img_size,
                    metadata={'engine': 'table_transformer', 'message': 'No tables detected'}
                )
            
            # Analyze structure for each detected table
            table_results = []
            for table_detection in detected_tables:
                structure_data = self._analyze_table_structure(image, table_detection['bbox'])
                cells = self._create_table_cells(structure_data)
                
                table_results.append({
                    'bbox': table_detection['bbox'],
                    'confidence': table_detection['confidence'],
                    'cells': cells,
                    'structure_confidence': np.mean([e['confidence'] for e in structure_data['elements']])
                })
            
            # Convert to standardized format
            result = self.postprocess_output({'tables': table_results}, img_size)
            
            if self.show_log:
                logger.info(f"Extracted {len(result.tables)} tables using Table Transformer")
            
            return result
            
        except Exception as e:
            logger.error("Error during Table Transformer extraction", exc_info=True)
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
            logger.error("Error during Table Transformer prediction", exc_info=True)
            return []