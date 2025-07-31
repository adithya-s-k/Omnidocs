#!/usr/bin/env python3
"""
Simple layout detectors test - just pass an image and test each detector.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_yolo():
    """Test YOLO Layout Detector"""
    print("\n" + "="*50)
    print("Testing YOLOLayoutDetector")
    print("="*50)
    
    try:
        from omnidocs.tasks.layout_analysis.extractors.doc_layout_yolo import YOLOLayoutDetector
        
        detector = YOLOLayoutDetector(show_log=True)
        image_path = "tests/layout_detectors/assets/news_paper.png"
        
        annotated_image, layout_output = detector.detect(image_path)
        print(f"Detected {len(layout_output.bboxes)} elements")
        
        # Visualize
        output_path = "tests/layout_detectors/output/yolo_result.png"
        detector.visualize((annotated_image, layout_output), output_path)
        print(f"Saved visualization to {output_path}")
        
    except Exception as e:
        print(f"YOLO failed: {e}")

def test_florence():
    """Test Florence Layout Detector"""
    print("\n" + "="*50)
    print("Testing FlorenceLayoutDetector")
    print("="*50)
    
    try:
        from omnidocs.tasks.layout_analysis.extractors.florence import FlorenceLayoutDetector
        
        detector = FlorenceLayoutDetector(show_log=True)
        image_path = "tests/layout_detectors/assets/news_paper.png"
        
        annotated_image, layout_output = detector.detect(image_path)
        print(f"Detected {len(layout_output.bboxes)} elements")
        
        # Visualize
        output_path = "tests/layout_detectors/output/florence_result.png"
        detector.visualize((annotated_image, layout_output), output_path)
        print(f"Saved visualization to {output_path}")
        
    except Exception as e:
        print(f"Florence failed: {e}")

def test_paddle():
    """Test Paddle Layout Detector"""
    print("\n" + "="*50)
    print("Testing PaddleLayoutDetector")
    print("="*50)
    
    try:
        print("Importing PaddleLayoutDetector...")
        from omnidocs.tasks.layout_detectors.extractors.paddle import PaddleLayoutDetector
        print("✓ Import successful")
        
        print("Initializing detector...")
        detector = PaddleLayoutDetector(show_log=True)
        print("✓ Initialization successful")
        
        print("Running detection...")
        image_path = "tests/layout_detectors/assets/news_paper.png"
        annotated_image, layout_output = detector.detect(image_path)
        print(f"✓ Detected {len(layout_output.bboxes)} elements")
        
        # Visualize
        print("Saving visualization...")
        output_path = "tests/layout_detectors/output/paddle_result.png"
        detector.visualize((annotated_image, layout_output), output_path)
        print(f"✓ Saved visualization to {output_path}")
        
    except ImportError as e:
        print(f"Paddle import failed: {e}")
    except Exception as e:
        print(f"Paddle runtime error: {e}")
        import traceback
        traceback.print_exc()

def test_rtdetr():
    """Test RTDETR Layout Detector"""
    print("\n" + "="*50)
    print("Testing RTDETRLayoutDetector")
    print("="*50)
    
    try:
        from omnidocs.tasks.layout_analysis.extractors.rtdetr import RTDETRLayoutDetector
        
        detector = RTDETRLayoutDetector(show_log=True)
        image_path = "tests/layout_analysis/assets/news_paper.png"
        
        annotated_image, layout_output = detector.detect(image_path)
        print(f"Detected {len(layout_output.bboxes)} elements")
        
        # Visualize
        output_path = "tests/layout_analysis/output/rtdetr_result.png"
        detector.visualize((annotated_image, layout_output), output_path)
        print(f"Saved visualization to {output_path}")
        
    except Exception as e:
        print(f"RTDETR failed: {e}")

def test_surya():
    """Test Surya Layout Detector"""
    print("\n" + "="*50)
    print("Testing SuryaLayoutDetector")
    print("="*50)
    
    try:
        from omnidocs.tasks.layout_analysis.extractors.surya import SuryaLayoutDetector
        
        detector = SuryaLayoutDetector(show_log=True)
        image_path = "tests/layout_analysis/assets/news_paper.png"
        
        annotated_image, layout_output = detector.detect(image_path)
        print(f"Detected {len(layout_output.bboxes)} elements")
        
        # Visualize
        output_path = "tests/layout_analysis/output/surya_result.png"
        detector.visualize((annotated_image, layout_output), output_path)
        print(f"Saved visualization to {output_path}")
        
    except Exception as e:
        print(f"Surya failed: {e}")

def main():
    """Simple test - just test each detector one by one."""
    
    print("Simple Layout Detectors Test")
    print("=" * 50)
    
    # Create output directory
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    
    # Test each detector
    test_yolo()
    test_florence()
    test_paddle()
    test_rtdetr()
    test_surya()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("Check tests/output/ for visualizations")

if __name__ == "__main__":
    main()