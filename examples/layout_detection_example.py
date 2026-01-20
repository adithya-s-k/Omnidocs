"""
Layout Detection Example - End-to-End PDF Processing

This example demonstrates:
1. Loading a PDF document
2. Running layout detection with DocLayoutYOLO or RT-DETR
3. Visualizing and saving results
4. Accessing normalized coordinates

Usage:
    # Run with default (DocLayoutYOLO)
    python examples/layout_detection_example.py

    # Run with RT-DETR
    python examples/layout_detection_example.py --model rtdetr

    # Specify device
    python examples/layout_detection_example.py --device cuda

    # Process specific pages
    python examples/layout_detection_example.py --pages 1-3
"""

import argparse
from pathlib import Path

from omnidocs import Document
from omnidocs.tasks.layout_extraction import (
    DocLayoutYOLO,
    DocLayoutYOLOConfig,
    LayoutLabel,
    RTDETRConfig,
    RTDETRLayoutExtractor,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run layout detection on a PDF document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["yolo", "rtdetr"],
        default="yolo",
        help="Layout detection model to use (default: yolo)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on: cuda, mps, or cpu (default: cpu)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to PDF file (default: uses test fixture)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to process, e.g., '1-3' or '1,3,5' (default: all pages)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold (default: model-specific)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for visualizations (default: examples/output)",
    )
    return parser.parse_args()


def parse_page_range(page_str: str, max_pages: int) -> list:
    """Parse page range string into list of page indices (0-indexed)."""
    if not page_str:
        return list(range(max_pages))

    pages = []
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start = int(start) - 1  # Convert to 0-indexed
            end = int(end) - 1
            pages.extend(range(start, min(end + 1, max_pages)))
        else:
            page = int(part) - 1  # Convert to 0-indexed
            if 0 <= page < max_pages:
                pages.append(page)

    return sorted(set(pages))


def create_extractor(model: str, device: str, confidence: float = None):
    """Create layout extractor based on model choice."""
    if model == "yolo":
        conf = confidence if confidence is not None else 0.25
        print(f"🔧 Initializing DocLayoutYOLO (confidence={conf})...")
        return DocLayoutYOLO(
            config=DocLayoutYOLOConfig(
                device=device,
                confidence=conf,
                img_size=1024,
            )
        )
    else:  # rtdetr
        conf = confidence if confidence is not None else 0.4
        print(f"🔧 Initializing RT-DETR (confidence={conf})...")
        return RTDETRLayoutExtractor(
            config=RTDETRConfig(
                device=device,
                confidence=conf,
                image_size=640,
            )
        )


def main():
    args = parse_args()

    # ===========================================
    # Configuration
    # ===========================================

    # Determine PDF path
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdf_path = Path(__file__).parent.parent / "tests" / "fixtures" / "pdfs" / "research_paper_1.pdf"

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"📄 Loading PDF: {pdf_path.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖥️  Device: {args.device}")
    print(f"🤖 Model: {args.model.upper()}")
    print("-" * 50)

    # ===========================================
    # Step 1: Load the PDF Document
    # ===========================================

    doc = Document.from_pdf(str(pdf_path))
    print(f"✅ Loaded document: {doc.page_count} pages")
    print(f"   File: {doc.metadata.file_name}")
    print(f"   Size: {doc.metadata.file_size / 1024:.1f} KB")

    # ===========================================
    # Step 2: Initialize Layout Detector
    # ===========================================

    layout_extractor = create_extractor(args.model, args.device, args.confidence)
    print("✅ Layout detector ready")

    # ===========================================
    # Step 3: Determine Pages to Process
    # ===========================================

    pages_to_process = parse_page_range(args.pages, doc.page_count)
    print(f"\n🔍 Processing {len(pages_to_process)} page(s)...")

    # ===========================================
    # Step 4: Process Each Page
    # ===========================================

    total_elements = 0

    for page_num in pages_to_process:
        print(f"\n--- Page {page_num + 1}/{doc.page_count} ---")

        # Get page image
        page_image = doc.get_page(page_num)
        print(f"   Image size: {page_image.size[0]}x{page_image.size[1]}")

        # Run layout detection
        result = layout_extractor.extract(page_image)
        total_elements += result.element_count
        print(f"   Detected {result.element_count} layout elements")

        # Print detected elements
        if result.bboxes:
            # Group by label
            label_counts = {}
            for box in result.bboxes:
                label = box.label.value
                label_counts[label] = label_counts.get(label, 0) + 1

            print("   Elements found:")
            for label, count in sorted(label_counts.items()):
                print(f"      - {label}: {count}")

        # ===========================================
        # Step 5: Save Visualization
        # ===========================================

        output_path = output_dir / f"page_{page_num + 1:02d}_layout_{args.model}.png"
        result.visualize(
            page_image,
            output_path=output_path,
            show_labels=True,
            show_confidence=True,
            line_width=2,
        )
        print(f"   💾 Saved: {output_path.name}")

        # ===========================================
        # Step 6: Access Normalized Coordinates
        # ===========================================

        # Get normalized coordinates (0-1024 range)
        normalized_bboxes = result.get_normalized_bboxes()

        # Example: Print first 3 elements with normalized coords
        if normalized_bboxes:
            print("   Sample normalized coordinates (0-1024 range):")
            for box in normalized_bboxes[:3]:
                coords = [f"{c:.1f}" for c in box["bbox"]]
                print(f"      {box['label']}: [{', '.join(coords)}]")

        # ===========================================
        # Step 7: Filter Results (Example)
        # ===========================================

        # Filter by specific label
        tables = result.filter_by_label(LayoutLabel.TABLE)
        figures = result.filter_by_label(LayoutLabel.FIGURE)

        if tables:
            print(f"   📊 Found {len(tables)} table(s)")
        if figures:
            print(f"   🖼️  Found {len(figures)} figure(s)")

        # Filter by confidence
        high_conf = result.filter_by_confidence(0.8)
        if high_conf:
            print(f"   ⭐ {len(high_conf)} elements with >80% confidence")

    # ===========================================
    # Summary
    # ===========================================

    print("\n" + "=" * 50)
    print("✅ Processing complete!")
    print(f"   Model: {args.model.upper()}")
    print(f"   Pages processed: {len(pages_to_process)}")
    print(f"   Total elements detected: {total_elements}")
    print(f"   Visualizations saved to: {output_dir}")
    print("=" * 50)

    # Clean up
    doc.close()


if __name__ == "__main__":
    main()
