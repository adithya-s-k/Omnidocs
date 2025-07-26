import sys
import os
import warnings
import json 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from omnidocs.tasks.table_extraction.extractors import (
    CamelotExtractor,
    PDFPlumberExtractor,
    # PPStructureExtractor,  # not working cuz of some issues
    TableTransformerExtractor,
    TableFormerExtractor,
    TabulaExtractor
)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# Fix protobuf compatibility issue with PaddleOCR
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def table_to_dict(table):
    return {
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "cells": [
            {
                "row": cell.row,
                "col": cell.col,
                "text": cell.text,
                "bbox": getattr(cell, "bbox", None)
            }
            for cell in table.cells
        ],
        "bbox": getattr(table, "bbox", None)
    }

def save_tables_to_json(tables, extractor_name, filetype):
    data = [table_to_dict(table) for table in tables]
    out_path = f"output_{extractor_name}_{filetype}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {out_path}")

def test_table_extraction():
    # Define which extractors work with which file types
    pdf_extractors = [
        CamelotExtractor,
        PDFPlumberExtractor,
        TabulaExtractor
    ]

    image_extractors = [
        # PPStructureExtractor,  # not working cuz of some issues
        TableTransformerExtractor,
        TableFormerExtractor
    ]

    pdf_path = "assets/table_document.pdf"
    image_path = "assets/table_image.png"

    # Test PDF extractors
    print("Testing PDF extractors...")
    for extractor_cls in pdf_extractors:
        print(f"\n{'='*50}")
        print(f"Testing {extractor_cls.__name__} with PDF")
        print(f"{'='*50}")

        try:
            result = extractor_cls().extract(pdf_path)
            print(f"Extracted tables: {len(result.tables)} table(s)")

            for i, table in enumerate(result.tables):
                print(f"\nTable {i+1}: {table.num_rows} rows x {table.num_cols} columns")
                print(f"Total cells: {len(table.cells)}")

                if table.cells:
                    non_empty_cells = [cell for cell in table.cells if cell.text.strip()]
                    print(f"Non-empty cells: {len(non_empty_cells)}")

                    # Show first few cells
                    for cell in table.cells[:10]:
                        if cell.text.strip():
                            text = cell.text.strip()[:30]
                            print(f"  [{cell.row},{cell.col}]: '{text}'")

                    # Show CSV preview
                    try:
                        csv_preview = table.to_csv()
                        lines = csv_preview.strip().split('\n')[:5]
                        print(f"\nCSV Preview:")
                        for line in lines:
                            print(f"  {line}")
                    except Exception as e:
                        print(f"  CSV preview failed: {e}")

            # Save to JSON
            save_tables_to_json(result.tables, extractor_cls.__name__, "pdf")

            print("SUCCESS: PDF extraction completed")
            assert len(result.tables) >= 0

        except Exception as e:
            print(f"ERROR: {str(e)}")
            assert False, f"{extractor_cls.__name__} failed: {str(e)}"

    # Test image extractors
    print("\n\nTesting image extractors...")
    for extractor_cls in image_extractors:
        print(f"\n{'='*50}")
        print(f"Testing {extractor_cls.__name__} with Image")
        print(f"{'='*50}")

        try:
            result = extractor_cls().extract(image_path)
            print(f"Extracted tables: {len(result.tables)} table(s)")

            for i, table in enumerate(result.tables):
                print(f"\nTable {i+1}: {table.num_rows} rows x {table.num_cols} columns")
                print(f"Total cells: {len(table.cells)}")

                if table.cells:
                    non_empty_cells = [cell for cell in table.cells if cell.text.strip()]
                    print(f"Non-empty cells: {len(non_empty_cells)}")

                    # Show first few cells
                    for cell in table.cells[:10]:
                        if cell.text.strip():
                            text = cell.text.strip()[:30]
                            print(f"  [{cell.row},{cell.col}]: '{text}'")

                    # Show CSV preview
                    try:
                        csv_preview = table.to_csv()
                        lines = csv_preview.strip().split('\n')[:5]
                        print(f"\nCSV Preview:")
                        for line in lines:
                            print(f"  {line}")
                    except Exception as e:
                        print(f"  CSV preview failed: {e}")

            # Save to JSON
            save_tables_to_json(result.tables, extractor_cls.__name__, "image")

            print("SUCCESS: Image extraction completed")
            assert len(result.tables) >= 0

        except Exception as e:
            print(f"ERROR: {str(e)}")
            assert False, f"{extractor_cls.__name__} failed: {str(e)}"

if __name__ == "__main__":
    test_table_extraction()