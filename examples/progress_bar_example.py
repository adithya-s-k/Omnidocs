"""
Example script demonstrating Document.iter_pages() with progress bar.

This shows how to enable the progress bar when iterating over pages.
"""

from omnidocs import Document

# Example 1: Using iter_pages with progress bar
doc = Document.from_pdf("sample.pdf")

# With progress bar (requires tqdm to be installed: uv add --optional progress)
print("Processing pages with progress bar...")
for page in doc.iter_pages(progress=True):
    # Process each page
    # result = some_model.process(page)
    pass

# Example 2: Without progress bar (default)
print("\nProcessing pages without progress bar...")
for page in doc.iter_pages():
    # Process each page
    # result = some_model.process(page)
    pass

# Example 3: Progress bar with custom processing
print("\nProcessing pages with custom logic and progress...")
page_results = []
for page in doc.iter_pages(progress=True):
    # You can do any processing here
    width, height = page.size
    page_results.append({
        "width": width,
        "height": height,
        "mode": page.mode
    })

print(f"Processed {len(page_results)} pages")
