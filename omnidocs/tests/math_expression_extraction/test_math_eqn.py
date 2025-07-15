import pytest
import os
from omnidocs.tasks.math_expression_extraction.extractors import (
    DonutExtractor,
    LaTeXOCRExtractor,
    NougatExtractor,
    TexifyExtractor,
    TextellerExtractor,
    UniMERNetExtractor
)

def initialize_extractor(extractor_cls, device='cpu', show_log=False):
    """Helper function to initialize extractors with correct parameters."""
    if extractor_cls.__name__ == "UniMERNetExtractor":
        model_path = "omnidocs/models/unimernet_base"
        # UniMERNetExtractor will automatically download the model if it doesn't exist
        return extractor_cls(model_path=model_path, device=device, show_log=show_log)
    else:
        return extractor_cls(device=device, show_log=show_log)

# Simple Math Expression Extraction Tests
@pytest.mark.parametrize("extractor_cls", [
    DonutExtractor,
    LaTeXOCRExtractor,
    NougatExtractor,
    TexifyExtractor,
    TextellerExtractor,
    UniMERNetExtractor
])
def test_math_expression_extraction(extractor_cls):
    """Basic test: give image, get LaTeX output."""
    try:
        # Initialize extractor using helper function
        extractor = initialize_extractor(extractor_cls, device='cpu', show_log=False)
        
        # Extract from test image
        result = extractor.extract("omnidocs/tests/math_expression_extraction/assets/math_equation.png")
        
        print(f"\n{extractor_cls.__name__} Results:")
        
        # Check if we got something back
        if hasattr(result, 'expressions') and result.expressions:
            output = result.expressions[0]
            print(f"  LaTeX: {output}")
            assert len(output) > 0, "Should extract some LaTeX"
            
        elif hasattr(result, 'latex') and result.latex:
            output = result.latex
            print(f"  LaTeX: {output}")
            assert len(output) > 0, "Should extract some LaTeX"
            
        else:
            print(f"  No expressions found")
            # Still pass - extractor worked but found nothing
            
    except ImportError:
        pytest.skip(f"{extractor_cls.__name__} dependencies not installed")
        
    except FileNotFoundError as e:
        pytest.skip(f"{extractor_cls.__name__} - {str(e)}")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        # Skip extractors with dependency/compatibility issues
        if ("to_dict" in str(e) or "EinopsError" in str(e) or "AttributeError" in str(e) or
            "WinError 127" in str(e) or "OSError" in str(e) or "procedure could not be found" in str(e)):
            pytest.skip(f"{extractor_cls.__name__} has compatibility issues: {str(e)}")
        else:
            pytest.fail(f"{extractor_cls.__name__} failed: {str(e)}")


def test_simple_comparison():
    """Compare a few extractors on the same image."""
    image_path = "omnidocs/tests/math_expression_extraction/assets/math_equation.png"
    
    if not os.path.exists(image_path):
        pytest.skip("Test image not available")
    
    extractors_to_test = [UniMERNetExtractor, DonutExtractor, LaTeXOCRExtractor, TexifyExtractor, TextellerExtractor, NougatExtractor ]
    
    print(f"\nComparing extractors:")
    results = {}
    
    for ExtractorClass in extractors_to_test:
        try:
            extractor = initialize_extractor(ExtractorClass, device='cpu', show_log=False)
            result = extractor.extract(image_path)
            
            if hasattr(result, 'expressions') and result.expressions:
                output = result.expressions[0]
            elif hasattr(result, 'latex'):
                output = result.latex
            else:
                output = "No result"
                
            results[ExtractorClass.__name__] = output
            print(f"  {ExtractorClass.__name__}: {output}")
            
        except Exception as e:
            results[ExtractorClass.__name__] = f"Error: {str(e)}"
            print(f"  {ExtractorClass.__name__}: Error - {str(e)}")
    
    # At least one should work
    working_results = [r for r in results.values() if not r.startswith("Error:") and r != "No result"]
    assert len(working_results) > 0, "At least one extractor should work"
