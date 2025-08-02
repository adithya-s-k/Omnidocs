"""
Comprehensive test for ALL math expression extractors
One file to test them all!
Run with: python tests/math_expression_extraction/test_all_math_extractors.py
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

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_all_math_extractors():
    """Test all available math expression extractors."""
    
    print("🧮 COMPREHENSIVE MATH EXPRESSION EXTRACTION TEST")
    print("=" * 70)
    
    # Test image path
    image_path = "tests/math_expression_extraction/assets/math_equation.png"
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    print(f"📸 Test image: {image_path}")
    print("=" * 70)
    
    extractors_tested = 0
    extractors_working = 0
    working_extractors = []
    failed_extractors = []
    
    # 1. Test UniMERNet
    print("\n1️⃣ Testing UniMERNet...")
    try:
        from omnidocs.tasks.math_expression_extraction.extractors.unimernet import UniMERNetExtractor
        
        print("✅ UniMERNetExtractor imported successfully!")
        extractor = UniMERNetExtractor(device='cpu', show_log=False)
        print("✅ UniMERNetExtractor initialized!")
        
        result = extractor.extract(image_path)
        print(f"✅ UniMERNet: Found {len(result.expressions)} expressions")
        
        if result.expressions:
            expr = result.expressions[0]
            print(f"   LaTeX: {expr[:80]}...")
        
        extractors_tested += 1
        extractors_working += 1
        working_extractors.append("UniMERNet")
        
    except Exception as e:
        print(f"❌ UniMERNet error: {e}")
        extractors_tested += 1
        failed_extractors.append(f"UniMERNet: {str(e)[:50]}...")
    
    # 2. Test Nougat
    print("\n2️⃣ Testing Nougat...")
    try:
        from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor
        
        print("✅ NougatExtractor imported successfully!")
        extractor = NougatExtractor(device='cpu', show_log=False)
        print("✅ NougatExtractor initialized!")
        
        result = extractor.extract(image_path)
        print(f"✅ Nougat: Found {len(result.expressions)} expressions")
        
        if result.expressions:
            expr = result.expressions[0]
            print(f"   LaTeX: {expr[:80]}...")
        
        extractors_tested += 1
        extractors_working += 1
        working_extractors.append("Nougat")
        
    except Exception as e:
        print(f"❌ Nougat error: {e}")
        extractors_tested += 1
        failed_extractors.append(f"Nougat: {str(e)[:50]}...")
    
    # 3. Test Donut
    print("\n3️⃣ Testing Donut...")
    try:
        from omnidocs.tasks.math_expression_extraction.extractors.donut import DonutExtractor
        
        print("✅ DonutExtractor imported successfully!")
        extractor = DonutExtractor(device='cpu', show_log=False)
        print("✅ DonutExtractor initialized!")
        
        result = extractor.extract(image_path)
        print(f"✅ Donut: Found {len(result.expressions)} expressions")
        
        if result.expressions:
            expr = result.expressions[0]
            print(f"   LaTeX: {expr[:80]}...")
        
        extractors_tested += 1
        extractors_working += 1
        working_extractors.append("Donut")
        
    except Exception as e:
        print(f"❌ Donut error: {e}")
        extractors_tested += 1
        failed_extractors.append(f"Donut: {str(e)[:50]}...")
    
    # 4. Test Surya 
    try:
        from omnidocs.tasks.math_expression_extraction.extractors import SuryaMathExtractor
        print("Surya imported successfully!")
        extractor = SuryaMathExtractor(device='cpu', show_log=False)
        print("Surya Extractor initialized!")
        result = extractor.extract(image_path)
        print(f"Sruya : Found {len(result.expressions)} expressions")
        
        if result.expressions:
            expr = result.expressions[0]
            print(f"   LaTeX: {expr[:80]}...")
        
        extractors_tested += 1
        extractors_working += 1
        working_extractors.append("Surya")
    except Exception as e:
        print(f"Surya error: {e}")
        extractors_tested += 1
        failed_extractors.append(f"Surya: {str(e)[:50]}...")

        
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 MATH EXTRACTION TEST SUMMARY")
    print("=" * 70)
    print(f"📊 Extractors tested: {extractors_tested}")
    print(f"✅ Extractors working: {extractors_working}")
    print(f"❌ Extractors failed: {extractors_tested - extractors_working}")
    print(f"📈 Success rate: {(extractors_working/extractors_tested)*100:.1f}%" if extractors_tested > 0 else "No extractors tested")
    
    print(f"\n✅ Working extractors: {', '.join(working_extractors)}")
    if failed_extractors:
        print(f"❌ Failed extractors: {', '.join(failed_extractors)}")
    
    if extractors_working > 0:
        print(f"\n🎉 SUCCESS! {extractors_working} math extractors are working!")
    else:
        print(f"\n😞 No math extractors are currently working")
    
    print("=" * 70)
    
    return {
        'tested': extractors_tested,
        'working': extractors_working,
        'working_list': working_extractors,
        'failed_list': failed_extractors
    }

if __name__ == "__main__":
    test_all_math_extractors()
