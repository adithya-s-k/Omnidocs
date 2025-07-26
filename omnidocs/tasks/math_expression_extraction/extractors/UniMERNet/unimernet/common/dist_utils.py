"""

 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

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


Minimal distributed utilities for inference-only UniMERNet.
Contains only the essential functions needed for model loading.
"""

import os
import logging
from unimernet.common.utils import download_url


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    # For inference-only, we always return False
    return False


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally.
    Simplified version for inference-only use.
    """
    try:
        from unimernet.common.utils import download_url
        import tempfile
        
        # Create a temporary directory for caching
        cache_dir = os.path.join(tempfile.gettempdir(), "unimernet_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Extract filename from URL
        filename = os.path.basename(url.split('?')[0])
        cached_file = os.path.join(cache_dir, filename)
        
        # Download if not already cached
        if not os.path.exists(cached_file):
            logging.info(f"Downloading {url} to {cached_file}")
            download_url(url, cache_dir, filename)
        
        return cached_file
        
    except Exception as e:
        logging.error(f"Failed to download cached file: {e}")
        return None
