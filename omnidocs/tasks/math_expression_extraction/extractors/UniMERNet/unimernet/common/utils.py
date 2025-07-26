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


Minimal utilities for inference-only UniMERNet.
Contains only the essential functions needed for model loading.
"""

import os
import re
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from unimernet.common.registry import registry
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
)


def is_url(input_url):
    """Check if an input string is a url."""
    return re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None


def get_abs_path(rel_path):
    """Get absolute path from relative path using registry."""
    return os.path.join(registry.get_path("library_root"), rel_path)


def makedir(dir_path):
    """Create directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)


def get_redirected_url(url):
    """Get the final URL after redirects."""
    try:
        response = urllib.request.urlopen(url)
        return response.url
    except:
        return url


def _get_google_drive_file_id(url):
    """Extract Google Drive file ID from URL."""
    # Simple check for Google Drive URLs
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
    return None


def _urlretrieve(url, fpath):
    """Download file from URL."""
    urllib.request.urlretrieve(url, fpath)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download a file from a url and place it in root."""
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = get_redirected_url(url)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead."
                " Downloading " + url + " to " + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
