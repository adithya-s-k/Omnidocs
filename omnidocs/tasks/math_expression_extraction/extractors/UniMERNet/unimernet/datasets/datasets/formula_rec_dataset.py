"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
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
from PIL import Image
import torch
from torch.utils.data import Dataset


class FormulaRecDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.annotation = ann_paths

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def _add_instance_ids(self, key="instance_id"):
        # For inference, we don't need actual data loading
        pass

    def __len__(self):
        return 0

    def collater(self, samples):
        return samples

    def __getitem__(self, index):
        # For inference, this won't be called
        pass
