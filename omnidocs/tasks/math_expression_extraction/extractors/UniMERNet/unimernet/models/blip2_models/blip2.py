"""
 Copyright (c) 2023, salesforce.com, inc.
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

import torch
import torch.nn as nn
from unimernet.models.base_model import BaseModel


class Blip2Base(BaseModel):
    """
    Minimal Blip2Base class for inference only.
    """
    
    def __init__(self):
        super().__init__()
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            import contextlib
            return contextlib.nullcontext()
    
    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
