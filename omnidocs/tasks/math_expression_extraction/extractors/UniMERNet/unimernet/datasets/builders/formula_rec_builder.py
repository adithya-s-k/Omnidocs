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

from unimernet.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from unimernet.datasets.datasets.formula_rec_dataset import FormulaRecDataset
from unimernet.common.registry import registry


@registry.register_builder("formula_rec_eval")
class FormulaRecBuilder(BaseDatasetBuilder):
    train_dataset_cls = FormulaRecDataset
    eval_dataset_cls = FormulaRecDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/formula/default.yaml",
    }

    def build_datasets(self):
        # For inference, we don't need actual datasets, just return empty dict
        return {"eval": None}

    def build_processors(self):
        # Build processors from config
        super().build_processors()
