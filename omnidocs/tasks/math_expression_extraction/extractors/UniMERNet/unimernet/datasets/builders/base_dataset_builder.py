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

import logging
import os
import shutil
import warnings
from omegaconf import OmegaConf
from unimernet.common.registry import registry
from unimernet.common.utils import get_abs_path, get_cache_path
from torch.utils.data import Dataset


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": None, "eval": None}
        self.text_processors = {"train": None, "eval": None}

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        # this is a concatenated dataset.

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=self.config.build_info.images,
            ann_paths=self.config.build_info.annotation,
        )

        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=self.config.build_info.images,
            ann_paths=self.config.build_info.annotation,
        )

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        """
        Download annotation files if not exist.
        """
        pass

    def _download_vis(self):
        """
        Download visual files if not exist.
        """
        pass


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg
