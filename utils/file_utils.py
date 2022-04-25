# coding=utf-8
# Copyright (C) 2010-2013 Alibaba Group Holding Limited. All rights reserved.
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
"""
Functions to handle checkpoints

"""
import os
import logging
import torch
import shutil
from packaging import version

import transformers
from transformers.file_utils import WEIGHTS_NAME, add_code_sample_docstrings
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer import TRAINER_STATE_NAME, OPTIMIZER_NAME, SCHEDULER_NAME, SCALER_NAME


logger = logging.getLogger(__name__)


def rename_ckpt_params(ckpt_folder: str, source_key_patterns, target_key_patterns):
    """
    Do:
        - Rename the checkpint file to from ckpt_folder/WEIGHTS_NAME to ckpt_folder/copy_WEIGHTS_NAME.
        - Rename the keys in checkpoint by converting source_key_patterns to target_key_patterns.
        - Save the resultant model to ckpt_folder/WEIGHTS_NAME.
    
    Args:
        ckpt_folder: folder of checkpoint
        source_key_patterns: str or list of str, e.g., ['LayerNorm.gamma', 'LayerNorm.beta']
        target_key_patterns: str or list of str matching source_key_patterns, e.g., ['LayerNorm.weight', 'LayerNorm.bias']
    """
    # check inputs
    assert os.path.isdir(ckpt_folder), ValueError('{} is not a valid folder'.format(ckpt_folder))
    ckpt_path = os.path.join(ckpt_folder, WEIGHTS_NAME)
    assert os.path.isfile(ckpt_path), ValueError(f"Cannot find a valid checkpoint at {ckpt_path}.")
    if isinstance(source_key_patterns, str):
        source_key_patterns = [source_key_patterns]
    if isinstance(target_key_patterns, str):
        target_key_patterns = [target_key_patterns]

    # map keys with source_key_patterns to keys with target_key_patterns, while keeping the rest the same
    state_dict = torch.load(ckpt_path)
    key_map = dict(zip(state_dict.keys(), state_dict.keys()))
    for skp, tkp in zip(source_key_patterns, target_key_patterns):
        for key in key_map.keys():
            if skp in key:
                key_map[key] = key_map[key].replace(skp, tkp)
    # log the changes keys 
    changed_keys = {key:value for key, value in key_map.items() if key != value}
    if changed_keys:
        message = 'The following keys  are changed: \n'
        for key, value in changed_keys.items():
            message += '{} -----> {}\n'.format(key, value)
        print(message)

        # rename the origial file ans save new checkpoint
        ckpt_copy_path = os.path.join(ckpt_folder, 'copy_' + WEIGHTS_NAME)
        os.rename(ckpt_path, ckpt_copy_path)
        print("File {} is renamed to {}.".format(ckpt_path, ckpt_copy_path))
        changed_state_dict = {key_map[key]:value for key, value in state_dict.items()}
        torch.save(changed_state_dict, ckpt_path)
    else:
        print('No key is changed. Nothing is done.')


class RenameCKPTFiles(object):
    def __init__(self, model_name_or_path):
        if model_name_or_path is not None:
            self.trainer_state_name = os.path.join(model_name_or_path, TRAINER_STATE_NAME)
            self._trainer_state_name = os.path.join(model_name_or_path, 'copy_' + TRAINER_STATE_NAME)

            self.optimizer_name = os.path.join(model_name_or_path, OPTIMIZER_NAME)
            self._optimizer_name = os.path.join(model_name_or_path, 'copy_' + OPTIMIZER_NAME)

            self.scheduler_name = os.path.join(model_name_or_path, SCHEDULER_NAME)
            self._scheduler_name = os.path.join(model_name_or_path, 'copy_' + SCHEDULER_NAME)

            self.scaler_name = os.path.join(model_name_or_path, SCALER_NAME)
            self._scaler_name = os.path.join(model_name_or_path, 'copy_' + SCALER_NAME)
        else:
            self.trainer_state_name = None
            self._trainer_state_name = None
            self.optimizer_name = None
            self._optimizer_name = None
            self.scheduler_name = None
            self._scheduler_name = None
            self.scaler_name = None
            self._scaler_name = None

    def rename_one_file(self, source_file, target_file):
        if source_file is not None and os.path.isfile(source_file):
            try:
                os.rename(source_file, target_file)
            except FileNotFoundError:
                if os.path.isfile(target_file):
                    print('RenameCKPTFiles: {} cannot be renamed as it has been renamed by other process.'.format(source_file))
                else:
                    raise FileNotFoundError('RenameCKPTFiles: {} cannot be renamed due to unexpected reason. To be debugged.'.format(source_file))

    def rename_files(self):
        self.rename_one_file(self.trainer_state_name, self._trainer_state_name)
        self.rename_one_file(self.optimizer_name, self._optimizer_name)
        self.rename_one_file(self.scheduler_name, self._scheduler_name)
        self.rename_one_file(self.scaler_name, self._scaler_name)

    def restore_file_names(self):
        self.rename_one_file(self._trainer_state_name, self.trainer_state_name)
        self.rename_one_file(self._optimizer_name, self.optimizer_name)
        self.rename_one_file(self._scheduler_name, self.scheduler_name)
        self.rename_one_file(self._scaler_name, self.scaler_name) 


class FileName(str):
    """ Customized string type to handle file name and extension.

    It accepts file path string as input and does:
    - replace '~' with '/root';
    - creates path and extension property.
    """
    def __new__(cls, *args, **kwargs):
        obj = str.__new__(cls, *args, **kwargs)  # type: FileName
        if obj.startswith('~'):
            obj = cls(os.path.expanduser("~") + obj[1:])  # type: FileName

        split_list = obj.split(".")
        obj._path = ''
        for s in split_list[:-1]:
            obj._path += s
        obj._extension = split_list[-1]
        return obj

    @property
    def path(self):
        return self._path

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, value):
        self._extension = value


def copy_simcse_ckpt_to_huggingface(source_folder, target_folder, model_type='roberta'):
    """ Convert SimCSE GitHub checkpoint to Huggingface style.
    """
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    onlyfiles = {f: os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))}
    for filename, source_path in onlyfiles.items():
        target_path = os.path.join(target_folder, filename)
        if os.path.isfile(target_path):
            print('{} exists in folder {}, thus is skipped.'.format(filename, target_folder))
        else:
            if filename == 'pytorch_model.bin':
                state_dict = torch.load(source_path)
                state_dict2 = {(model_type + '.' + key):value for key, value in state_dict.items() if not key.startswith('pooler.dense')}
                state_dict2['embedding_projector.dense.weight'] = state_dict['pooler.dense.weight']
                state_dict2['embedding_projector.dense.bias'] = state_dict['pooler.dense.bias']
                print('Save {} to {}.'.format(filename, target_folder))
                torch.save(state_dict2, target_path)
            else:
                print('Copy {} to {}.'.format(filename, target_folder))
                shutil.copyfile(source_path, target_path)


def add_custom_code_sample_docstrings(
    *docstr,
    processor_class=None,
    checkpoint=None,
    output_type=None,
    config_class=None,
    mask=None,
    model_cls=None,
    modality=None
    ):
    ver = transformers.__version__
    if version.parse(ver) < version.parse('4.12.0'):
        return add_code_sample_docstrings(
            *docstr, 
            tokenizer_class=processor_class,
            checkpoint=checkpoint,
            output_type=output_type,
            config_class=config_class,
            mask=mask,
            model_cls=model_cls,
            )
    else:
        return add_code_sample_docstrings(
            *docstr, 
            processor_class=processor_class,
            checkpoint=checkpoint,
            output_type=output_type,
            config_class=config_class,
            mask=mask,
            model_cls=model_cls,
            modality=modality,
            )