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
Callbacks

"""
import warnings
# import logging
import torch
from datetime import datetime
from transformers.integrations import WandbCallback, TensorBoardCallback
from transformers.trainer_callback import (
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PrinterCallback, 
    TrainerCallback,
)


# logger = logging.get_logger(__name__)


class EditableLogPrinterCallback(PrinterCallback):
    """ TrainerCallback with editable log info.
    """
    def __init__(self):
        super().__init__()
        self.log = dict()

    def add_to_log(self, log):
        if isinstance(log, dict):
            log = {k:v.item() if isinstance(v, torch.Tensor) else v for k, v in log.items()}
            self.log.update(log)
        else:
            warnings.warn('Expect log to be dict, got {}'.format(type(log)), RuntimeWarning)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs = {**logs, **self.log}
        _ = logs.pop("total_flos", None)
        epoch = logs.pop('epoch', 'unknown')  # in case training is not done, logs does not have 'epoch'
        if state.is_local_process_zero:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('[{}  Epoch {} - Step {}]: {}.'.format(dt_string, epoch, state.global_step, logs))


class EditableLogWandbCallback(WandbCallback):
    """ WandbCallback with editable log info.
    """
    def __init__(self):
        super().__init__()
        self.log = dict()

    def add_to_log(self, log):
        if isinstance(log, dict):
            self.log.update(log)
        else:
            warnings.warn('Expect log to be dict, got {}'.format(type(log)), RuntimeWarning)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        if self._wandb is not None:
            self._wandb.run.name = args.output_dir.split("/")[-1][0:60]
            self._wandb.run.notes = args.output_dir.split("/")[-1]

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model, reinit=False)
        if state.is_world_process_zero:
            self._wandb.log({**logs, **self.log}, step=state.global_step)


class EditableLogTensorBoardCallback(TensorBoardCallback):
    """ TensorBoardCallback with editable log info.
    """
    def __init__(self):
        super().__init__()
        self.log = dict()

    def add_to_log(self, log):
        if isinstance(log, dict):
            self.log.update(log)
        else:
            warnings.warn('Expect log to be dict, got {}'.format(type(log)), RuntimeWarning)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs = {**logs, **self.log}
        super().on_log(args, state, control, logs, **kwargs)


class HaltTrainingCallback(TrainerCallback):
    """ Halt training if passing certain number of steps or epochs

    Args:
        halt_step: 
        halt_epoch: 
    """
    def __init__(self, halt_step=-1, halt_epoch=-1):
        self.halt_step = halt_step
        self.halt_epoch = halt_epoch

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        if self.halt_epoch >= 0 and state.epoch >= self.halt_epoch:
            control.should_training_stop = True
            if state.is_local_process_zero:
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print('[{}  Epoch {}]: halt_epoch reached. Stop training now.'.format(dt_string, state.epoch))

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        if self.halt_step >= 0 and state.global_step >= self.halt_step:
            control.should_training_stop = True
            if state.is_local_process_zero:
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print('[{}  Step {}]: halt_step reached. Stop training now.'.format(dt_string, state.global_step))