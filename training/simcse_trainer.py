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

import os
import numpy as np
import torch
from torch import nn
from packaging import version
from transformers import Trainer
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset

from data.metrics.senteval.engine import SE
from utils.retriever import BaseFaissIPRetriever

SENTEVAL_TASKS = [
    'CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
    'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
    'STS14', 'STS15', 'STS16',
    'Length', 'WordContent', 'Depth', 'TopConstituents',
    'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    'OddManOut', 'CoordinationInversion', 'SICKRelatedness-finetune', 'STSBenchmark-finetune', 'STSBenchmark-fix'
    ]
STS_TASKS = ['STSBenchmark', 'SICKRelatedness', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']
TRANS_TASK = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']


logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval: bool = False,
    ) -> Dict[str, float]:
        # prepare
        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        # when using amp, this makes get_parameter_dtype return float16
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            # among all STS tasks, only SICK-R has upper-case words in the sentences.
            # To enlarge the impact, in case ignore_upper_case_phase is train, we make downstream tasks upper-cased.
            if self.args.ignore_upper_case_phase == 'train':  
                sentences = [s[0].upper()+s[1:] for s in sentences]
            if self.args.ignore_upper_case_phase in {'eval', 'both'}:
                sentences = [s.lower() for s in sentences]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                if self.use_amp and self.args.fp16_full_eval:
                    with autocast():
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                else:
                    outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        
        PATH_TO_DATA = self.SentEval_data_dir
        if PATH_TO_DATA.startswith('~'):
            PATH_TO_DATA = os.path.expanduser("~") + PATH_TO_DATA[1:]
        # Set params for SentEval
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        if eval_senteval:  # during test, use full mode
            params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                    'tenacity': 5, 'epoch_size': 4}
        else:  # when eval_during_train, use fast mode
            params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                    'tenacity': 3, 'epoch_size': 2}
        params['get_align_uniform'] = self.args.get_align_uniform

        se = SE(params, batcher, prepare)

        if self.args.metric_for_best_model == 'stsb_spearman':
            tasks = ['STSBenchmark']
        elif self.args.metric_for_best_model == 'sickr_spearman':
            tasks = ['SICKRelatedness']
        elif self.args.metric_for_best_model == 'avg_sts':
            tasks = ['STSBenchmark', 'SICKRelatedness']
        elif self.args.metric_for_best_model == 'avg_transfer':
            tasks = TRANS_TASK
        elif self.args.metric_for_best_model == 'avg_transfer':
            tasks = [self.args.metric_for_best_model.upper()]
        else:
            raise NotImplementedError('Metrics for {} have not been added yet.')
        if eval_senteval or self.args.eval_during_train:
            if self.args.task_set == 'sts':
                tasks = list(set(tasks + STS_TASKS))
            elif self.args.task_set == 'transfer':
                tasks = list(set(tasks + TRANS_TASK))
            elif self.args.task_set == 'all':
                tasks = STS_TASKS + TRANS_TASK
            else:
                if isinstance(self.args.task_set, str):
                    assert self.args.task_set in SENTEVAL_TASKS, ValueError(
                        'Unrecognized senteval task: {}'.format(self.args.task_set))
                    tasks += [self.args.task_set]
                elif isinstance(self.args.task_set, list):
                    assert all(task in SENTEVAL_TASKS for task in self.args.task_set), ValueError(
                        'Unrecognized senteval task set: {}'.format(self.args.task_set))
                    tasks += self.args.task_set
                else:
                    raise TypeError('task_set must be str or list, got {}'.format(type(self.args.task_set)))

        model.eval()
        results = se.eval(tasks)
        
        # for STSBenchmark and SICKRelatedness results[task] format 
        # {
        #   'train': {'metric1': (...), 'metric2': (...), ..., 'nsamples': nsamples}
        #   'dev': {...}
        #   'test': {...}
        # }
        # Others may be different
        # record results on both dev and test sets.
        metrics = {}
        for task in tasks:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                # STS12-STS16 do not have dev set
                metrics['test_{}_spearman'.format(task)] = results[task]['all']["spearman"]['all'] * 100
            elif task == 'STSBenchmark':
                metrics.update(
                    {
                        "eval_stsb_spearman": results[task]['dev']['spearman'][0] * 100,
                        "test_stsb_spearman": results[task]['test']['spearman'][0] * 100,
                    }
                )
                if self.args.get_align_uniform:
                    metrics.update(
                        {
                            "eval_stsb_alignment": results[task]['dev']['alignment'],
                            "eval_stsb_uniformity": results[task]['dev']['uniformity'],
                            "test_stsb_alignment": results[task]['test']['alignment'],
                            "test_stsb_uniformity": results[task]['test']['uniformity'],
                        }
                    )
            elif task == 'SICKRelatedness':
                metrics.update(
                    {
                        "eval_sickr_spearman": results[task]['dev']['spearman'][0] * 100,
                        "test_sickr_spearman": results[task]['test']['spearman'][0] * 100,
                    }
                )
            elif task in TRANS_TASK:
                metrics.update(
                    {
                        'eval_{}'.format(task): results[task]['devacc'],
                        'test_{}'.format(task): results[task]['acc'],
                    }
                )
            else:
                raise NotImplementedError('Metrics for other tasks have not been added yet.')
        # for certain task_set combination, we can add their average results
        if 'STSBenchmark' in tasks and 'SICKRelatedness' in tasks:
            metrics['eval_avg_sts'] = (metrics["eval_stsb_spearman"] + metrics["eval_sickr_spearman"]) / 2
        if all(task in tasks for task in STS_TASKS):
            avg_sts = metrics['test_stsb_spearman'] + metrics['test_sickr_spearman']
            for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                avg_sts += metrics['test_{}_spearman'.format(task)]
            avg_sts /= 7
            metrics['test_avg_sts'] = avg_sts
        if all(task in tasks for task in TRANS_TASK):
            # dev
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                metrics['eval_{}'.format(task)] = results[task]['devacc']
                avg_transfer += results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer
            # test
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                metrics['test_{}'.format(task)] = results[task]['acc']
                avg_transfer += results[task]['acc']
            avg_transfer /= 7
            metrics['test_avg_transfer'] = avg_transfer

        self.log(metrics)

        return metrics

    def get_dataset_representations(self, dataset=None, cache_file=None, load_from_cache_file=True, discard_embd_proj=True):
        if (
            load_from_cache_file
            and cache_file is not None 
            and os.path.exists(cache_file)
        ):
            logger.info('Loading dataset representations from file {}'.format(cache_file))
            representations = torch.load(cache_file)
            return representations.numpy()

        elif dataset is not None:
            logger.info('Calculating dataset representations...')
            self.model.get_sent_emb_only = True
            # The self.model.embedding_projector may be random here. 
            # If so, discard it to get a proper representation.
            original_discard_embd_proj = None
            if self.model.discard_embd_proj != discard_embd_proj:
                original_discard_embd_proj = self.model.discard_embd_proj
                self.model.discard_embd_proj = discard_embd_proj
            outputs = self.predict(dataset)  # PredictionOutput(predictions, label_ids, metrics)
            self.model.get_sent_emb_only = False
            if original_discard_embd_proj is not None:
                self.model.discard_embd_proj = original_discard_embd_proj
            representations = outputs.predictions  # numpy.ndarray
            if (
                cache_file is not None 
                and not os.path.exists(cache_file)
            ):
                logger.info('Save dataset representations to file {}'.format(cache_file))
                torch.save(torch.tensor(representations), cache_file)
            return representations

        else:
            raise ValueError('Unrecognized inputs. Either dataset or cache_file must be provided.')

    def get_representation_retrieved_results(
        self, 
        representations=None, 
        cache_file=None, 
        sent_rep_cache_file=None, 
        load_from_cache_file=True,
        retrieve_hard_negatives=2,  # at least 2 to exclude the same itself
        discard_embd_proj=None,
        ):
        if (
            load_from_cache_file
            and cache_file is not None 
            and os.path.exists(cache_file)
        ):
            logger.info('Loading retrieved results from file {}'.format(cache_file))
            scores, retrieved_indices = np.load(cache_file)
            return scores.astype(np.float32), retrieved_indices.astype(np.int)

        elif representations is None:
            representations = self.get_dataset_representations(
                self.train_dataset, 
                cache_file=sent_rep_cache_file, 
                load_from_cache_file=load_from_cache_file,
                discard_embd_proj=discard_embd_proj,
                ).astype(np.float32)  # numpy.ndarray

        retriever = BaseFaissIPRetriever(representations)
        retriever.add(representations)  # retriever only accepts inputs in float32
        scores, retrieved_indices = retriever.batch_search(representations, retrieve_hard_negatives, batch_size=2048)

        if (
            cache_file is not None 
            and not os.path.exists(cache_file)
        ):
            np.save(cache_file, (scores, retrieved_indices))

        return scores, retrieved_indices

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

        if getattr(self.control, 'should_update_buffer_representations', False):
            self._update_buffer_representations()

    def _update_buffer_representations(self):
        if self.model.config.sample_retrieved_dynamic_negatives > 0 and self.model.buffer_representations is not None:
            representations = self.get_dataset_representations(dataset=self.train_dataset, discard_embd_proj=True)  # numpy.ndarray

            # update model.buffer_representations
            device = self.model.buffer_representations.device
            dtype = self.model.buffer_representations.dtype
            self.model.buffer_representations = torch.tensor(representations, device=device, dtype=dtype)

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        # In distributed mode, all_gather requires all tensors to be contiguous. Otherwise, the following error happens
        # RuntimeError: Tensors must be non-overlapping and dense.
        if isinstance(tensors, torch.Tensor):
            tensors = tensors.contiguous()

        return super()._nested_gather(tensors, name)