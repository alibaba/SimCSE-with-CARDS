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
Data collator

"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from data.text_augmentation import randomly_switch_case_batch


class DataCollatorForPairedLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator used for paried language modeling. Modified from:
        SimCSE.simcse.train.OurDataCollatorWithPadding
    Inputs are dynamically padded to the maximum length of a batch or max_length if specified.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.
            Set to 0 to avoid the masking process.
        mask_token_rate (:obj:`float`, `optional`, defaults to 0.8):
            The ratio of tokens to be replaced by [MASK] token.
        max_length (:obj:`int`, `optional`):
            If set, sequences longer than this will be truncated, sequences shorter will be padded.
            Else, will pad the samples dynamically when batching to the maximum length in the batch.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set, will pad the sequence so that its length is a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        mlm_probability: float = 0.15, 
        mask_token_rate: float = 0.8, 
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        dtype=None, 
        **kwargs,
        ):

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_rate = mask_token_rate
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.dtype = dtype or torch.long

        # data augmentation related
        self.switch_case_probability = kwargs.get('switch_case_probability', 0.0)
        self.switch_case_pattern = kwargs.get('switch_case_pattern', '00' if self.switch_case_probability == 0 else '01')
        self.switch_case_method = kwargs.get('switch_case_method', 'v1')

        # buffer
        self.special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']

    def __call__(self, examples: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # examples: List[Dict[
        #   'attention_mask': [[...], [...], ..., [...]], 
        #   'input_ids': [[...], [...], ..., [...]], 
        #   'ids': [[...], [...], ..., [...]], 
        # ]] (N, S, Ln)
        batch_size = len(examples)
        if batch_size == 0:
            return 

        # text augmentation
        if self.switch_case_pattern != '00':  # randomly switch case of words
            examples = randomly_switch_case_batch(examples, self.tokenizer, self.switch_case_probability, self.switch_case_method)
            if self.switch_case_pattern == '11':
                examples = randomly_switch_case_batch(examples, self.tokenizer, self.switch_case_probability, self.switch_case_method)

        num_sentences = len(examples[0]['input_ids'])
        # flat_examples: List[Dict['attention_mask': [...], 'input_ids': [...], 'ids': [...], ...]], (N*S, L, Ln)
        flat_examples = [
            {k: ex[k][i] if k in self.special_keys else ex[k] for k in ex} for ex in examples for i in range(num_sentences)
            ]

        batch = self.tokenizer.pad(
            flat_examples,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["input_ids"] = batch["input_ids"].to(self.dtype)

        if self.mlm_probability > 0:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.torch_mask_tokens(batch["input_ids"])

        batch = {k: batch[k].view(batch_size, num_sentences, -1) if k in self.special_keys else batch[k].view(batch_size, num_sentences, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

    def collate_flat_examples(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]], do_augment=False) -> Dict[str, torch.Tensor]:
        # examples: List[Dict[
        #   'attention_mask': [[...], [...], ..., [...]], 
        #   'input_ids': [[...], [...], ..., [...]], 
        #   'ids': [[...], [...], ..., [...]], 
        # ]] (N, S, Ln)
        batch_size = len(examples)
        if batch_size == 0:
            return 

        # text augmentation
        if self.switch_case_probability > 0 and do_augment:  # randomly switch case of words
            examples = randomly_switch_case_batch(examples, self.tokenizer, self.switch_case_probability)

        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["input_ids"] = batch["input_ids"].to(self.dtype)

        if self.mlm_probability > 0:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.torch_mask_tokens(batch["input_ids"])

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # for self.mask_token_rate (default 80%) of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_token_rate)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # for 0.5*(1-self.mask_token_rate) (default 10%) of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=self.dtype)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels