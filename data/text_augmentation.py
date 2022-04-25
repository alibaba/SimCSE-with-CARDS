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
Augmentation for texts

"""
from multiprocessing.sharedctypes import Value
import torch
from typing import List, Dict, Union, Optional
import random
import numpy as np

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


SPECIAL_TOKENS = {
    'bert': ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'],
    'roberta': ['<s>', '</s>', '<unk>', '<pad>', "<mask>"],
}


def randomly_switch_case_str_v1(input_str: str, switch_case_probability=0.2):
    """ Given a sentence, randomly switch the first letter of the word to lower case or upper case. The number of words to be 
    switched is max(1, int(round(num_words * switch_case_probability))).

    Note: this function only works for tokenizer that distinguishes lower and upper cases.

    e.g.,
    input_str (roberta)
    Thebes held onto power until the 12th Dynasty, when its first king, Amenemhet Iwho reigned between 1980 1951 b.c. established a capital near Memphis.
    output
    Thebes held onto power until the 12th dynasty, When its first king, amenemhet Iwho reigned Between 1980 1951 b.c. established a capital near Memphis.

    input_str (bert)
    thebes held onto power until the 12th dynasty, when its first king, amenemhet iwho reigned between 1980 1951 b. c. established a capital near memphis.
    output
    thebes held onto power Until the 12th dynasty, When its first king, amenemhet Iwho reigned Between 1980 1951 b. C. established a capital near memphis.
    """
    tokens = [t for t in input_str.split(' ') if t != '']  # remove empty tokens
    num_words = len(tokens)
    cand_indexes = list(range(num_words))
    num_switch_words = max(1, int(round(num_words * switch_case_probability)))
    random.shuffle(cand_indexes)
    num_siwtched_words = 0
    for word_indexes in cand_indexes:
        if num_siwtched_words < num_switch_words:
            token = tokens[word_indexes]
            if token[0].isalpha():  # isalpha checks a-zA-Z
                if token[0].islower():
                    tokens[word_indexes] = token[0].upper() + token[1:]
                else:
                    tokens[word_indexes] = token[0].lower() + token[1:]
                num_siwtched_words += 1
    output_str = ' '.join(tokens)

    return output_str


def randomly_switch_case_str_v2(input_str: str, switch_case_probability=0.2):
    """ Given a sentence, randomly switch the first letter of the word to lower case or upper case. Different from 
    randomly_switch_case_str_v1, this function independently checks if each word should be switched. So in case the 
    input string is very short, it is likely that the output_str equals the input_str.

    Note: this function only works for tokenizer that distinguishes lower and upper cases.

    e.g.,
    input_str (roberta)
    Thebes held onto power until the 12th Dynasty, when its first king, Amenemhet Iwho reigned between 1980 1951 b.c. established a capital near Memphis.
    output
    Thebes held onto power until the 12th dynasty, When its first king, amenemhet Iwho reigned Between 1980 1951 b.c. established a capital near Memphis.

    input_str (bert)
    thebes held onto power until the 12th dynasty, when its first king, amenemhet iwho reigned between 1980 1951 b. c. established a capital near memphis.
    output
    thebes held onto power Until the 12th dynasty, When its first king, amenemhet Iwho reigned Between 1980 1951 b. C. established a capital near memphis.
    """
    tokens = [t for t in input_str.split(' ') if t != '']  # remove empty tokens
    num_words = len(tokens)
    cand_indexes = np.where(np.random.uniform(size=(num_words,)) < switch_case_probability)[0]
    for word_indexes in cand_indexes:
        token = tokens[word_indexes]
        if token[0].isalpha():  # isalpha checks a-zA-Z
            if token[0].islower():
                tokens[word_indexes] = token[0].upper() + token[1:]
            else:
                tokens[word_indexes] = token[0].lower() + token[1:]
    output_str = ' '.join(tokens)

    return output_str


def check_switch_case_type(tokenizer, word):
    """ Detect the difference of tokenization before and after switch case.

    Returns:
    sc_type: one of four types:
    - 0: Substitution
    - 1: Division
    - 2: Fusion
    - 3: Regrouping
    len_diff: len(new_word) - len(word)

    """ 
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    if word[0].islower():
        new_word = word[0].upper() + word[1:]
        new_word_ids = tokenizer.encode(new_word, add_special_tokens=False)
    else:
        new_word = word[0].lower() + word[1:]
        new_word_ids = tokenizer.encode(new_word, add_special_tokens=False)
    len_diff = len(new_word_ids) - len(word_ids)

    if new_word_ids[1:] == word_ids[1:]:
        sc_type = 0  # Substitution
    elif len_diff > 0 and word_ids[1:] == new_word_ids[1+len_diff:]:
        sc_type = 1  # Division
    elif len_diff < 0 and word_ids[1-len_diff:] == new_word_ids[1:]:
        sc_type = 2  # Fusion
    else:
        sc_type = 3  # Regrouping
    
    return sc_type, new_word_ids


def randomly_switch_case_str_v3(tokenizer, input_str: str, switch_case_probability=0.2, sc_rule: str = None):
    """ Given a sentence, randomly switch the first letter of the word to lower case or upper case. Same as 
    randomly_switch_case_str_v2, this function independently checks if each word should be switched. So in case the 
    input string is very short, it is likely that the output_str equals the input_str. Different to
    randomly_switch_case_str_v2, this function support rule-based switch case defined in string sc_rule.

    Note: this function only works for tokenizer that distinguishes lower and upper cases.

    e.g.,
    input_str (roberta)
    Thebes held onto power until the 12th Dynasty, when its first king, Amenemhet Iwho reigned between 1980 1951 b.c. established a capital near Memphis.
    output
    Thebes held onto power until the 12th dynasty, When its first king, amenemhet Iwho reigned Between 1980 1951 b.c. established a capital near Memphis.

    input_str (bert)
    thebes held onto power until the 12th dynasty, when its first king, amenemhet iwho reigned between 1980 1951 b. c. established a capital near memphis.
    output
    thebes held onto power Until the 12th dynasty, When its first king, amenemhet Iwho reigned Between 1980 1951 b. C. established a capital near memphis.
    """
    tokens = [t for t in input_str.split(' ') if t != '']  # remove empty tokens
    num_words = len(tokens)
    cand_indexes = np.where(np.random.uniform(size=(num_words,)) < switch_case_probability)[0]
    for word_indexes in cand_indexes:
        token = tokens[word_indexes]
        if isinstance(sc_rule, str):
            if sc_rule == 'substitution':
                sc_type, _ = check_switch_case_type(tokenizer, token)
                if sc_type == 0 and token[0].isalpha():  # isalpha checks a-zA-Z
                    if token[0].islower():
                        tokens[word_indexes] = token[0].upper() + token[1:]
                    else:
                        tokens[word_indexes] = token[0].lower() + token[1:]
            elif sc_rule == 'retokenization':
                sc_type, new_word_ids = check_switch_case_type(tokenizer, token)
                if sc_type in {1, 3} and token[0].isalpha():  # Division, Regrouping
                    new_subtokens = tokenizer.convert_ids_to_tokens(new_word_ids)
                    new_token = tokenizer.mask_token.join(new_subtokens)
                    # switch the case back
                    if new_token[0].islower():
                        tokens[word_indexes] = new_token[0].upper() + new_token[1:]
                    else:
                        tokens[word_indexes] = new_token[0].lower() + new_token[1:]
            else:
                raise NotImplementedError('sc_rule {} not implemented.'.format(sc_rule))
        else:
            if token[0].isalpha():  # isalpha checks a-zA-Z
                if token[0].islower():
                    tokens[word_indexes] = token[0].upper() + token[1:]
                else:
                    tokens[word_indexes] = token[0].lower() + token[1:]
    output_str = ' '.join(tokens)

    return output_str


def randomly_switch_case_batch(
    examples: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]], 
    tokenizer: PreTrainedTokenizerBase, 
    switch_case_probability: float = 0.2, 
    switch_case_method: str = 'v1',
    ):
    """ Randomly switch the case of the first letter of a word for a batch of sequences.

    Args:
        examples: List[Dict['attention_mask': [[...], [...], ...], 'input_ids': [[...], [...], ...]]], (N, S, Ln)
        tokenizer:
        switch_case_probability:
        switch_case_method: v1 or v2
    """
    batch_size = len(examples)
    if batch_size == 0:
        raise ValueError('Empty examples.')

    if switch_case_method == 'v1':
        randomly_switch_case_str = randomly_switch_case_str_v1
    elif switch_case_method == 'v2':
        randomly_switch_case_str = randomly_switch_case_str_v2
    else:
        randomly_switch_case_str = lambda x, p: randomly_switch_case_str_v3(tokenizer, x, p, switch_case_method)

    is_nested = isinstance(examples[0]["input_ids"][0], list)
    if is_nested:  # do augmentation and append the results to ex["input_ids"]
        for ex in examples:
            input_ids = ex["input_ids"][0]  # [digits]
            input_str = tokenizer.decode(input_ids, skip_special_tokens=True)  # str
            output_str = randomly_switch_case_str(input_str, switch_case_probability)
            output_ids = tokenizer.encode(output_str, add_special_tokens=True)  # [digits]
            if switch_case_method == 'retokenization':
                output_ids = [ids for ids in output_ids if ids != tokenizer.mask_token_id]
            ex["input_ids"].append(output_ids)
            ex["attention_mask"].append([1]*len(output_ids))
            if "token_type_ids" in ex:
                ex["token_type_ids"].append([0]*len(output_ids))

        return examples

    else:  # do augmentation and append the results to ex["input_ids"]
        aug_examples = []
        for ex in examples:
            aug_examples.append(ex)
            input_ids = ex["input_ids"] 
            input_str = tokenizer.decode(input_ids, skip_special_tokens=True)  # str
            output_str = randomly_switch_case_str(input_str, switch_case_probability)
            output_ids = tokenizer.encode(output_str, add_special_tokens=True)  # [digits]
            if switch_case_method == 'retokenization':
                output_ids = [ids for ids in output_ids if ids != tokenizer.mask_token_id]
            aug_ex = {"input_ids": output_ids, "attention_mask": [1]*len(output_ids)}
            if "token_type_ids" in ex:
                aug_ex["token_type_ids"] = [0]*len(output_ids)
            aug_examples.append(aug_ex)

        return aug_examples
