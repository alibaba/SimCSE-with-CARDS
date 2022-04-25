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
Pre-training of SimCSE using the Huggingface Transformer library. 

"""
import logging
import math
import os
import sys

sys.path.append(os.getcwd())

from dataclasses import asdict, dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import shutil
import ast
import copy

import transformers
from transformers import (
    # CONFIG_MAPPING,  # use customized config
    MODEL_FOR_MASKED_LM_MAPPING,
    # AutoConfig,  # use customized config
    AutoTokenizer,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    DefaultFlowCallback,
    default_data_collator,
    set_seed,
    BertForPreTraining,
)
from transformers.integrations import WandbCallback, TensorBoardCallback
from transformers.trainer_callback import PrinterCallback, TrainerState, TrainerControl
from transformers.trainer_utils import is_main_process
from transformers.trainer import TRAINER_STATE_NAME

from data.data_collator import DataCollatorForPairedLanguageModeling
from models.bert import BertConfig, RobertaConfig
from models.bert.modeling_simcse import SimCSEForSequenceEmbedding
from training.simcse_trainer import CLTrainer as Trainer
from utils.callbacks import HaltTrainingCallback
from utils.file_utils import RenameCKPTFiles, FileName

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
CONFIG_MAPPING = {'bert': BertConfig, 'roberta': RobertaConfig}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # SimCSE's arguments
    cl_temperature: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, demean, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    mlm_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight for MLM auxiliary objective; assign a positive value to enable mlm."
        }
    )
    discard_embd_proj: bool = field(
        default=False,
        metadata={
            "help": "Whether to discard contrastive learning projector head during evaluation."
        }
    )
    hidden_dropout_prob: Optional[str] = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    embedding_dropout_prob: Optional[str] = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "label smoothing used in CrossEntropyLoss; 0.0 for no label smoothing."
        }
    )
    switch_case_pattern: str = field(
        default='00',
        metadata={
            "help": "Assign 01, 10 or 11 to consider samples from switch case augmentation."
        }
    )
    symmetric_contrastive_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use symmetric contrastive loss."
        }
    )
    retriever_metric: str = field(
        default='ip',
        metadata={
            "help": "Metric for retriever, support ip (inner product), l2 and cos."
        }
    )
    retrieve_dynamic_negatives: int = field(
        default=0, 
        metadata={
            "help": "The number of hard negatives to retrieve using the model representations; 0 to disable it."
            "Different from data_args.retrieve_hard_negatives, the hard negatives are retrieved during training."
            }
    )
    sample_retrieved_dynamic_negatives: int = field(
        default=0, 
        metadata={
            "help": "The number of hard negatives to sample from retrieved negatives; 0 to disable it."
            "Different from data_args.sample_retrieved_hard_negatives, the hard negatives are samples during training."
            }
    )
    ignore_num_top_retrieved_dynamic_negatives: int = field(
        default=0, 
        metadata={
            "help": "The number of hard negatives to ignore from sampling."
            }
    )
    share_sampled_dynamic_negatives: bool = field(
        default=True,
        metadata={"help": "Whether to share the sampled dynamic negatives among batch samples."}
    )
    update_retrieved_negatives_representations: bool = field(
        default=False,
        metadata={"help": "Whether to update the representations of retrieved negatives based on current model."}
    )
    detach_retrieved_negatives_representations: bool = field(
        default=False,
        metadata={"help": "Whether to detach the representations of retrieved negatives after updating it."}
    )
    retrieve_using_buffer_representations: bool = field(
        default=False,
        metadata={"help": "Whether to call retriever before getting the most updated representations."}
    )
    save_buffer_representations: bool = field(
        default=False,
        metadata={"help": "Whether to save the updated representations in ckeckpoint files."}
    )
    retrieved_and_sampled_knn_type: str = field(
        default='default',
        metadata={"help": "How to retrieve and sample knn_ids; one of default, topk, random."}
    )

    def __post_init__(self):
        if self.model_name_or_path in {'None', '', ' '}:
            self.model_name_or_path = None
        if isinstance(self.model_name_or_path, str) and self.model_name_or_path.startswith('~'):
            self.model_name_or_path = os.path.expanduser("~") + self.model_name_or_path[1:]
        if self.config_name in {'None', '', ' '}:
            self.config_name = None
        if isinstance(self.config_name, str) and self.config_name.startswith('~'):
            self.config_name = os.path.expanduser("~") + self.config_name[1:]
        if self.tokenizer_name in {'None', '', ' '}:
            self.tokenizer_name = None
        if isinstance(self.tokenizer_name, str) and self.tokenizer_name.startswith('~'):
            self.tokenizer_name = os.path.expanduser("~") + self.tokenizer_name[1:]
        if self.embedding_dropout_prob is not None:
            self.embedding_dropout_prob = eval(self.embedding_dropout_prob)
        if self.hidden_dropout_prob is not None:
            self.hidden_dropout_prob = eval(self.hidden_dropout_prob)
        if self.sample_retrieved_dynamic_negatives > 0:
            if self.retrieve_dynamic_negatives + self.ignore_num_top_retrieved_dynamic_negatives < self.sample_retrieved_dynamic_negatives + 1:
                raise ValueError(
                    'retrieve_hard_negatives has to be at least sample_retrieved_hard_negatives + ignored_num + 1. got {}, {} and {}'.format(
                        self.retrieve_dynamic_negatives, self.ignore_num_top_retrieved_dynamic_negatives, self.sample_retrieved_dynamic_negatives))

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case validation_file is not provided."
        },
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mask_token_rate: float = field(
        default=0.8, metadata={"help": "Ratio of masked tokens to be replaced with [MASK]."}
    )
    # extended arguments
    switch_case_probability: float = field(
        default=0.0, metadata={"help": "Ratio of words to switch case."}
    )
    switch_case_method: str = field(
        default='v1', metadata={
            "help": "Assign 01, 10 or 11 to consider samples from switch case augmentation."
        }
    )
    retrieve_hard_negatives: int = field(
        default=0, 
        metadata={"help": "The number of hard negatives to retrieve using the model representations; 0 to disable it."}
    )
    sample_retrieved_hard_negatives: int = field(
        default=0, 
        metadata={"help": "The number of hard negatives to sample from retrieved negatives; 0 to disable it."}
    )
    sent_rep_cache_file: Optional[str] = field(
        default=None, metadata={"help": "Path to saved sentence representation file."})
    retriever_results_cache_file: Optional[str] = field(
        default=None,
        metadata={"help": "Filename of the saved retrieved scores and indices."})
    SentEval_data_dir: str = field(
        default=None, metadata={"help": "Path to SentEval data folder (which contains downstream folder)."})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.validation_file in {'None', '', ' '}:
                self.validation_file = None
            if self.train_file is not None:
                self.train_file = FileName(self.train_file)
                assert self.train_file.extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                self.validation_file = FileName(self.validation_file)
                assert self.validation_file.extension in ["csv", "json", "txt", 'npy'], "`validation_file` should be a csv, a json or a txt file."
        if self.sent_rep_cache_file in {'None', '', ' '}:
            self.sent_rep_cache_file = None
        if self.retriever_results_cache_file in {'None', '', ' '}:
            self.retriever_results_cache_file = None
        if self.preprocessing_num_workers is not None and self.preprocessing_num_workers < 0:
            self.preprocessing_num_workers = None
        if self.sample_retrieved_hard_negatives > 0:
            if self.retrieve_hard_negatives < self.sample_retrieved_hard_negatives + 1:
                raise ValueError(
                    'retrieve_hard_negatives has to be at least sample_retrieved_hard_negatives + 1. got {} and {}'.format(
                        self.retrieve_hard_negatives, self.sample_retrieved_hard_negatives))
        assert self.switch_case_method in {'v1', 'v2', 'substitution', 'retokenization'}, NotImplementedError(
            'switch_case_method {} not implemented.'.format(self.switch_case_method))
        assert os.path.isdir(self.SentEval_data_dir), FileExistsError('A valid SentEval data folder must be provided, got {}'.format(self.SentEval_data_dir))

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.
    See all possible arguments in transformers/training_args.py or by passing the --help flag to this script.

    Usage:
    training_args = ExtendedTrainingArguments(...)

    """
    eval_during_train: bool = field(
        default=False,
        metadata={
            "help": "By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate "
            "both STS and transfer tasks (dev) at the end of training. Using --eval_during_train will allow evaluating"
            "both STS and transfer tasks (dev) during training."
        }
    )
    task_set: str = field(
        default='all',
        metadata={
            "help": "Pass task_set to select STS and transfer tasks (dev) during evaluation."
            }
    )
    ignore_trainer_state: bool = field(
        default=True,
        metadata={
            "help": "When continue training from a checkpoint, whether to load the trainer_state.json or not."
            "If False, will continue from the epochs_trained in trainer_state.json"
        },
    )
    debug_mode: int = field(
        default=0, metadata={"help": "In debug mode, norms of parameters will be added to tensorboard."},
    )
    halt_step: Optional[int] = field(
        default=-1, metadata={"help": "Whether to stop training after a certain number of steps."},
    )
    remove_checkpoints_at_end: bool = field(
        default=True, metadata={"help": "Whether to remove all checkpoints after loading the best one."})
    update_buffer_representations_steps: Optional[int] = field(
        default=-1, metadata={"help": "Whether to stop training after a certain number of steps."},
    )
    ignore_upper_case_phase: Optional[str] = field(
        default='never', metadata={"help": "Force model to ignore the word case during 'train', 'eval' or 'both'."})
    get_align_uniform: Optional[bool] = field(
        default=None, metadata={"help": "Whether to calculate alignment and uniformity during evaluation."})

    def __post_init__(self):
        super().__post_init__()
        if self.task_set.startswith('[') and self.task_set.endswith(']'):
            self.task_set = ast.literal_eval(self.task_set)
        assert self.ignore_upper_case_phase in {'train', 'eval', 'both', 'never'}, ValueError(
            'Invalid value {} for ignore_upper_case_phase'.format(self.ignore_upper_case_phase))


class UpdateBufferRepresentationsFlowCallback(DefaultFlowCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # DefaultFlowCallback does not implement on_step_begin
        control.should_update_buffer_representations = False

        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        # update buffer_representations
        if state.global_step % args.update_buffer_representations_steps == 0 and model.config.sample_retrieved_dynamic_negatives > 0:
            control.should_update_buffer_representations = True

        return control


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtendedTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    # __post_init__ check across args
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    if model_args.sample_retrieved_dynamic_negatives > 0 and data_args.sample_retrieved_hard_negatives > 0:
        raise ValueError(
            "Do not support sample_retrieved_hard_negatives and sample_retrieved_dynamic_negatives to be both positive.")
    if (
        (data_args.switch_case_probability == 0 and model_args.switch_case_pattern != '00')
        or (data_args.switch_case_probability != 0 and model_args.switch_case_pattern == '00')
    ):
        logger.warning('When eith switch_case_probability is 0 or switch_case_pattern is 00, the other one must also be 00 or 0.')
        data_args.switch_case_probability = 0
        model_args.switch_case_pattern = '00'
    if (
        training_args.ignore_upper_case_phase in {'train', 'both'}
        and model_args.switch_case_pattern != '00' 
        and data_args.switch_case_probability > 0
    ):
        raise ValueError('Cannot simutaneously ignore upper case and switch case.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Data args %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.train_file.extension == "csv":
        datasets = load_dataset(
            './data/datasets/huggingface_csv.py', 
            data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(
            './data/datasets/huggingface_txt.py', 
            data_files=data_files, cache_dir="./data/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = CONFIG_MAPPING[model_args.model_type].from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        **config_kwargs)
    assert model_args.model_type == config.model_type, ValueError(
        'model_args.model_type does not match config.model_type, got {} and {}.'.format(model_args.model_type, config.model_type))
    # add new model_args if they have not been added.
    new_keys = [  # new_keys are not in roberta/bert config
        'cl_temperature', 'pooler_type', 'hard_negative_weight', 'mlm_loss_weight', 'discard_embd_proj',
        'label_smoothing',
        'switch_case_pattern', 'symmetric_contrastive_loss', 
        'retrieve_dynamic_negatives', 'sample_retrieved_dynamic_negatives', 'ignore_num_top_retrieved_dynamic_negatives',
        'share_sampled_dynamic_negatives', 'retriever_metric',
        'update_retrieved_negatives_representations', 'detach_retrieved_negatives_representations', 
        'retrieve_using_buffer_representations', 'save_buffer_representations', 'retrieved_and_sampled_knn_type',
    ]
    for key in new_keys:
        if not hasattr(config, key):
            setattr(config, key, getattr(model_args, key, None))
    overwrite_keys =[  # overwrite_keys are in roberta/bert config
        'hidden_dropout_prob', 'embedding_dropout_prob'
    ]
    for key in overwrite_keys:
        if hasattr(config, key) and getattr(model_args, key, None) is not None:
            setattr(config, key, getattr(model_args, key, None))
    logger.info("Updated model config %s", config)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    logger.info("========================= Tokenizer configured. =========================")

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]  # 'text'
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples, indices=None):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        if training_args.ignore_upper_case_phase in {'train', 'both'}:
            sentences = [s.lower() for s in sentences]

        # sent_features (:obj:`transformers.tokenization_utils_base.BatchEncoding`)
        # - sent_features['input_ids'] is a list of sentences (in digits), not padded
        # - sent_features['attention_mask'] is a list of [1,...,1]s; since sent_features['input_ids'] are not padded, 
        #   sent_features['attention_mask'] contains only 1s. 
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        
        # features is sent_features rearranged, as dict.
        # - features['input_ids'] is a list of list of sentences (in digits), not padded
        # - features['attention_mask'] is a list of list of [1,...,1]s; since features['input_ids'] are not padded, 
        #   features['attention_mask'] contains only 1s. 
        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        if indices is not None:
            features['ids'] = indices
            
        return features

    if training_args.do_train:
        """
        train_dataset has the following structure:
            '_info': DatasetInfo()
            '_split': NamedSplit('train')
            '_indexes': {}
            '_data': MemoryMappedTable['attention_mask': pyarrow.lib.ChunkedArray, 'input_ids': pyarrow.lib.ChunkedArray]
            '_format_type': None
            '_format_kwargs': {}
            '_format_columns': None
            '_output_all_columns': False
            '_fingerprint': ...
        """
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            batch_size= 2048,
            writer_batch_size= 2048,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            with_indices=data_args.sample_retrieved_hard_negatives > 0 or model_args.sample_retrieved_dynamic_negatives > 0,
            # somehow does not work with cache_file_name
            # cache_file_name=f"{data_args.train_file.path}_{model_args.model_type}_cached_{sent0_cname}_{sent1_cname}_{sent2_cname}_MaxLen{data_args.max_seq_length}",
        )
    logger.info("========================= Datasets loaded. =========================")
    if data_args.pad_to_max_length:
        data_collator = default_data_collator 
    else: 
        data_collator = DataCollatorForPairedLanguageModeling(
            tokenizer, 
            mlm_probability=data_args.mlm_probability if model_args.mlm_loss_weight > 0 else 0, 
            mask_token_rate=data_args.mask_token_rate,
            # max_length=data_args.max_seq_length,  # TODO check this
            # pad_to_multiple_of=data_args.pad_to_multiple_of,
            switch_case_pattern=model_args.switch_case_pattern,
            switch_case_probability=data_args.switch_case_probability if model_args.switch_case_pattern != '00' else 0,
            switch_case_method=data_args.switch_case_method,
            )
    logger.info("========================= Data_collator configured. =========================")

    SimCSEForSequenceEmbedding.base_model_prefix = model_args.model_type
    SimCSEForSequenceEmbedding.config_class = CONFIG_MAPPING[model_args.model_type]
    model = SimCSEForSequenceEmbedding.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        **config_kwargs,
        )
    if model_args.mlm_loss_weight > 0 and model_args.model_type == 'bert':  # TODO check if this works for roberta
        pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
        model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

    # Resizes input token embeddings matrix of the model if :obj:`num_tokens != config.vocab_size`.
    # See modeling_utils.PreTrainedModel.resize_token_embeddings for details.
    num_tokens = len(tokenizer)
    if num_tokens != config.vocab_size:
        model.resize_token_embeddings(num_tokens)
        logger.info(
            "config.vocab_size {} does not match tokenizer length {}. "\
                "Token embeddings are resized.".format(config.vocab_size, num_tokens))
    else:
        logger.info("Tokenizer length is {}".format(num_tokens))
    model.tokenizer = tokenizer  # for debugging purpose

    logger.info("========================= Model configured. =========================")

    # Initialize our Trainer with custom callbacks
    # note if custom callbacks are implemented, the default callbacks need to be removed.
    callbacks = []
    callbacks_to_remove = []
    if training_args.halt_step > -1:
        halt_callback = HaltTrainingCallback(halt_step=training_args.halt_step)
        callbacks.append(halt_callback)
    if training_args.disable_tqdm and hasattr(model, 'printer_callback'):
        callbacks.append(model.printer_callback)
        callbacks_to_remove.append(PrinterCallback)
    if training_args.update_buffer_representations_steps > 0:
        callbacks.append(UpdateBufferRepresentationsFlowCallback)
        callbacks_to_remove.append(DefaultFlowCallback)
    if training_args.debug_mode:
        if hasattr(model, "trainer_callback"):
            callbacks.append(model.trainer_callback)
            callbacks_to_remove.append(DefaultFlowCallback)
        if hasattr(model, "tensorboard_callback"):
            callbacks.append(model.tensorboard_callback)
            callbacks_to_remove.append(TensorBoardCallback)
        if hasattr(model, "wandb_callback"):
            callbacks.append(model.wandb_callback)
            callbacks_to_remove.append(WandbCallback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    # trainer.model_args = model_args  # we do not need this any more
    trainer.SentEval_data_dir = data_args.SentEval_data_dir
    for cb in [cb for cb in trainer.callback_handler.callbacks if type(cb) in callbacks_to_remove]:
        trainer.remove_callback(cb)
    model.set_amp(trainer.use_amp)
    logger.info("========================= Trainer configured. =========================")

    # Training
    if training_args.do_train:
        if data_args.sample_retrieved_hard_negatives > 0:
            trainer = update_train_dataset_with_retrieved_negatives(trainer, data_args)

        if model_args.sample_retrieved_dynamic_negatives > 0:
            representations = trainer.get_dataset_representations(
                trainer.train_dataset, 
                cache_file=data_args.sent_rep_cache_file, 
                load_from_cache_file=True,
                discard_embd_proj=True,
                ).astype(np.float32)  # numpy.ndarray
            model.buffer_dataset = trainer.train_dataset
            model.buffer_representations = torch.tensor(
                representations, device=training_args.device, 
                dtype=torch.float16 if training_args.fp16 else torch.float32, 
                requires_grad=False)
            if model.config.retriever_metric == 'cos':
                model.buffer_representations = F.normalize(model.buffer_representations, dim=1)
            model.buffer_collator = data_collator
            if hasattr(model, 'buffer_retrieve_counts'):
                model.buffer_retrieve_counts = torch.zeros(
                    (representations.shape[0],), device=training_args.device, dtype=torch.int64, requires_grad=False)

        logger.info("*** Training ***")
        # rename trainer_state.json, so to avoid trainer.train loading trainer_state.json to check epochs_trained
        ckpt_file_handler = RenameCKPTFiles(model_args.model_name_or_path)
        if (
            model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            and training_args.ignore_trainer_state
        ):
            ckpt_file_handler.rename_files()

        train_result = trainer.train(
            resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        # rename trainer_state.json back
        # if (
        #     model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        #     and training_args.ignore_trainer_state
        # ):
        #     ckpt_file_handler.restore_file_names()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

        logger.info("========================= Training completed. =========================")
    
    if training_args.load_best_model_at_end and training_args.remove_checkpoints_at_end and trainer.is_world_process_zero():
        checkpoints = trainer._sorted_checkpoints(training_args.output_dir)
        for checkpoint in checkpoints:
            logger.info("Deleting older checkpoint [{}] ".format(checkpoint))
            shutil.rmtree(checkpoint)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        eval_results = trainer.evaluate(eval_senteval=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(eval_results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
        logger.info("========================= Evaluation completed. =========================")

    return eval_results


def update_train_dataset_with_retrieved_negatives(trainer, data_args):

    logger.info("========================= Updating datasets with retrieved negatives. =========================")

    _, retrieved_indices = trainer.get_representation_retrieved_results(
        cache_file=data_args.retriever_results_cache_file, 
        sent_rep_cache_file=data_args.sent_rep_cache_file, 
        load_from_cache_file=True,
        retrieve_hard_negatives=data_args.retrieve_hard_negatives,
        discard_embd_proj=True,
        )

    def prepare_retrieved_negatives(examples):
        # examples is sent_features rearranged, as dict.
        # - examples['input_ids'] is a list of list of sentences (in digits), not padded.
        # - examples['attention_mask'] is a list of list of [1,...,1]s; since examples['input_ids'] are not padded, 
        #   examples['attention_mask'] contains only 1s. 
        # - examples['ids'] is a list of indices, showing the position of samples in datasets.
        
        total = len(examples['input_ids'])  # N
        # get retrieved negatives
        retrieved_nn_ids = retrieved_indices[examples['ids']]  # (N, retrieve_hard_negatives), numpy.ndarray, may include the query sample itself
        # exclude itself
        # knn_ids, (N, K)
        knn_ids = [np.array([k for k in kplus1 if k!= query], dtype=np.int) if query in kplus1 else kplus1[:-1] for kplus1, query in zip(retrieved_nn_ids, examples['ids'])]
        knn_ids = np.array(knn_ids, dtype=np.int)  # (N, retrieve_hard_negatives - 1), numpy.ndarray
        # shuffle to select from the first sample_retrieved_hard_negatives
        if data_args.sample_retrieved_hard_negatives < data_args.retrieve_hard_negatives - 1:
            idx = np.random.rand(*knn_ids.shape).argsort(axis=1)
            knn_ids = np.take_along_axis(knn_ids, idx, axis=1)
        features = copy.deepcopy(examples)  # avoid corrupting the dataset
        # convert a list to a nested list
        features['ids'] = [[ids,] for ids in features['ids']]
        # append kth_neighbors info to features
        for k in range(data_args.sample_retrieved_hard_negatives):
            kth_ids = knn_ids[:, k]  # (N,), numpy.ndarray
            kth_neighbors = trainer.train_dataset[kth_ids]  # N OrderedDict('input_ids':, 'attention_mask')
            features['input_ids'] = [features['input_ids'][i] + [kth_neighbors['input_ids'][i][0],] for i in range(total)]
            features['attention_mask'] = [features['attention_mask'][i] + [kth_neighbors['attention_mask'][i][0],] for i in range(total)]
            features['ids'] = [features['ids'][i] + [kth_neighbors['ids'][i],] for i in range(total)]

        return features

    trainer.train_dataset = trainer.train_dataset.map(
        prepare_retrieved_negatives,
        batched=True,
        batch_size= 2048,
        writer_batch_size= 2048,
        num_proc=data_args.preprocessing_num_workers,
        # load_from_cache_file=not data_args.overwrite_cache,
        load_from_cache_file=False,
        # somehow does not work with cache_file_name
        # cache_file_name=f"{data_args.train_file.path}_{model_args.model_type}_cached_{sent0_cname}_{sent1_cname}_{sent2_cname}_MaxLen{data_args.max_seq_length}",
    )
    logger.info("========================= Datasets updated with retrieved negatives. =========================")
    logger.info("Example: {}".format(trainer.train_dataset[0]))

    return trainer


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
