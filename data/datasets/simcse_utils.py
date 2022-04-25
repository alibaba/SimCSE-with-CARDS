# coding=utf-8
# Copyright 2022 The DAMO Academy All rights reserved.
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
Code to generate pooler output given data and checkpoint

"""
import os
import sys
sys.path.append(os.getcwd())  # cd to working directory before doing this
import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from training.simcse_trainer import CLTrainer as Trainer
from transformers.trainer_callback import PrinterCallback
from transformers import TrainingArguments
from data.data_collator import DataCollatorForPairedLanguageModeling
from models.bert.modeling_simcse import SimCSEForSequenceEmbedding
from models.bert import RobertaConfig, BertConfig

CONFIG_MAPPING = {'bert': BertConfig, 'roberta': RobertaConfig}


def generate_pooler_output_from_txt_data(
    model_name_or_path, 
    pooler_output_path,
    train_file, 
    model_type='roberta', 
    max_seq_length=32, 
    config_name=None,
    pooler_type=None,
    ):
    data_files = {"train": train_file}
    datasets = load_dataset('./data/datasets/huggingface_txt.py', data_files=data_files, cache_dir="./data/")

    tokenizer_kwargs = {
        "cache_dir": '../.cache',
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    column_names = datasets["train"].column_names
    sent0_cname = column_names[0]  # 'text'
    sent1_cname = column_names[0]
    sent2_cname = None

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

        # sent_features (:obj:`transformers.tokenization_utils_base.BatchEncoding`)
        # - sent_features['input_ids'] is a list of sentences (in strings), not padded
        # - sent_features['attention_mask'] is a list of [1,...,1]s; since sent_features['input_ids'] are not padded, 
        #   sent_features['attention_mask'] contains only 1s. 
        sent_features = tokenizer(
            sentences,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
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

    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        batch_size= 2048,
        writer_batch_size= 2048,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        with_indices=True,
        # somehow does not work with cache_file_name
        # cache_file_name=f"{data_args.train_file.path}_{model_args.model_type}_cached_{sent0_cname}_{sent1_cname}_{sent2_cname}_MaxLen{data_args.max_seq_length}",
    )

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"), "Data loaded, #entries {}.".format(len(train_dataset)))

    data_collator = DataCollatorForPairedLanguageModeling(
        tokenizer,
        mlm_probability=0,
        mask_token_rate=0.8,
    )

    config_kwargs = {
        "cache_dir": '../.cache',
        "revision": "main",
        "use_auth_token": None,
    }

    if config_name is None:
        config_name = os.path.join(model_name_or_path, 'config.json')
    config = CONFIG_MAPPING[model_type].from_pretrained(
        config_name,
        **config_kwargs)
    config.symmetric_contrastive_loss = False
    config.mlm_loss_weight = 0
    if getattr(config, 'pooler_type', None) is None:
        config.pooler_type = pooler_type or 'cls'
    if getattr(config, 'label_smoothing', None) is None:
        config.label_smoothing = 0
    if getattr(config, 'save_buffer_representations', None) is None:
        config.save_buffer_representations = False
    if getattr(config, 'sample_retrieved_dynamic_negatives', None) is None:
        config.sample_retrieved_dynamic_negatives = 0
    if getattr(config, 'discard_embd_proj', None) is None:
        config.discard_embd_proj = True
    if getattr(config, 'update_retrieved_negatives_representations', None) is None:
        config.update_retrieved_negatives_representations = False 

    SimCSEForSequenceEmbedding.base_model_prefix = model_type
    SimCSEForSequenceEmbedding.config_class = CONFIG_MAPPING[model_type]
    model = SimCSEForSequenceEmbedding.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config,
        **config_kwargs
        )
    model.tokenizer = tokenizer  # for debugging purpose

    callbacks = []
    callbacks_to_remove = []
    callbacks.append(model.printer_callback)
    callbacks_to_remove.append(PrinterCallback)

    training_args = TrainingArguments(
        output_dir='../output/debug',
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=1e-5,
        num_train_epochs=1,
        evaluation_strategy='steps',
        eval_steps=250,
        do_train=True,
        metric_for_best_model='stsb_spearman',
        logging_steps=50,
        dataloader_num_workers=4,
        fp16=True,
        fp16_opt_level='O2',
        fp16_full_eval=True,
        disable_tqdm=True,
        report_to='tensorboard',
        ddp_find_unused_parameters=False,
        )
    training_args.eval_during_train=False
    training_args.task_set='all'
    # training_args.logging_steps=50
    training_args.halt_step=-1

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    for cb in [cb for cb in trainer.callback_handler.callbacks if type(cb) in callbacks_to_remove]:
        trainer.remove_callback(cb)
    model.set_amp(trainer.use_amp)

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"), "Model loaded. Getting pooler_output.")

    model.get_sent_emb_only = True
    original_discard_embd_proj = model.discard_embd_proj
    model.discard_embd_proj = True

    outputs = trainer.predict(train_dataset)  # PredictionOutput(predictions, label_ids, metrics)

    model.get_sent_emb_only = False
    model.discard_embd_proj = original_discard_embd_proj

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"), "Pooler_output predicted, info {}.".format(
        (type(outputs.predictions), outputs.predictions.shape, outputs.predictions.dtype)
        ))

    torch.save(torch.tensor(outputs.predictions), pooler_output_path)
    print('pooler_output saved to', pooler_output_path)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # train_file="../../dataset/SimCSE/wiki1m_for_simcse.txt"
    # train_file="../../dataset/SimCSE/wiki1m_deduplicate_for_simcse.txt"
    train_file = '../../dataset/SimCSE/wiki1m_deduplicate_for_simcse.937207.selected.txt'
    
    # model_name_or_path = "../output/bert_large/pretraining/roberta_large_baseline"
    model_name_or_path = '../output/bert_base/pretraining/roberta_base_baseline'
    # model_name_or_path = '../output/bert_large/pretraining/bert_large_cased_baseline'
    config_name = None

    # pooler_output_path = '../../dataset/SimCSE/roberta_large_cached_pooler_output/wiki1m_deduplicate_for_simcse.937207.selected.roberta_large_pooler_output.pt'
    pooler_output_path = '../../dataset/SimCSE/roberta_base_cached_pooler_output/wiki1m_deduplicate_for_simcse.937207.selected.roberta_base_avg_pooler_output.pt'
    # pooler_output_path = '../../dataset/SimCSE/bert_large_cached_pooler_output/wiki1m_deduplicate_for_simcse.937207.selected.bert_large_cased_avg_pooler_output.pt'
    # model_type = 'roberta'
    model_type = 'bert'
    pooler_type = 'avg'  # cls, cls_before_pooler, avg, demean, avg_top2, avg_first_last

    generate_pooler_output_from_txt_data(
        model_name_or_path, 
        pooler_output_path,
        train_file, 
        model_type=model_type, 
        max_seq_length=32, 
        config_name=None,
        pooler_type=pooler_type,
    )