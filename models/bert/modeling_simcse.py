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
"""PyTorch SimCSE model. """

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel, get_parameter_dtype
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from .modeling_roberta import RobertaModel, RobertaLMHead, RobertaEncoder, RobertaConfig
from .modeling_bert import BertModel, BertLMPredictionHead, BertEncoder, BertConfig
from utils.callbacks import EditableLogPrinterCallback
from utils.distributed import all_gather
from data.metrics import get_accuracy

import faiss.contrib.torch_utils  # this will replace faiss.knn_gpu with torch_utils.torch_replacement_knn_gpu
import faiss

logger = logging.get_logger(__name__)

CONFIG_MAPPING = {'bert': BertConfig, 'roberta': RobertaConfig}
RETRIEVER_METRIC = {'ip': faiss.METRIC_INNER_PRODUCT, 'l2': faiss.METRIC_L2, 'cos': faiss.METRIC_INNER_PRODUCT}


class EmbeddingProjectionLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class SentenceEmbeddingPooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding.

    Args:
        config.pooler_type (:obj:`str`, defaults to :obj:`"cls"`):
            "cls": extract [CLS] representation, to be used with BERT/RoBERTa's MLP embedding projector.
            "cls_before_pooler": extract [CLS] representation, to be used without the original MLP embedding projector.
            "avg": get average of the last layers' hidden states at each token.
            "avg_before_pooler": extract average of the last layers' hidden states, to be used without the original MLP embedding projector.
            "avg_top2": get average of the last two layers.
            "avg_first_last": get average of the first and the last layers.
    """
    def __init__(self, config):
        super().__init__()
        self.pooler_type = config.pooler_type.lower()
        assert self.pooler_type in [
            "cls", "cls_before_pooler", 
            "avg", "avg_before_pooler", 
            "avg_first_last", "avg_first_last_before_pooler",
            "avg_top2", "avg_top2_before_pooler",
            "demean",
            ], NotImplementedError("Unrecognized pooling type {}".format(self.pooler_type))

    def forward(self, outputs, attention_mask, return_dict):
        if return_dict:
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
        else:
            last_hidden_state = outputs[0]
            hidden_states = outputs[2] if len(outputs) > 2 else None

        if self.pooler_type in ["cls", "cls_before_pooler"]:
            pooled_result = last_hidden_state[:, 0]
        elif self.pooler_type in ["avg", "avg_before_pooler"]:
            pooled_result = ((last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type in ["demean"]:
            pooled_result = last_hidden_state[:, 0] - ((last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type in ["avg_first_last", "avg_first_last_before_pooler"]:
            first_hidden_state = hidden_states[0]
            last_hidden_state = hidden_states[-1]
            pooled_result = ((first_hidden_state + last_hidden_state) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type in ["avg_top2", "avg_top2_before_pooler"]:
            second_last_hidden_state = hidden_states[-2]
            last_hidden_state = hidden_states[-1]
            pooled_result = ((last_hidden_state + second_last_hidden_state) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError

        return pooled_result


class SimCSEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = None  # needs to be set before calling cls.from_pretrained
    base_model_prefix = None  # needs to be set before calling cls.from_pretrained
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        if self.config_class is None:
            self.config_class = CONFIG_MAPPING[config.model_type]
        if self.base_model_prefix is None:
            self.base_model_prefix = config.model_type
        
        super().__init__(config)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BertEncoder, RobertaEncoder)):
            module.gradient_checkpointing = value

    @property
    def dtype(self) -> torch.dtype:
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).

        Note that under torch.cuda.amp, the module parameters will be float32, so deciding self.dtype based on 
        parameters dtype is not accurate.
        """
        return torch.float16 if self.use_amp and self.training else get_parameter_dtype(self)

    def set_amp(self, use_amp):
        self.use_amp = use_amp
        self.apply(self._set_amp)

    def _set_amp(self, module):
        if isinstance(module, PreTrainedModel):
            module.use_amp = self.use_amp


class SimCSEForSequenceEmbedding(SimCSEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = None
        self.roberta = None
        self.lm_head = None
        if config.model_type == "bert":
            self.bert = BertModel(config, add_pooling_layer=False)
            if config.mlm_loss_weight > 0:
                self.lm_head = BertLMPredictionHead(config)
        elif config.model_type == "roberta":
            self.roberta = RobertaModel(config, add_pooling_layer=False)
            if config.mlm_loss_weight > 0:
                self.lm_head = RobertaLMHead(config)
        else:
            raise NotImplementedError("SimCSE for {} is not implemneted.".format(config.model_type))

        self.pooler = SentenceEmbeddingPooler(config)
        if self.config.pooler_type in {"cls", "demean", "avg", "avg_first_last", "avg_top2"}:
            self.embedding_projector = EmbeddingProjectionLayer(config)
        
        self.cos = nn.CosineSimilarity(dim=-1)
        self.printer_callback = EditableLogPrinterCallback()
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.init_weights()

        # extended attributes
        self.tokenizer = None  # will be assigned in run_simcse.py

        # buffer
        self.buffer_dataset = None
        if self.config.update_retrieved_negatives_representations and config.save_buffer_representations:
            self.register_buffer("buffer_representations", None)  # if stay None, will not be saved
            self.register_buffer("buffer_retrieve_counts", None)
        else:
            self.buffer_representations = None
        
        if self.config.sample_retrieved_dynamic_negatives > 0:
            self.buffer_res = faiss.StandardGpuResources()
            self.buffer_res.noTempMemory()  # by default, StandardGpuResources reserve 1.5 GB memory

        self.buffer_collator = None
        # flags
        self.get_sent_emb_only = False
        self.discard_embd_proj = config.discard_embd_proj

    def get_sentence_embedding(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids.ndim == 3:  # (N, K, L), when given a set of inputs, use the first one
            input_ids = input_ids[:, 0]
            attention_mask = attention_mask[:, 0]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, 0]

        encoder = self.bert or self.roberta
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=return_dict,
        )

        pooler_output = self.pooler(outputs, attention_mask, return_dict)
        if self.config.pooler_type in {"cls", "demean", "avg", "avg_first_last", "avg_top2"} and not self.discard_embd_proj:
            pooler_output = self.embedding_projector(pooler_output)

        if not return_dict:
            if self.get_sent_emb_only:
                return (pooler_output,)
            else:
                return (outputs[0], pooler_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=None if self.get_sent_emb_only else outputs.last_hidden_state,
            hidden_states=None if self.get_sent_emb_only else outputs.hidden_states,
        )

    def _gather(self, tensor):
        # Note we used all_gather that permits passing gradients to other gpus to reduce memory usage.
        # This is different to that of SimCSE Github
        # In our case, the contribution of each samples to loss is scaled by per_device_train_batch_size; then gradient 
        # is scaled by the total batch size.
        # In the case of SimCSE code, the sample is scaled by per_device_train_batch_size * num_gpus, and the gradient 
        # per_device_train_batch_size * num_gpus * num_gpus.
        # So, we need to downscale the learning rate in order to be comparable to that of SimCSE.
        if dist.is_initialized() and self.training:
            # with torch.no_grad():  # all_gather does not pass gradients; we can make this more explicit using no_grad
            #     # Dummy vectors for all_gather
            #     tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            #     # all_gather
            #     dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
            # # Since all_gather results do not have gradients, we replace the
            # # current process's corresponding embeddings with original tensors
            # tensor_list[dist.get_rank()] = tensor
            tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            tensor_list = all_gather(tensor_list, tensor.contiguous())
            # Get full batch embeddings: (bs x N, hidden)
            tensor = torch.cat(tensor_list, 0)

        return tensor

    @torch.no_grad()
    def get_knn_ids(self, query_tensors=None, query_ids=None):
        """ Search in self.buffer_representations the nearest neighbors to query_tensors.

        Args:
            query_tensors: (N, D), note faiss.knn_gpu requires it be contiguous.
            query_ids: (N, ), index of query_tensors in self.buffer_representations.
        """
        
        if query_tensors is None:
            query_tensors = self.buffer_representations[query_ids]

        if self.config.retrieved_and_sampled_knn_type == 'random':
            batch_size = query_tensors.shape[0]
            index_size = self.buffer_representations.shape[0]
            retrieved_nn_ids = torch.randint(
                index_size, (batch_size, self.config.retrieve_dynamic_negatives), 
                dtype=torch.int64, device=self.device)
        else:
            if not query_tensors.is_contiguous():
                query_tensors = query_tensors.contiguous()

            if self.config.retriever_metric == 'cos':
                query_tensors = F.normalize(query_tensors, dim=1).to(query_tensors.dtype)
                # query_tensors /= query_tensors.norm(dim=1, keepdim=True)

            # retrieved_nn_ids: tensor, torch.int64
            _, retrieved_nn_ids = faiss.knn_gpu(  # (N, K1)
                self.buffer_res, query_tensors, 
                self.buffer_representations, 
                self.config.retrieve_dynamic_negatives, 
                metric=RETRIEVER_METRIC[self.config.retriever_metric])

        if self.config.retrieved_and_sampled_knn_type == 'topk':
            batch_size = query_tensors.shape[0]
            decision_matrix = torch.arange(  # (N, K1)
                self.config.retrieve_dynamic_negatives, device=self.device).unsqueeze(dim=0).repeat(batch_size, 1)
            decision_matrix[torch.eq(retrieved_nn_ids, query_ids.unsqueeze(1))] = self.config.retrieve_dynamic_negatives
        else:
            decision_matrix = torch.rand_like(retrieved_nn_ids, dtype=query_tensors.dtype)
            decision_matrix[torch.eq(retrieved_nn_ids, query_ids.unsqueeze(1))] = 1.01
        _, indices = decision_matrix.sort(dim=1, descending=False)
        knn_ids = torch.gather(retrieved_nn_ids, 1, indices)  # (N, K1)

        ind_start = self.config.ignore_num_top_retrieved_dynamic_negatives
        ind_end = self.config.ignore_num_top_retrieved_dynamic_negatives + self.config.sample_retrieved_dynamic_negatives
        sampled_knn_ids = knn_ids[:, ind_start:ind_end]  # (N, K2)

        return sampled_knn_ids.contiguous()  # (N, K2)

    def get_cos_sim(self, z1, z2, z2_weight=None, z1_ids=None, z2_ids=None):
        """ Get pairwise weighted cos_sim matrix between z1 and z2.

        Args:
            z1: shape (N, D).
            z2: shape (N, D), or (N, K, D); in the latter case, each z1[i] corresponds to K z2[i]. 
            z2_weight: optional; if provided, for each z1[i], its corresponding z2[i] will be weighted.
            z1_ids, z2_ids: optional, ids of z1, z2 in the dataset; if provided, for each z1[i], z2[j] (i != j) 
                will be excluded from softmax calculation, if z1_ids[i] = z2_ids[j]. z1_ids shape (N, ); z2_ids
                shape (N, ) or (N, K).
        """
        batch_size = z1.shape[0]
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        # Gather all embeddings if using distributed training to increase the number of negative samples
        # z1 = self._gather(z1)  # no need to gather z1
        z2 = self._gather(z2)  # (N * num_gpus, D), or (N * num_gpus, K, D)

        if z2.ndim == 3:
            k = z2.shape[1]
            z2 = z2.reshape(-1, self.config.hidden_size)  # ((K * N) * num_gpus, D)
        else:
            k = 1

        cos_sim = self.cos(  # (N, (K * N) * num_gpus)
            z1.unsqueeze(1).to(torch.float32), z2.unsqueeze(0).to(torch.float32)) / self.config.cl_temperature
        
        # apply weights to z2[i] for each z1[i]
        if z2_weight is not None:
            if dist.is_initialized() and dist.get_rank() > 0:
                weights = torch.zeros(batch_size, batch_size * num_gpus, dtype=cos_sim.dtype, device=cos_sim.device)  # (N, N * num_gpus)
                weights[(torch.arange(batch_size), dist.get_rank()*batch_size+torch.arange(batch_size))] = z2_weight
            else:
                weights = torch.eye(batch_size, batch_size * num_gpus, dtype=cos_sim.dtype, device=cos_sim.device) * z2_weight  # (N, N * num_gpus)
            if k > 1:
                weights = weights.unsqueeze(2).repeat(1, 1, k).reshape(cos_sim.shape[0], -1)  # (N, (K * N) * num_gpus)
            cos_sim += weights

        # apply weights to z2[j] for each z1[i] if z1_ids[i] = z2_ids[j] and i != j
        # this is to avoid false negative
        if z1_ids is not None and z2_ids is not None:
            # Gather all ids if using distributed training
            # z1_ids = self._gather(z1_ids).view(-1, 1)  # (N * num_gpus, 1)
            z1_ids = z1_ids.view(-1, 1)  # (N, 1)
            z2_ids = self._gather(z2_ids).view(-1, 1)  # (N * num_gpus, 1) or ((K * N) * num_gpus, 1)
            weights = torch.eq(z1_ids, z2_ids.T).to(cos_sim.dtype) * -10000 # (N, (K * N) * num_gpus)
            cos_sim += weights

        return cos_sim

    def get_samplewise_cos_sim(self, z1, z2, z2_weight=None):
        """ Get samplewise weighted cos_sim matrix between each row of z1 and z2. We do not do all_gather in this case.

        Args:
            z1: shape (N, D).
            z2: shape (N, D), or (N, K, D); in the latter case, each z1[i] corresponds to K z2[i]. 
            z2_weight: optional; if provided, for each z1[i], its corresponding z2[i] will be weighted.
        """
        if z2.ndim == 2:
            z2 = z2.view(-1, 1, self.config.hidden_size)  # (N, 1, D)

        cos_sim = self.cos(  # (N, K)
            z1.unsqueeze(1).to(torch.float32), z2.to(torch.float32)) / self.config.cl_temperature

        # apply weights to z2[i] for each z1[i]
        if z2_weight is not None:
            cos_sim += z2_weight

        return cos_sim

    @torch.no_grad()
    def get_input_ids(self, indices):
        """
        Args: 
            indices: tensor of shape (N, K)
        """
        batch_size, k = indices.shape
        flat_indices = indices.view(-1)  # (N * K, )

        sliced_dataset = self.buffer_dataset[flat_indices]  # NK OrderedDict('input_ids': ..., 'attention_mask': ...,)
        # convert OrderedDict to tensors
        examples = [{'input_ids': sliced_dataset['input_ids'][i][0], 'attention_mask': sliced_dataset['attention_mask'][i][0]} for i in range(batch_size * k)]
        batch = self.buffer_collator.collate_flat_examples(examples, do_augment=False)
        input_ids, attention_mask = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)  # (N * K, D)

        input_ids = input_ids.reshape(batch_size, k, -1)
        attention_mask = attention_mask.reshape(batch_size, k, -1)

        return input_ids, attention_mask

    @torch.no_grad()
    def update_buffer_representations(self, query_tensors, query_ids, update_counts=False):
        """ Update self.buffer_representations using query_tensors.

        Args:
            query_tensors: (N, D), note faiss.knn_gpu requires it be contiguous.
            query_ids: (N, ), index of query_tensors in self.buffer_representations.
        """
        if self.training:
            all_query_ids = self._gather(query_ids)
            if self.config.retriever_metric == 'cos':
                query_tensors = F.normalize(query_tensors, dim=1).to(query_tensors.dtype)
                # query_tensors /= query_tensors.norm(dim=1, keepdim=True)
            all_query_tensors = self._gather(query_tensors)
            self.buffer_representations[all_query_ids] = all_query_tensors

            # record the frequency of each index
            if update_counts and hasattr(self, 'buffer_retrieve_counts'):
                unique_ids, counts = torch.unique(all_query_ids, return_counts=True)
                self.buffer_retrieve_counts[unique_ids] += counts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        ids=None,  # required by trainer._remove_unused_columns so as to avoid treating ids as unused_columns.
        **kwargs,
    ):
        if sent_emb or self.get_sent_emb_only:
            return self.get_sentence_embedding(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        batch_size, num_seq = input_ids.shape[:2]  # (N, S, L)
        inputs_embeds = None

        if self.config.switch_case_pattern != '00':
            batch_size, num_seq = input_ids.shape[:2]  # (N, S+1, L)
            if self.config.switch_case_pattern in ('01', '10'):
                num_seq = num_seq - 1
                if self.config.switch_case_pattern == '10':
                    input_ids = torch.cat((input_ids[:, -1:], input_ids[:, 1:-1]), dim=1)
                    attention_mask = torch.cat((attention_mask[:, -1:], attention_mask[:, 1:-1]), dim=1)
                    if token_type_ids is not None:
                        token_type_ids = torch.cat((token_type_ids[:, -1:], token_type_ids[:, 1:-1]), dim=1)
                else:  # 01
                    input_ids = torch.cat((input_ids[:, 0:1], input_ids[:, -1:], input_ids[:, 2:-1]), dim=1)
                    attention_mask = torch.cat((attention_mask[:, 0:1], attention_mask[:, -1:], attention_mask[:, 2:-1]), dim=1)
                    if token_type_ids is not None:
                        token_type_ids = torch.cat((token_type_ids[:, 0:1], token_type_ids[:, -1:], token_type_ids[:, 2:-1]), dim=1)
            elif self.config.switch_case_pattern == '11':
                num_seq = num_seq - 2
                input_ids = torch.cat((input_ids[:, -2:], input_ids[:, 2:-2]), dim=1)
                attention_mask = torch.cat((attention_mask[:, -2:], attention_mask[:, 2:-2]), dim=1)
                if token_type_ids is not None:
                    token_type_ids = torch.cat((token_type_ids[:, -2:], token_type_ids[:, 2:-2]), dim=1)

        if self.config.sample_retrieved_dynamic_negatives > 0 and self.config.retrieve_using_buffer_representations:
            query_ids = ids.squeeze(1)  # (N,)
            ids = None  # to avoid being used again
            sampled_knn_ids = self.get_knn_ids(query_ids=query_ids)  # (N, K2)
            knn_input_ids, knn_attention_mask = self.get_input_ids(sampled_knn_ids)  # (N, K2, L)
            if knn_input_ids.shape[2] < input_ids.shape[2]:  # in case input_ids is obtained after augmentation, its shape may change.
                pads = torch.empty(
                    batch_size, self.config.sample_retrieved_dynamic_negatives, input_ids.shape[2] - knn_input_ids.shape[2], 
                    dtype=knn_input_ids.dtype, device=knn_input_ids.device).fill_(self.tokenizer.pad_token_id)
                knn_input_ids = torch.cat((knn_input_ids, pads), dim=2)
                knn_attention_mask = torch.cat((knn_attention_mask, torch.zeros_like(pads)), dim=2)
            input_ids = torch.cat((input_ids, knn_input_ids), dim=1)  # (N, S+K2, L)
            attention_mask = torch.cat((attention_mask, knn_attention_mask), dim=1)  # (N, S+K2, L)
            if token_type_ids is not None:
                token_type_ids = torch.cat((token_type_ids, torch.zeros_like(knn_input_ids)), dim=1)  # (N, S+K2, L)
        
        # Flatten input for encoding
        flat_input_ids = input_ids.view((-1, input_ids.size(-1)))  # (N * S, L), or (N, S+K2, L)
        flat_attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (N * S, L), or (N*(S+K2), L)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (N * S, L), or (N*(S+K2), L)

        # Get raw embeddings
        encoder = self.bert or self.roberta
        outputs = encoder(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=return_dict,
        )

        # Pooling
        pooler_output = self.pooler(outputs, flat_attention_mask, return_dict)  # (N * S, D), or (N*(S+K2), D)
        if self.config.sample_retrieved_dynamic_negatives > 0 and self.config.retrieve_using_buffer_representations:
            pooler_output = pooler_output.view(
                (batch_size, num_seq + self.config.sample_retrieved_dynamic_negatives, pooler_output.size(-1)))  # (N, S+K2, D)
        else:
            pooler_output = pooler_output.view((batch_size, num_seq, pooler_output.size(-1)))  # (N, S, D)

        # select the negatives based on pooler_output
        if self.config.sample_retrieved_dynamic_negatives > 0:
            if self.config.retrieve_using_buffer_representations:
                if self.config.update_retrieved_negatives_representations:
                    # update database
                    self.update_buffer_representations(pooler_output[:,0], query_ids)
                    knn_pooler_output = pooler_output[:, -self.config.sample_retrieved_dynamic_negatives:]  # (N, K2, D)
                    self.update_buffer_representations(
                        knn_pooler_output.reshape(-1, knn_pooler_output.size(-1)), sampled_knn_ids.view(-1), update_counts=True)
            else:
                # update database
                query_ids = ids.squeeze(1)  # (N,)
                ids = None  # to avoid being used again
                self.update_buffer_representations(pooler_output[:,0], query_ids)
                sampled_knn_ids = self.get_knn_ids(pooler_output[:,0], query_ids)  # (N, K2)
                
                if self.config.update_retrieved_negatives_representations:
                    knn_input_ids, knn_attention_mask = self.get_input_ids(sampled_knn_ids)
                    flat_knn_input_ids = knn_input_ids.view((-1, knn_input_ids.size(-1)))  # (N * K2, L)
                    flat_knn_attention_mask = knn_attention_mask.view((-1, knn_attention_mask.size(-1)))  # (N * K2, L)
                    if self.config.detach_retrieved_negatives_representations:
                        with torch.no_grad():  # no_grad saves more memory than simply detaching knn_pooler_output
                            knn_outputs = encoder(
                                flat_knn_input_ids,
                                attention_mask=flat_knn_attention_mask,
                                output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                                return_dict=return_dict,
                            )
                            # Pooling
                            knn_pooler_output = self.pooler(knn_outputs, flat_knn_attention_mask, return_dict)  # (N * K2, D)
                    else:
                        knn_outputs = encoder(
                            flat_knn_input_ids,
                            attention_mask=flat_knn_attention_mask,
                            output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                            return_dict=return_dict,
                        )
                        # Pooling
                        knn_pooler_output = self.pooler(knn_outputs, flat_knn_attention_mask, return_dict)  # (N * K2, D)
                    # update database
                    self.update_buffer_representations(knn_pooler_output, sampled_knn_ids.view(-1), update_counts=True)
                    knn_pooler_output = knn_pooler_output.view((batch_size, self.config.sample_retrieved_dynamic_negatives, knn_pooler_output.size(-1)))  # (N, K2, D)
                else:
                    knn_pooler_output = self.buffer_representations[sampled_knn_ids].detach()  # (N, K2, D)
                
                pooler_output = torch.cat((pooler_output, knn_pooler_output), dim=1)  # (N, S+K2, D)

        # Add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.config.pooler_type in {"cls", "demean", "avg", "avg_first_last", "avg_top2"}:
            pooler_output = self.embedding_projector(pooler_output)  # (N, S, D), or (N, S+K2, D)

        # Separate representation
        z1, z2 = pooler_output[:,0], pooler_output[:,1]  # (N, D), (N, D)
        cos_sim = self.get_cos_sim(z1, z2)  # (N, N * num_gpus)
        if self.config.symmetric_contrastive_loss:
            cos_sim_T = self.get_cos_sim(z2, z1)  # (N, N * num_gpus)

        # Hard negative
        # cases of num_seq == 3: 
        #   - supervised learning
        #   - unsupervised learning, with sample_retrieved_hard_negatives==1
        if num_seq == 3:
            z3 = pooler_output[:, 2]  # (N, D)
            z1z2_ids = ids  # (N, 2)
            ids = None
            if z1z2_ids is not None:
                z1_ids, z2_ids = z1z2_ids[:, 0], z1z2_ids[:, 1]
            else:
                z1_ids, z2_ids = None, None
            # Note that weights are actually logits of weights, thus hard_negative_weight = 0 actually means 1
            z1_z3_cos = self.get_cos_sim(  # (N, N * num_gpus)
                z1, z3, self.config.hard_negative_weight, z1_ids, z2_ids)
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)  # (N, 2 * N * num_gpus)

            if self.config.symmetric_contrastive_loss:
                z2_z3_cos = self.get_cos_sim(z2, z3, self.config.hard_negative_weight, z1_ids, z2_ids)  # (N, N * num_gpus)
                cos_sim_T = torch.cat([cos_sim_T, z2_z3_cos], 1)  # (N, 2 * N * num_gpus)

        if self.config.sample_retrieved_dynamic_negatives > 0:
            zknn = pooler_output[:, -self.config.sample_retrieved_dynamic_negatives:]  # (N, K2, D)
            if self.config.share_sampled_dynamic_negatives:
                z1_zknn_cos = self.get_cos_sim(z1, zknn, self.config.hard_negative_weight, query_ids, sampled_knn_ids)  # (N, (K2 * N) * num_gpus)
            else:
                z1_zknn_cos = self.get_samplewise_cos_sim(z1, zknn, self.config.hard_negative_weight)  # (N, K2)
            cos_sim = torch.cat([cos_sim, z1_zknn_cos], 1)  # (N, x + K2)

        if dist.is_initialized():
            labels = dist.get_rank() * cos_sim.shape[0] + torch.arange(cos_sim.shape[0], dtype=torch.int64, device=self.device)  # (N, )
        else:
            labels = torch.arange(cos_sim.shape[0], dtype=torch.int64, device=self.device)  # (N, )
        loss = self.loss_fct(cos_sim, labels)

        if self.config.symmetric_contrastive_loss:
            loss = (loss + self.loss_fct(cos_sim_T, labels))/2.

        # Calculate loss for MLM auxiliary objective
        mlm_outputs = None
        if self.config.mlm_loss_weight > 0 and mlm_input_ids is not None:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1))) # (N * S, L)
            mlm_outputs = encoder(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True if self.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
            if mlm_labels is not None:  # TODO check if correct
                mask_indices = (mlm_labels != -100)
                masked_labels = mlm_labels.view(-1, mlm_labels.size(-1))[mask_indices]
                prediction_scores = self.lm_head(mlm_outputs.last_hidden_state[mask_indices])
                masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
                loss = loss + self.config.mlm_loss_weight * masked_lm_loss
                mlm_acc = get_accuracy(prediction_scores, masked_labels.view(-1))
                self.printer_callback.add_to_log({'loss/mlm_loss': masked_lm_loss})
                self.printer_callback.add_to_log({'metrics/mlm_acc': mlm_acc})

        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
