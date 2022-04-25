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
"""classification metrics"""
import torch
import logging

logger = logging.getLogger(__name__)


def get_accuracy(logits, labels):
    """ Calculate the accuracy for multi-class or binary classification problem.

    Args:
        logits: the model predictions, dtype float,
            - shape (N, ..., C): will be converted to (N', C) and considered as multi-class logits
            - shape (N, ): considered as two-class logits; logits > 0 means the sample is predicted to be positive
        labels: the target class labels, dtype long
            - shape (N, ...) will be converted to (N', ); contain the index of classes, e.g., [3, 5, 99, 1]
            - in case logits is in shape (N, ), labels contain 0 or 1, e.g., [1, 0, 1, 1, 0]
    """
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    batch_size = logits.shape[0]
    predictions = logits.argmax(dim=1) if logits.ndim == 2 else (logits>0).to(torch.long)  # (N, )
    acc = (predictions == labels).sum().to(torch.float32) / batch_size

    return acc


def get_binary_classification_metrics(logits, labels, eps=1e-10):
    """ calculate the accuracy, precision, recall, f1 and confusion matrix for binary classification.

    Args:
        logits: the model predictions, dtype float, shape (N, ); logits > 0 means the sample is predicted to be positive
        labels: the target class labels, dtype long, contain 0 or 1, e.g., [1, 0, 1, 1, 0]
        eps: add eps in denominator to avoid division by 0

    Returns;
        a dict with keys: 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'
    """
    assert logits.ndim == 1, ValueError('Expect logits.ndim to be 1, got {}'.format(logits.ndim))

    logits = (logits>0).to(torch.long)
    positives = (labels == 1)
    negatives = ~positives
    tp = logits[positives].sum().to(torch.float32)
    fp = logits[negatives].sum().to(torch.float32)
    fn = (logits[positives] == 0).sum().to(torch.float32)
    tn = (logits[negatives] == 0).sum().to(torch.float32)
    batch_size = tp + tn + fp + fn
    accuracy = (tp + tn) / batch_size
    precision = tp / (tp + fp + eps)  # add eps to avoid division by 0
    recall = tp / (tp + fn + eps)  # add eps to avoid division by 0
    f1 = 2*precision*recall/(precision + recall + eps)  # add eps to avoid division by 0
    confusion_mat = torch.tensor([[tp/batch_size, fp/batch_size], [fn/batch_size, tn/batch_size]], dtype=torch.float32)

    metric_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_mat,
    }

    return metric_dict
