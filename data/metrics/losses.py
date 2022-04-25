# coding=utf-8
# Copyright 2021 The Alibaba Damo Academy.
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
"""losses"""
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_alignment(fx, fy, alpha=2, weights=None):
    """ Calculate the alignment between two paired batch of samples.

    Args:
        fx: tensor of shape (N, D).
        fy: tensor of shape (N, D).
        alpha: scalar.
        weights: tensor of shape (N,).
    """

    fx = F.normalize(fx, p=2, dim=1)
    fy = F.normalize(fy, p=2, dim=1)
    if weights is None:
        return (fx - fy).norm(dim=1).pow(alpha).mean()
    else:
        weights = weights.to(fx.dtype)
        return ((fx - fy).norm(dim=1).pow(alpha) * weights).sum() / (weights.sum() + 1e-8)


def get_uniformity(fx, temperature=2):
    """ Calculate the uniformity among samples.

    Args:
        x: tensor of shape (N, D).
        temperature: scalar.
    """
    fx = F.normalize(fx, p=2, dim=1)
    sq_pdist = torch.pdist(fx, p=2).pow(2)
    return sq_pdist.mul(-temperature).exp().mean().log()
