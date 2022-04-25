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
from ast import Not
import numpy as np
from tqdm import tqdm
import faiss.contrib.torch_utils  # this will replace faiss.knn_gpu with torch_utils.torch_replacement_knn_gpu
import faiss
import torch

import logging
logger = logging.getLogger(__name__)


RETRIEVER_METRIC = {'ip': faiss.METRIC_INNER_PRODUCT, 'l2': faiss.METRIC_L2, 'cos': faiss.METRIC_INNER_PRODUCT}


class BaseFaissIPRetriever:
    def __init__(self, p_reps, device=0, shard=True):
        if p_reps is None: 
            self.index = None
            return
        dim = p_reps.shape[1]
        if hasattr(faiss, 'StandardGpuResources'):
            logger.info(f"FAISS search in GPU mode.")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False  # use precomputed tables?
            # Whether to shard the index across GPUs (slower but can handle large datasets), versus 
            # replication across GPUs (faster)
            co.shard = shard  
            cpu_index = faiss.IndexFlatIP(dim)
            index = faiss.index_cpu_to_all_gpus(cpu_index, co)

            # # gpu mode
            # res = faiss.StandardGpuResources()
            # res.noTempMemory()
            # config = faiss.GpuIndexFlatConfig()
            # config.useFloat16 = True
            # print(f"GPU mode: config={config}; res={res}")
            # config.device = device
            # index = faiss.GpuIndexFlatIP(res, dim, config)

            # nlist = 1024  # nlist: The feature space is partitioned into nlist cells. The database vectors are assigned to one of these cells thanks using a quantization function (in the case of k-means, the assignment to the centroid closest to the query), and stored in an inverted file structure formed of nlist inverted lists.
            # index = faiss.IndexIVFFlat(index0, dim, nlist)
            # index.set_direct_map_type(faiss.DirectMap.Hashtable)  # this can only apply to GPU index, if the shard = False
            # index = faiss.index_cpu_to_all_gpus(index,co)

            # index.train(p_reps)
            # index.nprobe = 16  # default nprobe is 1. At query time, a set of nprobe inverted lists is selected. Doing so, only a fraction (nprobe/nlist) of the database is compared to the query: as a first approximation

        else:
            # cpu mode
            logger.info(f"FAISS search in CPU mode.")
            index = faiss.IndexFlatIP(dim)   # create a CPU index
        self.index = index

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def search_and_reconstruct(self, q_reps: np.ndarray, k: int):
        return self.index.search_and_reconstruct(q_reps, k)

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def add_with_ids(self, p_reps: np.ndarray, ids: np.ndarray):
        self.index.add_with_ids(p_reps, ids)

    # all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), mininterval=10):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices
