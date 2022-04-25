# Improving Contrastive Learning of Sentence Embeddings with Case-Augmented Positives and Retrieved Negatives

This repository implements switch-case augmentation and hard negative retrieval from the paper "Improving Contrastive Learning of Sentence Embeddings with Case-Augmented Positives and Retrieved Negatives". Combining the two approaches with SimCSE leads to the model called Contrastive learning with Augmented and Retrieved Data for Sentence embedding (CARDS).

## Results and Checkpoints
Table 1. Performance on sentence embedding tasks

| Pretraining    | Finetuning   | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
|----------------|--------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|
| roberta-base   | SimCSE+CARDS | 72.65 | 84.26 | 76.52 | 82.98 | 82.73 | 82.04 | 70.66  | 78.83 |
| roberta-large  | SimCSE+CARDS | 74.63 | 86.27 | 79.25 | 85.93 | 83.17 | 83.86 | 72.77  | 80.84 |

Download link:
CARDS-roberta-base ([Download, 440MB](http://dirl-sas-open.oss-cn-hangzhou.aliyuncs.com/cards_roberta_base.zip)), CARDS-roberta-large ([Download, 1.23GB](http://dirl-sas-open.oss-cn-hangzhou.aliyuncs.com/cards_roberta_large.zip)).

Table 2. Performance on GLUE tasks

| Pretraining        | Finetuning         | MNLI-m | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | Avg. |
|--------------------|--------------------|:------:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:----:|
| debertav2-xxlarge  | R-drop+Switch-case | 92.0   | 93.0 | 96.3 | 97.2  | 75.5 | 93.6  | 93.9 | 94.2 | 91.7 |

## Usage
This repo is built based on [Huggingface Transformers](https://github.com/huggingface/transformers) and [SimCSE](https://github.com/princeton-nlp/SimCSE). See requirements.txt for package versions. 

### Data Preparation
```shell
# 1. Download wiki-1m dataset: 
# - use wget -P target_folder in data/datasets/download_wiki.sh, and run
bash data/datasets/download_wiki.sh
# - modify train_file in scripts/bert/run_simcse_pretraining_v2.sh

# 2. preprocess wiki-1m dataset for negative retrieval
# - deduplicate the wiki-1m dataset, and (optionally) remove sentences with less than three words
# - modify paths in data/datasets/simcse_utils.py then run it to get model representations for all sentences in dataset
python data/datasets/simcse_utils.py

# 3. Download SentEval evaluation data:
# - use wget -P target_folder in data/datasets/download_senteval.sh, and run
bash data/datasets/download_senteval.sh
```

### Fine-tune Roberta with CARDS
Before running the code, the user may need to change default model checkpoint and I/O paths, including: 
- ``scripts/bert/run_simcse_grid.sh``: line 42-50 (train_file, train_file_dedupl (optional), output_dir, tensorboard_dir, sent_rep_cache_file, SentEval_data_dir)
- ``scripts/bert/run_simcse_pretraining.sh``: line 17-20 (train_file, output_dir, tensorboard_dir, SentEval_data_dir), line 45 (sent_rep_cache_files), line 166-213 (model_name_or_path, config_name).

#### Fine-tune + evaluation
```shell
# MUST cd to the folder which contains data/, examples/, models/, scripts/, training/ and utils/
cd YOUR_CARDS_WORKING_DIRECTORY

# roberta-base
new_train_file=path_to_wiki1m
sent_rep_cache_file=path_to_sentence_representation_file  # generated by data/datasets/simcse_utils.py 

# run a model with a single set of hyper-parameters
# when running the model for the very first time, need to add overwrite_cache=True, this will produce a processed training data cache.
bash scripts/bert/run_simcse_grid.sh \
    model_type=roberta model_size=base \
    cuda=0,1,2,3 seed=42 learning_rate=4e-5 \
    new_train_file=${new_train_file} sent_rep_cache_file=${sent_rep_cache_file} \
    dyn_knn=65 sample_k=1 knn_metric=cos \
    switch_case_probability=0.05 switch_case_method=v2 \
    print_only=False

# grid-search on hyper-parameters
bash scripts/bert/run_simcse_grid.sh \
    model_type=roberta model_size=base \
    cuda=0,1,2,3 seed=42 learning_rate=1e-5,2e-5,4e-5 \
    new_train_file=${new_train_file} sent_rep_cache_file=${sent_rep_cache_file} \
    dyn_knn=0,9,65 sample_k=1 knn_metric=cos \
    switch_case_probability=0,0.05,0.1,0.15 switch_case_method=v2 \
    print_only=False

# roberta-large
bash scripts/bert/run_simcse_grid.sh \
    model_type=roberta model_size=large \
    cuda=0,1,2,3 seed=42 learning_rate=7.5e-6 \
    new_train_file=${new_train_file} sent_rep_cache_file=${sent_rep_cache_file} \
    dyn_knn=9 sample_k=1 knn_metric=cos \
    switch_case_probability=0.1 switch_case_method=v1 \
    print_only=False
```
#### Evaluation only
```shell
# provide train_file, output_dir, tensorboard_dir if different to the default values
model_name=name_of_saved_mdoel  # e.g., roberta_large_bs128x4_lr2e-5_switchcase0.1_v2
bash ./scripts/bert/run_simcse_pretraining.sh \
    model_name_or_path=${output_dir}/${model_name} model_name=${model_name} config_name=${output_dir}/${model_name}/config.json \
    train_file=${train_file} output_dir=${output_dir}/test_only tensorboard_dir=${tensorboard_dir} \
    model_type=roberta model_size=base do_train=False \
    cuda=0 ngpu=1
```

#### Known issues
For unknown reasons, the set of good model hyper-parameters were different when working with Huggingface Transformers v4.11.3 and v4.15.0. The hyper-parameters listed above were grid-searched on Transformers 4.11.3.

## Citation
```
@inproceedings{cards,
    title = "Improving Contrastive Learning of Sentence Embeddings with Case-Augmented Positives and Retrieved Negatives",
    author = "Wei Wang and Liangzhu Ge and Jingqiao Zhang and Cheng Yang",
    booktitle = "The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)",
    year = "2022"
}
```