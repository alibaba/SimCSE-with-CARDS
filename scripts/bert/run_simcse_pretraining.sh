#!/bin/bash

# echo "************************** Below are command to use in teriminal (un-comment,  copy and paste to termimal to run) **************************"
# nohup bash ./scripts/bert/run_simcse_pretraining_v2.sh train_file=unsupervised model_type=roberta model_size=large num_train_epochs=1 learning_rate=1e-5 cuda=0 ngpu=1 model_name=MODEL &
# nohup bash ./scripts/bert/run_simcse_pretraining_v2.sh train_file=supervised model_type=roberta model_size=large num_train_epochs=3 learning_rate=1e-5 per_device_train_batch_size=128 cuda=0,1,2,3 ngpu=4 model_name=MODEL &

# echo "************************** Below are examples of frequently used commands **************************"
# kill -9 `ps auxfww | grep examples/bert/run_simcse.py | grep -v nothing | grep -v grep | awk '{print $2}'`
# kill -9 `ps -ef | grep whatever | grep -v nothing | grep -v grep | awk '{print $2}'`

# cd YOUR_CARDS_WORKING_DIRECTORY  # cwd, which contains data/, examples/, models/, scripts/, training/ and utils/

# default values
# paths
model_name_or_path=None;
config_name=None;
train_file=../../dataset/SimCSE/wiki1m_for_simcse.txt;
output_dir=../output/bert_large/sentence_embedding/unsupervised;
tensorboard_dir=../output/bert_large/tensorboard/sentence_embedding/unsupervised;
SentEval_data_dir=../../dataset/SentEval;
# data
overwrite_cache=False;
mlm_probability=0.15;
mask_token_rate=0.8;
max_seq_length=32;
# model
model_type=roberta;  # roberta, bert, bert_cased
model_size=large;  # large, base
cl_temperature=0.05;
pooler_type=cls;
hard_negative_weight=0;
mlm_loss_weight=0;
discard_embd_proj=true;
hidden_dropout_prob=0.1;
embedding_dropout_prob=None;
# ---------------- note: newly added arguments should repeat for at least 7 times; check to see if properly done ------------- #
label_smoothing=0.0;
switch_case_pattern=00;  # 00, 01, 10, 11
switch_case_probability=0.0;
switch_case_method=v1;  # v1, v2
ignore_upper_case_phase=never;  # never, train, eval, both
symmetric_contrastive_loss=False;
retriever_metric=ip;  # ip (inner product), or cos
retrieve_hard_negatives=0;
sample_retrieved_hard_negatives=0;
sent_rep_cache_file=None;
retriever_results_cache_file=None;
retrieve_dynamic_negatives=0;
sample_retrieved_dynamic_negatives=0;
ignore_num_top_retrieved_dynamic_negatives=0;
share_sampled_dynamic_negatives=True;
update_retrieved_negatives_representations=False;
detach_retrieved_negatives_representations=False;
retrieve_using_buffer_representations=True;
update_buffer_representations_steps=-1;
save_buffer_representations=False;
retrieved_and_sampled_knn_type=default;
weight_decay=0.0;
# training and evaluation
per_device_train_batch_size=128;
per_device_eval_batch_size=512;
learning_rate=1e-5;
num_train_epochs=1;
evaluation_strategy=steps;
eval_steps=125;
do_train=true;
metric_for_best_model=stsb_spearman;  # stsb_spearman, sickr_spearman, avg_sts
get_align_uniform=False;
remove_checkpoints_at_end=True;
eval_during_train=False;
task_set=sts;  # sts, all, transfer
logging_steps=50;
halt_step=-1;
# misc
cuda=0,1,2,3; ngpu=4; seed=42;
dataloader_num_workers=4;
preprocessing_num_workers=-1;
master_port=29500;
fp16_opt_level=O2;
disable_tqdm=True;
report_to=tensorboard;
ddp_find_unused_parameters=False;

# pass arguments as key=value pairs
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
        model_name)                         model_name=${VALUE} ;;
        model_name_or_path)                 model_name_or_path=${VALUE} ;;
        config_name)                        config_name=${VALUE} ;;
        train_file)                         train_file=${VALUE} ;;
        output_dir)                         output_dir=${VALUE} ;;
        tensorboard_dir)                    tensorboard_dir=${VALUE} ;;
        SentEval_data_dir)                  SentEval_data_dir=${VALUE} ;;
        overwrite_cache)                    overwrite_cache=${VALUE} ;;
        mlm_probability)                    mlm_probability=${VALUE} ;;
        mask_token_rate)                    mask_token_rate=${VALUE} ;;
        max_seq_length)                     max_seq_length=${VALUE} ;;
        model_type)                         model_type=${VALUE} ;;
        model_size)                         model_size=${VALUE} ;;
        cl_temperature)                     cl_temperature=${VALUE} ;;
        pooler_type)                        pooler_type=${VALUE} ;;
        hard_negative_weight)               hard_negative_weight=${VALUE} ;;
        mlm_loss_weight)                    mlm_loss_weight=${VALUE} ;;
        discard_embd_proj)                  discard_embd_proj=${VALUE} ;;
        hidden_dropout_prob)                hidden_dropout_prob=${VALUE} ;;
        embedding_dropout_prob)             embedding_dropout_prob=${VALUE} ;;
        label_smoothing)                    label_smoothing=${VALUE} ;;
        switch_case_pattern)                switch_case_pattern=${VALUE} ;;
        switch_case_probability)            switch_case_probability=${VALUE} ;;
        switch_case_method)                 switch_case_method=${VALUE} ;;
        ignore_upper_case_phase)            ignore_upper_case_phase=${VALUE} ;;
        symmetric_contrastive_loss)         symmetric_contrastive_loss=${VALUE} ;;
        retriever_metric)                   retriever_metric=${VALUE} ;;
        retrieve_hard_negatives)            retrieve_hard_negatives=${VALUE} ;;
        sample_retrieved_hard_negatives)    sample_retrieved_hard_negatives=${VALUE} ;;
        sent_rep_cache_file)                sent_rep_cache_file=${VALUE} ;;
        retriever_results_cache_file)       retriever_results_cache_file=${VALUE} ;;
        retrieve_dynamic_negatives)         retrieve_dynamic_negatives=${VALUE} ;;
        sample_retrieved_dynamic_negatives) sample_retrieved_dynamic_negatives=${VALUE} ;;
        ignore_num_top_retrieved_dynamic_negatives) ignore_num_top_retrieved_dynamic_negatives=${VALUE} ;;
        share_sampled_dynamic_negatives)    share_sampled_dynamic_negatives=${VALUE} ;;
        update_retrieved_negatives_representations) update_retrieved_negatives_representations=${VALUE} ;;
        detach_retrieved_negatives_representations) detach_retrieved_negatives_representations=${VALUE} ;;
        retrieve_using_buffer_representations)      retrieve_using_buffer_representations=${VALUE} ;;
        update_buffer_representations_steps) update_buffer_representations_steps=${VALUE} ;;
        save_buffer_representations)        save_buffer_representations=${VALUE} ;;
        retrieved_and_sampled_knn_type)     retrieved_and_sampled_knn_type=${VALUE} ;;
        weight_decay)                       weight_decay=${VALUE} ;;
        per_device_train_batch_size)        per_device_train_batch_size=${VALUE} ;;
        per_device_eval_batch_size)         per_device_eval_batch_size=${VALUE} ;;
        learning_rate)                      learning_rate=${VALUE} ;;
        num_train_epochs)                   num_train_epochs=${VALUE} ;;
        evaluation_strategy)                evaluation_strategy=${VALUE} ;;
        eval_steps)                         eval_steps=${VALUE} ;;
        do_train)                           do_train=${VALUE} ;;
        metric_for_best_model)              metric_for_best_model=${VALUE} ;;
        get_align_uniform)                  get_align_uniform=${VALUE} ;;
        remove_checkpoints_at_end)          remove_checkpoints_at_end=${VALUE} ;;
        eval_during_train)                  eval_during_train=${VALUE} ;;
        task_set)                           task_set=${VALUE} ;;
        logging_steps)                      logging_steps=${VALUE} ;;
        halt_step)                          halt_step=${VALUE} ;;
        cuda)                               cuda=${VALUE} ;;
        ngpu)                               ngpu=${VALUE} ;;
        seed)                               seed=${VALUE} ;;
        dataloader_num_workers)             dataloader_num_workers=${VALUE} ;;
        preprocessing_num_workers)          preprocessing_num_workers=${VALUE} ;;
        master_port)                        master_port=${VALUE} ;;
        fp16_opt_level)                     fp16_opt_level=${VALUE} ;;
        disable_tqdm)                       disable_tqdm=${VALUE} ;;
        report_to)                          report_to=${VALUE} ;;
        ddp_find_unused_parameters)         ddp_find_unused_parameters=${VALUE} ;;
        *)                                  echo "$KEY not supported" && exit ;;
    esac

done

# check inputs
if [[ ${model_name} = "" ]]; then
    echo "No model_name has been provided. Pretraining process ends now."
    exit
fi
# The default paths to saved roberta and bert checkpoints
if [[ ${model_type} = "roberta" ]]; then
    if [[ ${model_size} = "large" ]]; then
        if [[ ${model_name_or_path} = "None" ]]; then
            model_name_or_path=../output/bert_large/pretraining/roberta_large_baseline;
        fi
        if [[ ${config_name} = "None" ]]; then
            config_name=../output/bert_large/pretraining/roberta_large_baseline/config.json;
        fi
    elif [[ ${model_size} = "base" ]]; then
        if [[ ${model_name_or_path} = "None" ]]; then
            model_name_or_path=../output/bert_base/pretraining/roberta_base_baseline;
        fi
        if [[ ${config_name} = "None" ]]; then
            config_name=../output/bert_base/pretraining/roberta_base_baseline/config.json;
        fi
    else
        echo "${model_type}-${model_size} not implemented yet. training process ends."
        exit
    fi
elif [[ ${model_type} = "bert" ]]; then
    if [[ ${model_size} = "large" ]]; then
        if [[ ${model_name_or_path} = "None" ]]; then
            model_name_or_path=../output/bert_large/pretraining/bert_large_baseline;
        fi
        if [[ ${config_name} = "None" ]]; then
            config_name=../output/bert_large/pretraining/bert_large_baseline/config.json;
        fi
    else
        echo "${model_type}-${model_size} not implemented yet. training process ends."
        exit
    fi
elif [[ ${model_type} = "bert_cased" ]] || [[ ${model_type} = "bert-cased" ]]; then
    model_type=bert;
    if [[ ${model_size} = "large" ]]; then
        if [[ ${model_name_or_path} = "None" ]]; then
            model_name_or_path=../output/bert_large/pretraining/bert_large_cased_baseline;
        fi
        if [[ ${config_name} = "None" ]]; then
            config_name=../output/bert_large/pretraining/bert_large_cased_baseline/config.json;
        fi
    else
        echo "${model_type}-${model_size} not implemented yet. training process ends."
        exit
    fi
else
    echo "${model_type}-${model_size} not implemented yet. training process ends."
    exit
fi

# set up output folder;
mkdir -p ${output_dir}/${model_name};
mkdir -p ${tensorboard_dir}/${model_name};
NOW=$( date '+%F %H:%M:%S' )
echo $NOW,"Saving to "${output_dir}/${model_name}
echo "************************** Pre-training starts **************************"
export CUDA_VISIBLE_DEVICES=${cuda}
export WANDB_DISABLED=true
(
    if [[ ${ngpu} = 1 ]]; then
        python examples/bert/run_simcse.py \
            --model_name_or_path=${model_name_or_path} --config_name=${config_name} \
            --train_file=${train_file} --output_dir=${output_dir}/${model_name} --logging_dir ${tensorboard_dir}/${model_name} --cache_dir=../.cache --SentEval_data_dir=${SentEval_data_dir} \
            --mlm_probability=${mlm_probability} --max_seq_length=${max_seq_length} --mask_token_rate=${mask_token_rate} \
            --ignore_data_skip --do_train=${do_train} --do_eval=true --overwrite_output_dir --overwrite_cache=${overwrite_cache} \
            --model_type=${model_type} --cl_temperature=${cl_temperature} --pooler_type=${pooler_type} \
            --hard_negative_weight=${hard_negative_weight} --mlm_loss_weight=${mlm_loss_weight} --discard_embd_proj=${discard_embd_proj} \
            --hidden_dropout_prob=${hidden_dropout_prob} --embedding_dropout_prob=${embedding_dropout_prob} \
            --label_smoothing=${label_smoothing} \
            --switch_case_pattern=${switch_case_pattern} --switch_case_probability=${switch_case_probability} --switch_case_method=${switch_case_method} \
            --ignore_upper_case_phase=${ignore_upper_case_phase} \
            --symmetric_contrastive_loss=${symmetric_contrastive_loss} \
            --retriever_metric=${retriever_metric} \
            --retrieve_hard_negatives=${retrieve_hard_negatives} --sample_retrieved_hard_negatives=${sample_retrieved_hard_negatives} \
            --sent_rep_cache_file=${sent_rep_cache_file} --retriever_results_cache_file=${retriever_results_cache_file} \
            --retrieve_dynamic_negatives=${retrieve_dynamic_negatives} --sample_retrieved_dynamic_negatives=${sample_retrieved_dynamic_negatives} \
            --ignore_num_top_retrieved_dynamic_negatives=${ignore_num_top_retrieved_dynamic_negatives} \
            --share_sampled_dynamic_negatives=${share_sampled_dynamic_negatives} \
            --update_retrieved_negatives_representations=${update_retrieved_negatives_representations} \
            --detach_retrieved_negatives_representations=${detach_retrieved_negatives_representations} \
            --retrieve_using_buffer_representations=${retrieve_using_buffer_representations} \
            --update_buffer_representations_steps=${update_buffer_representations_steps} --save_buffer_representations=${save_buffer_representations} \
            --retrieved_and_sampled_knn_type=${retrieved_and_sampled_knn_type} \
            --weight_decay=${weight_decay} \
            --per_device_train_batch_size=${per_device_train_batch_size} --per_device_eval_batch_size=${per_device_eval_batch_size} \
            --learning_rate=${learning_rate} --num_train_epochs=${num_train_epochs} \
            --evaluation_strategy=${evaluation_strategy} --eval_steps=${eval_steps} --metric_for_best_model=${metric_for_best_model} \
            --get_align_uniform=${get_align_uniform} \
            --load_best_model_at_end=True --remove_checkpoints_at_end=${remove_checkpoints_at_end} \
            --eval_during_train=${eval_during_train} --task_set=${task_set} --logging_steps=${logging_steps} \
            --dataloader_num_workers=${dataloader_num_workers} --preprocessing_num_workers=${preprocessing_num_workers} \
            --seed=${seed} --halt_step=${halt_step} \
            --fp16 --fp16_opt_level ${fp16_opt_level} \
            --report_to=${report_to} --disable_tqdm=${disable_tqdm} \
            --ddp_find_unused_parameters=${ddp_find_unused_parameters} \
            > ${output_dir}/${model_name}.log 2>&1;
    else
        export TOKENIZERS_PARALLELISM=true
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=${ngpu} --rdzv_id=${model_name} --rdzv_backend=c10d --rdzv_endpoint=localhost:${master_port} examples/bert/run_simcse.py \
            --model_name_or_path=${model_name_or_path} --config_name=${config_name} \
            --train_file=${train_file} --output_dir=${output_dir}/${model_name} --logging_dir ${tensorboard_dir}/${model_name} --cache_dir=../.cache --SentEval_data_dir=${SentEval_data_dir} \
            --mlm_probability=${mlm_probability} --max_seq_length=${max_seq_length} --mask_token_rate=${mask_token_rate} \
            --ignore_data_skip --do_train=${do_train} --do_eval=true --overwrite_output_dir --overwrite_cache=${overwrite_cache} \
            --model_type=${model_type} --cl_temperature=${cl_temperature} --pooler_type=${pooler_type} \
            --hard_negative_weight=${hard_negative_weight} --mlm_loss_weight=${mlm_loss_weight} --discard_embd_proj=${discard_embd_proj} \
            --hidden_dropout_prob=${hidden_dropout_prob} --embedding_dropout_prob=${embedding_dropout_prob} \
            --label_smoothing=${label_smoothing} \
            --switch_case_pattern=${switch_case_pattern} --switch_case_probability=${switch_case_probability} --switch_case_method=${switch_case_method} \
            --ignore_upper_case_phase=${ignore_upper_case_phase} \
            --symmetric_contrastive_loss=${symmetric_contrastive_loss} \
            --retriever_metric=${retriever_metric} \
            --retrieve_hard_negatives=${retrieve_hard_negatives} --sample_retrieved_hard_negatives=${sample_retrieved_hard_negatives} \
            --sent_rep_cache_file=${sent_rep_cache_file} --retriever_results_cache_file=${retriever_results_cache_file} \
            --retrieve_dynamic_negatives=${retrieve_dynamic_negatives} --sample_retrieved_dynamic_negatives=${sample_retrieved_dynamic_negatives} \
            --ignore_num_top_retrieved_dynamic_negatives=${ignore_num_top_retrieved_dynamic_negatives} \
            --share_sampled_dynamic_negatives=${share_sampled_dynamic_negatives} \
            --update_retrieved_negatives_representations=${update_retrieved_negatives_representations} \
            --detach_retrieved_negatives_representations=${detach_retrieved_negatives_representations} \
            --retrieve_using_buffer_representations=${retrieve_using_buffer_representations} \
            --update_buffer_representations_steps=${update_buffer_representations_steps} --save_buffer_representations=${save_buffer_representations} \
            --retrieved_and_sampled_knn_type=${retrieved_and_sampled_knn_type} \
            --weight_decay=${weight_decay} \
            --per_device_train_batch_size=${per_device_train_batch_size} --per_device_eval_batch_size=${per_device_eval_batch_size} \
            --learning_rate=${learning_rate} --num_train_epochs=${num_train_epochs} \
            --evaluation_strategy=${evaluation_strategy} --eval_steps=${eval_steps} --metric_for_best_model=${metric_for_best_model} \
            --get_align_uniform=${get_align_uniform} \
            --load_best_model_at_end=True --remove_checkpoints_at_end=${remove_checkpoints_at_end} \
            --eval_during_train=${eval_during_train} --task_set=${task_set} --logging_steps=${logging_steps} \
            --dataloader_num_workers=${dataloader_num_workers} --preprocessing_num_workers=${preprocessing_num_workers} \
            --seed=${seed} --halt_step=${halt_step} \
            --fp16 --fp16_opt_level ${fp16_opt_level} \
            --report_to=${report_to} --disable_tqdm=${disable_tqdm} \
            --ddp_find_unused_parameters=${ddp_find_unused_parameters} \
            > ${output_dir}/${model_name}.log 2>&1;
    fi
    sleep 11s
    NOW=$( date '+%F %H:%M:%S' )
    echo $NOW,"Saved to "${output_dir}/${model_name}
    echo "************************** Pre-training ends **************************"
) # & # comment & to run tasks sequentially
sleep 77  # start the next task after a break