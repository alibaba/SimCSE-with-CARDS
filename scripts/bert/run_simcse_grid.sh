# ---------------------------------------
# Grid search pretraining
# ---------------------------------------
# nohup bash scripts/bert/test_simcse_grid.sh alphas=0.25,0.5,0.75 betas=0.5,1,1.5 cuda=0,1,2,3 &
# nohup bash scripts/bert/test_simcse_grid.sh alphas=0.75,1,1.25 betas=-0.5,0,0.5 cuda=4,5,6,7 &

# kill -9 `ps -ef | grep scripts/bert/test_simcse_grid.sh | grep -v nothing | grep -v grep | awk '{print $2}'`
# kill -9 `ps -ef | grep ./scripts/bert/run_simcse_pretraining.sh | grep -v nothing | grep -v grep | awk '{print $2}'`

# cd YOUR_CARDS_WORKING_DIRECTORY  # cwd, which contains data/, examples/, models/, scripts/, training/ and utils/

cuda=0,1,2,3;
ngpu=4;
master_port=29500;
model_type=roberta;  # roberta, bert, bert_cased
model_size=large;
batch_size=128;
learning_rate=3e-5;  # 1e-5,3e-5,5e-5
hidden_dropout_prob=0.1;  # 0.1,0.2,0.3
label_smoothing=0.0;  # 0.0,0.1,0.2,0.3
switch_case_pattern=00;  # one of 00, 01, 10, 11
switch_case_probability=0.0;  # 0.0,0.05,0.1,0.2
switch_case_method=v1;  # v1, v2, substitution, retokenization
ignore_upper_case_phase=never;  # never, train, eval, both
knn=0;  # 0,9,65,513
dyn_knn=0;  # 0,9,65,513
knn_metric=ip;  # ip, l2, cos
sample_k=1;  # any number between 0 and dyn_knn - 1
skip_knn=0;  # any number between 0 and dyn_knn - sample_k - 1
share_dyn_knn=True;  # either True or False
update_rep=False;  # either True or False
detach_rep=False;  # either True or False
retri_on_old_rep=True;  # either True or False
update_buf_rep=-1;  # integer, update_buf_rep_steps
knn_type=default;  # default, random, topk
negw=0.0;  # float
temp=0.05;  # positive float
pooler_type=cls;  # cls,cls_before_pooler,avg,avg_before_pooler,avg_first_last,avg_first_last_before_pooler,avg_top2,avg_top2_before_pooler,demean
weight_decay=0.0;  # 0.0,0.1
# normalized=0,1;  # TODO, normalized representation
seed=42;  # 42,49,99
train_file=../../dataset/SimCSE/wiki1m_for_simcse.txt;
train_file_dedupl=../../dataset/SimCSE/wiki1m_deduplicate_for_simcse.txt;
new_train_file=None;
output_dir=../output/bert_large/sentence_embedding/unsupervised;
new_output_dir=None;
tensorboard_dir=../output/bert_large/sentence_embedding/tensorboard/unsupervised;
new_tensorboard_dir=None;
sent_rep_cache_file=../../dataset/SimCSE/roberta_large_cached_pooler_output/wiki1m_deduplicate_roberta_large_pooler_output.pt;
SentEval_data_dir=../../dataset/SentEval;
overwrite_cache=False;
test_only=False;
metric_for_best_model=stsb_spearman;
postfix=None;
print_only=False;

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
        cuda)                    cuda=${VALUE} ;;
        model_type)              model_type=${VALUE} ;;
        model_size)              model_size=${VALUE} ;;
        learning_rate)           learning_rate=${VALUE} ;;
        hidden_dropout_prob)     hidden_dropout_prob=${VALUE} ;;
        weight_decay)            weight_decay=${VALUE} ;;
        label_smoothing)         label_smoothing=${VALUE} ;;
        switch_case_pattern)     switch_case_pattern=${VALUE} ;;
        switch_case_probability) switch_case_probability=${VALUE} ;;
        switch_case_method)      switch_case_method=${VALUE} ;;
        ignore_upper_case_phase) ignore_upper_case_phase=${VALUE} ;;
        knn)                     knn=${VALUE} ;;
        dyn_knn)                 dyn_knn=${VALUE} ;;
        knn_metric)              knn_metric=${VALUE} ;;
        sample_k)                sample_k=${VALUE} ;;
        skip_knn)                skip_knn=${VALUE} ;;
        share_dyn_knn)           share_dyn_knn=${VALUE} ;;
        update_rep)              update_rep=${VALUE} ;;
        detach_rep)              detach_rep=${VALUE} ;;
        retri_on_old_rep)        retri_on_old_rep=${VALUE} ;;
        update_buf_rep)          update_buf_rep=${VALUE} ;;
        knn_type)                knn_type=${VALUE} ;;
        negw)                    negw=${VALUE} ;;
        temp)                    temp=${VALUE} ;;
        pooler_type)             pooler_type=${VALUE} ;;
        seed)                    seed=${VALUE} ;;
        test_only)               test_only=${VALUE} ;;
        metric_for_best_model)   metric_for_best_model=${VALUE} ;;
        postfix)                 postfix=${VALUE} ;;
        print_only)              print_only=${VALUE} ;;
        new_train_file)          new_train_file=${VALUE} ;;
        new_output_dir)          output_dir=${VALUE} ;;
        new_tensorboard_dir)     tensorboard_dir=${VALUE} ;;
        sent_rep_cache_file)     sent_rep_cache_file=${VALUE} ;;
        overwrite_cache)         overwrite_cache=${VALUE} ;;
        *)                       echo "$KEY not supported" && exit ;;
    esac

done

learning_rate=($(echo $learning_rate | tr "," "\n"));
hidden_dropout_prob=($(echo $hidden_dropout_prob | tr "," "\n"));
weight_decay=($(echo $weight_decay | tr "," "\n"));
label_smoothing=($(echo $label_smoothing | tr "," "\n"));
switch_case_probability=($(echo $switch_case_probability | tr "," "\n"));
knn=($(echo $knn | tr "," "\n"));
dyn_knn=($(echo $dyn_knn | tr "," "\n"));
negw=($(echo $negw | tr "," "\n"));
temp=($(echo $temp | tr "," "\n"));
pooler_type=($(echo $pooler_type | tr "," "\n"));
seed=($(echo $seed | tr "," "\n"));

if [ "${cuda}" == "0,1,2,3" ]; then
    master_port=29500;
elif [ "${cuda}" == "0,1,2,3,4,5,6,7" ]; then
    batch_size=64;
    ngpu=8;
    master_port=29500;
elif [ "${cuda}" == "4,5,6,7" ]; then
    master_port=29501;
fi

if [ "${new_train_file}" != "None" ]; then
    train_file=${new_train_file};
fi

if [ "${new_output_dir}" != "None" ]; then
    output_dir=${new_output_dir}
elif [ "${model_size}" != "large" ]; then
    output_dir=../output/bert_${model_size}/sentence_embedding/unsupervised;
fi

if [ "${new_tensorboard_dir}" != "None" ]; then
    tensorboard_dir=${new_tensorboard_dir}
elif [ "${model_size}" != "large" ]; then
    tensorboard_dir=../output/bert_${model_size}/sentence_embedding/tensorboard/unsupervised;
fi

base_model=${model_type}_${model_size}_bs${batch_size}x${ngpu};
# params_set is an array of strings, each for one hyper-param setting
params_set=("model_name=${base_model} train_file=${train_file}");

function learning_rate_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params);

        for lr in ${learning_rate[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ $param == model_name=* ]]; then
                    new_params_arr+=("${param}_lr${lr}");
                else
                    new_params_arr+=("$param");
                fi
            done
            new_params_arr+=("learning_rate=${lr}");
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}");
}

function hidden_dropout_prob_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params);

        for hdp in ${hidden_dropout_prob[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do
                if [[ ! $param == model_name=* ]] || [ "${hdp}" == "0.1" ]; then
                    new_params_arr+=("$param");
                else
                    new_params_arr+=("${param}_dropout${hdp}");
                fi
            done
            new_params_arr+=("hidden_dropout_prob=${hdp}");
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function weight_decay_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for wd in ${weight_decay[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${wd}" == "0.0" ]; then
                    new_params_arr+=("$param");
                else
                    new_params_arr+=("${param}_wd${wd}");
                    new_params_arr+=("weight_decay=${wd}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function label_smoothing_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for ls in ${label_smoothing[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${ls}" == "0.0" ]; then
                    new_params_arr+=("$param");
                else
                    new_params_arr+=("${param}_labelsmooth${ls}");
                    new_params_arr+=("label_smoothing=${ls}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function switch_case_loop(){
    local new_params_set=();
    for params in "${params_set[@]}";
    do
        params_arr=($params)

        for scp in ${switch_case_probability[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${scp}" == "0.0" ]; then
                    new_params_arr+=("$param");
                else
                    local model_name_string=${param}_switchcase${scp};
                    if [ "${switch_case_pattern}" == "00" ]; then
                        switch_case_pattern=01;
                    fi
                    if [ "${switch_case_pattern}" != "01" ]; then
                        model_name_string=${model_name_string}_${switch_case_pattern};
                    fi
                    if [ "${switch_case_method}" != "v1" ]; then
                        model_name_string=${model_name_string}_${switch_case_method};
                    fi
                    new_params_arr+=("${model_name_string}");
                    new_params_arr+=("switch_case_probability=${scp}");
                    new_params_arr+=("switch_case_pattern=${switch_case_pattern}");
                    new_params_arr+=("switch_case_method=${switch_case_method}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function ignore_case_handler(){
    if [ ! "${ignore_upper_case_phase}" == "never" ]; then
        local new_params_set=();
        for params in "${params_set[@]}"; 
        do
            params_arr=($params)

            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]]; then
                    new_params_arr+=("$param")
                else
                    new_params_arr+=("${param}_ignore_upper_case_${ignore_upper_case_phase}");
                    new_params_arr+=("ignore_upper_case_phase=${ignore_upper_case_phase}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
        params_set=("${new_params_set[@]}")
    fi
}

function knn_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for rhn in ${knn[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [ "${rhn}" == "0" ]; then
                    new_params_arr+=("$param");
                elif [[ $param == train_file=* ]] && [ "${new_train_file}" == "None" ]; then
                    new_params_arr+=("train_file=${train_file_dedupl}");
                elif [[ $param == model_name=* ]]; then
                    srhn=1;
                    new_params_arr+=("${param}_retri${rhn}sample${srhn}_dedupl");
                    retriever_results_cache_file=../../dataset/SimCSE/roberta_large_cached_pooler_output/wiki1m_deduplicate_roberta_large_pooler_output_retriever_${rhn}neg_results.npy;
                    new_params_arr+=("retrieve_hard_negatives=${rhn}");
                    new_params_arr+=("sample_retrieved_hard_negatives=${srhn}");
                    new_params_arr+=("sent_rep_cache_file=${sent_rep_cache_file}");
                    new_params_arr+=("retriever_results_cache_file=${retriever_results_cache_file}");
                else
                    new_params_arr+=("$param");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function dyn_knn_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for rdn in ${dyn_knn[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do
                if [ "${rdn}" == "0" ]; then
                    new_params_arr+=("$param");
                elif [[ $param == train_file=* ]] && [ "${new_train_file}" == "None" ]; then
                    new_params_arr+=("train_file=${train_file_dedupl}");
                elif [[ $param == model_name=* ]]; then
                    if [ "${knn_type}" == "random" ]; then
                        local model_name_string=${param}_retri${rdn}random
                        new_params_arr+=("retrieved_and_sampled_knn_type=${knn_type}");
                    elif [ "${knn_type}" == "topk" ]; then
                        local model_name_string=${param}_retri${rdn}top_dynamic
                        new_params_arr+=("retrieved_and_sampled_knn_type=${knn_type}");
                    else
                        local model_name_string=${param}_retri${rdn}dynamic
                    fi
                    if [ "${skip_knn}" != "0" ]; then
                        model_name_string=${model_name_string}_skip${skip_knn};
                        new_params_arr+=("ignore_num_top_retrieved_dynamic_negatives=${skip_knn}");
                    fi
                    srdn=${sample_k};
                    model_name_string=${model_name_string}_sample${srdn}_dedupl;
                    if [ "${retri_on_old_rep}" == "True" ]; then
                        model_name_string=${model_name_string}_retri_then;
                        new_params_arr+=("retrieve_using_buffer_representations=${retri_on_old_rep}");
                    fi
                    if [ "${update_rep}" == "True" ]; then
                        model_name_string=${model_name_string}_update;
                    else
                        model_name_string=${model_name_string}_no_update;
                        new_params_arr+=("update_retrieved_negatives_representations=${update_rep}");
                    fi
                    if [ ! "${knn_metric}" == "ip" ]; then
                        model_name_string=${model_name_string}_metric_${knn_metric};
                        new_params_arr+=("retriever_metric=${knn_metric}");
                    fi
                    if [ "${share_dyn_knn}" == "False" ]; then
                        model_name_string=${model_name_string}_samplewise_cos;
                        new_params_arr+=("share_sampled_dynamic_negatives=${share_dyn_knn}");
                    fi
                    if [ "${detach_rep}" == "True" ]; then
                        model_name_string=${model_name_string}_detach;
                        new_params_arr+=("detach_retrieved_negatives_representations=${detach_rep}");
                    fi
                    if [ ! "${update_buf_rep}" == "-1" ]; then
                        model_name_string=${model_name_string}_updatebuf${update_buf_rep};
                        new_params_arr+=("update_buffer_representations_steps=${update_buf_rep}");
                    fi
                    new_params_arr+=("${model_name_string}");
                    new_params_arr+=("retrieve_dynamic_negatives=${rdn}");
                    new_params_arr+=("sample_retrieved_dynamic_negatives=${srdn}");
                    new_params_arr+=("sent_rep_cache_file=${sent_rep_cache_file}");
                else
                    new_params_arr+=("$param");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function hard_negative_weight_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for hnw in ${negw[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${hnw}" == "0.0" ]; then
                    new_params_arr+=("$param")
                else
                    new_params_arr+=("${param}_negw${hnw}");
                    new_params_arr+=("hard_negative_weight=${hnw}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function cl_temperature_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for ct in ${temp[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${ct}" == "0.05" ]; then
                    new_params_arr+=("$param")
                else
                    new_params_arr+=("${param}_temp${ct}");
                    new_params_arr+=("cl_temperature=${ct}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function pooler_type_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for pt in ${pooler_type[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${pt}" == "cls" ]; then
                    new_params_arr+=("$param")
                else
                    new_params_arr+=("${param}_pooler${pt}");
                    new_params_arr+=("pooler_type=${pt}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function seed_loop(){
    local new_params_set=();
    for params in "${params_set[@]}"; 
    do
        params_arr=($params)

        for s in ${seed[*]};
        do
            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]] || [ "${s}" == "42" ]; then
                    new_params_arr+=("$param")
                else
                    new_params_arr+=("${param}_s${s}");
                    new_params_arr+=("seed=${s}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
    done
    params_set=("${new_params_set[@]}")
}

function postfix_handler(){
    if [ ! "${postfix}" == "None" ]; then
        local new_params_set=();
        for params in "${params_set[@]}"; 
        do
            params_arr=($params)

            local new_params_arr=();
            for param in "${params_arr[@]}"; do 
                if [[ ! $param == model_name=* ]]; then
                    new_params_arr+=("$param")
                else
                    new_params_arr+=("${param}_${postfix}");
                fi
            done
            new_params_set+=("${new_params_arr[*]}");
        done
        params_set=("${new_params_set[@]}")
    fi
}


################################
# do hyper-parameter configuration
# echo "#params_set: "${#params_set[@]}
learning_rate_loop
hidden_dropout_prob_loop
weight_decay_loop
label_smoothing_loop
switch_case_loop
ignore_case_handler
knn_loop
dyn_knn_loop
hard_negative_weight_loop
cl_temperature_loop
pooler_type_loop
seed_loop
postfix_handler


echo "#params_set: "${#params_set[@]}
count_effective_param_set=0;
# string=$( IFS=$'\n'; echo "${params_set[*]}" )
# echo "$string"

################################
# run model
for params in "${params_set[@]}"; 
do  
    # find certain parameters
    params_arr=($params)
    for param in "${params_arr[@]}"; do 
        if [[ $param == model_name=* ]]; then
            model_name=${param#*=}
        elif [[ $param == train_file=* ]]; then
            train_file=${param#*=}
        elif [[ $param == hidden_dropout_prob=* ]]; then
            hdp=${param#*=}
        fi
    done
    model_folder=${output_dir}/${model_name};
    model_path=${model_folder}/pytorch_model.bin;
    if [ -f "$model_path" ]; then
        if [ "${test_only}" == "True" ]; then
            echo "Test "${model_folder}". cuda "$cuda
            bash ./scripts/bert/run_simcse_pretraining.sh \
                model_name_or_path=${model_folder} config_name=${model_folder}/config.json \
                model_type=${model_type} model_size=${model_size} do_train=False \
                train_file=${train_file} output_dir=${output_dir}/test_only tensorboard_dir=${tensorboard_dir} SentEval_data_dir=${SentEval_data_dir} \
                hidden_dropout_prob=${hdp} \
                cuda=${cuda} ngpu=1 \
                model_name=${model_name}
        else
            echo $model_name" has been trained previously and skipped this time."
        fi
    else
        if [ "${test_only}" == "True" ]; then
            echo $model_name" has not been trained yet and will be skipped for test this time."
        else    
            if [ "${print_only}" == "False" ]; then
                echo "Train "$model_name". cuda "$cuda" port "$master_port
                bash ./scripts/bert/run_simcse_pretraining.sh \
                    output_dir=${output_dir} tensorboard_dir=${tensorboard_dir} SentEval_data_dir=${SentEval_data_dir} overwrite_cache=${overwrite_cache} \
                    model_type=${model_type} model_size=${model_size} num_train_epochs=1 discard_embd_proj=true eval_steps=125 \
                    per_device_train_batch_size=${batch_size} cuda=${cuda} ngpu=${ngpu} master_port=${master_port} seed=${s} \
                    $params
            else
                echo "Train "$model_path". cuda "$cuda" port "$master_port" train_file "${train_file}
                # echo "params ""output_dir=${output_dir} tensorboard_dir=${tensorboard_dir} model_type=${model_type} model_size=${model_size} num_train_epochs=1 discard_embd_proj=true eval_steps=125 per_device_train_batch_size=${batch_size} cuda=${cuda} ngpu=${ngpu} master_port=${master_port} seed=${s} $params"
                count_effective_param_set=$(expr ${count_effective_param_set} + 1)
            fi
        fi
    fi
    wait
done
if [ "${print_only}" == "True" ]; then
    echo "#effective params_set: "${count_effective_param_set}
fi