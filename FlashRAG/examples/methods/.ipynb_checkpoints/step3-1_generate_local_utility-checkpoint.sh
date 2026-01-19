export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29508

# ------------------------------ iteration0 ------------------------------
agent_name=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em
for dataset in hotpotqa triviaqa
do
    python step3-1_generate_local_utility.py \
        --model_name_or_path {WORKSPACE_DIR}/llm/Qwen2.5-72B-Instruct \
        --num_gpus 8 \
        --input_data_path training_data/passage_candidates/${agent_name}/${dataset}.json \
        --subanswer_data_path training_data/trajectory_subanswer/${agent_name}/${dataset}.jsonl \
        --result_path training_data/local_utility/${agent_name}/${dataset}.jsonl \
        --batch_size 512 \
        --batch_passages_size 10 \
        --use_subanswer True \

done
# ------------------------------ iteration1 ------------------------------
# agent_name=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
# dataset=triviaqa
# python step3-1_generate_local_utility.py \
#     --model_name_or_path /root/paddlejob/workspace/llm/Qwen2.5-72B-Instruct \
#     --num_gpus 8 \
#     --input_data_path training_data/passage_candidates/${agent_name}/${dataset}.json \
#     --subanswer_data_path training_data/trajectory_subanswer/${agent_name}/${dataset}.jsonl \
#     --result_path training_data/local_utility/${agent_name}/${dataset}.jsonl \
#     --batch_size 512 \
#     --batch_passages_size 10 \
#     --use_subanswer True \
