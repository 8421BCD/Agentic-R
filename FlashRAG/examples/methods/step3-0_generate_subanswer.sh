export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29508

# ------------------------------ iteration0 ------------------------------
for dataset in hotpotqa triviaqa
do
    agent_name=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em
    python step3-0_generate_subanswer.py \
        --model_name_or_path {WORKSPACE_DIR}/llm/Qwen2.5-72B-Instruct \
        --num_gpus 8 \
        --input_data_path training_data/trajectory/${agent_name}/${dataset}.json \
        --result_path training_data/trajectory_subanswer/${agent_name}/${dataset}.jsonl \
        --batch_size 512 \

done

# ------------------------------ iteration1 ------------------------------
# agent_name=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
# for dataset in triviaqa
# do
#     python step3-0_generate_subanswer.py \
#         --model_name_or_path /root/paddlejob/workspace/llm/Qwen2.5-72B-Instruct \
#         --num_gpus 8 \
#         --input_data_path training_data/trajectory/${agent_name}/${dataset}.json \
#         --result_path training_data/trajectory_subanswer/${agent_name}/${dataset}.jsonl \
#         --batch_size 512 
# done
