export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MASTER_PORT=29506

# ------------------------------ iteration0 ------------------------------
for dataset_name in hotpotqa triviaqa
do
    start_qid=0
    end_qid=9999999999
    generator_name=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em
    python step3-2_generate_global_utility.py \
        --generator_model_path {WORKSPACE_DIR}/trained_models/$generator_name \
        --retrieval_model_path {WORKSPACE_DIR}/llm/e5-base-v2 \
        --trajectory_path training_data/passage_candidates/$generator_name/${dataset_name}.json \
        --start_qid $start_qid \
        --end_qid $end_qid \
        --result_path training_data/global_utility/$generator_name/${dataset_name}.jsonl \
        --index_path {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/e5-base-v2_Flat.index \
        --method_name search-r1 \
        --split train \
        --dataset_name $dataset_name \
        --gpu_id "0,1,2,3" \

done

# ------------------------------ iteration1 ------------------------------
# dataset_name=triviaqa
# start_qid=0
# end_qid=99999999
# generator_name=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
# retriever=deepr-e5_triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em_global-local_question-currentq
# python step3-2_generate_global_utility.py \
#     --generator_model_path {WORKSPACE_DIR}/trained_models/$generator_name \
#     --retrieval_model_path {WORKSPACE_DIR}/trained_models/$retriever \
#     --trajectory_path training_data/passage_candidates/$generator_name/${dataset_name}.json \
#     --start_qid $start_qid \
#     --end_qid $end_qid \
#     --result_path training_data/global_utility/$generator_name/${dataset_name}.jsonl \
#     --index_path {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
#     --method_name search-r1 \
#     --split train \
#     --dataset_name $dataset_name \
#     --gpu_id "0,1,2,3" \
#     --agentic_retriever_input \