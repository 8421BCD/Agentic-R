export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MASTER_PORT=29505

# ------------------------------ iteration0 ------------------------------
generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em
retriever=e5-base-v2
for dataset_name in triviaqa hotpotqa
do
    python step1_generate_trajectory.py \
        --generator_model_path /root/paddlejob/workspace/trained_models/$generator_model \
        --retrieval_model_path /root/paddlejob/workspace/llm/$retriever \
        --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.index \
        --method_name search-r1 \
        --split train \
        --dataset_name $dataset_name \
        --gpu_id "0,1,2,3" \

done

# ------------------------------ iteration1 ------------------------------
# generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
# retriever=Agentic-R_iter0    # the retriever trained after the first iteration
# for dataset_name in triviaqa hotpotqa
# do
    # python step1_generate_trajectory.py \
    #     --generator_model_path /root/paddlejob/workspace/trained_models/$generator_model \
    #     --retrieval_model_path /root/paddlejob/workspace/trained_models/$retriever \
    #     --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
    #     --method_name search-r1 \
    #     --retrieval_method e5 \
    #     --split train \
    #     --dataset_name $dataset \
    #     --gpu_id "0,1,2,3" \
    #     --agentic_retriever_input \
# done
