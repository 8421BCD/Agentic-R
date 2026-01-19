export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29508

# ------------------------------ iteration0 ------------------------------
generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em
retriever=e5-base-v2
python step2_generate_passage_candidates.py \
    --generator_model $generator_model \
    --retrieval_model_path "{WORKSPACE_DIR}/llm/${retriever}" \
    --index_path "{WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index" \
    --dataset_names triviaqa hotpotqa

# ------------------------------ iteration1 ------------------------------
# generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
# retriever=deepr-e5_triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em_global-local_question-currentq
# python step2_generate_passage_candidates.py \
#     --generator_model $generator_model \
#     --retrieval_model_path "{WORKSPACE_DIR}/trained_models/${retriever}" \
#     --index_path "{WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index" \
#     --agentic_retriever_input True \
#     --dataset_names triviaqa hotpotqa

