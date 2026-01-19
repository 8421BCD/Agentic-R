export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MASTER_PORT=12334

########################################################################################################################################################
# --------------------- retrievers based on our agent triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1 (trained for 2 iterations) ---------------------
########################################################################################################################################################
################ Agentic-R_e5 #################
generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
retriever=Agentic-R_e5
dataset_names_all="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle"
# dataset_names_all="nq"
python run_exp.py \
    --generator_model_path /root/paddlejob/workspace/trained_models/$generator_model \
    --retrieval_model_path /root/paddlejob/workspace/trained_models/$retriever \  
    --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
    --method_name search-r1 \
    --retrieval_method e5 \
    --dataset_names_all $dataset_names_all \
    --gpu_id "0,1,2,3" \
    --agentic_retriever_input \


################ e5 #################
# generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
# retriever=e5-base-v2
# dataset_names_all="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle"
# # dataset_names_all="nq"
# python run_exp.py \
#     --generator_model_path /root/paddlejob/workspace/trained_models/$generator_model \
#     --retrieval_model_path /root/paddlejob/workspace/llm/$retriever \
#     --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
#     --method_name search-r1 \
#     --retrieval_method e5 \
#     --dataset_names_all $dataset_names_all \
#     --gpu_id "0,1,2,3" \



########################################################################################################################################################
# --------------------- retrievers based on R1-Searcher ---------------------
########################################################################################################################################################
################ Agentic-R_e5 #################
# generator_model=Qwen-2.5-7B-base-RAG-RL
# retriever=Agentic-R_e5
# retrieval_method=e5
# # dataset_names_all="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle"
# dataset_names_all="nq"
# python run_exp.py \
#     --generator_model_path /root/paddlejob/workspace/llm/$generator_model \
#     --retrieval_model_path /root/paddlejob/workspace/trained_models/$retriever \
#     --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
#     --retrieval_method $retrieval_method \
#     --method_name r1-searcher \
#     --dataset_names_all $dataset_names_all \
#     --gpu_id "0,1,2,3" \
#     --agentic_retriever_input \


################ e5 #################
# generator_model=Qwen-2.5-7B-base-RAG-RL
# retriever=e5-base-v2
# retrieval_method=e5
# # dataset_names_all="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle"
# dataset_names_all="nq"
# python run_exp.py \
#     --generator_model_path /root/paddlejob/workspace/llm/$generator_model \
#     --retrieval_model_path /root/paddlejob/workspace/llm/$retriever \
#     --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
#     --retrieval_method $retrieval_method \
#     --method_name r1-searcher \
#     --dataset_names_all $dataset_names_all \
#     --gpu_id "0,1,2,3" \
