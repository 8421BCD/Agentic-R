bash /root/paddlejob/workspace/env_run/train_bert/clear_gpu.sh
bash /root/paddlejob/workspace/env_run/train_bert/stop_gpu.sh
# current_dir=$PWD
# cd /root/paddlejob/workspace/env_run/train_bert/
# bash train_bert_small.sh
# cd $current_dir

# for ours Agentic-R
# model_name=deepr-e5_triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em_local_currentq
# # ckpt=checkpoint-1986
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
#     --retrieval_method ${model_name} \
#     --model_path /root/paddlejob/workspace/trained_models/${model_name} \
#     --corpus_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
#     --save_dir /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/ \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 128 \
#     --faiss_type Flat \
#     --sentence_transformer \
#     --instruction "passage: "

# # for e5
model_name=e5-large-v2
model_path=/root/paddlejob/workspace/llm/$model_name
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
    --retrieval_method ${model_name} \
    --model_path $model_path \
    --corpus_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
    --save_dir /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/ \
    --use_fp16 \
    --max_length 256 \
    --batch_size 256 \
    --faiss_type Flat \
    --sentence_transformer \

# for bge
# model_name=scarlet_bge_base_en_v1.5
# # model_name=bge_base_en_v1.5
# model_path=/root/paddlejob/workspace/llm/$model_name
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
#     --retrieval_method ${model_name} \
#     --model_path $model_path \
#     --corpus_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
#     --save_dir /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/ \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 256 \
#     --faiss_type Flat \
#     --sentence_transformer \

# for llm-embedder (also bge-based)
# model_name=llm-embedder
# model_path=/root/paddlejob/workspace/llm/$model_name
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
#     --retrieval_method ${model_name} \
#     --model_path $model_path \
#     --corpus_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
#     --save_dir /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/ \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 256 \
#     --faiss_type Flat \
#     --sentence_transformer \
#     --instruction "Represent this document for retrieval: "


# for contriever (rescore)
# model_name=iqatr-hotpotqa
# model_path=/root/paddlejob/workspace/llm/$model_name
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
#     --retrieval_method ${model_name} \
#     --model_path $model_path \
#     --corpus_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
#     --save_dir /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/ \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 256 \
#     --faiss_type Flat \
#     --sentence_transformer \

# for qwen3
# model_name=Qwen3-Embedding-0.6B
# model_path=/root/paddlejob/workspace/llm/$model_name
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m flashrag.retriever.index_builder \
#     --retrieval_method ${model_name} \
#     --model_path $model_path \
#     --corpus_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
#     --save_dir /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/ \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 64 \
#     --faiss_type Flat \
#     --sentence_transformer \
#     --instruction "" \


bash /root/paddlejob/workspace/env_run/train_bert/clear_gpu.sh
bash /root/paddlejob/workspace/env_run/train_bert/stop_gpu.sh
current_dir=$PWD
cd /root/paddlejob/workspace/env_run/train_bert/
bash train_bert.sh
cd $current_dir