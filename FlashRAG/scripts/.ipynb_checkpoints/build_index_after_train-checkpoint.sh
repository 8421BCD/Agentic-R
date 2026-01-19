model_name=xxx
ckpt=xxx
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
    --retrieval_method ${model_name}_${ckpt} \
    --model_path {WORKSPACE_DIR}/trained_models/${model_name}/${ckpt} \
    --corpus_path {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
    --save_dir {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/ \
    --use_fp16 \
    --max_length 256 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat \
    --sentence_transformer \
    --instruction "passage: " \
    # --save_embedding

