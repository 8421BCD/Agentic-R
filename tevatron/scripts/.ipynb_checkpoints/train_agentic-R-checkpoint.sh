workspace_dir=$(grep "WORKSPACE_DIR" ../../config.py | cut -d "'" -f 2)
project_dir=$(grep "PROJECT_DIR" ../../config.py | cut -d "'" -f 2)

# ------------------------------ iteration0 ------------------------------
save_name=Agentic-R_e5_iter0
training_data_name=retriever_training_data.jsonl
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60000 --module tevatron.retriever.driver.train \
    --deepspeed ${project_dir}/tevatron/deepspeed/ds_zero1_config.json \
    --output_dir ${workspace_dir}/trained_models/$save_name \
    --model_name_or_path ${workspace_dir}/llm/e5-base-v2 \
    --attn_implementation eager \
    --dataset_path ${project_dir}/FlashRAG/examples/methods/training_data/$training_data_name \
    --corpus_path ${workspace_dir}/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
    --agentic_retriever_input True \
    --save_strategy epoch \
    --num_train_epochs 2 \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --fp16 \
    --pooling mean \
    --normalize \
    --temperature 0.01 \
    --per_device_train_batch_size 32 \
    --gradient_checkpointing \
    --train_group_size 16 \
    --learning_rate 2e-5 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --save_only_model True \
    --corpus_name custom_corpus \


# ------------------------------ iteration1 ------------------------------
# save_name=Agentic-R_e5
# training_data_name=retriever_training_data_iter1.jsonl
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60000 --module tevatron.retriever.driver.train \
#     --deepspeed ${project_dir}/tevatron/deepspeed/ds_zero1_config.json \
#     --output_dir ${workspace_dir}/trained_models/$save_name \
#     --model_name_or_path ${workspace_dir}/llm/e5-base-v2 \
#     --attn_implementation eager \
#     --dataset_path ${workspace_dir}/env_run/deep-retriever/FlashRAG/examples/methods/training_data/$training_data_name \
#     --corpus_path ${workspace_dir}/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
#     --agentic_retriever_input True \
#     --save_strategy epoch \
#     --num_train_epochs 2 \
#     --query_prefix "query: " \
#     --passage_prefix "passage: " \
#     --fp16 \
#     --pooling mean \
#     --normalize \
#     --temperature 0.01 \
#     --per_device_train_batch_size 32 \
#     --gradient_checkpointing \
#     --train_group_size 16 \
#     --learning_rate 2e-5 \
#     --query_max_len 512 \
#     --passage_max_len 512 \
#     --logging_steps 10 \
#     --overwrite_output_dir \
#     --save_only_model True \
#     --corpus_name custom_corpus \

