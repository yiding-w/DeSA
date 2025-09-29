#!/bin/bash

# ==== 数据与模型配置 ====
TOPK=3

# ==== 路径设置 ====
VLLM_OUTPUT="vllm_outputs.json"
SAVE_PATH=""

RETRIEVAL_MODEL_PATH="intfloat/e5-base-v2"
RETRIEVAL_METHOD="e5"

INDEX_PATH=""
CORPUS_PATH="${INDEX_PATH}/wiki-18.jsonl"

# ==== 执行过滤脚本 ====
CUDA_VISIBLE_DEVICES=0,1,2,3 python filter.py \
    --vllm_output "$VLLM_OUTPUT" \
    --save_path "$SAVE_PATH" \
    --retrieval_method "$RETRIEVAL_METHOD" \
    --retrieval_topk "$TOPK" \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_path "$RETRIEVAL_MODEL_PATH" \
    --retrieval_pooling_method "mean" \
    --retrieval_query_max_length 256 \
    --retrieval_use_fp16 \
    --retrieval_batch_size 512
