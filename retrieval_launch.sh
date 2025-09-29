#!/bin/bash
#SBATCH --job-name=retriever_server
#SBATCH --output=logs/retriever_%j.out
#SBATCH --error=logs/retriever_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00

# Load modules or activate environment
source ~/.bashrc
conda activate retriever

# Print local IP address (useful for connecting to the server externally)
echo "[INFO] Node hostname: $(hostname)"
echo "[INFO] IP address: $(hostname -I | awk '{print $1}')"

# Define variables
file_path=
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Start the retrieval server
python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
