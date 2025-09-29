import os
import re
import json
import faiss
import torch
import datasets
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel, AutoConfig


# ---------- Retriever Core Components ----------

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus

def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError

class Encoder:
    def __init__(self, model_path, pooling_method, max_length, use_fp16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
        if use_fp16:
            self.model = self.model.half()
        self.pooling_method = pooling_method
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: List[str], is_query=True):
        if "e5" in self.model.name_or_path.lower():
            if is_query:
                texts = [f"query: {x}" for x in texts]
            else:
                texts = [f"passage: {x}" for x in texts]
        elif "bge" in self.model.name_or_path.lower() and is_query:
            texts = [f"Represent this sentence for searching relevant passages: {x}" for x in texts]

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model(**inputs, return_dict=True)
        emb = pooling(outputs.pooler_output, outputs.last_hidden_state, inputs['attention_mask'], self.pooling_method)
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

class DenseRetriever:
    def __init__(self, args):
        self.index = faiss.read_index(args.index_path)
        if args.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co)
        self.corpus = load_corpus(args.corpus_path)
        self.encoder = Encoder(
            model_path=args.retrieval_model_path,
            pooling_method=args.retrieval_pooling_method,
            max_length=args.retrieval_query_max_length,
            use_fp16=args.retrieval_use_fp16
        )
        self.topk = args.retrieval_topk

    def search(self, query: str, num=None):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode([query])
        scores, idxs = self.index.search(query_emb, k=num)
        return load_docs(self.corpus, idxs[0])

def get_retriever(args):
    return DenseRetriever(args)

# ---------- Filtering Logic ----------

def extract_search_query(text: str):
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def contains_gold_answer(ctxs, answers):
    for ctx in ctxs:
        context_text = ctx.get("contents", "") + "\n" + ctx.get("title", "")
        if any(ans.lower() in context_text.lower() for ans in answers):
            return True
    return False

def filter_and_retrieve(data_path, retriever, save_path, topk=10):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    filtered = []
    for sample in tqdm(data, desc="Filtering"):
        query = extract_search_query(sample.get("model_answer", ""))
        if not query:
            print("Without Query")
            continue

        ctxs = retriever.search(query, num=topk)
        if contains_gold_answer(ctxs, sample.get("gold_answer", [])):
            sample["retrieved_ctxs"] = ctxs
            filtered.append(sample)
            print("with answer")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(filtered)} filtered samples to: {save_path}")
    print("Filter rate:", 1-len(filtered)/len(data))
# ---------- CLI Entry ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vllm_output", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--retrieval_topk", type=int, default=10)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)

    parser.add_argument("--retrieval_model_path", type=str, required=True)
    parser.add_argument("--retrieval_pooling_method", default="mean")
    parser.add_argument("--retrieval_query_max_length", type=int, default=256)
    parser.add_argument("--retrieval_use_fp16", action="store_true")
    parser.add_argument("--retrieval_batch_size", type=int, default=512)
    parser.add_argument("--faiss_gpu", default=True, type=bool)

    args = parser.parse_args()

    # 自动补 index 路径
    if args.retrieval_method != "bm25":
        args.index_path = os.path.join(args.index_path, f"{args.retrieval_method}_Flat.index")
    else:
        args.index_path = os.path.join(args.index_path, "bm25")

    retriever = get_retriever(args)
    filter_and_retrieve(args.vllm_output, retriever, args.save_path, topk=args.retrieval_topk)
