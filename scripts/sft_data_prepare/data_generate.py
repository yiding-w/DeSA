import json
import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np

def safe_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# 1. 加载 parquet 文件
df = pd.read_parquet('/scratch/mrm2vx/Search-R1/data/nq_hotpotqa_query_generation_sft/train.parquet')

# 2. 初始化 vLLM 模型
llm = LLM(model='Qwen/Qwen2.5-3B-Instruct',gpu_memory_utilization=0.80)  # 你可以换成自己的 checkpoint
sampling_params = SamplingParams(temperature=1, top_p=0.9, max_tokens=512,)

# 3. 构造 batch prompt 列表
prompts = [item[0]['content'] for item in df['prompt']]  # 从 prompt list 里取出 content

# 4. 推理（建议使用 tqdm 显示进度）
batch_size = 400
results = []
for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i:i+batch_size]
    outputs = llm.generate(batch_prompts, sampling_params)
    # outputs = [None] * batch_size
    for j, output in enumerate(outputs):
        question = df.iloc[i + j]['prompt'][0]['content']
        # print(question)
        gold_answer = df.iloc[i + j]['reward_model']['ground_truth']['target']
        # print(gold_answer)
        # exit(0)
        if isinstance(gold_answer, np.ndarray):
            gold_answer = gold_answer.tolist()
        print(gold_answer)
        result = {
            "question": df.iloc[i + j]['prompt'][0]['content'],
            "gold_answer": df.iloc[i + j]['reward_model']['ground_truth']['target'],
            "model_answer": output.outputs[0].text.strip(),  # 默认只取第一个生成结果
        }
        results.append(result)

# 5. 保存为 JSONL 格式
with open("vllm_outputs.json", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=safe_json) + "\n")

# 6. 验证：用 datasets 加载
dataset = Dataset.from_json("vllm_outputs.json")
print(dataset[0])