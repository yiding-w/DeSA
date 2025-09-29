# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import json
recall_log = [] 

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_information_blocks(solution_str):
    """Extract all <information> ... </information> blocks from solution string.
    
    Args:
        solution_str (str): The solution text containing information blocks
        
    Returns:
        list: List of extracted information block contents
    """
    info_pattern = r"<information>(.*?)</information>"
    return re.findall(info_pattern, solution_str, re.DOTALL)

def extract_documents_from_information(information_blocks):
    """Extract documents from information blocks by splitting on 'Doc' patterns.
    
    Args:
        information_blocks (list): List of information block contents
        
    Returns:
        list: List of extracted document contents
    """
    documents = []
    for info_block in information_blocks:
        # 按 "Doc" 分割，支持多种可能的格式
        doc_patterns = [
            r"Doc\s*\d+\([^)]*\):\s*(.*?)(?=Doc\s*\d+|$)",  # Doc 1(Title: "xxx"): content
            r"Doc\s*\d+[:\.]?\s*(.*?)(?=Doc\s*\d+|$)",  # Doc 1: content Doc 2: content
            r"Document\s*\d+[:\.]?\s*(.*?)(?=Document\s*\d+|$)",  # Document 1: content
            r"\[Doc\s*\d+\]\s*(.*?)(?=\[Doc\s*\d+\]|$)",  # [Doc 1] content [Doc 2] content
        ]
        
        found_docs = False
        for pattern in doc_patterns:
            matches = re.findall(pattern, info_block, re.DOTALL | re.IGNORECASE)
            if matches:
                documents.extend([doc.strip() for doc in matches if doc.strip()])
                found_docs = True
                break
        
        # 只有当信息块中明确包含Doc分割标记时才提取文档
        # 如果没有找到明确的文档分割，不添加任何文档
    
    return documents

def compute_retrieval_accuracy(documents, ground_truth):
    """Compute retrieval accuracy: proportion of documents containing correct answers.
    
    Args:
        documents (list): List of retrieved documents
        ground_truth (dict): Ground truth containing target answers
        
    Returns:
        float: Retrieval accuracy score (0.0 to 1.0)
    """
    if not documents:
        return 0.0
    
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]
    
    relevant_docs = 0
    for doc in documents:
        normalized_doc = normalize_answer(doc)
        if any(target in normalized_doc for target in normalized_targets):
            relevant_docs += 1
    
    return relevant_docs / len(documents)

def compute_mmr(documents, ground_truth):
    """Compute Mean Reciprocal Rank (MRR): reciprocal rank of first relevant document.
    
    Args:
        documents (list): List of retrieved documents in ranked order
        ground_truth (dict): Ground truth containing target answers
        
    Returns:
        float: MRR score (0.0 to 1.0)
    """
    if not documents:
        return 0.0
    
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]
    
    for i, doc in enumerate(documents):
        normalized_doc = normalize_answer(doc)
        if any(target in normalized_doc for target in normalized_targets):
            return 1.0 / (i + 1)  # 排名从1开始
    
    return 0.0  # 没有相关文档

def extract_solution(solution_str):
    """Extract the answer from the solution string.
    
    Args:
        solution_str (str): The solution text containing answer tags
        
    Returns:
        str or None: Extracted answer or None if not found/invalid
    """
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

import numpy as np

def make_serializable_target(target):
    if isinstance(target, str):
        return [target]
    elif isinstance(target, np.ndarray):
        return [str(x) for x in target.tolist()]
    elif isinstance(target, list):
        return [str(x) for x in target]
    else:
        return [str(target)]


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., log_recall=False, recall_log_path='./recall_result/alltest_nqhotpotqatrained.json', return_details=False):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        return_details: if True, return detailed metrics dict; if False, return single score
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    # Compute recall score from <information> blocks (independent of log_recall)
    information_blocks = extract_information_blocks(solution_str)
    normalized_info = normalize_answer(" ".join(information_blocks))
    recalled = any(
        normalize_answer(ans) in normalized_info
        for ans in ([ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target'])
    )
    recall_score = score if recalled else format_score

    # Extract documents and compute retrieval metrics
    documents = extract_documents_from_information(information_blocks)
    retrieval_accuracy = compute_retrieval_accuracy(documents, ground_truth)
    mmr = compute_mmr(documents, ground_truth)

    if do_print:
        print(f"Documents found: {len(documents)}")
        print(f"Retrieval accuracy: {retrieval_accuracy:.3f}")
        print(f"MMR: {mmr:.3f}")

    # Log recall (side effect, unchanged)
    if log_recall:
        single_record = {
            "golden_answers": ground_truth.get('target', []).tolist(),
            "solution_string": solution_str,
            "extracted_answer": answer,
            "information": information_blocks,
            "documents": documents,
            "recalled": recalled,
            "retrieval_accuracy": retrieval_accuracy,
            "mmr": mmr
        }

        with open(recall_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(single_record, ensure_ascii=False) + '\n')

    # Original EM score calculation (unchanged) - qa_em only cares about EM
    if answer is None:
        combined_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            combined_score = score
        else:
            combined_score = format_score

    if return_details:
        return {
            'recall_score': recall_score,  # Independent recall calculation
            'em_score': combined_score,  # EM is the main metric here
            'combined_score': combined_score,
            'retrieval_accuracy': retrieval_accuracy,
            'mmr': mmr,
            'num_documents': len(documents)
        }
    else:
        return combined_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1., return_details=False):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        return_details: if True, return detailed metrics dict; if False, return single score
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    # Compute recall score from <information> blocks
    information_blocks = extract_information_blocks(solution_str)
    normalized_info = normalize_answer(" ".join(information_blocks))
    recalled = any(
        normalize_answer(ans) in normalized_info
        for ans in ([ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target'])
    )
    recall_score = score if recalled else format_score

    # Original SubEM score calculation (unchanged)
    if answer is None:
        combined_score = 0
    else:
        if subem_check(answer, ground_truth['target']):
            combined_score = score
        else:
            combined_score = format_score

    if return_details:
        return {
            'recall_score': recall_score,
            'em_score': combined_score,  # SubEM is the EM score here
            'combined_score': combined_score
        }
    else:
        return combined_score
