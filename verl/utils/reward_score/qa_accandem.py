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
import os
from datetime import datetime
import numpy as np

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
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def make_serializable_target(target):
    if isinstance(target, str):
        return [target]
    elif isinstance(target, np.ndarray):
        return [str(x) for x in target.tolist()]
    elif isinstance(target, list):
        return [str(x) for x in target]
    else:
        return [str(target)]


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., 
                    log_recall=False, recall_log_path='./recall_result/qa_accandem_results.json', 
                    return_details=False):
    """The scoring function combining retrieval accuracy and EM with 0.5+0.5 weights.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for format failure
        score: the score for success
        log_recall: whether to log detailed results
        recall_log_path: path to save recall logs
        return_details: if True, return detailed metrics dict; if False, return single score
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"[QA_ACCANDEM] Golden answers: {ground_truth['target']}")
        print(f"[QA_ACCANDEM] Extracted answer: {answer}")

    # Extract documents and compute retrieval metrics
    information_blocks = extract_information_blocks(solution_str)
    documents = extract_documents_from_information(information_blocks)
    
    # Compute recall score from <information> blocks (like qa_search)
    normalized_info = normalize_answer(" ".join(information_blocks))
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]
    recalled = any(t in normalized_info for t in normalized_targets)
    recall_score = 1.0 if recalled else 0.0
    
    # Compute retrieval accuracy (separate metric)
    retrieval_accuracy = compute_retrieval_accuracy(documents, ground_truth)
    
    # Compute EM score
    if answer is None:
        em_success = False
        em_score = 0.0
    else:
        em_success = em_check(answer, ground_truth['target'])
        em_score = 1.0 if em_success else 0.0

    # Compute MMR for additional monitoring
    mmr = compute_mmr(documents, ground_truth)

    # Combined score: 0.5 * retrieval_accuracy + 0.5 * em_score
    combined_score = 0.5 * retrieval_accuracy * score + 0.5 * em_score * score
    if combined_score == 0:
        combined_score = format_score

    if do_print:
        print(f"[QA_ACCANDEM] Documents found: {len(documents)}")
        print(f"[QA_ACCANDEM] Recall success: {recalled}")
        print(f"[QA_ACCANDEM] Retrieval accuracy: {retrieval_accuracy:.3f}")
        print(f"[QA_ACCANDEM] EM success: {em_success}")
        print(f"[QA_ACCANDEM] Combined score: {combined_score:.3f} (0.5*{retrieval_accuracy:.3f} + 0.5*{em_score:.1f})")
        print(f"[QA_ACCANDEM] MMR: {mmr:.3f}")

    # Log results
    if log_recall:
        single_record = {
            "golden_answers": make_serializable_target(ground_truth['target']),
            "solution_string": solution_str,
            "extracted_answer": answer,
            "information": information_blocks,
            "documents": documents,
            "recalled": recalled,
            "recall_score": recall_score,
            "retrieval_accuracy": retrieval_accuracy,
            "em_success": em_success,
            "em_score": em_score,
            "combined_score": combined_score,
            "mmr": mmr,
            "num_documents": len(documents)
        }

        os.makedirs(os.path.dirname(recall_log_path) if os.path.dirname(recall_log_path) else '.', exist_ok=True)
        with open(recall_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(single_record, ensure_ascii=False) + '\n')

    if return_details:
        return {
            'recall_score': recall_score,  # Proper recall calculation from information blocks
            'em_score': em_score,  # EM score (0 or 1)
            'combined_score': combined_score,  # 0.5 * retrieval_accuracy + 0.5 * em_score
            'retrieval_accuracy': retrieval_accuracy,
            'mmr': mmr,
            'num_documents': len(documents),
            'recalled': recalled,
            'em_success': em_success,
            'answer': answer,
            'targets': make_serializable_target(ground_truth['target'])
        }
    else:
        return combined_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1., return_details=False):
    """The scoring function for substring exact match combined with retrieval accuracy.

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
        print(f"[QA_ACCANDEM_SUBEM] Golden answers: {ground_truth['target']}")
        print(f"[QA_ACCANDEM_SUBEM] Extracted answer: {answer}")

    # Extract documents and compute retrieval metrics
    information_blocks = extract_information_blocks(solution_str)
    documents = extract_documents_from_information(information_blocks)
    
    # Compute recall score from <information> blocks (like qa_search)
    normalized_info = normalize_answer(" ".join(information_blocks))
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]
    recalled = any(t in normalized_info for t in normalized_targets)
    recall_score = 1.0 if recalled else 0.0
    
    # Compute retrieval accuracy
    retrieval_accuracy = compute_retrieval_accuracy(documents, ground_truth)
    
    # Compute SubEM score
    if answer is None:
        subem_success = False
        subem_score = 0.0
    else:
        subem_success = subem_check(answer, ground_truth['target'])
        subem_score = 1.0 if subem_success else 0.0

    # Combined score: 0.5 * retrieval_accuracy + 0.5 * subem_score
    combined_score = 0.5 * retrieval_accuracy * score + 0.5 * subem_score * score
    if combined_score == 0:
        combined_score = format_score

    if do_print:
        print(f"[QA_ACCANDEM_SUBEM] Recall success: {recalled}")
        print(f"[QA_ACCANDEM_SUBEM] Retrieval accuracy: {retrieval_accuracy:.3f}")
        print(f"[QA_ACCANDEM_SUBEM] SubEM success: {subem_success}")
        print(f"[QA_ACCANDEM_SUBEM] Combined score: {combined_score:.3f}")
    
    if return_details:
        return {
            'recall_score': recall_score,  # Recall score from information blocks
            'em_score': subem_score,  # SubEM score
            'combined_score': combined_score,  # 0.5 * retrieval_accuracy + 0.5 * subem_score
            'retrieval_accuracy': retrieval_accuracy,
            'mmr': compute_mmr(documents, ground_truth),
            'num_documents': len(documents),
            'em_success': subem_success,
            'answer': answer,
            'targets': make_serializable_target(ground_truth['target'])
        }
    else:
        return combined_score
