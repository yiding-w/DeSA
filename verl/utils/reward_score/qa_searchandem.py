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

recall_log = [] 

# 添加全局变量来跟踪训练指标
training_metrics = {
    'recall_scores': [],
    'em_scores': [],
    'combined_scores': [],
    'step_count': 0
} 

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


import json
import random

def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., log_recall=False, recall_log_path='./recall_result/nqhotpot_after_train.json', return_details=False):
    """The scoring function based only on recall from information blocks.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: not used anymore, retained for compatibility
        format_score: score when recall fails
        score: score when recall succeeds
        return_details: if True, return detailed metrics dict; if False, return only the combined score
    """
    # Extract information blocks from solution string
    information_blocks = extract_information_blocks(solution_str)
    normalized_info = normalize_answer(" ".join(information_blocks))
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]
    answer = extract_solution(solution_str=solution_str)
    
    # Extract documents and compute retrieval metrics
    documents = extract_documents_from_information(information_blocks)
    retrieval_accuracy = compute_retrieval_accuracy(documents, ground_truth)
    mmr = compute_mmr(documents, ground_truth)
    
    # Check recall
    recalled = any(t in normalized_info for t in normalized_targets)
    recall_score = 1.0 if recalled else 0.0
    
    # Check EM
    if answer is None:
        em_score = 0.0
        em_success = False
    else:
        em_success = em_check(answer, ground_truth['target'])
        em_score = score if em_success else format_score

    if random.randint(1, 64) == 1:
        print(f"--------------------------------")
        print(f"Golden answers: {targets}")
        print(f"Normalized info: {normalized_info}")
        print(f"Recall success: {recalled}")
        print(f"EM success: {em_success}")
        print(f"Documents found: {len(documents)}")
        print(f"Retrieval accuracy: {retrieval_accuracy:.3f}")
        print(f"MMR: {mmr:.3f}")

    # Log recall result
    if log_recall:
        single_record = {
            "target": make_serializable_target(ground_truth['target']),
            "information": information_blocks,
            "documents": documents,
            "recalled": recalled,
            "em_success": em_success,
            "answer": answer,
            "retrieval_accuracy": retrieval_accuracy,
            "mmr": mmr
        }
        with open(recall_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(single_record, ensure_ascii=False) + '\n')
    
    # Compute combined score
    if recalled:
        combined_score = 0.5 * score + 0.5 * em_score
    else:
        combined_score = 0.5 * format_score + 0.5 * em_score
    
    if return_details:
        return {
            'combined_score': combined_score,
            'recall_score': recall_score,
            'em_score': em_score / score if score > 0 else em_score,  # normalize to 0-1
            'recalled': recalled,
            'em_success': em_success,
            'answer': answer,
            'targets': targets,
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
        return_details: if True, return detailed metrics dict; if False, return only the score
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        subem_success = False
        subem_score = 0
    else:
        subem_success = subem_check(answer, ground_truth['target'])
        subem_score = score if subem_success else format_score
    
    if return_details:
        return {
            'combined_score': subem_score,  # 使用统一的字段名
            'recall_score': 1.0 if subem_success else 0.0,  # subem可以看作是一种recall
            'em_score': subem_score / score if score > 0 else subem_score,  # normalize to 0-1
            'recalled': subem_success,
            'em_success': subem_success,
            'answer': answer,
            'targets': ground_truth['target'] if isinstance(ground_truth['target'], list) else [ground_truth['target']]
        }
    else:
        return subem_score


def log_training_metrics(recall_score, em_score, combined_score, step=None, log_path='./training_metrics.json'):
    """Log training metrics for recall and EM.
    
    Args:
        recall_score: recall score (0 or 1)
        em_score: exact match score (0 or 1) 
        combined_score: combined reward score
        step: training step number (optional)
        log_path: path to save the metrics log
    """
    global training_metrics
    
    if step is not None:
        training_metrics['step_count'] = step
    else:
        training_metrics['step_count'] += 1
    
    training_metrics['recall_scores'].append(recall_score)
    training_metrics['em_scores'].append(em_score)
    training_metrics['combined_scores'].append(combined_score)
    
    # Log current metrics
    current_metrics = {
        'step': training_metrics['step_count'],
        'recall_score': recall_score,
        'em_score': em_score,
        'combined_score': combined_score,
        'timestamp': datetime.now().isoformat()
    }
    
    # Append to log file
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(current_metrics, ensure_ascii=False) + '\n')


def get_training_metrics_summary():
    """Get summary statistics of training metrics.
    
    Returns:
        dict: Summary statistics including mean, std, latest values
    """
    global training_metrics
    
    if not training_metrics['recall_scores']:
        return {'error': 'No metrics recorded yet'}
    
    import numpy as np
    
    recall_scores = np.array(training_metrics['recall_scores'])
    em_scores = np.array(training_metrics['em_scores']) 
    combined_scores = np.array(training_metrics['combined_scores'])
    
    summary = {
        'total_samples': len(recall_scores),
        'latest_step': training_metrics['step_count'],
        'recall': {
            'mean': float(np.mean(recall_scores)),
            'std': float(np.std(recall_scores)),
            'latest': float(recall_scores[-1]) if len(recall_scores) > 0 else 0,
            'last_10_mean': float(np.mean(recall_scores[-10:])) if len(recall_scores) >= 10 else float(np.mean(recall_scores))
        },
        'em': {
            'mean': float(np.mean(em_scores)),
            'std': float(np.std(em_scores)),
            'latest': float(em_scores[-1]) if len(em_scores) > 0 else 0,
            'last_10_mean': float(np.mean(em_scores[-10:])) if len(em_scores) >= 10 else float(np.mean(em_scores))
        },
        'combined': {
            'mean': float(np.mean(combined_scores)),
            'std': float(np.std(combined_scores)),
            'latest': float(combined_scores[-1]) if len(combined_scores) > 0 else 0,
            'last_10_mean': float(np.mean(combined_scores[-10:])) if len(combined_scores) >= 10 else float(np.mean(combined_scores))
        }
    }
    
    return summary


def compute_score_em_with_logging(solution_str, ground_truth, method='strict', format_score=0., score=1., 
                                 log_recall=False, recall_log_path='./recall_result/nqhotpot_after_train.json',
                                 log_training_metrics_flag=True, training_metrics_path='./training_metrics.json',
                                 step=None):
    """Enhanced version of compute_score_em that automatically logs training metrics.
    
    This function computes the score and automatically logs recall and EM metrics for training monitoring.
    """
    # Get detailed metrics
    detailed_results = compute_score_em(
        solution_str, ground_truth, method, format_score, score, 
        log_recall, recall_log_path, return_details=True
    )
    
    # Log training metrics if enabled
    if log_training_metrics_flag:
        log_training_metrics(
            recall_score=detailed_results['recall_score'],
            em_score=detailed_results['em_score'], 
            combined_score=detailed_results['combined_score'],
            step=step,
            log_path=training_metrics_path
        )
    
    # Return just the combined score for compatibility with existing code
    return detailed_results['combined_score']
