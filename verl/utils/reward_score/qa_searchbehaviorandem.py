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
from collections import Counter
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


def normalize_text(text):
    """Normalize text for comparison purposes.
    
    Args:
        text (str): Input text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 移除多余空格并去掉首尾空格
    text = ' '.join(text.split())
    
    return text


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
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_original_question(solution_string):
    """Extract original question from solution string.
    
    Args:
        solution_string (str): The solution text containing the question
        
    Returns:
        str: Extracted original question or empty string if not found
    """
    try:
        # 查找 Question: 后面的内容
        question_match = re.search(r'Question:\s*(.+?)(?=\n|<|$)', solution_string, re.DOTALL)
        if question_match:
            return question_match.group(1).strip()
    except:
        pass
    return ""


def extract_search_queries(solution_str):
    """Extract search queries from solution string.
    
    Args:
        solution_str (str): The solution text containing search tags
        
    Returns:
        list: List of extracted search queries
    """
    search_queries = re.findall(r'<search>\s*(.+?)\s*</search>', solution_str)
    return search_queries


def is_valid_search_query(query: str) -> bool:
    """
    判断search query是否有效
    简单的启发式规则：
    - 不能为空或只有空格
    - 不能只是"and"
    - 不能只是标点符号
    - 长度应该合理（至少2个字符）
    """
    if not query or not query.strip():
        return False
    
    cleaned = query.strip().lower()
    
    # 只是"and"
    if cleaned == "and":
        return False
    
    # 只是标点符号
    if re.match(r'^[^\w\s]+$', cleaned):
        return False
    
    # 太短
    if len(cleaned) < 2:
        return False
    
    # 其他可能的无效模式
    invalid_patterns = [
        r'^\s*$',  # 只有空格
        r'^\.+$',  # 只有点
        r'^,+$',   # 只有逗号
        r'^\?+$',  # 只有问号
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, cleaned):
            return False
    
    return True


def check_query_rewriting(original_question, search_queries):
    """Check if query rewriting was performed.
    
    Args:
        original_question (str): The original question text
        search_queries (list): List of search queries to check
        
    Returns:
        tuple: (has_no_rewriting, no_rewriting_queries)
    """
    if not original_question or not search_queries:
        return False, []
    
    # 标准化原始问题
    normalized_original = normalize_text(original_question)
    
    # 检查每个查询是否与原始问题相同
    no_rewriting_queries = []
    for query in search_queries:
        if query.strip().lower() == 'query':  # 跳过无效的'query'
            continue
        
        normalized_query = normalize_text(query)
        if normalized_query == normalized_original:
            no_rewriting_queries.append(query)
    
    has_no_rewriting = len(no_rewriting_queries) > 0
    return has_no_rewriting, no_rewriting_queries


def analyze_search_behavior(solution_str, ground_truth):
    """Analyze search behavior and calculate penalty scores.
    
    Args:
        solution_str (str): The solution text to analyze
        ground_truth (dict): Ground truth containing target answers
        
    Returns:
        tuple: (penalty_score, behavior_flags)
    """
    
    # 提取搜索查询
    search_queries = extract_search_queries(solution_str)
    
    # 提取原始问题
    original_question = extract_original_question(solution_str)
    
    # 检查recall状态
    information_blocks = extract_information_blocks(solution_str)
    normalized_info = normalize_answer(" ".join(information_blocks))
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]
    recalled = any(t in normalized_info for t in normalized_targets)
    
    penalty = 0.0
    behavior_flags = {
        'has_invalid_search': False,
        'has_duplicate_queries': False,
        'no_search_and_no_recall': False,
        'no_rewriting_and_no_recall': False
    }
    
    # 1. 检查无效搜索 (invalid search) - 一次减0.2
    if search_queries:
        for query in search_queries:
            if query.strip().lower() != 'query' and not is_valid_search_query(query):
                behavior_flags['has_invalid_search'] = True
                penalty += 0.2
                break  # 只减一次
    
    # 2. 检查重复查询 (duplicate queries) - 有重复就减0.2
    if search_queries:
        actual_queries = [q.strip() for q in search_queries if q.strip().lower() != 'query' and q.strip()]
        if actual_queries:
            query_counts = Counter(actual_queries)
            if any(count > 1 for count in query_counts.values()):
                behavior_flags['has_duplicate_queries'] = True
                penalty += 0.2
    
    # 3. 检查没有搜索且没有recall - 减0.2
    has_no_search = (len(search_queries) == 0 or 
                    (len(search_queries) == 1 and search_queries[0].strip().lower() == 'query') or
                    all(q.strip().lower() == 'query' for q in search_queries))
    
    if has_no_search and not recalled:
        behavior_flags['no_search_and_no_recall'] = True
        penalty += 0.2
    
    # 4. 检查没有查询重写且没有recall - 减0.2
    if original_question and search_queries and not recalled:
        has_no_rewriting, _ = check_query_rewriting(original_question, search_queries)
        if has_no_rewriting:
            behavior_flags['no_rewriting_and_no_recall'] = True
            penalty += 0.2
    
    return penalty, behavior_flags


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
                    log_recall=False, recall_log_path='./recall_result/qa_searchbehavior_results.json', 
                    return_details=False):
    """The scoring function based on recall from information blocks with search behavior penalties.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: not used anymore, retained for compatibility
        format_score: score when recall fails
        score: score when recall succeeds
        log_recall: whether to log detailed results
        recall_log_path: path to save recall logs
        return_details: if True, return detailed metrics dict; if False, return single score
    """
    # Extract information blocks from solution string
    information_blocks = extract_information_blocks(solution_str)
    normalized_info = normalize_answer(" ".join(information_blocks))
    targets = [ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target']
    normalized_targets = [normalize_answer(t) for t in targets]

    # Extract documents and compute retrieval metrics
    documents = extract_documents_from_information(information_blocks)
    retrieval_accuracy = compute_retrieval_accuracy(documents, ground_truth)
    mmr = compute_mmr(documents, ground_truth)

    # Check recall
    recalled = any(t in normalized_info for t in normalized_targets)
    
    # Original score calculation (unchanged) - qa_search only cares about recall
    base_score = score if recalled else format_score
    
    # Analyze search behavior and apply penalties
    penalty, behavior_flags = analyze_search_behavior(solution_str, ground_truth)
    
    # Apply penalty to the score (ensure score doesn't go below format_score)
    combined_score = max(base_score - penalty, format_score)

    # Also compute EM score for detailed metrics
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        em_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            em_score = score
        else:
            em_score = format_score

    final_score = 0.5*combined_score + 0.5*em_score  # 混合得分，recall和EM各占50%

    if random.randint(1, 64) == 1:
        print(f"--------------------------------")
        print(f"[QA_SEARCHBEHAVIOR] Golden answers: {targets}")
        print(f"[QA_SEARCHBEHAVIOR] Normalized info: {normalized_info}")
        print(f"[QA_SEARCHBEHAVIOR] Recall success: {recalled}")
        print(f"[QA_SEARCHBEHAVIOR] Base score: {base_score}")
        print(f"[QA_SEARCHBEHAVIOR] Behavior penalty: {penalty}")
        print(f"[QA_SEARCHBEHAVIOR] Final score: {final_score}")
        print(f"[QA_SEARCHBEHAVIOR] Behavior flags: {behavior_flags}")
        print(f"[QA_SEARCHBEHAVIOR] Search queries: {extract_search_queries(solution_str)}")
        print(f"[QA_SEARCHBEHAVIOR] Documents found: {len(documents)}")
        print(f"[QA_SEARCHBEHAVIOR] Retrieval accuracy: {retrieval_accuracy:.3f}")
        print(f"[QA_SEARCHBEHAVIOR] MMR: {mmr:.3f}")

    # Log recall result
    if log_recall:
        single_record = {
            "target": make_serializable_target(ground_truth['target']),
            "information": information_blocks,
            "documents": documents,
            "recalled": recalled,
            "base_score": base_score,
            "behavior_penalty": penalty,
            "final_score": final_score,
            "behavior_flags": behavior_flags,
            "search_queries": extract_search_queries(solution_str),
            "original_question": extract_original_question(solution_str),
            "retrieval_accuracy": retrieval_accuracy,
            "mmr": mmr,
            "num_documents": len(documents)
        }
        
        import os
        os.makedirs(os.path.dirname(recall_log_path) if os.path.dirname(recall_log_path) else '.', exist_ok=True)
        with open(recall_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(single_record, ensure_ascii=False) + '\n')

    if return_details:
        return {
            'recall_score': combined_score,  # In qa_searchbehavior, recall with penalty is the main metric
            'em_score': em_score,  # But we also compute EM for tracking
            'combined_score': final_score,
            'base_score': base_score,
            'behavior_penalty': penalty,
            'behavior_flags': behavior_flags,
            'retrieval_accuracy': retrieval_accuracy,
            'mmr': mmr,
            'num_documents': len(documents),
            'search_queries': extract_search_queries(solution_str),
            'original_question': extract_original_question(solution_str)
        }
    else:
        return final_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1., return_details=False):
    """The scoring function for substring exact match (EM) with search behavior penalties.

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
        print(f"[QA_SEARCHBEHAVIOR_SUBEM] Golden answers: {ground_truth['target']}")
        print(f"[QA_SEARCHBEHAVIOR_SUBEM] Extracted answer: {answer}")

    # Compute recall score from <information> blocks
    information_blocks = extract_information_blocks(solution_str)
    normalized_info = normalize_answer(" ".join(information_blocks))
    recalled = any(
        normalize_answer(ans) in normalized_info
        for ans in ([ground_truth['target']] if isinstance(ground_truth['target'], str) else ground_truth['target'])
    )
    recall_score = score if recalled else format_score

    # Original SubEM score calculation
    if answer is None:
        base_subem_score = 0
    else:
        if subem_check(answer, ground_truth['target']):
            base_subem_score = score
        else:
            base_subem_score = format_score
    
    # Apply search behavior penalties
    penalty, behavior_flags = analyze_search_behavior(solution_str, ground_truth)
    combined_score = max(base_subem_score - penalty, format_score)

    if do_print:
        print(f"[QA_SEARCHBEHAVIOR_SUBEM] Base SubEM score: {base_subem_score}")
        print(f"[QA_SEARCHBEHAVIOR_SUBEM] Behavior penalty: {penalty}")
        print(f"[QA_SEARCHBEHAVIOR_SUBEM] Final score: {combined_score}")
        print(f"[QA_SEARCHBEHAVIOR_SUBEM] Behavior flags: {behavior_flags}")

    if return_details:
        return {
            'recall_score': recall_score,
            'em_score': combined_score,  # SubEM with penalty is the EM score here
            'combined_score': combined_score,
            'base_score': base_subem_score,
            'behavior_penalty': penalty,
            'behavior_flags': behavior_flags
        }
    else:
        return combined_score
