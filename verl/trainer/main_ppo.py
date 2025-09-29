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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em,qa_search,qa_searchandem,qa_accandem,qa_searchbehavior,qa_searchbehaviorandem
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_search.compute_score_em
    elif data_source in ['accandem']:
        return qa_accandem.compute_score_em  # 使用检索准确率+EM的组合评分
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def get_detailed_metrics(self, data: DataProto):
        """Enhanced version that returns both reward tensor and detailed metrics"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores'], None

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 用于收集本批次的指标
        batch_recall_scores = []
        batch_em_scores = []
        batch_combined_scores = []
        batch_retrieval_accuracy_scores = []
        batch_mmr_scores = []
        batch_num_documents = []
        batch_behavior_penalties = []
        batch_behavior_flags = {
            'has_invalid_search': [],
            'has_duplicate_queries': [],
            'no_search_and_no_recall': [],
            'no_rewriting_and_no_recall': []
        }

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # 获取详细指标
            detailed_metrics = compute_score_fn(
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                format_score=self.format_score,
                return_details=True
            )
            
            # 提取各项指标
            score = detailed_metrics['combined_score']
            recall_score = detailed_metrics['recall_score']
            em_score = detailed_metrics['em_score']
            
            # 提取检索指标（如果存在）
            retrieval_accuracy = detailed_metrics.get('retrieval_accuracy', 0.0)
            mmr = detailed_metrics.get('mmr', 0.0)
            num_documents = detailed_metrics.get('num_documents', 0)
            
            # 提取行为惩罚指标（如果存在）
            behavior_penalty = detailed_metrics.get('behavior_penalty', 0.0)
            behavior_flags = detailed_metrics.get('behavior_flags', {})
            
            # 收集指标
            batch_recall_scores.append(recall_score)
            batch_em_scores.append(em_score)
            batch_combined_scores.append(score)
            batch_retrieval_accuracy_scores.append(retrieval_accuracy)
            batch_mmr_scores.append(mmr)
            batch_num_documents.append(num_documents)
            batch_behavior_penalties.append(behavior_penalty)
            
            # 收集行为标志
            for flag_name, flag_value in behavior_flags.items():
                if flag_name in batch_behavior_flags:
                    batch_behavior_flags[flag_name].append(flag_value)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
                print(f"[METRICS] Recall: {recall_score:.3f}, EM: {em_score:.3f}, Combined: {score:.3f}")
                print(f"[RETRIEVAL] Acc: {retrieval_accuracy:.3f}, MMR: {mmr:.3f}, Docs: {num_documents}")
                if behavior_penalty > 0:
                    print(f"[BEHAVIOR] Penalty: {behavior_penalty:.3f}, Flags: {behavior_flags}")

        # 返回详细指标给trainer记录到wandb
        detailed_metrics_summary = {
            'recall_mean': np.mean(batch_recall_scores) if batch_recall_scores else 0.0,
            'em_mean': np.mean(batch_em_scores) if batch_em_scores else 0.0,
            'combined_mean': np.mean(batch_combined_scores) if batch_combined_scores else 0.0,
            'retrieval_accuracy_mean': np.mean(batch_retrieval_accuracy_scores) if batch_retrieval_accuracy_scores else 0.0,
            'mmr_mean': np.mean(batch_mmr_scores) if batch_mmr_scores else 0.0,
            'num_documents_mean': np.mean(batch_num_documents) if batch_num_documents else 0.0,
            'behavior_penalty_mean': np.mean(batch_behavior_penalties) if batch_behavior_penalties else 0.0,
            'invalid_search_rate': np.mean(batch_behavior_flags['has_invalid_search']) if batch_behavior_flags['has_invalid_search'] else 0.0,
            'duplicate_queries_rate': np.mean(batch_behavior_flags['has_duplicate_queries']) if batch_behavior_flags['has_duplicate_queries'] else 0.0,
            'no_search_and_no_recall_rate': np.mean(batch_behavior_flags['no_search_and_no_recall']) if batch_behavior_flags['no_search_and_no_recall'] else 0.0,
            'no_rewriting_and_no_recall_rate': np.mean(batch_behavior_flags['no_rewriting_and_no_recall']) if batch_behavior_flags['no_rewriting_and_no_recall'] else 0.0,
            'batch_size': len(batch_recall_scores)
        }

        return reward_tensor, detailed_metrics_summary

        return reward_tensor, detailed_metrics_summary

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # 使用标准模式（不要详细指标）
            score = compute_score_fn(
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                format_score=self.format_score
            )

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra
import os

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        try:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        except:
            ray.init(
                        _temp_dir=os.environ.get("RAY_TEMP_DIR", "/tmp/ray"),
                            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}
                            )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 创建reward函数
    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=1
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
