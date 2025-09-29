data_name=nq_hotpotqa_train

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=data/${data_name} # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

WAND_PROJECT="Search-R1"

export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-3b-em-structureformat
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-3b-it-em-structureformat
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-7b-em-structureformat
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-7b-it-em-structureformat
# export BASE_MODEL='Qwen/Qwen2.5-14B'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-14b-em-structureformat
# export BASE_MODEL='Qwen/Qwen2.5-14B-Instruct'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-qwen2.5-14b-it-em-structureformat

# export BASE_MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-deepseekr1-7b-em-structureformat
# export BASE_MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
# export EXPERIMENT_NAME=${data_name}-search-r1-grpo-deepseekr1-14b-em-structureformat

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/home/peterjin/verl_checkpoints/$EXPERIMENT_NAME \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee /home/peterjin/rl_logs/$EXPERIMENT_NAME.log
