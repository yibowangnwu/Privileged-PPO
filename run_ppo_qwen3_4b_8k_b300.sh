#!/usr/bin/env bash
# Short throughput run for Qwen3-4B PPO on 2x B300.
# Each training step samples 256 prompts x 8 responses = 2048 trajectories.

set -euo pipefail

cd "$(dirname "$0")"

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
export CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}

export NNODES=${NNODES:-1}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}

export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_N=${ROLLOUT_N:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.80}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}

export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIMIZER_OFFLOAD=False
export CRITIC_PARAM_OFFLOAD=False
export CRITIC_OPTIMIZER_OFFLOAD=False
export REF_PARAM_OFFLOAD=False

export PRIVILEGED_CRITIC_ENABLE=False

# Keep the timing run focused on rollout and PPO updates.
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=False

export PROJECT_NAME=${PROJECT_NAME:-verl_ppo_qwen3_4b_b300_timing}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_ppo_8k_n8_bs256_2xb300_$(date +%Y%m%d_%H%M)}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

exec ./run_ppo_7b.sh \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    critic.model.enable_activation_offload=False \
    critic.model.enable_gradient_checkpointing=False \
    "$@"
