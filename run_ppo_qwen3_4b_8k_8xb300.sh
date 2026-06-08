#!/usr/bin/env bash
# Qwen3-4B privileged-critic PPO with 8K responses on 8x B300.
# Scans GAE lambda from high to low; completed runs are merged to Hugging Face
# format and uploaded.

set -euo pipefail

cd "$(dirname "$0")"

# Replace these placeholders, or export the variables before running.
export HF_TOKEN="${HF_TOKEN:-hf_REPLACE_WITH_YOUR_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:-REPLACE_WITH_YOUR_WANDB_API_KEY}"
export HF_REPO_ID="${HF_REPO_ID:-YOUR_HF_USERNAME/qwen3-4b-pvc-ppo-8k-8xb300}"

if [[ "$HF_TOKEN" == "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
    echo "Set HF_TOKEN before running." >&2
    exit 1
fi
if [[ "$WANDB_API_KEY" == "REPLACE_WITH_YOUR_WANDB_API_KEY" ]]; then
    echo "Set WANDB_API_KEY before running." >&2
    exit 1
fi
if [[ "$HF_REPO_ID" == YOUR_HF_USERNAME/* ]]; then
    echo "Set HF_REPO_ID to the destination Hugging Face model repository." >&2
    exit 1
fi

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
export CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}

export NNODES=${NNODES:-1}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

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

export PRIVILEGED_CRITIC_ENABLE=True
export PRIVILEGED_CRITIC_KEY=${PRIVILEGED_CRITIC_KEY:-reference_trace}
export PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH=${PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH:-2048}

GAE_LAMBDAS=(${GAE_LAMBDAS:-0.999 0.995 0.990})

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
# Any positive save frequency makes verl save the final training step.
export SAVE_FREQ=${SAVE_FREQ:-$TOTAL_TRAINING_STEPS}
export TEST_FREQ=${TEST_FREQ:-10}
export VAL_BEFORE_TRAIN=False
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

export PROJECT_NAME=${PROJECT_NAME:-verl_pvc_ppo_qwen3_4b_b300}
HF_PRIVATE=${HF_PRIVATE:-true}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M)}
SCAN_STATUS=()

run_one_lambda() {
    local gae_lambda="$1"
    local lambda_tag="${gae_lambda//./p}"
    local experiment_name="${EXPERIMENT_NAME_PREFIX:-qwen3_4b_pvc_ppo_8k_n8_bs256}_lam${lambda_tag}_8xb300_${RUN_STAMP}"
    local checkpoint_root="${CHECKPOINT_ROOT_BASE:-$PWD/checkpoints/$PROJECT_NAME}/$experiment_name"
    local hf_target_dir="${HF_TARGET_DIR_BASE:-$checkpoint_root/huggingface_actor}"
    local hf_repo_id="${HF_REPO_ID_PREFIX:-$HF_REPO_ID}-lam${lambda_tag}"
    local latest_step_file="$checkpoint_root/latest_checkpointed_iteration.txt"
    local latest_step
    local actor_checkpoint

    echo "===== Starting GAE_LAMBDA=${gae_lambda} ====="
    export GAE_LAMBDA="$gae_lambda"
    export EXPERIMENT_NAME="$experiment_name"

    if ! ./run_ppo_7b.sh \
        trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
        trainer.default_local_dir="${checkpoint_root}" \
        actor_rollout_ref.model.enable_activation_offload=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        critic.model.enable_activation_offload=False \
        critic.model.enable_gradient_checkpointing=False \
        "${@:2}"; then
        echo "Training failed for GAE_LAMBDA=${gae_lambda}; continuing." >&2
        SCAN_STATUS+=("lambda=${gae_lambda}: training_failed")
        return 0
    fi

    if [[ ! -f "$latest_step_file" ]]; then
        echo "Final checkpoint marker not found for GAE_LAMBDA=${gae_lambda}: $latest_step_file" >&2
        SCAN_STATUS+=("lambda=${gae_lambda}: missing_checkpoint_marker")
        return 0
    fi

    latest_step=$(<"$latest_step_file")
    actor_checkpoint="$checkpoint_root/global_step_${latest_step}/actor"
    if [[ ! -d "$actor_checkpoint" ]]; then
        echo "Final actor checkpoint not found for GAE_LAMBDA=${gae_lambda}: $actor_checkpoint" >&2
        SCAN_STATUS+=("lambda=${gae_lambda}: missing_actor_checkpoint")
        return 0
    fi

    local merge_args=(
        merge
        --backend fsdp
        --local_dir "$actor_checkpoint"
        --target_dir "$hf_target_dir"
        --hf_upload_path "$hf_repo_id"
    )
    if [[ "$HF_PRIVATE" == "true" || "$HF_PRIVATE" == "True" || "$HF_PRIVATE" == "1" ]]; then
        merge_args+=(--private)
    fi

    if ! python3 scripts/legacy_model_merger.py "${merge_args[@]}"; then
        echo "Merge/upload failed for GAE_LAMBDA=${gae_lambda}; continuing." >&2
        SCAN_STATUS+=("lambda=${gae_lambda}: train_ok_upload_failed")
        return 0
    fi

    SCAN_STATUS+=("lambda=${gae_lambda}: ok")
}

for gae_lambda in "${GAE_LAMBDAS[@]}"; do
    run_one_lambda "$gae_lambda" "$@"
done

echo "===== Lambda scan summary ====="
printf '%s\n' "${SCAN_STATUS[@]}"
