#!/usr/bin/env bash
# PVC/PPO for Qwen2.5-7B-Instruct on OpenR1-Math-220k, tuned for 2x H200 140G.
# This wrapper keeps the normal PPO script as the single source of truth and
# only changes defaults needed for the privileged reference-solution critic.

set -euo pipefail

cd "$(dirname "$0")"

export PRIVILEGED_CRITIC_ENABLE=${PRIVILEGED_CRITIC_ENABLE:-True}
export PRIVILEGED_CRITIC_KEY=${PRIVILEGED_CRITIC_KEY:-reference_trace}
export PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH=${PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH:-2048}

export PROJECT_NAME=${PROJECT_NAME:-verl_pvc_ppo_openr1_math}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_pvc_ppo_2xh200_${MAX_RESPONSE_LENGTH:-4096}_$(date +%Y%m%d_%H%M)}

exec ./run_ppo_7b.sh "$@"
