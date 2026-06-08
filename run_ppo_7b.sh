#!/usr/bin/env bash
# PPO for Qwen2.5-7B-Instruct on OpenR1-Math-220k, tuned for 2x H200 140G.
# Default is intentionally conservative for first-run stability at 4K responses.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}

cd "$(dirname "$0")"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}

DATASET_NAME=${DATASET_NAME:-open-r1/OpenR1-Math-220k}
DATASET_CONFIG=${DATASET_CONFIG:-default}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATA_DIR=${DATA_DIR:-$HOME/data/openr1_math_220k_default}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_DATA_DIR=${VAL_DATA_DIR:-$HOME/data/math_eval}
AMC23_FILE=${AMC23_FILE:-$VAL_DATA_DIR/amc23/test.parquet}
AIME24_FILE=${AIME24_FILE:-$VAL_DATA_DIR/aime24/test.parquet}
AIME25_FILE=${AIME25_FILE:-$VAL_DATA_DIR/aime25/test.parquet}
AIME26_FILE=${AIME26_FILE:-$VAL_DATA_DIR/aime26/test.parquet}
VAL_FILES=${VAL_FILES:-"['$AMC23_FILE', '$AIME24_FILE', '$AIME25_FILE', '$AIME26_FILE']"}
PREPARE_DATA=${PREPARE_DATA:-auto}
PREPARE_VAL_DATA=${PREPARE_VAL_DATA:-auto}
REWARD_FN_FILE=${REWARD_FN_FILE:-$PWD/custom_math_eval_reward.py}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.65}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-12288}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_N=${ROLLOUT_N:-1}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-True}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}

ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-5e-6}
KL_COEF=${KL_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
GAE_LAMBDA=${GAE_LAMBDA:-1.0}

PRIVILEGED_CRITIC_ENABLE=${PRIVILEGED_CRITIC_ENABLE:-False}
PRIVILEGED_CRITIC_KEY=${PRIVILEGED_CRITIC_KEY:-reference_trace}
PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH=${PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH:-2048}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
LOGGER=${LOGGER:-'["console","wandb"]'}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
VAL_N=${VAL_N:-8}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-1.0}

PROJECT_NAME=${PROJECT_NAME:-verl_ppo_openr1_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_ppo_2xh200_${MAX_RESPONSE_LENGTH}_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=${MAX_MODEL_LEN}
fi

if [[ "$PREPARE_DATA" == "auto" && ! -f "$TRAIN_FILE" ]]; then
    PREPARE_DATA=True
fi

if [[ "$PREPARE_VAL_DATA" == "auto" && ( ! -f "$AMC23_FILE" || ! -f "$AIME24_FILE" || ! -f "$AIME25_FILE" || ! -f "$AIME26_FILE" ) ]]; then
    PREPARE_VAL_DATA=True
fi

if [[ "$PRIVILEGED_CRITIC_ENABLE" == "True" || "$PRIVILEGED_CRITIC_ENABLE" == "true" || "$PRIVILEGED_CRITIC_ENABLE" == "1" ]]; then
    if [[ -f "$TRAIN_FILE" ]]; then
        if ! TRAIN_FILE="$TRAIN_FILE" PRIVILEGED_CRITIC_KEY="$PRIVILEGED_CRITIC_KEY" python3 - <<'PY'
import os
import pyarrow.parquet as pq

train_file = os.environ["TRAIN_FILE"]
key = os.environ["PRIVILEGED_CRITIC_KEY"]
schema = pq.ParquetFile(train_file).schema_arrow
raise SystemExit(0 if key in schema.names else 1)
PY
        then
            echo "Privileged critic is enabled but $TRAIN_FILE lacks column $PRIVILEGED_CRITIC_KEY; regenerating data."
            PREPARE_DATA=True
        fi
    fi
fi

cat > "$REWARD_FN_FILE" <<'PY'
import json
import os
import re
import time

from verl.utils.reward_score import math_dapo, math_verify

EVAL_SOURCES = {"amc23", "aime24", "aime25", "aime26"}
REWARD_DEBUG_FILE = os.environ.get("REWARD_DEBUG_FILE", "")
REWARD_DEBUG_MAX = int(os.environ.get("REWARD_DEBUG_MAX", "2000"))
_reward_debug_count = 0


def _extract_boxed(text):
    boxed = math_dapo.last_boxed_only_string(str(text))
    if boxed is None:
        return None
    try:
        return math_dapo.remove_boxed(boxed)
    except Exception:
        return None


def _extract_answer(text):
    text = str(text)
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed, "boxed"

    matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip(), "answer"

    matches = re.findall(r"####\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip(), "hash"

    return "[INVALID]", "invalid"


def _math_train_score(solution_str, ground_truth):
    pred, extractor = _extract_answer(solution_str)
    pred_norm = math_dapo.normalize_final_answer(pred)
    gt_norm = math_dapo.normalize_final_answer(str(ground_truth))
    correct = pred_norm == gt_norm
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred_norm,
        "extractor": extractor,
    }


def _write_debug(data_source, solution_str, ground_truth, result, extra_info=None, error=None):
    global _reward_debug_count
    if not REWARD_DEBUG_FILE or _reward_debug_count >= REWARD_DEBUG_MAX:
        return
    _reward_debug_count += 1
    record = {
        "time": time.time(),
        "pid": os.getpid(),
        "data_source": data_source,
        "score": result.get("score") if isinstance(result, dict) else result,
        "acc": result.get("acc") if isinstance(result, dict) else None,
        "pred": result.get("pred") if isinstance(result, dict) else None,
        "extractor": result.get("extractor") if isinstance(result, dict) else None,
        "verifier": result.get("verifier") if isinstance(result, dict) else None,
        "ground_truth": ground_truth,
        "solution_tail": str(solution_str)[-1200:],
        "extra_info": extra_info,
        "error": error,
    }
    try:
        os.makedirs(os.path.dirname(REWARD_DEBUG_FILE), exist_ok=True)
        with open(REWARD_DEBUG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ground_truth = str(ground_truth)
    if data_source in EVAL_SOURCES:
        try:
            score = math_verify.compute_score(solution_str, ground_truth, timeout=10.0)
            result = {"score": float(score), "acc": bool(score), "verifier": "math_verify"}
            _write_debug(data_source, solution_str, ground_truth, result, extra_info=extra_info)
            return result
        except Exception as exc:
            fallback = _math_train_score(solution_str=solution_str, ground_truth=ground_truth)
            fallback["verifier"] = f"train_parser_fallback:{type(exc).__name__}"
            _write_debug(data_source, solution_str, ground_truth, fallback, extra_info=extra_info, error=repr(exc))
            return fallback

    result = _math_train_score(solution_str=solution_str, ground_truth=ground_truth)
    _write_debug(data_source, solution_str, ground_truth, result, extra_info=extra_info)
    return result
PY

if [[ "$PREPARE_DATA" == "True" || "$PREPARE_DATA" == "true" || "$PREPARE_DATA" == "1" ]]; then
    export DATASET_NAME DATASET_CONFIG DATASET_SPLIT DATA_DIR
    python3 - <<'PY'
import os

from datasets import load_dataset

dataset_name = os.environ.get("DATASET_NAME", "open-r1/OpenR1-Math-220k")
dataset_config = os.environ.get("DATASET_CONFIG", "default")
dataset_split = os.environ.get("DATASET_SPLIT", "train")
data_dir = os.path.expanduser(os.environ.get("DATA_DIR", "~/data/openr1_math_220k_default"))

os.makedirs(data_dir, exist_ok=True)
train_path = os.path.join(data_dir, "train.parquet")
print(f"Loading {dataset_name} ({dataset_config}, split={dataset_split}) from Hugging Face...")
ds = load_dataset(dataset_name, dataset_config, split=dataset_split)

instruction = "Please reason step by step, and put your final answer within \\boxed{}."

def has_problem_and_answer(example):
    problem = example.get("problem") or example.get("question")
    answer = example.get("answer") or example.get("final_answer")
    return problem is not None and str(problem).strip() and answer is not None and str(answer).strip()

def process(example, idx):
    problem = example.get("problem") or example.get("question")
    answer = example.get("answer") or example.get("final_answer")
    reference_trace = example.get("solution") or example.get("reference_solution") or ""
    question = str(problem).strip()
    if "\\boxed{}" not in question:
        question = f"{question}\n\n{instruction}"
    return {
        "data_source": "math",
        "prompt": [{"role": "user", "content": question}],
        "reference_trace": str(reference_trace).strip(),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
        "extra_info": {
            "split": "train",
            "index": idx,
            "source": example.get("source"),
            "uuid": example.get("uuid"),
            "problem_type": example.get("problem_type"),
        },
    }

ds = ds.filter(has_problem_and_answer)
ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)
ds.to_parquet(train_path)
print(f"Wrote {len(ds)} rows from split={dataset_split} to {train_path}")
PY
fi

if [[ "$PREPARE_VAL_DATA" == "True" || "$PREPARE_VAL_DATA" == "true" || "$PREPARE_VAL_DATA" == "1" ]]; then
    export VAL_DATA_DIR
    python3 - <<'PY'
import os
import re

from datasets import load_dataset

val_data_dir = os.path.expanduser(os.environ.get("VAL_DATA_DIR", "~/data/math_eval"))
instruction = "Please reason step by step, and put your final answer within \\boxed{}."

benchmarks = {
    "amc23": "math-ai/amc23",
    "aime24": "math-ai/aime24",
    "aime25": "math-ai/aime25",
    "aime26": "math-ai/aime26",
}


def strip_boxed(text):
    text = str(text).strip()
    match = re.search(r"\\boxed\{([^{}]*)\}", text)
    return match.group(1).strip() if match else text


def process(example, idx, name):
    problem = example.get("problem") or example.get("question")
    answer = example.get("answer") or example.get("solution")
    reference_trace = ""
    if problem is None or answer is None:
        return None
    question = str(problem).strip()
    if "\\boxed{}" not in question:
        question = f"{question}\n\n{instruction}"
    return {
        "data_source": name,
        "prompt": [{"role": "user", "content": question}],
        "reference_trace": str(reference_trace).strip(),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": strip_boxed(answer)},
        "extra_info": {
            "split": "test",
            "index": idx,
            "id": None if example.get("id") is None else str(example.get("id")),
            "url": None if example.get("url") is None else str(example.get("url")),
        },
    }

for name, dataset_name in benchmarks.items():
    out_dir = os.path.join(val_data_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.parquet")
    print(f"Loading {dataset_name} test for {name}...")
    ds = load_dataset(dataset_name, split="test")
    ds = ds.map(lambda ex, idx: process(ex, idx, name), with_indices=True, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: ex["prompt"] is not None and ex["reward_model"]["ground_truth"] != "")
    ds.to_parquet(out_path)
    print(f"Wrote {len(ds)} rows to {out_path}")
PY
fi

DATA=(
    algorithm.adv_estimator=gae
    algorithm.lam=${GAE_LAMBDA}
    algorithm.use_kl_in_reward=True
    algorithm.kl_ctrl.type=fixed
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    algorithm.privileged_critic.enable=${PRIVILEGED_CRITIC_ENABLE}
    algorithm.privileged_critic.key=${PRIVILEGED_CRITIC_KEY}
    algorithm.privileged_critic.max_reference_length=${PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH}
    data.train_files="['$TRAIN_FILE']"
    data.val_files="$VAL_FILES"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.truncation='error'
)

REWARD=(
    reward.custom_reward_function.path="$REWARD_FN_FILE"
    reward.custom_reward_function.name=compute_score
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.ppo_epochs=2
    actor_rollout_ref.actor.shuffle=True
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}
)

CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.model.enable_activation_offload=True
    critic.model.trust_remote_code=True
    critic.optim.lr=${CRITIC_LR}
    critic.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    critic.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    critic.forward_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    critic.fsdp.param_offload=${CRITIC_PARAM_OFFLOAD}
    critic.fsdp.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD}
    critic.fsdp.reshard_after_forward=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger="$LOGGER"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${REWARD[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "$@"
