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
