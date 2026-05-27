# Privileged Critic / PVC Notes

This repo has an optional privileged critic path for PPO math RL. The goal is to let the critic see a training-only reference solution while keeping the actor as normal deployment-compatible PPO.

## Short Version

Normal PPO actor input stays unchanged:

```text
[prompt][student_response]
```

When enabled, critic input is rebuilt as:

```text
[critic instruction + problem + reference solution][same student_response]
```

The student response tokens, batch order, and `response_mask` are unchanged. This is what keeps critic values aligned with actor logprobs and PPO advantages.

## How To Run

Use the PVC wrapper:

```bash
cd /root/verl
./run_pvc_ppo_7b.sh
```

This wrapper calls `run_ppo_7b.sh` with these defaults:

```bash
PRIVILEGED_CRITIC_ENABLE=True
PRIVILEGED_CRITIC_KEY=reference_trace
PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH=2048
PROJECT_NAME=verl_pvc_ppo_openr1_math
```

Normal PPO is still the default for `run_ppo_7b.sh`:

```bash
PRIVILEGED_CRITIC_ENABLE=False
```

## Files Changed

### `run_ppo_7b.sh`

Adds user-facing config:

```bash
PRIVILEGED_CRITIC_ENABLE=${PRIVILEGED_CRITIC_ENABLE:-False}
PRIVILEGED_CRITIC_KEY=${PRIVILEGED_CRITIC_KEY:-reference_trace}
PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH=${PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH:-2048}
```

Passes these through to Hydra:

```bash
algorithm.privileged_critic.enable=${PRIVILEGED_CRITIC_ENABLE}
algorithm.privileged_critic.key=${PRIVILEGED_CRITIC_KEY}
algorithm.privileged_critic.max_reference_length=${PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH}
```

During OpenR1 data preparation, stores dataset `solution` as a top-level parquet column:

```python
"reference_trace": str(reference_trace).strip()
```

The reward target remains `reward_model.ground_truth` from `answer/final_answer`. Do not use `reference_trace` as the reward target.

The script also checks stale local train parquet files. If PVC is enabled and the existing parquet lacks `reference_trace`, it forces data regeneration.

### `run_pvc_ppo_7b.sh`

Thin wrapper around `run_ppo_7b.sh`. It only changes PVC-related defaults and experiment naming, then executes the normal script. This avoids maintaining two diverging PPO scripts.

### `verl/trainer/config/algorithm.py`

Adds:

```python
@dataclass
class PrivilegedCriticConfig(BaseConfig):
    enable: bool = False
    key: str = "reference_trace"
    max_reference_length: int = 2048
```

And attaches it to `AlgoConfig`:

```python
privileged_critic: PrivilegedCriticConfig = field(default_factory=PrivilegedCriticConfig)
```

### `verl/trainer/config/ppo_trainer.yaml`

Adds default YAML config:

```yaml
algorithm:
  privileged_critic:
    enable: False
    key: reference_trace
    max_reference_length: 2048
```

### `verl/trainer/ppo/ray_trainer.py`

Adds helper methods:

```python
_use_privileged_critic()
_tokenize_privileged_critic_prefix(problem, reference_trace)
_make_privileged_critic_batch(batch)
```

The core helper builds a critic-only batch:

```python
critic_batch.batch["prompts"] = critic_prompts
critic_batch.batch["input_ids"] = torch.cat([critic_prompts, responses], dim=1)
critic_batch.batch["attention_mask"] = torch.cat([prompt_mask, response_mask], dim=1)
critic_batch.batch["position_ids"] = torch.cat([prompt_position_ids, response_position_ids], dim=1)
```

It does not change:

```python
responses
response_mask
batch order
actor input_ids
actor logprobs
```

Value computation now uses the privileged batch only when enabled:

```python
values = self._compute_values(self._make_privileged_critic_batch(batch))
batch = batch.union(values)
```

Critic update also uses the privileged batch only when enabled:

```python
critic_output = self._update_critic(self._make_privileged_critic_batch(batch))
```

Actor update still receives the original non-privileged batch.

## Important Alignment Detail

`verl.workers.utils.padding.no_padding_2_padding()` slices model outputs back to response positions using `prompts`, `responses`, and `attention_mask`.

That means the privileged critic batch must update `prompts` to the new critic prefix, while keeping `responses` as the original sampled response. If `prompts` is not updated, the returned value tensor will be sliced from the wrong positions.

The returned `values` tensor shape is still:

```text
[batch_size, max_response_length]
```

So it can be unioned back into the original actor batch and used by GAE/PPO normally.

## Important Causal-LM Detail

The reference solution must appear before the student response:

```text
[problem + reference solution][student response]
```

Do not format critic input as:

```text
[problem][student response][reference solution]
```

For a decoder-only LM, value states at response-token positions cannot attend to tokens on their right. If the reference is placed after the response, the critic effectively cannot use it for token values.

## Important Batch Plumbing Detail

`_get_gen_batch()` pops most non-tensor fields before rollout. Initially this removed `reference_trace`, causing a crash later in `_make_privileged_critic_batch()`:

```text
KeyError: non_tensor_batch has no key 'reference_trace'
```

Fix: when PVC is enabled, `_get_gen_batch()` keeps the configured privileged key:

```python
keys_to_keep = {"data_source", "reward_model", "extra_info", "uid"}
if self._use_privileged_critic():
    keys_to_keep.add(self.config.algorithm.privileged_critic.get("key", "reference_trace"))
```

This does not leak the reference into the actor prompt. It only keeps the non-tensor field available for later critic batch construction.

## Known Limitations / Next Steps

This is PVC-V only: privileged value baseline for standard GAE/PPO. It does not implement PVC-Delta token credit yet.

Current implementation decodes actor prompt ids back to text, then tokenizes the critic prompt. This is simple and clear for a first experiment, but not optimal. If it becomes a bottleneck, pre-tokenize critic prefixes during data preparation.

`PRIVILEGED_CRITIC_MAX_REFERENCE_LENGTH` caps only the reference-solution token count. The full critic prefix length is approximately problem tokens plus capped reference tokens plus instruction tokens. If OOM occurs, reduce this cap or increase critic token length limits.

Validation parquet rows include an empty `reference_trace` for schema consistency, but validation does not use the critic path.

## Quick Sanity Checks

Shell/Python checks used after implementation:

```bash
bash -n /root/verl/run_ppo_7b.sh
bash -n /root/verl/run_pvc_ppo_7b.sh
python3 -m py_compile /root/verl/verl/trainer/ppo/ray_trainer.py /root/verl/verl/trainer/config/algorithm.py
git -C /root/verl diff --check
```

To check the local train parquet has the required column:

```bash
python3 - <<'PY'
import pyarrow.parquet as pq
print(pq.ParquetFile('/root/data/openr1_math_220k_default/train.parquet').schema_arrow.names)
PY
```

Expected to include:

```text
reference_trace
```
