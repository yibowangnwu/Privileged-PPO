# Privileged Critic PPO Handoff

## One-line Summary

This project tests whether PPO's critic can recover the credit-assignment benefits often attributed to self-distillation by giving the critic the same privileged information that a self-distillation teacher sees, while keeping the actor prompt and deployment behavior unchanged.

## Motivation

Knowledge distillation is usually framed as a teacher-student setup: the student learns not only from hard labels or scalar outcomes, but from richer teacher outputs. Hinton et al.'s distillation work is the classic reference, and later self-distillation work such as Born-Again Networks showed that the teacher and student can even share the same architecture and still improve the next model generation.

For RLVR-style math training, the interesting question is not just "does the teacher produce better logits?" but "what role is the teacher functionally playing?" My current hypothesis is:

> In self-distillation for RL/math, the teacher is often acting like a value/credit-assignment function. It has access to extra information and uses that to provide a denser training signal than a scalar reward.

That raises an apparent mismatch with PPO actor-critic training. PPO already has a component whose job is credit assignment: the critic. If credit assignment is the issue, why does a self-distillation teacher sometimes help while a standard PPO critic does not?

The working guess is that the teacher is not inherently stronger because it is a "teacher." It is stronger because it has privileged information. Standard PPO critics are trained on the same prompt/response trajectory the actor sees, while a self-distillation teacher may condition on richer feedback, reference reasoning, verifier traces, corrected trajectories, or other training-only information. If the actor and critic have the same base architecture, there is no principled reason the critic should be weaker than the teacher once it receives the same information.

## Core Idea

Give privileged, training-only information to the critic, not to the actor.

Actor input remains deployment-compatible:

```text
[problem][student_response]
```

Critic input becomes:

```text
[critic instruction + problem + reference solution][same student_response]
```

The critic therefore computes token values for the same generated response tokens, but with access to the reference solution before those response tokens. This matters for a causal decoder-only LM: value states at response positions can attend leftward to the reference, so the critic can judge which parts of the response are useful or off-track with much better context.

The actor never sees the reference solution. The reference is only used to improve value estimates, GAE, and PPO credit assignment.

## Hypothesis

If self-distillation helps because the teacher has privileged information, then a privileged PPO critic should capture at least part of the same benefit:

- faster critic convergence
- lower-variance or better-shaped advantages
- better token-level credit assignment
- possibly better final RL performance at the same compute budget

Early experiments already suggest that adding the privileged reference information makes the critic converge noticeably faster. The next question is whether that translates into policy improvement rather than only cleaner critic loss curves.

## Current Implementation

The repo has a PVC path documented in `privileged_critic.md`.

Key behavior:

- `run_ppo_7b.sh` supports `PRIVILEGED_CRITIC_ENABLE=True`.
- Training data stores the reference solution in the top-level parquet column `reference_trace`.
- The actor rollout/generation path does not include `reference_trace`.
- During value computation and critic update, `ray_trainer.py` rebuilds a critic-only batch with:
  - new critic prompts containing the problem plus capped reference trace
  - the same sampled actor responses
  - the same response mask and batch ordering
- Actor logprobs and actor updates still use the original non-privileged actor batch.

Important alignment detail: the privileged batch must update `prompts` to the critic prefix while keeping `responses` unchanged. verl slices values back to response positions using `prompts`, `responses`, and `attention_mask`; if `prompts` stays as the actor prompt, value slicing is wrong.

## Planned Experiment

The current experiment script is:

```bash
./run_ppo_qwen3_4b_8k_8xb300.sh
```

It is a hypothetical/ready-to-run 8x B300 script for:

- model: `Qwen/Qwen3-4B-Instruct-2507`
- privileged critic enabled
- max prompt length: `2048`
- max response length: `8192`
- rollout responses per prompt: `8`
- train batch size: `256`
- test frequency: every `10` training steps
- final actor checkpoint save enabled

The script scans GAE lambda from high to low:

```bash
GAE_LAMBDAS="0.999 0.995 0.990"
```

The scan order is intentional: try `0.999` first, then `0.995`, then `0.990`. If a run fails or the job dies partway through, the script records that failure and continues to the next lambda when possible. The goal is not to guarantee all settings finish, but to get as much signal as possible from available machine time.

Each completed run:

1. Saves the final actor checkpoint.
2. Finds the latest checkpoint via `latest_checkpointed_iteration.txt`.
3. Merges the FSDP actor checkpoint into Hugging Face format with `scripts/legacy_model_merger.py`.
4. Uploads the merged actor to Hugging Face.

The script expects these to be set before a real run:

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
export HF_REPO_ID=your-user-or-org/qwen3-4b-pvc-ppo-8k-8xb300
```

By default, per-lambda uploads get suffixes like:

```text
...-lam0p999
...-lam0p995
...-lam0p990
```

## What To Watch

Metrics to compare against normal PPO or non-privileged critic baselines:

- critic loss convergence speed
- value explained variance, if logged
- advantage scale and variance
- train reward vs validation reward
- pass rates on AMC/AIME eval sets
- whether the faster critic convergence actually improves actor updates

Potential failure modes:

- Privileged critic improves value loss but does not improve policy.
- Reference traces are too long and increase memory pressure.
- The critic overfits reference formatting and gives brittle value estimates.
- Actor/critic batch alignment bugs silently corrupt values.
- Upload may fail even when training succeeds; the script records this separately.

## Why This Is Interesting

This experiment reframes self-distillation as privileged value learning. If it works, the lesson is not necessarily that PPO needs a teacher-like actor target. It may be that PPO's critic was under-informed. Giving the critic training-only reference information could be a cleaner way to get dense credit assignment while preserving the standard actor policy interface.

## References

- Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network": https://arxiv.org/abs/1503.02531
- Furlanello et al., "Born Again Neural Networks": https://arxiv.org/abs/1805.04770
- Self-Distillation Policy Optimization discussion page: https://huggingface.co/papers/2601.20802
