# Proposal: Privileged Prefix-Value Critics for Verifiable LLM Reasoning

**Working title:** *From Self-Teacher to Explicit Critic: Privileged Value Credit for LLM Reasoning RL*  
**Short name:** PVC — *Privileged Value Credit*  
**Status:** research proposal / implementation plan  
**Target system:** verl PPO / RLVR pipeline  
**Date:** 2026-05-26

---

## 0. One-sentence thesis

Current self-distillation methods often use a privileged self-teacher as an implicit token-level value or advantage proxy. This proposal makes that role explicit: train a privileged prefix-value critic that sees reference trajectories, answers, verifier traces, or other training-only information, while the actor remains deployment-compatible.

In autoregressive generation, action value is simply next-prefix value:

\[
Q(h_t, a_t, z) = V(h_t \circ a_t, z),
\]

so a privileged prefix-value critic can produce token-level credit by comparing pre-token and post-token values:

\[
\Delta V_t^{\text{priv}}
= V_\phi(h_t \circ a_t, z) - V_\phi(h_t, z).
\]

This replaces the self-teacher likelihood-ratio proxy with a verifier-aligned, explicitly trained value estimator.

---

## 1. Motivation

### 1.1 The problem: standard RLVR credit assignment is too coarse

In verifiable LLM reasoning tasks, the reward is often sparse and terminal:

\[
R(y) \in \{0,1\}.
\]

A rollout receives a scalar correctness signal only after completing a long chain of reasoning. This creates three bottlenecks:

1. **Long-horizon credit assignment.** Early tokens may decide success or failure, but the reward arrives only at the end.
2. **Sparse reward collapse.** On hard prompts, all sampled rollouts may fail, producing little or no positive gradient.
3. **Weak value estimation.** A normal critic sees only the prompt and current prefix:
   \[
   V_\phi(x, y_{<t}),
   \]
   and must predict future correctness from an extremely underdetermined state.

For example, the prefix `Let's solve step by step` contains little information about whether the final answer will be correct. A standard PPO critic is therefore often just a noisy variance-reduction baseline rather than a meaningful reasoning progress estimator.

### 1.2 What self-distillation is really doing

Recent self-distillation RLVR methods address the same bottleneck by introducing a self-teacher conditioned on privileged information:

\[
\pi_T(a_t \mid h_t, z),
\]

where \(z\) may be feedback, a reference answer, a successful rollout, or a verifier-side signal. Importantly, in many of these methods the teacher does **not** perform a new rollout. It re-scores the student's sampled trajectory.

The resulting signal often has the form:

\[
\Delta_t^{\text{teacher}}
= \log \pi_\theta(a_t \mid h_t, z)
- \log \pi_\theta(a_t \mid h_t).
\]

Although this is written as a likelihood difference, algorithmically it behaves like a token-level advantage estimate: it says whether a token becomes more or less plausible after the model sees privileged information.

This suggests a sharper interpretation:

> The self-teacher is an implicit privileged advantage estimator implemented through policy logits.

### 1.3 Why not let the critic do the critic's job?

The self-teacher is not a pure value function. It mixes several signals:

\[
\text{teacher likelihood}
= \text{value signal}
+ \text{language prior}
+ \text{teacher style}
+ \text{reference-path bias}
+ \text{prompt artifact}.
\]

A token may receive high self-teacher probability because it matches the teacher's style, not because it improves final verifier reward. Conversely, a token on an alternative valid reasoning path may receive low teacher probability simply because it diverges from the reference trajectory.

The proposal is therefore:

> Replace the implicit likelihood-ratio teacher with an explicit privileged value critic trained against verifier-aligned returns.

The critic uses the same privileged information channel as the self-teacher, but its training objective is value estimation rather than next-token imitation.

---

## 2. Core idea

Let:

- \(x\): prompt / problem
- \(y_{<t}\): generated reasoning prefix
- \(h_t = (x, y_{<t})\): autoregressive history
- \(a_t = y_t\): next token
- \(z\): privileged training-only information
- \(R\): verifier reward

Privileged information \(z\) may include:

- correct final answer
- reference reasoning trajectory
- human or stronger-model CoT
- verifier feedback
- execution trace
- proof trace
- search trajectory
- successful sibling rollout
- retrieved decomposition
- program unit-test trace

The actor remains unchanged:

\[
\pi_\theta(a_t \mid h_t).
\]

The critic is privileged:

\[
V_\phi(h_t, z).
\]

The actor never sees \(z\) at inference time. The critic sees \(z\) only during training.

This is an asymmetric actor-critic setup specialized for LLM reasoning. The goal is not to make the critic magically smarter than the actor; the goal is to make value estimation easier by conditioning it on privileged structure.

---

## 3. Key autoregressive observation: Q and V differ mostly by one token

In ordinary RL, one may distinguish state value and action value:

\[
V(s_t), \quad Q(s_t, a_t).
\]

In an autoregressive language model, the transition after choosing a token is deterministic:

\[
h_{t+1} = h_t \circ a_t.
\]

Therefore:

\[
Q(h_t, a_t, z)
= \mathbb{E}[R \mid h_t, a_t, z]
= \mathbb{E}[R \mid h_t \circ a_t, z]
= V(h_{t+1}, z).
\]

Thus, a prefix-value critic can naturally become a token-level credit estimator:

\[
A_t^{\text{PVC}}
\approx
V_\phi(h_{t+1}, z) - V_\phi(h_t, z).
\]

This is the central algorithmic bridge between self-distillation and actor-critic RL:

| Self-distillation | Privileged critic proposal |
|---|---|
| Teacher sees privileged information | Critic sees privileged information |
| Teacher re-scores student rollout | Critic evaluates student prefixes |
| Signal is likelihood ratio | Signal is value difference |
| Objective is partially imitation-like | Objective is verifier-aligned value learning |
| Teacher style can leak into update | Critic can be trained to ignore style and predict return |

---

## 4. Proposed methods

The proposal has two levels. The first is a conservative implementation that should be easy to add to verl. The second is the more conceptually important version that directly replaces self-teacher token credit.

### 4.1 PVC-V: privileged value critic for PPO / GAE

This is the minimal version.

Train:

\[
V_\phi(h_t, z) \approx \mathbb{E}[R \mid h_t, z].
\]

Use it in the normal PPO / GAE pipeline:

\[
\hat{A}_t^{\text{GAE}}
= \sum_{l \ge 0} (\gamma \lambda)^l
\left(r_{t+l} + \gamma V_\phi(h_{t+l+1}, z) - V_\phi(h_{t+l}, z)\right).
\]

Actor update remains standard PPO:

\[
\mathcal{L}_{\text{actor}}
= \mathbb{E}_t
\left[
\min\left(
\rho_t \hat{A}_t,
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right],
\]

where:

\[
\rho_t = \frac{\pi_\theta(a_t \mid h_t)}{\pi_{\theta_{old}}(a_t \mid h_t)}.
\]

Here the only change from standard PPO is that the value model receives privileged context.

This version is theoretically cleanest when \(z\) is prompt-level fixed and action-independent, such as a precomputed reference solution or verifier trace.

### 4.2 PVC-Δ: privileged value-difference token credit

This is the stronger version and the one that most directly competes with self-distillation.

Compute:

\[
\Delta V_t^{\text{priv}}
= V_\phi(h_{t+1}, z) - V_\phi(h_t, z).
\]

Use this value difference as a token-level credit signal.

A conservative RLVR-compatible version is:

\[
A_{i,t}^{\text{PVC-}\Delta}
= A_i^{\text{env}} \cdot g\left(\operatorname{norm}(\Delta V_{i,t}^{\text{priv}})\right),
\]

where:

- \(A_i^{\text{env}}\) is a sequence-level or group-relative advantage from verifier reward;
- \(g\) is a positive clipping / gating function;
- the verifier reward controls update direction;
- the privileged critic controls fine-grained token weighting.

This mirrors the stronger self-distillation design pattern: environment reward decides whether the trajectory should be reinforced or penalized, while the dense privileged signal decides which tokens deserve more or less credit.

Example gating function:

\[
g(u) = \operatorname{clip}\left(\exp(\alpha u), 1-c, 1+c\right).
\]

Then:

\[
A_{i,t}^{\text{PVC-}\Delta}
= A_i^{\text{env}} \cdot
\operatorname{clip}\left(
\exp\left(\alpha \cdot \operatorname{norm}(\Delta V_{i,t})\right),
1-c,
1+c
\right).
\]

This avoids directly imitating privileged teacher logits while preserving the benefit of dense token-level credit.

### 4.3 PVC-Routing: sample-dependent critic usage

Not every sample should use the privileged critic equally.

Possible routing rules:

- **failed rollouts:** use PVC-Δ strongly to identify where the trajectory lost progress;
- **successful rollouts:** use PVC-Δ carefully, because reference mismatch may suppress alternative valid reasoning;
- **high-entropy critic predictions:** downweight privileged signal;
- **easy prompts:** rely more on standard RLVR / GRPO;
- **hard prompts or all-failed groups:** allow stronger privileged credit or candidate-level scoring.

A routing objective can interpolate between group-relative reward advantage and privileged value credit:

\[
A_{i,t}
= (1 - \lambda_i) A_i^{\text{GRPO}}
+ \lambda_i A_{i,t}^{\text{PVC-}\Delta}.
\]

Here \(\lambda_i\) can depend on reward, critic confidence, rollout entropy, or agreement with reference structure.

---

## 5. Why this is meaningfully different from self-distillation

Self-distillation uses:

\[
\log \pi_\theta(a_t \mid h_t, z)
- \log \pi_\theta(a_t \mid h_t)
\]

as a proxy for token value.

PVC uses:

\[
V_\phi(h_t \circ a_t, z) - V_\phi(h_t, z)
\]

as a directly trained value signal.

The difference is not cosmetic. Teacher logits answer:

> After seeing privileged information, would the model be more likely to emit this token?

The privileged critic answers:

> After appending this token, does the prefix become more likely to lead to verifier success under the privileged structure?

The second question is closer to RL credit assignment.

This reframes recent self-distillation methods as partial, implicit implementations of a more general idea:

> Privileged information should be used for credit assignment, not necessarily for teacher imitation.

---

## 6. Implementation in verl

### 6.1 Goal

If the dataset already contains a correct trajectory string, e.g. `reference_trace`, then add it to the critic context without leaking it to the actor.

Actor input:

```text
[prompt] [sampled_response_prefix]
```

Critic input:

```text
[critic_instruction]
[problem / prompt]
[reference_trace]
[student sampled_response_prefix]
```

The actor rollout and actor update must never see `reference_trace`.

### 6.2 Dataset schema

Each training example should contain at least:

```json
{
  "prompt": [
    {"role": "user", "content": "Solve the problem..."}
  ],
  "reference_trace": "A correct reasoning trajectory string...",
  "reward_model": {
    "ground_truth": "final answer or verifier target"
  },
  "data_source": "my_dataset"
}
```

For parquet training data, `reference_trace` can be a string column. verl's default collation path can keep non-tensor fields as object arrays, so the field should be accessible as:

```python
batch.non_tensor_batch["reference_trace"]
```

If the default dataset drops the field, use a custom dataset class and preserve `reference_trace` in the returned sample dictionary.

### 6.3 Where to patch the PPO loop

The cleanest patch is in the trainer, not the rollout worker.

Standard verl PPO flow is approximately:

```python
gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)
batch = batch.union(gen_batch_output)

# reward / old_log_probs / ref_log_probs ...

values = critic_wg.compute_values(batch)
batch = batch.union(values)

batch = compute_advantage(batch, ...)

critic_output = critic_wg.update_critic(batch)
actor_output = actor_rollout_wg.update_actor(batch)
```

Modify only the critic calls:

```python
gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)
batch = batch.union(gen_batch_output)

# reward / old_log_probs / ref_log_probs ...

critic_batch = make_privileged_critic_batch(batch)
values = critic_wg.compute_values(critic_batch)
batch = batch.union(values)

batch = compute_advantage(batch, ...)

critic_train_batch = make_privileged_critic_batch(batch)
critic_output = critic_wg.update_critic(critic_train_batch)

# actor still uses the original non-privileged batch
actor_output = actor_rollout_wg.update_actor(batch)
```

### 6.4 Important causal-LM detail: reference must appear before response

If the critic is a decoder-only causal LM with a value head, the value at response-token positions can only attend to tokens on the left.

Therefore, do **not** format critic input as:

```text
[prompt]
[student response]
[reference trace]
```

because response-position value states cannot attend to the reference trace.

Use:

```text
[critic instruction]
[prompt]
[reference trace]
[student response]
```

This ensures each response token's hidden state can condition on the privileged reference.

### 6.5 Minimal helper function

Conceptual implementation:

```python
def make_privileged_critic_batch(batch):
    """
    Return a copy of the PPO batch whose input_ids / attention_mask / position_ids
    are rebuilt for the critic only.

    Actor batch:
        [prompt] [response]

    Critic batch:
        [critic instruction] [prompt] [reference_trace] [response]
    """
    refs = batch.non_tensor_batch["reference_trace"]
    prompt_ids = batch.batch["prompts"]
    responses = batch.batch["responses"]

    prompt_texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

    rows = []
    for prompt_text, ref_text, response_ids in zip(prompt_texts, refs, responses):
        critic_prefix = f"""
You are a value critic for reasoning RL.
Use the reference trajectory only to estimate whether the student solution prefix
is making progress toward a verifier-correct answer. Do not generate an answer.

[Problem]
{prompt_text}

[Reference trajectory]
{ref_text}

[Student solution]
"""
        prefix_ids = tokenizer.encode(critic_prefix, add_special_tokens=False)
        full_ids = prefix_ids + response_ids.tolist()
        rows.append(full_ids)

    input_ids, attention_mask, position_ids = left_pad_and_make_position_ids(rows)

    critic_batch = batch.clone()
    critic_batch.batch["input_ids"] = input_ids
    critic_batch.batch["attention_mask"] = attention_mask
    critic_batch.batch["position_ids"] = position_ids
    return critic_batch
```

Production improvements:

- pre-tokenize `critic_prefix_ids` in dataset preprocessing;
- avoid decode-encode in the training loop;
- cap `reference_trace` length separately from prompt and response;
- add `reference_dropout_prob`;
- support shuffled-reference sanity check through config;
- assert actor batch never contains privileged tokens in `input_ids`.

### 6.6 Config sketch

```yaml
algorithm:
  adv_estimator: gae
  use_privileged_critic: true
  privileged_critic:
    key: reference_trace
    mode: concat_prefix
    max_ref_tokens: 2048
    reference_dropout_prob: 0.1
    shuffled_reference_prob: 0.0
    use_delta_value_credit: false
    delta_value_alpha: 0.5
    delta_value_clip: 0.2
```

Phase 1 should set:

```yaml
use_delta_value_credit: false
```

and only test whether privileged critic improves value estimation and PPO sample efficiency.

Phase 2 can enable:

```yaml
use_delta_value_credit: true
```

and compare against self-distillation token-ratio methods.

---

## 7. Experiments

### 7.1 Main research questions

1. Does privileged critic conditioning improve value prediction?
2. Does improved value prediction translate into better sample efficiency and final accuracy?
3. Does value-difference token credit compete with self-teacher likelihood-ratio credit?
4. Does the method avoid reference imitation collapse better than self-distillation?
5. Which privileged information is most useful: final answer, full trajectory, verifier trace, execution trace, or multiple references?

### 7.2 Baselines

Minimum baseline set:

1. **GRPO / RLVR without critic**
2. **PPO with normal critic**
   \[
   V(h_t)
   \]
3. **PPO with privileged critic**
   \[
   V(h_t, z)
   \]
4. **PPO with shuffled privileged critic**
   \[
   V(h_t, z_{\text{wrong prompt}})
   \]
5. **Final-answer-only critic**
6. **Full-reference-trace critic**
7. **Verifier-trace / execution-trace critic**
8. **Self-distillation baseline** using teacher likelihood ratio
9. **RLSD-style baseline** using reward direction + teacher-ratio magnitude
10. **PVC-Δ** using reward direction + value-difference magnitude

The shuffled-reference baseline is crucial. If the shuffled version performs similarly to the correct-reference version, the critic is probably exploiting length, format, or stylistic shortcuts rather than reasoning structure.

### 7.3 Ablations

Privileged input ablations:

| Condition | Critic context |
|---|---|
| Normal critic | prompt + response |
| Answer only | prompt + final answer + response |
| Full trace | prompt + reference_trace + response |
| Execution trace | prompt + tests / runtime / verifier trace + response |
| Multi-reference | prompt + multiple reference traces + response |
| Shuffled trace | prompt + wrong reference_trace + response |
| Dropout trace | random spans removed from reference_trace |

Algorithmic ablations:

| Condition | Advantage / credit |
|---|---|
| PPO-GAE | standard GAE |
| PVC-V | GAE with privileged V |
| PVC-Δ | token weight from value difference |
| Teacher-ratio | token weight from self-teacher likelihood ratio |
| Hybrid routing | sample-dependent mixture |

### 7.4 Metrics

Policy metrics:

- pass@1
- pass@k
- average verifier reward
- sample efficiency curve
- final accuracy under fixed compute
- response length
- policy KL
- entropy

Critic metrics:

- value RMSE
- explained variance
- advantage variance
- calibration curve: predicted value vs empirical success
- correlation between \(\Delta V_t\) and eventual success

Diversity / collapse metrics:

- distinct reasoning templates
- n-gram diversity
- solution-path clustering
- off-reference correct rate
- pass@k vs pass@1 gap

Safety / leakage checks:

- actor input contains no `reference_trace` tokens;
- validation actor runs without privileged fields;
- shuffled-reference performance drops;
- final-answer-only vs full-trace comparison clarifies whether CoT is actually useful.

---

## 8. Expected outcomes

### 8.1 Strong positive result

Privileged critic improves:

- critic explained variance;
- advantage stability;
- PPO sample efficiency;
- final pass@1;
- stability relative to self-distillation;
- diversity relative to teacher imitation.

This would support the main thesis:

> The useful part of self-distillation is privileged credit assignment, and an explicit critic is a better parameterization for that role.

### 8.2 Mixed result

Privileged critic improves value metrics but not final policy performance.

Interpretation:

- value estimates are better but not used effectively by PPO;
- GAE may be insufficient;
- need PVC-Δ or routing;
- the privileged signal may be too late or too weak for all-failed prompts.

### 8.3 Negative result

Privileged critic improves neither value prediction nor policy learning.

Possible causes:

- reference traces are noisy or stylistically inconsistent;
- critic overfits lexical overlap;
- context length truncation destroys useful information;
- critic target is too sparse;
- standard PPO critic architecture cannot use reference context effectively.

Follow-up:

- use final-answer-only or verifier-trace input;
- train critic with pairwise ranking between successful and failed prefixes;
- add multi-reference alignment;
- move from concatenation to dual-encoder / cross-attention.

---

## 9. Risks and mitigations

### 9.1 Action leakage and biased baselines

The safest privileged information is prompt-level fixed:

- gold final answer;
- human reference solution;
- precomputed teacher solution;
- static verifier trace;
- independent search trace.

Riskier information includes:

- future tokens from the current actor rollout;
- best-of-N selected from the same current batch;
- traces generated after seeing the current sampled action;
- any hindsight signal tightly coupled to the current trajectory.

Mitigation:

- freeze privileged traces before the PPO update;
- use traces independent of the current sampled action;
- run shuffled-reference and delayed-buffer ablations;
- clearly separate PVC-V as a baseline estimator from PVC-Δ as an auxiliary credit modifier.

### 9.2 Reference lexical matching

The critic may learn string overlap with the reference instead of semantic progress.

Mitigation:

- reference dropout;
- multiple references;
- final-answer-only ablation;
- wrong-reference ablation;
- trace abstraction into key lemmas / execution states;
- evaluate off-reference correct solutions.

### 9.3 Exploration suppression

A single privileged trajectory may penalize valid alternative paths.

Mitigation:

- multi-reference training;
- do not directly imitate reference tokens;
- route successful off-reference rollouts away from strong privileged correction;
- maintain entropy / KL constraints;
- measure reasoning diversity.

### 9.4 Critic exploitation

Actor may learn to exploit critic errors if value-difference credit is too strong.

Mitigation:

- keep environment reward as update direction;
- clip value-difference weights;
- use uncertainty gating;
- periodically evaluate with pure verifier reward;
- avoid using PVC-Δ before PVC-V is calibrated.

### 9.5 Context and compute cost

Adding reference traces increases critic sequence length.

Mitigation:

- cap reference length;
- pre-tokenize critic prefixes;
- compress reference into answer + key steps;
- train final-answer-only and verifier-trace variants;
- use critic-only longer context, actor unchanged.

---

## 10. Implementation milestones

### Milestone 0: data validation

- Add `reference_trace` to parquet.
- Confirm it appears in `batch.non_tensor_batch`.
- Add assertion that actor rollout prompt does not include `reference_trace`.

### Milestone 1: PVC-V minimal patch

- Implement `make_privileged_critic_batch(batch)`.
- Use privileged batch for `critic_wg.compute_values`.
- Use privileged batch for `critic_wg.update_critic`.
- Keep actor rollout and actor update unchanged.
- Run normal critic vs privileged critic vs shuffled-reference critic.

### Milestone 2: critic diagnostics

- Log explained variance.
- Log value RMSE.
- Log advantage variance.
- Log reference length and truncation rate.
- Log shuffled-reference gap.

### Milestone 3: PVC-Δ token credit

- Compute \(\Delta V_t = V(h_{t+1},z)-V(h_t,z)\).
- Use \(\Delta V_t\) as token-level magnitude or auxiliary advantage.
- Compare against teacher likelihood-ratio self-distillation.

### Milestone 4: robustness improvements

- Add reference dropout.
- Add final-answer-only mode.
- Add multi-reference mode.
- Add routing based on rollout correctness and critic confidence.

### Milestone 5: paper-quality experiments

- Run across math and code verifier tasks.
- Compare against GRPO, PPO, SDPO-style, RLSD-style, and routing baselines.
- Report value diagnostics, policy performance, and diversity metrics.

---

## 11. Novelty and contribution

This proposal is not simply “give the critic the answer.”

The central contribution is the reinterpretation and replacement of self-distillation's teacher signal:

1. **Conceptual contribution:** self-teacher likelihood ratios are implicit privileged advantage estimates.
2. **Algorithmic contribution:** in autoregressive generation, token values can be represented as next-prefix values, so an explicit privileged prefix-value critic can produce dense token credit through value differences.
3. **Systems contribution:** the method can be implemented in verl with a trainer-level critic-only batch transformation, without changing rollout or actor deployment.
4. **Empirical contribution:** test whether verifier-aligned privileged value credit is more stable and less style-biased than self-teacher distillation.

The cleanest claim is:

> Self-distillation uses privileged information as a likelihood-based proxy for credit assignment. PVC uses the same information channel but trains an explicit verifier-aligned value function, then derives token credit from autoregressive prefix-value differences.

---

## 12. Related work framing

### Reinforcement Learning via Self-Distillation / SDPO

SDPO uses feedback-conditioned self-teaching to convert feedback into dense learning signals. It demonstrates that privileged or hindsight context can provide useful token-level guidance, but the signal is expressed through teacher next-token predictions.

### Self-Distilled RLVR / RLSD

RLSD identifies information leakage and instability when learning solely from a privileged teacher. It keeps RLVR reward as the update direction and uses self-distillation for token-level magnitude. PVC follows the same spirit but replaces teacher likelihood ratio with explicit privileged value difference.

### Sample Routing / SRPO

SRPO shows that different samples benefit from different optimization regimes, routing correct samples and failed samples differently. PVC can adopt the same idea by routing when and how strongly privileged value credit is used.

### Rebellious Student / RLRT

RLRT shows that teacher signals can suppress useful student reasoning on successful rollouts. This supports PVC's core criticism of teacher likelihood as an impure value proxy and motivates measuring off-reference correct paths.

### On-Policy Distillation lens

The OPD perspective suggests that on-policy sampling and optimization geometry matter as much as the teacher distribution itself. PVC pushes this further: if the teacher mostly acts as a credit assignment device, use an explicit critic rather than a policy teacher.

### Asymmetric actor-critic / CTDE

PVC is an LLM reasoning instance of asymmetric actor-critic: the actor uses deployment-time information, while the critic receives training-only privileged context. The key difference is that the privileged context is reasoning-specific: answer traces, verifier feedback, execution traces, or search trajectories.

---

## 13. Recommended first experiment

The fastest experiment that can validate or falsify the core idea:

### Dataset

Math or code RLVR dataset where each prompt has:

- prompt;
- verifier reward;
- correct final answer;
- correct reasoning string or execution trace.

### Conditions

1. PPO with normal critic.
2. PPO with final-answer-only critic.
3. PPO with full-reference-trace critic.
4. PPO with shuffled-reference-trace critic.

### Hypothesis

If the idea is valid:

\[
\text{full reference critic} > \text{normal critic}
\]

and:

\[
\text{full reference critic} > \text{shuffled reference critic}.
\]

If full reference is not better than final-answer-only, then full CoT may be unnecessary and potentially harmful. If shuffled reference performs similarly to correct reference, the critic is likely learning shortcuts.

### Success criteria

- higher value explained variance;
- lower advantage variance;
- faster pass@1 improvement;
- no significant collapse in reasoning diversity;
- no actor-side reference leakage.

---

## 14. Final summary

The proposal is best understood as:

> Turn the privileged self-teacher into an explicit privileged critic.

Current self-distillation methods already suggest that privileged context is useful for token-level credit assignment. But they express that credit through policy likelihoods, which are contaminated by style, teacher bias, and reference-path preference.

PVC instead trains:

\[
V_\phi(h_t, z)
\]

and uses the autoregressive identity:

\[
Q(h_t,a_t,z)=V(h_t\circ a_t,z)
\]

so token credit becomes:

\[
\Delta V_t = V_\phi(h_t\circ a_t,z)-V_\phi(h_t,z).
\]

This gives a cleaner and more RL-native alternative to self-distillation: privileged information is used for value estimation and credit assignment, while the actor remains deployable without privileged context.

---

## References

- Reinforcement Learning via Self-Distillation: https://arxiv.org/abs/2601.20802
- Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing: https://arxiv.org/abs/2604.02288
- Self-Distilled RLVR: https://arxiv.org/abs/2604.03128
- Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR: https://arxiv.org/abs/2605.10781
- SFT, RL, and OPD: https://nrehiew.github.io/blog/sft_rl_opd/
- verl project: https://github.com/verl-project/verl
- verl PPO trainer docs: https://verl.readthedocs.io/en/latest/workers/ray_trainer.html
- verl data interface docs: https://verl.readthedocs.io/en/latest/api/data.html
