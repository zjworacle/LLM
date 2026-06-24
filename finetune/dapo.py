"""DAPO — Decoupled-clip and Dynamic-sAmpling Policy Optimization (ByteDance, 2025).

DAPO ("DAPO: An Open-Source LLM Reinforcement Learning System at Scale", Yu et al.,
2025) is a set of pragmatic fixes to GRPO (see llm.finetune.grpo) that make
long-chain reasoning RL stable and sample-efficient. It keeps GRPO's group-relative
advantages but changes *how* the policy-gradient loss is formed:

1. **Clip-Higher** — decouple the lower and upper PPO clip ranges (eps_low <
   eps_high). The tight lower bound still guards against collapse, while the looser
   upper bound leaves room to *raise* low-probability tokens — preserving the
   exploration that a single symmetric clip silently kills.
2. **Dynamic Sampling** — drop prompt groups whose rollouts are all-correct or
   all-wrong: their rewards have zero variance, so their advantages (and gradients) are
   exactly zero. The trainer keeps generating until the batch is full of *informative*
   groups, so no compute is wasted on dead batches.
3. **Token-level loss** — average the loss over *all tokens in the batch* rather than
   per-sequence-then-per-group. Long (often the most important reasoning) responses are
   no longer down-weighted by GRPO's per-sequence length normalization.
4. **Overlong filtering** — mask the loss of truncated rollouts (those that hit the
   generation budget without emitting EOT) so a length cutoff is not mistaken for a
   bad answer.

DAPO drops the KL-to-reference penalty entirely (no reference model needed), relying on
clipping to keep updates well-behaved.

This module provides:
* dapo_loss — the decoupled-clip surrogate, returned as unreduced sums so the
  trainer can normalize at the *batch* (token) level.
* has_reward_variance — the dynamic-sampling keep/skip test.
* DAPOTrainer — a GRPOTrainer subclass implementing
  dynamic sampling, token-level reduction, and overlong filtering.
"""

from __future__ import annotations

from typing import Callable

import torch

from .grpo import GRPOTrainer, RewardFn, group_advantages, render_reasoning_prompt


def has_reward_variance(rewards: torch.Tensor, eps: float = 1e-6) -> bool:
    """Answers: is this group informative, or are all its rollouts equally (in)correct?

    Dynamic-sampling test: a group contributes a non-zero gradient only when its rewards
    differ (otherwise every advantage is zero). Returns True when the group should be
    kept.
    """
    return bool((rewards.max() - rewards.min()).item() > eps)


def dapo_loss(
    policy_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Answers: what (unreduced) clipped-surrogate loss does this group contribute, and
    over how many tokens?

    The decoupled-clip ("Clip-Higher") surrogate. Unlike grpo_loss
    this returns *sums* rather than a mean, so the trainer can divide by the total token
    count across the whole batch (DAPO's token-level normalization).

    Args:
        policy_logp: (batch_size, seq_length) per-token log-probs under the policy.
        old_logp: (batch_size, seq_length) log-probs from rollout time (detached).
        advantages: (batch_size,) group-normalized advantages, broadcast over tokens.
        mask: (batch_size, seq_length) 1.0 for completion tokens, 0.0 elsewhere.
        clip_eps_low: lower clip range (guards against collapse).
        clip_eps_high: upper clip range (>= low; leaves room to explore).

    Returns:
        (loss_sum, token_count, clipped_count) — all scalar tensors. loss_sum is
        the summed (masked) per-token loss; divide by token_count for the mean.
    """
    advantage = advantages.unsqueeze(1)  # (batch_size, 1) broadcast over tokens
    ratio = torch.exp(policy_logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * advantage
    per_token = -torch.min(unclipped, clipped)

    loss_sum = (per_token * mask).sum()
    token_count = mask.sum()
    with torch.no_grad():
        clipped_tokens = ((ratio < 1.0 - clip_eps_low) | (ratio > 1.0 + clip_eps_high)).float()
        clipped_count = (clipped_tokens * mask).sum()
    return loss_sum, token_count, clipped_count


class DAPOTrainer(GRPOTrainer):
    """Answers: how do we run dynamic sampling + token-level, decoupled-clip, KL-free
    updates while reusing GRPO's rollout/reward/log-prob machinery?

    DAPO trainer. Builds on GRPOTrainer but replaces the
    per-group GRPO step with: (1) dynamic sampling of informative groups, (2) a
    token-level loss summed across the whole batch, (3) decoupled clip ranges, and (4)
    optional overlong filtering. No reference model is used (DAPO is KL-free).

    Args:
        policy: the model being trained (optionally LoRA-wrapped).
        cfg: training hyper-parameters (batch_size = number of *kept* groups/step).
        get_batch: yields a list[dict] of prompt examples.
        tokenizer: encodes prompts / decodes completions.
        reward_fn: scores rollouts (defaults to the rule-based reasoning reward).
        group_size: rollouts sampled per prompt.
        max_new_tokens / temperature / top_k / top_p: rollout sampling controls.
        clip_eps_low / clip_eps_high: the decoupled (Clip-Higher) PPO clip ranges.
        dynamic_sampling: drop zero-variance groups and resample to fill the batch.
        overlong_filter: mask the loss of truncated (no-EOT) rollouts.
        max_resample_batches: cap on get_batch calls while filling one step.
        render_fn: prompt-text renderer.
    """

    def __init__(
        self,
        policy,
        cfg,
        get_batch,
        tokenizer,
        reward_fn: RewardFn | None = None,
        *,
        group_size: int = 8,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.28,
        dynamic_sampling: bool = True,
        overlong_filter: bool = True,
        max_resample_batches: int = 8,
        render_fn: Callable[[dict], str] = render_reasoning_prompt,
    ):
        super().__init__(
            policy,
            None,  # KL-free: no reference model
            cfg,
            get_batch,
            tokenizer,
            reward_fn,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            beta_kl=0.0,
            clip_eps=clip_eps_high,
            render_fn=render_fn,
        )
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.dynamic_sampling = dynamic_sampling
        self.overlong_filter = overlong_filter
        self.max_resample_batches = max_resample_batches
        self.kept_frac = 0.0  # fraction of sampled groups that survived dynamic sampling

    # ------------------------------------------------------------------
    def _group_data(self, example: dict):
        """Answers: for one prompt, what are its rollout sequences, completion mask
        (with overlong rollouts filtered out), and group-normalized advantages?"""
        prompt_text = self.render_fn(example)
        prompt_ids = self.tok.encode(prompt_text)

        sequences, mask = self._rollout(prompt_ids)
        completions = [
            self.tok.decode(row[len(prompt_ids):].tolist()) for row in sequences
        ]
        examples = [example] * self.group_size
        rewards = self.reward_fn([prompt_text] * self.group_size, completions, examples)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        advantages = group_advantages(rewards, self.group_size)

        token_mask = mask[:, 1:].float()
        if self.overlong_filter:
            # Truncated rollouts (no EOT before the budget) are masked out of the loss.
            generated = sequences[:, len(prompt_ids):]
            has_eot = (generated == self.eot).any(dim=1)
            token_mask = token_mask * has_eot.unsqueeze(1).float()
        return sequences, token_mask, advantages, rewards

    def _run_step(self) -> float:
        """Answers: how does one DAPO step fill a batch with informative groups and take
        a single token-level, decoupled-clip optimizer update?"""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)

        kept = []  # (sequences, token_mask, advantages) for informative groups
        total_reward = 0.0
        n_seen = 0
        batches = 0
        while len(kept) < cfg.batch_size and batches < self.max_resample_batches:
            batches += 1
            for example in self.get_batch():
                sequences, token_mask, advantages, rewards = self._group_data(example)
                n_seen += 1
                total_reward += rewards.mean().item()
                if not self.dynamic_sampling or has_reward_variance(rewards):
                    kept.append((sequences, token_mask, advantages))
                if len(kept) >= cfg.batch_size:
                    break

        self.reward_mean = total_reward / max(n_seen, 1)
        self.kept_frac = len(kept) / max(n_seen, 1)
        if not kept:
            # Every group was uniform this step; nothing to learn from.
            self.clip_frac = 0.0
            return 0.0

        self.model.train()
        loss_terms = []
        token_total = torch.zeros((), device=self.device)
        clip_total = torch.zeros((), device=self.device)
        with self._autocast():
            for sequences, token_mask, advantages in kept:
                policy_logp = self._logprobs(self.model, sequences)
                old_logp = policy_logp.detach()  # single on-policy step => ratio == 1
                loss_sum, token_count, clipped_count = dapo_loss(
                    policy_logp,
                    old_logp,
                    advantages,
                    token_mask,
                    self.clip_eps_low,
                    self.clip_eps_high,
                )
                loss_terms.append(loss_sum)
                token_total = token_total + token_count
                clip_total = clip_total + clipped_count
            # Token-level normalization: one denominator over the whole batch.
            loss = torch.stack(loss_terms).sum() / token_total.clamp_min(1.0)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.clip_frac = (clip_total / token_total.clamp_min(1.0)).item()
        return loss.item()

    def _log_extra(self) -> str:
        """Answers: what DAPO-specific metrics should appear in the step log?

        Mean rollout reward, the decoupled-clip fraction, and the dynamic-sampling keep
        rate (how many sampled groups were informative enough to train on).
        """
        return (
            f"| reward {self.reward_mean:+.3f} | clip {self.clip_frac:.3f} "
            f"| kept {self.kept_frac:.2f} "
        )
