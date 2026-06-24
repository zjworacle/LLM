"""GSPO — Group Sequence Policy Optimization (Qwen team, 2025).

GSPO ("Group Sequence Policy Optimization", Zheng et al., 2025; the algorithm behind
Qwen3) keeps GRPO's group-relative advantages but fixes where the importance ratio is
applied. GRPO (and PPO) weight each *token* by its own importance ratio
exp(logp_policy - logp_old); over a long response these per-token ratios compound
high variance and, for mixture-of-experts models, can swing wildly as routing shifts.
That instability is a major cause of RL collapse on long reasoning traces.

GSPO instead defines a single **sequence-level** importance ratio — the length-normalized
geometric mean of the per-token ratios:

    s_i = exp( (1 / |y_i|) * sum_t [ logp_policy(t) - logp_old(t) ] )

and clips at the sequence level:

    L = -mean_i  min( s_i * A_i,  clip(s_i, 1 - eps_low, 1 + eps_high) * A_i )

Because the ratio is averaged over the response, its variance is tiny, so the clip
ranges are correspondingly small (the paper uses ~3e-4 / 4e-4). The reward, the
group-normalized advantage A_i, and the rollout machinery are all identical to
GRPO; only the surrogate changes. GSPO needs no reference/KL term (the sequence clip
keeps updates in trust region), so no reference model is built.

This module provides:
* gspo_loss — the sequence-level clipped surrogate, with diagnostics.
* GSPOTrainer — a GRPOTrainer subclass that swaps
  in gspo_loss via the _compute_loss hook.
"""

from __future__ import annotations

from typing import Callable

import torch

from .grpo import GRPOTrainer, RewardFn, render_reasoning_prompt


def gspo_loss(
    policy_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps_low: float = 3e-4,
    clip_eps_high: float = 4e-4,
) -> tuple[torch.Tensor, dict]:
    """Answers: how likely is each *whole response* now vs. at rollout time, and how do
    we turn that into a sequence-level clipped objective?

    The GSPO sequence-level clipped surrogate.

    Args:
        policy_logp: (batch_size, seq_length) per-token log-probs under the policy.
        old_logp: (batch_size, seq_length) log-probs from rollout time (detached);
            equals policy_logp for a single on-policy step (ratio == 1).
        advantages: (batch_size,) group-normalized, per-sequence advantages.
        mask: (batch_size, seq_length) 1.0 for completion tokens, 0.0 elsewhere.
        clip_eps_low / clip_eps_high: sequence-level clip ranges (small by design).

    Returns:
        (loss, metrics) with the mean clip fraction and mean importance ratio.
    """
    token_logratio = (policy_logp - old_logp) * mask
    seq_length = mask.sum(dim=1).clamp_min(1.0)
    # Length-normalized mean log-ratio -> sequence-level importance ratio.
    seq_logratio = token_logratio.sum(dim=1) / seq_length
    importance = torch.exp(seq_logratio)  # (batch_size,)

    unclipped = importance * advantages
    clipped = torch.clamp(importance, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * advantages
    loss = -torch.min(unclipped, clipped).mean()

    with torch.no_grad():
        clipped_seqs = (
            (importance < 1.0 - clip_eps_low) | (importance > 1.0 + clip_eps_high)
        ).float()
        metrics = {
            "kl": 0.0,  # GSPO is KL-free; kept for log-line compatibility
            "clip_frac": clipped_seqs.mean().item(),
            "importance": importance.mean().item(),
        }
    return loss, metrics


class GSPOTrainer(GRPOTrainer):
    """Answers: how do we take GRPO rollouts but optimize the *sequence-level* GSPO
    surrogate instead of the per-token one?

    GSPO trainer. Reuses GRPOTrainer for rollouts, rewards,
    group advantages, optimizer, and logging, overriding only the loss hook. No
    reference model is used (GSPO is KL-free).

    Args:
        policy: the model being trained (optionally LoRA-wrapped).
        cfg: training hyper-parameters (batch_size = prompts per step).
        get_batch: yields a list[dict] of prompt examples.
        tokenizer: encodes prompts / decodes completions.
        reward_fn: scores rollouts (defaults to the rule-based reasoning reward).
        group_size: rollouts sampled per prompt (the group G).
        max_new_tokens / temperature / top_k / top_p: rollout sampling controls.
        clip_eps_low / clip_eps_high: the (small) sequence-level clip ranges.
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
        clip_eps_low: float = 3e-4,
        clip_eps_high: float = 4e-4,
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

    def _compute_loss(self, policy_logp, old_logp, ref_logp, advantages, token_mask):
        """Answers: which objective? GSPO's sequence-level clipped surrogate (ref unused)."""
        return gspo_loss(
            policy_logp,
            old_logp,
            advantages,
            token_mask,
            clip_eps_low=self.clip_eps_low,
            clip_eps_high=self.clip_eps_high,
        )

    def _log_extra(self) -> str:
        """Answers: what GSPO-specific metrics should appear in the step log?

        Mean rollout reward and the sequence-level clip fraction.
        """
        return f"| reward {self.reward_mean:+.3f} | seq_clip {self.clip_frac:.3f} "
