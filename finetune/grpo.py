"""GRPO — the reinforcement-learning stage of post-training (DeepSeek-R1 style).

DPO (see llm.finetune.dpo) aligns a model from *static* preference pairs. GRPO
(Group Relative Policy Optimization, Shao et al., "DeepSeekMath", 2024) instead trains
on the model's *own* rollouts: for each prompt it samples a group of completions,
scores them with a reward function, and pushes probability mass toward the
above-average completions and away from the below-average ones. Run on a reasoning
prompt set with a verifiable reward, this is exactly the recipe that turns a base/SFT
model into a *reasoning* model (R1-Zero style) — the policy learns to "think" because
longer correct chains earn higher reward.

GRPO drops PPO's value network: the baseline is just the per-group mean reward, so the
advantage of completion i in a group is:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

and the same scalar advantage is applied to every token of that completion. The token
objective is PPO's clipped surrogate plus a KL penalty to a frozen reference:

    L = -mean_t[ min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A) ] + beta_kl * KL(pi || ref)

with ratio_t = exp(logp_policy - logp_old) and the low-variance k3 KL estimator
KL = exp(ref-pi) - (ref-pi) - 1 (Schulman). We take a single on-policy gradient
step per batch, so logp_old equals the current policy at compute time (ratio starts
at 1) — the clip machinery is still correct and kicks in if you add inner epochs.

This module provides:
* reward helpers — format_reward, correctness_reward,
  make_reasoning_reward (rule-based, verifiable) and reward_model_scorer
  (wrap any scalar reward model). All share the pluggable RewardFn signature.
* group_advantages — group-normalized advantages.
* grpo_loss — the clipped surrogate + KL objective, with diagnostics.
* ReasoningPromptDataset — renders {question, answer} into reasoning prompts.
* GRPOTrainer — a Trainer subclass that samples
  rollouts, scores them, and takes GRPO steps. Works with full or LoRA fine-tuning.
"""

from __future__ import annotations

import re
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..infer.generate import generate
from ..train.trainer import Trainer

# A reward function scores a batch of rollouts. Given the prompts, the decoded
# completions, and the source examples (which may carry a reference answer), it
# returns one scalar reward per rollout. This single signature covers both rule-based
# verifiers and learned reward models.
RewardFn = Callable[[list[str], list[str], list[dict]], "list[float] | torch.Tensor"]

# Reasoning template: ask the model to think out loud, then commit to an answer. The
# tags give the verifier a structural hook and teach the model to separate scratch
# reasoning from its final answer.
# CUSTOMIZE: swap in the target model's chat template / system prompt for real runs.
REASONING_INSTRUCTION = (
    "Solve the problem. Show your reasoning inside <think> </think> tags, then give the "
    "final answer inside <answer> </answer> tags.\n\nProblem:\n{question}\n\n"
)

_THINK_ANSWER_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def render_reasoning_prompt(example: dict) -> str:
    """Answers: what prompt text should the model see for this reasoning example?

    Accepts an explicit {prompt: ...} or a {question: ...} reasoning example
    (rendered with REASONING_INSTRUCTION).
    """
    if example.get("prompt"):
        return example["prompt"]
    return REASONING_INSTRUCTION.format(question=example["question"])


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str | None:
    """Answers: what did the model commit to inside <answer>...</answer>?

    Returns the (stripped) contents of the last <answer> block, or None if absent.
    """
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def format_reward(prompts: list[str], completions: list[str], examples: list[dict]) -> list[float]:
    """Answers: did the completion follow the <think>.../<answer>... structure?

    Returns 1.0 for each completion containing a well-formed think+answer block, else
    0.0. This shapes the model toward the reasoning format before correctness can help.
    """
    return [1.0 if _THINK_ANSWER_RE.search(c) else 0.0 for c in completions]


def _normalize(text: str) -> str:
    """Loose normalization for answer comparison (case/space/trailing-punctuation)."""
    return re.sub(r"[\s.]+$", "", text.strip().lower())


def correctness_reward(
    prompts: list[str], completions: list[str], examples: list[dict]
) -> list[float]:
    """Answers: does the extracted answer match the example's reference answer?

    Returns 1.0 when the completion's <answer> matches example["answer"] (after
    loose normalization), else 0.0. This is the *verifiable* reward — no reward model
    needed when the ground truth is known (math, exact-match QA, unit tests, ...).
    """
    rewards = []
    for completion, example in zip(completions, examples):
        reference = example.get("answer")
        predicted = extract_answer(completion)
        if reference is None or predicted is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if _normalize(predicted) == _normalize(str(reference)) else 0.0)
    return rewards


def make_reasoning_reward(
    format_weight: float = 0.5, correctness_weight: float = 1.0
) -> RewardFn:
    """Answers: how do format + correctness combine into one verifiable reward signal?

    Build a rule-based reward that sums a structural format_reward and a
    ground-truth correctness_reward. Reward modeling without a reward model.
    """

    def reward(prompts, completions, examples):
        fmt = format_reward(prompts, completions, examples)
        correct = correctness_reward(prompts, completions, examples)
        return [format_weight * f + correctness_weight * c for f, c in zip(fmt, correct)]

    return reward


def reward_model_scorer(reward_model, tokenizer, device: torch.device | None = None) -> RewardFn:
    """Answers: how do we score rollouts with a *learned* scalar reward model instead?

    Wrap a reward model into the RewardFn interface. reward_model must map a
    batch of padded token ids (batch_size, seq_length) to a per-sequence scalar
    tensor (batch_size,) (e.g. a transformer with a scalar value head). Prompt and
    completion are concatenated, tokenized, and right-padded before scoring.
    """
    pad_id = 0
    model_device = device

    def reward(prompts, completions, examples):
        nonlocal model_device
        if model_device is None:
            model_device = next(reward_model.parameters()).device
        ids = [tokenizer.encode(p + c) for p, c in zip(prompts, completions)]
        max_len = max(len(seq) for seq in ids)
        batch = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
        for row, seq in enumerate(ids):
            batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        batch = batch.to(model_device)
        with torch.no_grad():
            scores = reward_model(batch)
        return scores.float().flatten().tolist()

    return reward


# ---------------------------------------------------------------------------
# GRPO math
# ---------------------------------------------------------------------------
def group_advantages(
    rewards: torch.Tensor, group_size: int, eps: float = 1e-6
) -> torch.Tensor:
    """Answers: how much better than its group's average is each rollout?

    Group-normalized advantages: reshape rewards into (num_groups, group_size),
    subtract each group's mean, and divide by its std. This per-group baseline is what
    lets GRPO skip PPO's value network.

    Args:
        rewards: (num_groups * group_size,) scalar rewards, grouped contiguously.
        group_size: number of rollouts sampled per prompt.

    Returns:
        (num_groups * group_size,) advantages (zero-mean within each group).
    """
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def grpo_loss(
    policy_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta_kl: float = 0.04,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Answers: given per-token log-probs and per-rollout advantages, what scalar do we
    minimize to make good completions more likely (without drifting from the reference)?

    The token-level clipped surrogate plus a KL penalty to the frozen reference.

    Args:
        policy_logp: (batch_size, seq_length) per-token log-probs under the policy
            (requires grad).
        old_logp: (batch_size, seq_length) log-probs from rollout time (detached);
            equals policy_logp for a single on-policy step (ratio == 1).
        ref_logp: (batch_size, seq_length) log-probs under the frozen reference.
        advantages: (batch_size,) per-rollout advantages, broadcast over tokens.
        mask: (batch_size, seq_length) 1.0 for completion tokens, 0.0 for prompt/pad.
        beta_kl: strength of the KL-to-reference penalty.
        clip_eps: PPO clip range on the importance ratio.

    Returns:
        (loss, metrics) where metrics carries the mean KL and the clip fraction.
    """
    advantage = advantages.unsqueeze(1)  # (batch_size, 1) -> broadcast over tokens
    ratio = torch.exp(policy_logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_term = -torch.min(unclipped, clipped)

    # k3 KL estimator (Schulman): unbiased, low-variance, and always non-negative.
    log_ratio_ref = ref_logp - policy_logp
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    per_token = policy_term + beta_kl * kl
    token_count = mask.sum().clamp_min(1.0)
    loss = (per_token * mask).sum() / token_count

    with torch.no_grad():
        mean_kl = (kl * mask).sum() / token_count
        clipped_tokens = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
        clip_frac = (clipped_tokens * mask).sum() / token_count
    return loss, {"kl": mean_kl.item(), "clip_frac": clip_frac.item()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ReasoningPromptDataset(Dataset):
    """Answers: how do raw {question, answer} items become prompt examples the
    trainer can roll out and verify?

    A thin dataset of reasoning examples. Unlike SFT/DPO datasets it does *not*
    pre-tokenize responses — GRPO generates the responses itself — so each item is just
    the source dict (carrying the rendered prompt and the reference answer).

    Args:
        examples: list of {question|prompt, answer} dicts.
    """

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class GRPOTrainer(Trainer):
    """Answers: how do we sample rollouts, score them, and take a GRPO step while
    reusing all the base training machinery (device/precision/FP8, optimizer, logging)?

    Trainer that optimizes the GRPO objective on its own rollouts against a frozen
    reference model.

    get_batch must return a list[dict] of prompt examples (length =
    cfg.batch_size); for each one the trainer samples group_size completions,
    scores them with reward_fn, and applies group-normalized advantages.

    Args:
        policy: the model being trained (a base/SFT checkpoint, optionally LoRA-wrapped).
        ref: a frozen copy of the reference model (kept in eval, no grads).
        cfg: training hyper-parameters (batch_size = prompts per step).
        get_batch: yields one list of prompt examples.
        reward_fn: scores rollouts (see RewardFn); defaults to the rule-based
            reasoning reward.
        tokenizer: encodes prompts / decodes completions.
        group_size: rollouts sampled per prompt (the GRPO group G).
        max_new_tokens: max tokens to generate per rollout.
        temperature / top_k / top_p: sampling controls for rollouts (exploration).
        beta_kl: KL-to-reference penalty strength.
        clip_eps: PPO clip range on the importance ratio.
        render_fn: prompt-text renderer (defaults to render_reasoning_prompt).
    """

    def __init__(
        self,
        policy,
        ref,
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
        beta_kl: float = 0.04,
        clip_eps: float = 0.2,
        render_fn: Callable[[dict], str] = render_reasoning_prompt,
    ):
        super().__init__(policy, cfg, get_batch)
        self.tok = tokenizer
        self.reward_fn = reward_fn or make_reasoning_reward()
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.beta_kl = beta_kl
        self.clip_eps = clip_eps
        self.render_fn = render_fn
        self.eot = tokenizer.eot_token

        # A frozen reference is only needed for the KL penalty; KL-free variants
        # (DAPO, GSPO) pass ref=None to skip it entirely.
        if ref is not None:
            self.ref = ref.to(self.device, dtype=self.precision.param_dtype).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        else:
            self.ref = None

        # Diagnostics surfaced in the step log.
        self.reward_mean = 0.0
        self.kl = 0.0
        self.clip_frac = 0.0

    # ------------------------------------------------------------------
    def _rollout(self, prompt_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Answers: what completions does the current policy produce for this prompt,
        and which positions are actual completion tokens?

        Samples group_size completions for a single prompt and returns the full
        (group_size, seq_length) token sequences plus a boolean completion mask
        (True for generated tokens up to and including the first EOT, False for the
        prompt and any post-EOT padding).
        """
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt = prompt.repeat(self.group_size, 1)
        sequences = generate(
            self.model,
            prompt,
            self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eot_token=self.eot,
            device=self.device,
        )

        prompt_len = len(prompt_ids)
        generated = sequences[:, prompt_len:]
        is_eot = generated == self.eot
        has_eot = is_eot.any(dim=1)
        first_eot = torch.argmax(is_eot.int(), dim=1)
        # Rows with no EOT keep every generated position.
        first_eot = torch.where(
            has_eot, first_eot, torch.full_like(first_eot, generated.size(1) - 1)
        )
        positions = torch.arange(generated.size(1), device=self.device).unsqueeze(0)
        keep = positions <= first_eot.unsqueeze(1)

        mask = torch.zeros_like(sequences, dtype=torch.bool)
        mask[:, prompt_len:] = keep
        return sequences, mask

    def _logprobs(self, model, sequences: torch.Tensor) -> torch.Tensor:
        """Answers: what per-token log-prob does this model assign along the sequence?

        Returns (batch_size, seq_length - 1) log-probs where entry t is the
        log-prob of token t+1 given tokens 0..t. Passing targets forces the
        model's full-logits path (its inference path returns only the last position).
        """
        inputs = sequences[:, :-1].contiguous()
        targets = sequences[:, 1:].contiguous()
        logits, _ = model(inputs, targets=targets)
        logp = F.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def _group_loss(self, example: dict) -> tuple[torch.Tensor, float, dict]:
        """Answers: for one prompt, how does a group of rollouts become a single GRPO
        loss term (plus its reward/KL diagnostics)?"""
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

        # Targets are sequences[:, 1:]; align the completion mask the same way.
        token_mask = mask[:, 1:].float()
        ref_logp = (
            self._logprobs(self.ref, sequences).detach() if self.ref is not None else None
        )

        self.model.train()
        with self._autocast():
            policy_logp = self._logprobs(self.model, sequences)
            old_logp = policy_logp.detach()  # single on-policy step => ratio starts at 1
            loss, metrics = self._compute_loss(
                policy_logp, old_logp, ref_logp, advantages, token_mask
            )
        return loss, rewards.mean().item(), metrics

    def _compute_loss(
        self,
        policy_logp: torch.Tensor,
        old_logp: torch.Tensor,
        ref_logp: torch.Tensor | None,
        advantages: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Answers: which objective turns these log-probs and advantages into a scalar?

        The GRPO objective. Subclasses (DAPO, GSPO) override this single hook to swap in
        their own variant while reusing the same rollout/reward/log-prob machinery.
        """
        return grpo_loss(
            policy_logp,
            old_logp,
            ref_logp,
            advantages,
            token_mask,
            beta_kl=self.beta_kl,
            clip_eps=self.clip_eps,
        )

    def _log_extra(self) -> str:
        """Answers: what GRPO-specific metrics should appear in the step log?

        Mean rollout reward (is the policy improving?), KL drift from the reference, and
        the PPO clip fraction (how often the ratio is being clipped).
        """
        return (
            f"| reward {self.reward_mean:+.3f} | kl {self.kl:.3f} "
            f"| clip {self.clip_frac:.2f} "
        )

    def _run_step(self) -> float:
        """Answers: how does one optimizer step turn freshly sampled rollouts into a
        weight update? For each prompt in the batch it rolls out a group, scores it,
        and backprops the GRPO loss (with grad accumulation, clipping, optional scaling)."""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_reward = 0.0
        total_kl = 0.0
        total_clip = 0.0
        n_groups = 0

        for _ in range(cfg.grad_accum_steps):
            examples = self.get_batch()
            for example in examples:
                loss, reward_mean, metrics = self._group_loss(example)
                # Average over all groups in the effective batch.
                scaled = loss / (cfg.grad_accum_steps * len(examples))
                if self.scaler is not None:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                total_loss += scaled.item()
                total_reward += reward_mean
                total_kl += metrics["kl"]
                total_clip += metrics["clip_frac"]
                n_groups += 1

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.reward_mean = total_reward / max(n_groups, 1)
        self.kl = total_kl / max(n_groups, 1)
        self.clip_frac = total_clip / max(n_groups, 1)
        return total_loss
