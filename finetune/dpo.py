"""Direct Preference Optimization (DPO) — the alignment stage of post-training.

After supervised fine-tuning (see llm.finetune.sft), DPO aligns the model to
human preferences *without* a reward model or RL loop. Given triples
(prompt, chosen, rejected) it nudges the policy to assign higher likelihood to
the chosen response than the rejected one, while a frozen *reference* model (usually
the SFT checkpoint) keeps the policy from drifting too far.

The loss (Rafailov et al., "Direct Preference Optimization", 2023) is:

    L = -log sigmoid( beta * [ (logp_chosen - logp_rejected)_policy
                               - (logp_chosen - logp_rejected)_ref ] )

where each logp is the summed next-token log-probability of a response under a
model, computed only over the response tokens (the prompt is masked, exactly as in
SFT). beta controls how strongly we trust the reference.

This module provides:
* sequence_logprob — summed log-prob of labels under logits, with masking.
* dpo_loss — the DPO objective plus the reward-accuracy metric.
* DPODataset — tokenizes preference triples into masked chosen/rejected pairs.
* DPOTrainer — a Trainer subclass running the
  paired policy/reference forward passes. Works with full or LoRA fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..train.trainer import Trainer
from .sft import PROMPT_NO_INPUT, PROMPT_WITH_INPUT

IGNORE_INDEX = -1  # must match the models' cross_entropy ignore_index


def render_prompt(example: dict) -> str:
    """Answers: what exact prompt text should the model see for this example?

    Build the prompt string from a preference example. Accepts either an explicit
    {prompt: ...} or an Alpaca-style {instruction: ..., input: ...}
    (rendered with the SFT templates).
    """
    if example.get("prompt"):
        return example["prompt"]
    if example.get("input"):
        return PROMPT_WITH_INPUT.format(
            instruction=example["instruction"], input=example["input"]
        )
    return PROMPT_NO_INPUT.format(instruction=example["instruction"])


def sequence_logprob(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Answers: what total log-probability does the model assign to the response
    tokens in each sequence?

    Summed log-prob of labels under logits, ignoring masked positions.

    Args:
        logits: (batch_size, seq_length, vocab_size) — already position-aligned with
            labels (the models do not shift internally; the dataset pre-shifts).
        labels: (batch_size, seq_length) — target ids, with ignore_index where the
            position should not contribute (prompt tokens and padding).

    Returns:
        (batch_size,) tensor of per-sequence summed log-probabilities.
    """
    logp = F.log_softmax(logits, dim=-1)
    mask = labels != ignore_index
    safe_labels = labels.clamp_min(0)  # gather needs valid indices; masked out below
    token_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Answers: did the policy raise the chosen response and lower the rejected one
    *more than the reference did*? Turns that into a single scalar to minimize.

    The DPO loss and the reward-accuracy metric.

    Args:
        *_logp: (batch_size,) summed response log-probs under the policy/reference.
        beta: temperature on the implicit reward (typically 0.1).

    Returns:
        (loss, reward_accuracy) where reward_accuracy is the fraction of pairs whose
        chosen implicit reward exceeds the rejected one — the key DPO health metric.
    """
    policy_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    reward_acc = (chosen_reward > rejected_reward).float().mean().item()
    return loss, reward_acc


class DPODataset(Dataset):
    """Answers: how do raw (prompt, chosen, rejected) text triples become the padded,
    prompt-masked tensors the trainer consumes?

    Tokenized preference triples with prompt tokens masked out of the loss.

    Each item is (chosen_x, chosen_y, rejected_x, rejected_y), where each
    x/y pair is built like SFTDataset: x is the
    prompt+response (pre-shifted) and y is the next-token targets with prompt and
    padding positions set to ignore_index.

    Args:
        examples: list of {prompt|instruction[+input], chosen, rejected}.
        tokenizer: a Tokenizer.
        block_size: fixed sequence length (truncate/pad to this).
        pad_id: token id used for padding inputs (its targets are always masked).
    """

    def __init__(self, examples: list[dict], tokenizer, block_size: int, pad_id: int = 0):
        self.examples = examples
        self.tok = tokenizer
        self.block_size = block_size
        self.pad_id = pad_id
        self.eot = tokenizer.eot_token

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_pair(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Answers: how does one prompt+response become an (input, target) pair where
        only the response tokens carry loss?"""
        prompt_ids = self.tok.encode(prompt)
        response_ids = self.tok.encode(response) + [self.eot]

        full = (prompt_ids + response_ids)[: self.block_size + 1]
        x = full[:-1]
        y = full[1:]

        # Mask prompt tokens: a target at position t predicts token t+1; mask it while
        # the predicted token still lies inside the prompt.
        n_prompt = len(prompt_ids)
        y = [IGNORE_INDEX if (i + 1) < n_prompt else tok for i, tok in enumerate(y)]

        pad = self.block_size - len(x)
        if pad > 0:
            x = x + [self.pad_id] * pad
            y = y + [IGNORE_INDEX] * pad

        return (
            torch.tensor(x[: self.block_size], dtype=torch.long),
            torch.tensor(y[: self.block_size], dtype=torch.long),
        )

    def __getitem__(self, idx: int):
        """Answers: what are the chosen and rejected branches for one preference example?"""
        example = self.examples[idx]
        prompt = render_prompt(example)
        chosen_x, chosen_y = self._encode_pair(prompt, example["chosen"])
        rejected_x, rejected_y = self._encode_pair(prompt, example["rejected"])
        return chosen_x, chosen_y, rejected_x, rejected_y


def make_collate_fn():
    """Answers: how do a list of per-example quadruples become four batched tensors?

    Return a collate fn that stacks preference quadruples into batched tensors.
    """

    def collate(batch):
        chosen_xs, chosen_ys, rejected_xs, rejected_ys = zip(*batch)
        return (
            torch.stack(chosen_xs),
            torch.stack(chosen_ys),
            torch.stack(rejected_xs),
            torch.stack(rejected_ys),
        )

    return collate


class DPOTrainer(Trainer):
    """Answers: how do we run the paired policy/reference passes and take one DPO
    optimizer step, reusing all the base training machinery?

    Trainer that optimizes the DPO objective against a frozen reference model.

    Reuses the base Trainer for device/precision/FP8,
    optimizer, LR schedule, logging, and checkpointing. get_batch must return a
    (chosen_x, chosen_y, rejected_x, rejected_y) tuple of LongTensors.

    Args:
        policy: the model being trained (the SFT checkpoint, optionally LoRA-wrapped).
        ref: a frozen copy of the reference model (kept in eval, no grads).
        cfg: training hyper-parameters.
        get_batch: yields one preference batch.
        beta: DPO temperature on the implicit reward.
    """

    def __init__(self, policy, ref, cfg, get_batch, beta: float = 0.1):
        super().__init__(policy, cfg, get_batch)
        self.beta = beta
        self.ref = ref.to(self.device, dtype=self.precision.param_dtype).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.reward_acc = 0.0
        self.kl_drift = 0.0  # mean policy-vs-reference log-ratio on the batch

    def _policy_logp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Answers: what log-prob does the *trainable* policy give this response?"""
        logits, _ = self.model(x, targets=y)
        return sequence_logprob(logits, y)

    @torch.no_grad()
    def _ref_logp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Answers: what log-prob does the *frozen* reference give this response?"""
        logits, _ = self.ref(x, targets=y)
        return sequence_logprob(logits, y)

    def _log_extra(self) -> str:
        """Answers: what DPO-specific metrics should appear in the step log?

        Reward accuracy (chosen beats rejected) and the policy-vs-reference log-ratio
        drift — a cheap KL-style estimate of how far the policy has moved.
        """
        return f"| reward_acc {self.reward_acc:.2f} | drift {self.kl_drift:+.3f} "

    def _run_step(self) -> float:
        """Answers: how does one optimizer step turn preference batches into a weight
        update? Scores chosen/rejected under policy and reference, then backprops the
        DPO loss (with grad accumulation, clipping, and optional loss scaling)."""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_acc = 0.0
        total_drift = 0.0

        for _ in range(cfg.grad_accum_steps):
            chosen_x, chosen_y, rejected_x, rejected_y = self.get_batch()
            chosen_x = chosen_x.to(self.device, non_blocking=True)
            chosen_y = chosen_y.to(self.device, non_blocking=True)
            rejected_x = rejected_x.to(self.device, non_blocking=True)
            rejected_y = rejected_y.to(self.device, non_blocking=True)

            with self._autocast():
                policy_chosen = self._policy_logp(chosen_x, chosen_y)
                policy_rejected = self._policy_logp(rejected_x, rejected_y)
                ref_chosen = self._ref_logp(chosen_x, chosen_y)
                ref_rejected = self._ref_logp(rejected_x, rejected_y)
                loss, acc = dpo_loss(
                    policy_chosen, policy_rejected, ref_chosen, ref_rejected, self.beta
                )
                loss = loss / cfg.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()
            total_acc += acc / cfg.grad_accum_steps
            # Mean log-ratio over both responses: a detached KL-style drift estimate.
            drift = 0.5 * (
                (policy_chosen - ref_chosen).mean() + (policy_rejected - ref_rejected).mean()
            )
            total_drift += drift.item() / cfg.grad_accum_steps

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.reward_acc = total_acc
        self.kl_drift = total_drift
        return total_loss