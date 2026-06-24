"""Evaluation harness — judging model *quality* independent of which loss trained it.

Pre-training, SFT, and DPO each minimize a different loss, so their loss values are
not comparable. This module provides out-of-band metrics that *are* comparable across
checkpoints:

* :func:`evaluate_loss` — held-out cross-entropy + perplexity (for pre-training/SFT).
* :func:`evaluate_multiple_choice` / :func:`score_choices` — task accuracy via
  log-prob scoring of candidate answers (works for *any* checkpoint).
* :func:`evaluate_preferences` — DPO health on held-out pairs: reward accuracy and a
  policy-vs-reference log-ratio drift (a cheap KL-style estimate).

Every metric reuses the same masked log-prob machinery as training
(:func:`llm.finetune.dpo.sequence_logprob`), so what you measure matches what you
optimized.
"""

from __future__ import annotations

import math

import torch

from ..finetune.dpo import IGNORE_INDEX, dpo_loss, sequence_logprob


@torch.no_grad()
def evaluate_loss(
    model,
    get_batch,
    num_batches: int = 50,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Answers: how well does the model predict held-out text?

    Averages the model's own cross-entropy over `num_batches` validation batches and
    reports perplexity. Comparable across pre-training and SFT (same loss form).

    Args:
        model: a model whose ``forward(x, targets=y)`` returns ``(logits, loss)``.
        get_batch: callable yielding one ``(input_ids, target_ids)`` validation batch.
        num_batches: how many batches to average over.
        device: optional device to move batches to.

    Returns:
        ``{"loss": avg_ce, "perplexity": exp(avg_ce)}``.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = get_batch()
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        _, loss = model(x, targets=y)
        total += loss.item()
    model.train(was_training)

    avg = total / max(num_batches, 1)
    return {"loss": avg, "perplexity": math.exp(avg)}


def _encode_prompt_choice(
    tokenizer, prompt: str, choice: str, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Answers: how does one (prompt, candidate-answer) pair become an (input, target)
    pair where only the answer tokens are scored? (Mirrors DPODataset masking.)"""
    prompt_ids = tokenizer.encode(prompt)
    choice_ids = tokenizer.encode(choice)

    full = (prompt_ids + choice_ids)[: block_size + 1]
    x = full[:-1]
    y = full[1:]

    n_prompt = len(prompt_ids)
    y = [IGNORE_INDEX if (i + 1) < n_prompt else tok for i, tok in enumerate(y)]

    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )


@torch.no_grad()
def score_choices(
    model,
    tokenizer,
    prompt: str,
    choices: list[str],
    block_size: int,
    device: torch.device | None = None,
    length_normalize: bool = True,
) -> tuple[list[float], int]:
    """Answers: which candidate completion does the model find most likely?

    Scores each candidate answer by the (optionally length-normalized) log-probability
    the model assigns to its tokens given the prompt — the standard way to do
    multiple-choice eval without generation.

    Args:
        model: forward returns ``(logits, loss)`` when given targets.
        tokenizer: a :class:`~llm.tokenizer.tiktoken_wrapper.Tokenizer`.
        prompt: the shared question/context string.
        choices: candidate answer strings.
        block_size: max sequence length (prompt+choice is truncated to this).
        device: optional device to run on.
        length_normalize: divide each score by its token count (avoids favoring short
            answers). Disable to compare raw sequence log-probs.

    Returns:
        ``(scores, best_index)`` where scores[i] is the log-prob of choices[i].
    """
    was_training = model.training
    model.eval()

    scores: list[float] = []
    for choice in choices:
        x, y = _encode_prompt_choice(tokenizer, prompt, choice, block_size)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        logits, _ = model(x, targets=y)
        logp = sequence_logprob(logits, y).item()
        if length_normalize:
            n_answer = int((y != IGNORE_INDEX).sum().item())
            logp = logp / max(n_answer, 1)
        scores.append(logp)

    model.train(was_training)
    best = max(range(len(scores)), key=lambda i: scores[i])
    return scores, best


@torch.no_grad()
def evaluate_multiple_choice(
    model,
    tokenizer,
    examples: list[dict],
    block_size: int,
    device: torch.device | None = None,
    length_normalize: bool = True,
) -> dict[str, float]:
    """Answers: how often does the model pick the correct answer?

    Args:
        examples: list of ``{"prompt": str, "choices": [str, ...], "answer": int}``,
            where ``answer`` is the index of the correct choice.
        (other args as in :func:`score_choices`.)

    Returns:
        ``{"accuracy": fraction_correct, "n": num_examples}``.
    """
    correct = 0
    for example in examples:
        _, pred = score_choices(
            model,
            tokenizer,
            example["prompt"],
            example["choices"],
            block_size,
            device=device,
            length_normalize=length_normalize,
        )
        correct += int(pred == example["answer"])

    n = len(examples)
    return {"accuracy": correct / max(n, 1), "n": n}


@torch.no_grad()
def evaluate_preferences(
    policy,
    ref,
    get_batch,
    num_batches: int = 20,
    beta: float = 0.1,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Answers: on held-out preferences, does the policy prefer chosen over rejected,
    and how far has it drifted from the reference?

    Args:
        policy: the (DPO-trained) model under test.
        ref: the frozen reference model.
        get_batch: yields ``(chosen_x, chosen_y, rejected_x, rejected_y)`` batches.
        num_batches: batches to average over.
        beta: DPO temperature (only affects the reward-accuracy threshold scale).
        device: optional device to run on.

    Returns:
        ``{"reward_accuracy": ..., "logratio_drift": ...}`` where logratio_drift is the
        mean ``policy_logp - ref_logp`` over responses — a cheap KL-style drift signal
        (near 0 at the start of DPO; growing positive as the policy diverges).
    """
    policy.eval()
    ref.eval()

    total_acc = 0.0
    total_drift = 0.0
    for _ in range(num_batches):
        chosen_x, chosen_y, rejected_x, rejected_y = get_batch()
        if device is not None:
            chosen_x, chosen_y = chosen_x.to(device), chosen_y.to(device)
            rejected_x, rejected_y = rejected_x.to(device), rejected_y.to(device)

        policy_chosen = sequence_logprob(policy(chosen_x, targets=chosen_y)[0], chosen_y)
        policy_rejected = sequence_logprob(policy(rejected_x, targets=rejected_y)[0], rejected_y)
        ref_chosen = sequence_logprob(ref(chosen_x, targets=chosen_y)[0], chosen_y)
        ref_rejected = sequence_logprob(ref(rejected_x, targets=rejected_y)[0], rejected_y)

        _, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        total_acc += acc
        drift = 0.5 * (
            (policy_chosen - ref_chosen).mean() + (policy_rejected - ref_rejected).mean()
        )
        total_drift += drift.item()

    n = max(num_batches, 1)
    return {"reward_accuracy": total_acc / n, "logratio_drift": total_drift / n}
