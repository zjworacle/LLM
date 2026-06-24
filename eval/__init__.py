"""Evaluation harness: held-out loss/perplexity, multiple-choice accuracy, and DPO
reward-accuracy/drift — metrics comparable across pre-training, SFT, and DPO."""

from .harness import (
    evaluate_loss,
    evaluate_multiple_choice,
    evaluate_preferences,
    score_choices,
)

__all__ = [
    "evaluate_loss",
    "evaluate_multiple_choice",
    "evaluate_preferences",
    "score_choices",
]
