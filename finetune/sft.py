"""Supervised fine-tuning (SFT) on instruction data, with prompt-token masking.

Instruction tuning trains the model to produce a *response* given a *prompt*. The key
detail is **loss masking**: we only compute the next-token loss over the response
tokens, not the prompt. We implement this by setting the target id to -1 (the
ignore_index used by the models' cross_entropy) for every prompt position.

Data format: a list of {"instruction": ..., "input": ..., "output": ...} dicts
(the Alpaca-style schema). :func:`format_example` renders them with a simple chat
template; swap in a model-specific template for real checkpoints.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

# Simple instruction template. CUSTOMIZE: replace with the target model's exact chat
# template (e.g. LLaMA-3's <|begin_of_text|>/<|start_header_id|> tokens) for best results.
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_example(example: dict) -> tuple[str, str]:
    """Split an Alpaca-style example into (prompt, response) strings."""
    if example.get("input"):
        prompt = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"], input=example["input"]
        )
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=example["instruction"])
    response = example["output"]
    return prompt, response


class SFTDataset(Dataset):
    """Tokenized instruction dataset with prompt tokens masked out of the loss.

    Each item is (input_ids, target_ids) of length block_size:
    * input_ids  — prompt + response (+ EOT), right-padded with pad_id.
    * target_ids — input shifted by one; prompt positions and padding are -1.

    Args:
        examples: list of Alpaca-style dicts.
        tokenizer: a :class:`~llm.tokenizer.tiktoken_wrapper.Tokenizer`.
        block_size: fixed sequence length (truncate/pad to this).
        pad_id: token id used for padding inputs (its targets are always masked).
    """

    IGNORE_INDEX = -1  # must match the models' cross_entropy ignore_index

    def __init__(self, examples: list[dict], tokenizer, block_size: int, pad_id: int = 0):
        self.examples = examples
        self.tok = tokenizer
        self.block_size = block_size
        self.pad_id = pad_id
        self.eot = tokenizer.eot_token

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, response = format_example(self.examples[idx])
        prompt_ids = self.tok.encode(prompt)
        response_ids = self.tok.encode(response) + [self.eot]

        # Full sequence = prompt followed by response.
        full = prompt_ids + response_ids
        full = full[: self.block_size + 1]  # +1 because we shift for next-token targets

        x = full[:-1]
        y = full[1:]

        # Mask the loss over prompt tokens: a target at position t is "predicting"
        # token t+1. Targets whose *predicted* token lies in the prompt are masked.
        n_prompt = len(prompt_ids)
        y = [
            self.IGNORE_INDEX if (i + 1) < n_prompt else tok
            for i, tok in enumerate(y)
        ]

        # Right-pad to block_size; padded targets are ignored.
        pad = self.block_size - len(x)
        if pad > 0:
            x = x + [self.pad_id] * pad
            y = y + [self.IGNORE_INDEX] * pad

        return (
            torch.tensor(x[: self.block_size], dtype=torch.long),
            torch.tensor(y[: self.block_size], dtype=torch.long),
        )


def make_collate_fn():
    """Return a collate fn that stacks (x, y) pairs into batched tensors."""

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)

    return collate
