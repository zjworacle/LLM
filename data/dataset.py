"""Datasets for language-model pretraining.

The core abstraction for pretraining is *packing*: concatenate all documents into one
long stream of token ids, then cut it into contiguous windows of block_size + 1
tokens. Each window yields:

* input  = tokens[i   : i + block_size]
* target = tokens[i+1 : i + block_size + 1]   (next-token prediction)

This is the simplest, most throughput-efficient layout and is what GPT-3 / LLaMA use
for the pretraining phase. (Instruction fine-tuning needs *masking* instead — that
lives in the SFT module.)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def tokens_from_text(text: str, encoding: str = "gpt2") -> torch.Tensor:
    """Tokenize raw text into a 1-D LongTensor of token ids using tiktoken.

    Imported lazily so the tokenizer dependency is only needed when actually used.
    """
    from llm.tokenizer.tiktoken_wrapper import Tokenizer

    tok = Tokenizer(encoding=encoding)
    return torch.tensor(tok.encode(text), dtype=torch.long)


class PackedDataset(Dataset):
    """Fixed-length windows over a 1-D token stream for next-token prediction.

    Args:
        tokens: a 1-D LongTensor of token ids (the packed corpus).
        block_size: context length; each item is block_size input tokens.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        assert tokens.dim() == 1, "tokens must be a 1-D tensor of ids"
        assert tokens.numel() > block_size, "need more tokens than block_size"
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of windows whose target stays in-bounds.
        return self.tokens.numel() - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:].clone()
        return x, y


def random_token_batch(
    vocab_size: int,
    batch_size: int,
    block_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a random (input, target) batch — handy for smoke tests / benchmarks.

    # CUSTOMIZE: replace this with a real DataLoader over PackedDataset for actual runs.
    """
    x = torch.randint(0, vocab_size, (batch_size, block_size), generator=generator)
    y = torch.randint(0, vocab_size, (batch_size, block_size), generator=generator)
    return x.to(device), y.to(device)
