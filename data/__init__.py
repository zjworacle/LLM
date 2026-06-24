"""Data loading utilities.

Two entry points:
* :class:`PackedDataset` — for pretraining: a long 1-D stream of token ids sliced into
  fixed-length (input, target) windows.
* :func:`tokens_from_text` — tokenize raw text into a token tensor with the project
  tokenizer (tiktoken).
"""

from .dataset import PackedDataset, random_token_batch, tokens_from_text

__all__ = ["PackedDataset", "random_token_batch", "tokens_from_text"]
