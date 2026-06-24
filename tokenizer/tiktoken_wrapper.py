"""A thin wrapper around tiktoken — the most widely used BPE tokenizer.

We standardize on tiktoken because it is the de-facto tokenizer in the market: it ships
the exact byte-pair-encoding vocabularies used by GPT-2/3/4. Using it means our token
ids line up with real OpenAI checkpoints, which matters when we later load pretrained
weights.

tiktoken is imported lazily (inside __init__) so that pure-architecture code and
tests that do not need tokenization can run without it installed.

# CUSTOMIZE: a from-scratch byte-level BPE implementation can be added alongside this
# wrapper for educational purposes; this class is the practical default.
"""

from __future__ import annotations


class Tokenizer:
    """Encode/decode text using a named tiktoken encoding.

    Args:
        encoding: tiktoken encoding name.
            * "gpt2"      -> 50257-token vocab (matches our GPT-3 default).
            * "cl100k_base" -> GPT-4 / ChatGPT vocab (~100k).
            * "o200k_base"  -> newest OpenAI vocab (~200k).
            # CUSTOMIZE: pick the encoding that matches the checkpoint you target.
    """

    def __init__(self, encoding: str = "gpt2"):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "tiktoken is required for tokenization. Install it with "
                "`pip install tiktoken`."
            ) from exc

        self._enc = tiktoken.get_encoding(encoding)
        self.encoding_name = encoding

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary.

        Note: this is the encoding's base vocab size. If you add custom special tokens
        you should size the model's embedding to max(token_id) + 1 instead.
        """
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        """The end-of-text token id used to separate documents during packing."""
        return self._enc.eot_token

    def encode(self, text: str, allowed_special: set[str] | str = set()) -> list[int]:
        """Encode a string to token ids.

        Args:
            allowed_special: which special tokens to honor in the input. Default is to
                treat everything as ordinary text (safest for untrusted input).
                # CUSTOMIZE: pass "all" to let <|endoftext|> etc. be parsed.
        """
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back to a string."""
        return self._enc.decode(tokens)
