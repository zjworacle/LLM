"""Autoregressive text generation with KV-cache decoding and sampling controls.

Supports greedy decoding plus temperature / top-k / top-p (nucleus) sampling. Uses the
model's per-layer KV cache so each new token costs one forward over a single position
rather than re-encoding the whole prefix.

The models return only the *last* position's logits at inference time (their fast
path), which is exactly what we need for incremental decoding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all but the top_k highest-probability logits (per row)."""
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative prob >= top_p."""
    if not (0.0 < top_p < 1.0):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    # Remove tokens once cumulative prob exceeds top_p, but always keep the first.
    remove = cumsum - probs > top_p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back to the original vocab order.
    return sorted_logits.gather(-1, sorted_idx.argsort(dim=-1))


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    eot_token: int | None = None,
    use_cache: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate max_new_tokens continuations for each row of idx.

    Args:
        model: a GPT3/Llama3 instance (must expose forward and init_caches).
        idx: prompt token ids, shape (batch, prompt_len).
        max_new_tokens: number of tokens to append.
        temperature: softmax temperature; 0 (or very small) => greedy argmax.
        top_k: keep only the top-k logits before sampling (0 disables).
        top_p: nucleus threshold in (0,1) (0 disables).
        eot_token: if all sequences emit this token, stop early.
        use_cache: use the KV cache for fast incremental decoding.

    Returns:
        Tensor of shape (batch, prompt_len + generated) of token ids.
    """
    model.eval()
    device = device or next(model.parameters()).device
    idx = idx.to(device)
    block_size = model.config.block_size

    caches = None
    if use_cache:
        dtype = next(model.parameters()).dtype
        caches = model.init_caches(idx.size(0), device=device, dtype=dtype)
        # Prime the cache with the full prompt in one forward.
        logits, _ = model(idx, caches=caches)
        next_logits = logits[:, -1, :]

    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        if not use_cache:
            # No cache: re-feed the (cropped) running sequence each step.
            cond = idx[:, -block_size:]
            logits, _ = model(cond)
            next_logits = logits[:, -1, :]
        elif step > 0:
            # Cache path: feed only the single previous token.
            logits, _ = model(idx[:, -1:], caches=caches)
            next_logits = logits[:, -1, :]
        # (for step 0 with cache, next_logits came from the prompt prime above)

        # Sampling transforms.
        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = next_logits / temperature
            scaled = _apply_top_k(scaled, top_k)
            scaled = _apply_top_p(scaled, top_p)
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Once a sequence has finished, keep appending the eot token.
        if eot_token is not None:
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eot_token),
                next_token,
            )
            finished = finished | (next_token.squeeze(1) == eot_token)

        idx = torch.cat([idx, next_token], dim=1)

        if eot_token is not None and bool(finished.all()):
            break
        if idx.size(1) >= block_size and not use_cache:
            # Without a cache we crop; with a cache we'd overflow the buffer, so stop.
            pass
        if use_cache and idx.size(1) >= block_size:
            break

    return idx


@torch.no_grad()
def generate_text(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 64,
    **kwargs,
) -> str:
    """Convenience wrapper: encode prompt, generate, and decode back to text."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out = generate(
        model,
        ids,
        max_new_tokens,
        eot_token=kwargs.pop("eot_token", tokenizer.eot_token),
        **kwargs,
    )
    return tokenizer.decode(out[0].tolist())
