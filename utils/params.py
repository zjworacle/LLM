"""Parameter counting and a component-level breakdown.

These helpers work for any of the from-scratch models (GPT3, Llama3): each named
parameter is bucketed into a component (embedding / attention / mlp / norm /
unembed) by matching the submodule names used in llm.models.common.

The "non-embedding" total is the quantity scaling laws care about (often written N):
the parameters inside the transformer blocks (attention + mlp + norms), excluding
the token/position embedding and the unembedding projection. Those embedding
matrices scale with vocab_size rather than depth/width, so they are excluded when
relating model size to loss.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch.nn as nn

# Ordered so the most specific module names are matched first. Each entry maps a
# component label to the substrings that identify it within a parameter's name.
_COMPONENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
    ("unembed", ("unembed",)),
    ("embedding", ("token_emb", "pos_emb")),
)

# Components excluded from the non-embedding (scaling-law N) total.
_EMBEDDING_COMPONENTS = ("embedding", "unembed")


def _classify(param_name: str) -> str:
    for component, needles in _COMPONENT_RULES:
        if any(needle in param_name for needle in needles):
            return component
    return "norm"  # LayerNorm / RMSNorm weights and biases


@dataclass
class ParamBreakdown:
    """Parameter counts grouped by component, plus convenience totals."""

    by_component: "OrderedDict[str, int]"

    def total(self) -> int:
        return sum(self.by_component.values())

    def non_embedding(self) -> int:
        """N for scaling laws: everything except embedding + unembedding matrices."""
        return sum(
            count
            for component, count in self.by_component.items()
            if component not in _EMBEDDING_COMPONENTS
        )

    def pretty(self) -> str:
        lines = []
        total = self.total()
        for component, count in self.by_component.items():
            share = 100.0 * count / total if total else 0.0
            lines.append(f"  {component:<10} {count:>14,}  ({share:5.1f}%)")
        lines.append(f"  {'total':<10} {total:>14,}")
        lines.append(f"  {'non-embed':<10} {self.non_embedding():>14,}")
        return "\n".join(lines)


def count_parameters(model: nn.Module) -> ParamBreakdown:
    """Group a model's parameters by component.

    Tied weights (e.g. GPT-3's unembed aliasing token_emb) are counted once:
    named_parameters() deduplicates shared tensors, keeping the first-registered name.
    """
    by_component: "OrderedDict[str, int]" = OrderedDict()
    for param_name, param in model.named_parameters():
        component = _classify(param_name)
        by_component[component] = by_component.get(component, 0) + param.numel()
    return ParamBreakdown(by_component)


# Tokens-per-parameter presets for common training regimes.
#   "chinchilla"   ~20x  -> minimizes *training* compute only (Hoffmann et al., 2022).
#   "overtrained"  ~150x -> trade extra training compute for cheaper inference
#                           (Llama-2 7B sits near here, ~285x).
#   "llama3"       ~2000x -> the heavily over-trained deployment regime
#                           (Llama-3 8B saw ~15T tokens, ~1875 tok/param).
TOKENS_PER_PARAM_REGIMES = {
    "chinchilla": 20,
    "overtrained": 150,
    "llama3": 2000,
}


def chinchilla_optimal_tokens(
    non_embedding_params: int,
    tokens_per_param: int | str = 20,
) -> int:
    """Estimate a training-token budget for a given non-embedding parameter count.

    The default 20 tok/param is the Chinchilla compute-*training*-optimal point: it
    minimizes loss for a fixed training FLOP budget, and is best read as a *lower
    bound* on how much data to use. It deliberately ignores inference cost.

    In practice deployed models are over-trained well past this — you spend more
    training compute once to get a smaller model that is permanently cheaper to
    serve (Llama-3 8B used ~100x the Chinchilla token count). Pass a regime name
    from TOKENS_PER_PARAM_REGIMES ("overtrained", "llama3") or any int to model
    that.
    """
    if isinstance(tokens_per_param, str):
        tokens_per_param = TOKENS_PER_PARAM_REGIMES[tokens_per_param]
    return non_embedding_params * tokens_per_param
