"""Load real pretrained weights into our from-scratch models.

Currently supports **GPT-2** (OpenAI / HuggingFace openai-community/gpt2 family),
whose architecture matches our :class:`~llm.models.gpt3.GPT3` exactly: learned absolute
position embeddings, LayerNorm, GELU MLP, tied embeddings, biases everywhere.

How to get the weights
----------------------
Download a model.safetensors for a GPT-2 checkpoint, e.g. from
https://huggingface.co/openai-community/gpt2 (or gpt2-medium/large/xl), then::

    from llm.infer.load_weights import build_gpt2, load_gpt2_safetensors
    model = build_gpt2("gpt2")
    load_gpt2_safetensors(model, "model.safetensors")

Key mapping notes
-----------------
* HF GPT-2 packs Q/K/V into a single c_attn projection; we split it into our
  separate q_proj/k_proj/v_proj.
* HF GPT-2 uses Conv1D layers whose weight is stored (in, out) — the transpose
  of an nn.Linear weight (out, in) — so all projection weights are transposed.
* unembed is tied to token_emb (no separate tensor needed).

# IMPROVE: GPT-2 uses the tanh ("new") GELU approximation; our GeluMLP uses exact GELU.
# For bit-level parity, switch nn.GELU(approximate="tanh") in common.GeluMLP.
"""

from __future__ import annotations

import torch

from ..models.gpt3 import GPT3, GPT3Config

# Standard GPT-2 sizes (vocab 50257, block 1024 for all).
GPT2_CONFIGS = {
    "gpt2": dict(n_layers=12, dim=768, n_heads=12),
    "gpt2-medium": dict(n_layers=24, dim=1024, n_heads=16),
    "gpt2-large": dict(n_layers=36, dim=1280, n_heads=20),
    "gpt2-xl": dict(n_layers=48, dim=1600, n_heads=25),
}


def build_gpt2(model_name: str = "gpt2") -> GPT3:
    """Construct a GPT3 model with the architecture of a named GPT-2 checkpoint."""
    if model_name not in GPT2_CONFIGS:
        raise KeyError(f"unknown GPT-2 size {model_name!r}; choices: {list(GPT2_CONFIGS)}")
    spec = GPT2_CONFIGS[model_name]
    cfg = GPT3Config(
        vocab_size=50257,
        block_size=1024,
        n_layers=spec["n_layers"],
        dim=spec["dim"],
        n_heads=spec["n_heads"],
        dropout=0.0,
        use_bias=True,
        tie_embeddings=True,
    )
    return GPT3(cfg)


def _load_safetensors_file(path: str) -> dict[str, torch.Tensor]:
    """Read a .safetensors file into a flat {name: tensor} dict (lazy import)."""
    from safetensors.torch import load_file

    return load_file(path)


def _strip_prefix(key: str) -> str:
    """Drop a leading transformer. if present (HF naming varies by export)."""
    return key[len("transformer.") :] if key.startswith("transformer.") else key


def convert_gpt2_state_dict(hf: dict[str, torch.Tensor], cfg: GPT3Config) -> dict[str, torch.Tensor]:
    """Map a HuggingFace GPT-2 state dict onto our GPT3 state_dict keys."""
    hf = {_strip_prefix(k): v for k, v in hf.items()}
    dim = cfg.dim
    out: dict[str, torch.Tensor] = {}

    # Embeddings.
    out["token_emb.weight"] = hf["wte.weight"]
    out["pos_emb.weight"] = hf["wpe.weight"]
    # Final norm.
    out["final_norm.weight"] = hf["ln_f.weight"]
    out["final_norm.bias"] = hf["ln_f.bias"]

    # These Conv1D weights are stored (in, out) and must be transposed to (out, in).
    transpose_keys = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}

    for i in range(cfg.n_layers):
        p = f"h.{i}."
        b = f"layers.{i}."

        # LayerNorms (attn_norm before attn, mlp_norm before mlp).
        out[b + "attn_norm.weight"] = hf[p + "ln_1.weight"]
        out[b + "attn_norm.bias"] = hf[p + "ln_1.bias"]
        out[b + "mlp_norm.weight"] = hf[p + "ln_2.weight"]
        out[b + "mlp_norm.bias"] = hf[p + "ln_2.bias"]

        # Attention: split the fused c_attn into q/k/v.
        c_attn_w = hf[p + "attn.c_attn.weight"].t()  # (3*dim, dim)
        c_attn_b = hf[p + "attn.c_attn.bias"]  # (3*dim,)
        qw, kw, vw = c_attn_w.split(dim, dim=0)
        qb, kb, vb = c_attn_b.split(dim, dim=0)
        out[b + "attn.q_proj.weight"] = qw
        out[b + "attn.q_proj.bias"] = qb
        out[b + "attn.k_proj.weight"] = kw
        out[b + "attn.k_proj.bias"] = kb
        out[b + "attn.v_proj.weight"] = vw
        out[b + "attn.v_proj.bias"] = vb
        out[b + "attn.o_proj.weight"] = hf[p + "attn.c_proj.weight"].t()
        out[b + "attn.o_proj.bias"] = hf[p + "attn.c_proj.bias"]

        # MLP (up_proj then down_proj).
        out[b + "mlp.up_proj.weight"] = hf[p + "mlp.c_fc.weight"].t()
        out[b + "mlp.up_proj.bias"] = hf[p + "mlp.c_fc.bias"]
        out[b + "mlp.down_proj.weight"] = hf[p + "mlp.c_proj.weight"].t()
        out[b + "mlp.down_proj.bias"] = hf[p + "mlp.c_proj.bias"]

    # unembed is tied to token_emb in our model, so no separate tensor is needed.
    _ = transpose_keys  # documented above; kept for clarity
    return out


def load_gpt2_safetensors(model: GPT3, path: str, strict: bool = False) -> GPT3:
    """Load GPT-2 weights from a .safetensors file into model (in place)."""
    hf = _load_safetensors_file(path)
    mapped = convert_gpt2_state_dict(hf, model.config)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # The tied unembed.weight is expected to be "missing" (it aliases token_emb).
    real_missing = [m for m in missing if not m.startswith("unembed")]
    if strict and (real_missing or unexpected):
        raise RuntimeError(f"weight load mismatch: missing={real_missing} unexpected={unexpected}")
    print(
        f"[load_weights] loaded GPT-2 weights ({model.config.n_layers} layers); "
        f"missing={len(real_missing)} unexpected={len(unexpected)}"
    )
    return model
