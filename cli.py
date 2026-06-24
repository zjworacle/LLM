"""Unified command-line interface: llm <command> [options].

This is the console entry point declared in pyproject.toml (llm = "llm.cli:main").
It dispatches to three subcommands that mirror the standalone scripts:

* llm train    — pretrain a model (see :mod:`scripts.train`)
* llm finetune — instruction fine-tune, full or LoRA (see :mod:`scripts.finetune`)
* llm generate — sample text from a checkpoint or real GPT-2 weights

Examples
--------
    llm train --run-config tiny-gpt3 --data random --max-steps 50
    llm finetune --run-config tiny-gpt3 --lora --max-steps 100
    llm generate --run-config tiny-gpt3 --ckpt out/tiny_gpt3/final.pt --prompt "Hi"

Each subcommand reuses the same building blocks as the scripts. We keep this thin so
the scripts remain usable on their own (handy when the package isn't installed).
"""

from __future__ import annotations

import argparse

import torch

from .infer.generate import generate_text
from .tokenizer.tiktoken_wrapper import Tokenizer
from .train.config import TrainConfig
from .train.trainer import Trainer
from .utils.device import pick_device


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_model(run_config: str, cfg):
    # Dispatch by run-config name: "llama4" -> MoE LLaMA-4, other "llama" -> LLaMA-3,
    # everything else -> GPT-3.
    name = run_config.lower()
    if "llama4" in name:
        from .models.llama4 import Llama4

        return Llama4(cfg)
    if "llama" in name:
        from .models.llama3 import Llama3

        return Llama3(cfg)
    from .models.gpt3 import GPT3

    return GPT3(cfg)


def _random_batch_fn(model_cfg, batch_size: int):
    from .data.dataset import random_token_batch

    def get_batch():
        return random_token_batch(
            model_cfg.vocab_size, batch_size, model_cfg.block_size, device="cpu"
        )

    return get_batch


def _text_batch_fn(model_cfg, batch_size: int, path: str, encoding: str, seed: int):
    from pathlib import Path

    from .data.dataset import PackedDataset, tokens_from_text

    tokens = tokens_from_text(Path(path).read_text(encoding="utf-8"), encoding=encoding)
    dataset = PackedDataset(tokens, model_cfg.block_size)
    gen = torch.Generator().manual_seed(seed)

    def get_batch():
        idx = torch.randint(0, len(dataset), (batch_size,), generator=gen)
        xs, ys = zip(*(dataset[i] for i in idx.tolist()))
        return torch.stack(xs), torch.stack(ys)

    return get_batch


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_train(args) -> None:
    from .runconfigs import get_run_config

    model_cfg, train_cfg = get_run_config(args.run_config)
    train_cfg.max_steps = args.max_steps
    train_cfg.batch_size = args.batch_size
    train_cfg.use_fp8 = args.fp8
    if args.precision:
        train_cfg.precision = args.precision
    if args.device:
        train_cfg.device = args.device
    if args.out_dir:
        train_cfg.out_dir = args.out_dir

    model = _build_model(args.run_config, model_cfg)
    if args.data == "random":
        get_batch = _random_batch_fn(model_cfg, train_cfg.batch_size)
    else:
        get_batch = _text_batch_fn(
            model_cfg, train_cfg.batch_size, args.data, args.encoding, train_cfg.seed
        )
    Trainer(model, train_cfg, get_batch).train()


def cmd_finetune(args) -> None:
    import json
    from pathlib import Path

    from .runconfigs import get_run_config
    from .finetune.lora import apply_lora, lora_state_dict
    from .finetune.sft import SFTDataset

    model_cfg, train_cfg = get_run_config(args.run_config)
    train_cfg.max_steps = args.max_steps
    train_cfg.batch_size = args.batch_size
    train_cfg.out_dir = args.out_dir or "out/sft"
    if args.lora:
        train_cfg.lr = 1e-3

    model = _build_model(args.run_config, model_cfg)
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    if args.lora:
        apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    tok = Tokenizer(encoding="gpt2")
    if args.data:
        examples = json.loads(Path(args.data).read_text(encoding="utf-8"))
    else:
        examples = [
            {"instruction": "Say hello.", "input": "", "output": "Hello!"},
            {"instruction": "What is 2+2?", "input": "", "output": "4."},
        ]
    dataset = SFTDataset(examples, tok, block_size=model_cfg.block_size)
    gen = torch.Generator().manual_seed(train_cfg.seed)

    def get_batch():
        idx = torch.randint(0, len(dataset), (train_cfg.batch_size,), generator=gen)
        xs, ys = zip(*(dataset[i] for i in idx.tolist()))
        return torch.stack(xs), torch.stack(ys)

    Trainer(model, train_cfg, get_batch).train()
    if args.lora:
        path = Path(train_cfg.out_dir) / "lora_adapters.pt"
        torch.save(lora_state_dict(model), path)
        print(f"[finetune] saved LoRA adapters -> {path}")


def cmd_generate(args) -> None:
    device = pick_device(args.device)
    if args.gpt2:
        from .infer.load_weights import build_gpt2, load_gpt2_safetensors

        model = build_gpt2(args.gpt2)
        if args.weights:
            load_gpt2_safetensors(model, args.weights)
        tok = Tokenizer(encoding="gpt2")
    else:
        from .runconfigs import get_run_config

        model_cfg, _ = get_run_config(args.run_config)
        model = _build_model(args.run_config, model_cfg)
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        tok = Tokenizer(encoding=args.encoding)

    model.to(device).eval()
    print(
        generate_text(
            model,
            args.prompt,
            tok,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm", description="GPT-3 / LLaMA-3 toolkit (MPS-first).")
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="pretrain a model")
    t.add_argument("--run-config", default="tiny-gpt3")
    t.add_argument("--data", default="random", help="'random' or path to a text file")
    t.add_argument("--encoding", default="gpt2")
    t.add_argument("--max-steps", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--precision", default=None)
    t.add_argument("--fp8", action="store_true")
    t.add_argument("--device", default=None)
    t.add_argument("--out-dir", default=None)
    t.set_defaults(func=cmd_train)

    # finetune
    f = sub.add_parser("finetune", help="instruction fine-tune (full or LoRA)")
    f.add_argument("--run-config", default="tiny-gpt3")
    f.add_argument("--data", default=None, help="JSON instruction file")
    f.add_argument("--init", default=None, help="base checkpoint to load")
    f.add_argument("--lora", action="store_true")
    f.add_argument("--lora-r", type=int, default=8)
    f.add_argument("--lora-alpha", type=int, default=16)
    f.add_argument("--max-steps", type=int, default=100)
    f.add_argument("--batch-size", type=int, default=4)
    f.add_argument("--out-dir", default=None)
    f.set_defaults(func=cmd_finetune)

    # generate
    g = sub.add_parser("generate", help="sample text from a model")
    g.add_argument("--prompt", default="Hello")
    g.add_argument("--max-new-tokens", type=int, default=50)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.0)
    g.add_argument("--run-config", default="tiny-gpt3")
    g.add_argument("--ckpt", default=None)
    g.add_argument("--gpt2", default=None, help="gpt2|gpt2-medium|gpt2-large|gpt2-xl")
    g.add_argument("--weights", default=None, help="path to GPT-2 model.safetensors")
    g.add_argument("--encoding", default="gpt2")
    g.add_argument("--device", default=None)
    g.set_defaults(func=cmd_generate)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
