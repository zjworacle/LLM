"""Fine-tuning: LoRA adapters, instruction/SFT training, DPO alignment, GRPO RL."""

from .dapo import DAPOTrainer, dapo_loss, has_reward_variance
from .dpo import DPODataset, DPOTrainer, dpo_loss, sequence_logprob
from .grpo import (
    GRPOTrainer,
    ReasoningPromptDataset,
    correctness_reward,
    extract_answer,
    format_reward,
    grpo_loss,
    group_advantages,
    make_reasoning_reward,
    render_reasoning_prompt,
    reward_model_scorer,
)
from .gspo import GSPOTrainer, gspo_loss
from .lora import (
    LoRALinear,
    apply_lora,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora,
    unmerge_lora,
)
from .sft import SFTDataset, format_example

__all__ = [
    "LoRALinear",
    "apply_lora",
    "lora_state_dict",
    "mark_only_lora_trainable",
    "merge_lora",
    "unmerge_lora",
    "SFTDataset",
    "format_example",
    "DPODataset",
    "DPOTrainer",
    "dpo_loss",
    "sequence_logprob",
    "GRPOTrainer",
    "ReasoningPromptDataset",
    "correctness_reward",
    "extract_answer",
    "format_reward",
    "grpo_loss",
    "group_advantages",
    "make_reasoning_reward",
    "render_reasoning_prompt",
    "reward_model_scorer",
    "DAPOTrainer",
    "dapo_loss",
    "has_reward_variance",
    "GSPOTrainer",
    "gspo_loss",
]
