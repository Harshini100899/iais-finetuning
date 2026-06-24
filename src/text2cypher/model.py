"""Model loading, PEFT/LoRA wrapping, and optimizer/scheduler configuration."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import MODEL_NAME
from .data import get_tokenizer


def load_model_and_tokenizer(checkpoint: str, device: torch.device) -> tuple:
    """
    Load base/fine-tuned model and tokenizer from checkpoint or fallback to base model.
    """
    print("\nLoading tokenizer ...")
    try:
        if checkpoint == "base":
            raise FileNotFoundError
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    except Exception:
        tokenizer = get_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {MODEL_NAME} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    if checkpoint == "base":
        print("Mode: BASE MODEL (no LoRA adapter)")
        model = base_model
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_path}\n"
                f"Run train.py first, or pass --checkpoint base for the base model."
            )
        print(f"Mode: FINE-TUNED MODEL  adapter={checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        model = model.merge_and_unload()

    model.to(device)
    model.eval()
    return model, tokenizer


def get_peft_lora_model(
    base_model,
    rank: int = 16,
    dropout: float = 0.05,
) -> PeftModel:
    """Wrap a causal LM base model in a PeftModel adapter."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    return model


def get_optimizer_and_scheduler(
    model,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    total_update_steps: int,
) -> tuple:
    """Create optimizer and schedule for LoRA parameter tuning."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    return optimizer, scheduler
