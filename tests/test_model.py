"""Unit tests for model loading helpers, LoRA wrappers, and optimizer setup."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from text2cypher.model import get_optimizer_and_scheduler, get_peft_lora_model


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.bias = nn.Parameter(torch.randn(2))


def test_get_optimizer_and_scheduler():
    model = DummyModel()
    model.weight.requires_grad = False

    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, lr=2e-4, weight_decay=0.01, warmup_steps=10, total_update_steps=100
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    opt_params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(opt_params) == 1
    assert opt_params[0] is model.bias


def test_get_peft_lora_model():
    base_model = MagicMock()
    with patch("text2cypher.model.get_peft_model") as mock_get_peft:
        get_peft_lora_model(base_model, rank=8, dropout=0.1)
        mock_get_peft.assert_called_once()
        config = mock_get_peft.call_args[0][1]
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert set(config.target_modules) == set(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
