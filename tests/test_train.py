"""Unit tests for training step logic and epoch routines."""

from unittest.mock import MagicMock

import torch

from text2cypher.train import compute_loss, run_epoch


def test_compute_loss():
    model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.loss = torch.tensor(1.5)
    model.return_value = mock_outputs

    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[1, 2]]),
    }

    loss = compute_loss(model, batch)
    assert loss.item() == 1.5
    model.assert_called_once_with(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )


def test_run_epoch_eval():
    model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.loss = torch.tensor(1.2)
    model.return_value = mock_outputs

    loader = [
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": torch.tensor([[1, 2]]),
        }
    ]

    optimizer = MagicMock()
    scheduler = MagicMock()

    avg_loss = run_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accum=1,
        device=torch.device("cpu"),
        train=False,
    )

    assert abs(avg_loss - 1.2) < 1e-6
    optimizer.step.assert_not_called()
