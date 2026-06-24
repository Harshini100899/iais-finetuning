"""Unit tests for evaluation inference routines."""

from unittest.mock import MagicMock

import torch

from text2cypher.evaluate import generate_cypher


def test_generate_cypher():
    model = MagicMock()
    tokenizer = MagicMock()

    tokenizer.apply_chat_template.return_value = "Mock Prompt"

    # Mock inputs from tokenizer to support .to(device) and shape checking
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    # Mock input_ids shape to be (1, 3) so that prompt_len is 3
    mock_inputs.__getitem__.return_value.shape = (1, 3)
    tokenizer.return_value = mock_inputs
    tokenizer.eos_token_id = 5

    model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11]])
    tokenizer.decode.return_value = "  MATCH (n) RETURN n  "

    generated = generate_cypher(
        model=model,
        tokenizer=tokenizer,
        question="What is the data?",
        schema="Node {name}",
        device=torch.device("cpu"),
        max_new_tokens=10,
        num_beams=1,
    )

    assert generated == "MATCH (n) RETURN n"
    model.generate.assert_called_once()

    # Verify tokenizer.decode args via torch.equal to avoid mock tensor compares
    tokenizer.decode.assert_called_once()
    args, kwargs = tokenizer.decode.call_args
    assert torch.equal(args[0], torch.tensor([10, 11]))
    assert kwargs == {"skip_special_tokens": True}
