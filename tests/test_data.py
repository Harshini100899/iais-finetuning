"""Unit tests for data loading, schema normalization, prompts, and collation."""

import torch

from text2cypher.data import (
    _compact_schema,
    _normalize_schema,
    build_full_messages,
    build_messages,
    collate_fn,
)


def test_compact_schema_verbose():
    verbose_schema = (
        "- **User**\n"
        "- `name`: STRING\n"
        "- `age`: INTEGER\n"
        "The relationships:\n"
        "(:User)-[:FRIEND]->(:User)"
    )
    compacted = _compact_schema(verbose_schema)
    assert "User {name, age}" in compacted
    assert "(:User)-[:FRIEND]->(:User)" in compacted


def test_normalize_schema():
    raw_schema = "Node properties: \n\n\n\nMovie {title: STRING}"
    normalized = _normalize_schema(raw_schema, "neo4j_custom")
    assert "\n\n\n\n" not in normalized
    assert normalized.startswith("Node properties:")


def test_build_messages():
    messages = build_messages("What is Alice's age?", "User {name, age}")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Alice's age" in messages[1]["content"]


def test_build_full_messages():
    messages = build_full_messages(
        "What is Alice's age?", "User {name, age}", "MATCH (n) RETURN n"
    )
    assert len(messages) == 3
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "MATCH (n) RETURN n"


def test_collate_fn():
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([4, 5]),
        },
    ]
    collated = collate_fn(batch, pad_token_id=0)

    assert collated["input_ids"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    assert collated["labels"].shape == (2, 3)

    assert collated["input_ids"][1, 2].item() == 0
    assert collated["attention_mask"][1, 2].item() == 0
    assert collated["labels"][1, 2].item() == -100
