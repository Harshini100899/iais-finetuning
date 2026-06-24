"""Preprocessing, prompt formatting, tokenization, and dataset loading."""

from __future__ import annotations

import re

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from .config import (
    CYSPIDER_PREFIX,
    DATASET_NAME,
    MAX_LENGTH,
    MODEL_NAME,
    get_system_prompt,
)


def _compact_schema(schema: str) -> str:
    """
    Compact a verbose schema into a minimal structural representation:
    User {label, area, size}
    (:User)-[:INTERACTED]->(:User)
    """
    lines = schema.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    compacted: list[str] = []
    current_node: str | None = None
    node_props: list[str] = []

    for line in lines:
        line_s = line.strip()

        # Match node/relationship label header: - **NodeLabel**
        m_node = re.match(r"-\s*\*\*([^*]+)\*\*", line_s)
        if m_node:
            if current_node:
                props_str = ", ".join(node_props)
                compacted.append(f"{current_node} {{{props_str}}}")
            current_node = m_node.group(1)
            node_props = []
            continue

        # Match property: - `label`: STRING ...
        if current_node:
            m_prop = re.match(r"-\s*`([^`:]+)", line_s)
            if m_prop:
                prop_name = m_prop.group(1).strip()
                node_props.append(prop_name)

        # Stop collecting node properties when we hit relationships
        if line_s.startswith("Relationship properties:") or line_s.startswith(
            "The relationships:"
        ):
            if current_node:
                props_str = ", ".join(node_props)
                compacted.append(f"{current_node} {{{props_str}}}")
                current_node = None

    if current_node:
        props_str = ", ".join(node_props)
        compacted.append(f"{current_node} {{{props_str}}}")

    # Extract relationship triplets
    for line in lines:
        line_s = line.strip()
        if line_s.startswith("(") and "-[" in line_s and "]-" in line_s:
            compacted.append(line_s)

    if not compacted:
        return schema.strip()

    return "\n".join(compacted)


def _normalize_schema(schema: str, data_source: str) -> str:
    """Convert any of the observed schema formats into one canonical text block."""
    if not schema or not schema.strip():
        return "Schema: (not provided)"

    if data_source.startswith("neo4jLabs_synthetic_") or data_source.startswith(
        "neo4j_rageval_"
    ):
        return _compact_schema(schema)

    s = schema.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in s.split("\n")]

    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def build_messages(
    question: str, schema: str, prompt_variant: str = "default"
) -> list[dict[str, str]]:
    """Build the chat messages list for apply_chat_template (without assistant turn)."""
    user_content = f"Graph Schema:\n{schema}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": get_system_prompt(prompt_variant)},
        {"role": "user", "content": user_content},
    ]


def build_full_messages(
    question: str, schema: str, cypher: str, prompt_variant: str = "default"
) -> list[dict[str, str]]:
    """Build the full chat messages list including the assistant (Cypher) turn."""
    messages = build_messages(question, schema, prompt_variant)
    messages.append({"role": "assistant", "content": cypher})
    return messages


def tokenize_and_mask(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
    prompt_variant: str = "default",
) -> dict:
    """Tokenize one example and apply loss masking to system/user prompt tokens."""
    question = example["question"]
    schema = _normalize_schema(example["schema"], example.get("data_source", ""))
    cypher = example["cypher"].strip()

    full_messages = build_full_messages(question, schema, cypher, prompt_variant)
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix_messages = build_messages(question, schema, prompt_variant)
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    prefix_enc = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    input_ids = full_enc["input_ids"]
    prefix_len = len(prefix_enc["input_ids"])

    labels = [-100] * min(prefix_len, len(input_ids)) + input_ids[prefix_len:]
    assert len(labels) == len(input_ids), (
        f"Label/input length mismatch: {len(labels)} vs {len(input_ids)}"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """
    Pad input_ids, attention_mask, and labels to the longest sequence in the batch.
    Labels are padded with -100 so padding positions don't contribute to loss.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids_list, attn_mask_list, labels_list = [], [], []
    for ex in batch:
        seq_len = len(ex["input_ids"])
        pad_len = max_len - seq_len

        input_ids_list.append(ex["input_ids"].tolist() + [pad_token_id] * pad_len)
        attn_mask_list.append(ex["attention_mask"].tolist() + [0] * pad_len)
        labels_list.append(ex["labels"].tolist() + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }


def get_tokenizer() -> AutoTokenizer:
    """Load and configure the SmolLM2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_datasets(
    tokenizer: AutoTokenizer | None = None,
    max_length: int = MAX_LENGTH,
    overfit_n: int | None = None,
    verbose: bool = True,
    prompt_variant: str = "default",
) -> tuple[DatasetDict, AutoTokenizer]:
    """Load, filter, normalize, tokenize, and return the dataset splits."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Loading dataset: {DATASET_NAME}")
        print(f"{'=' * 60}")

    raw = load_dataset(DATASET_NAME)

    if verbose:
        print("\nSplit sizes (before filtering):")
        for split, ds in raw.items():
            print(f"  {split:15s}: {len(ds):>5d} rows")

    def _not_cyspider(example):
        return not example["data_source"].startswith(CYSPIDER_PREFIX)

    filtered = raw.filter(_not_cyspider, desc="Dropping cyspider rows")

    if verbose:
        print("\nAfter dropping cyspider rows:")
        for split, ds in filtered.items():
            before = len(raw[split])
            after = len(ds)
            dropped = before - after
            print(f"  {split:15s}: {after:>4d} kept, {dropped:>4d} dropped")

    if overfit_n is not None:
        if verbose:
            print(f"\nOverfit mode: using first {overfit_n} training examples only.")
        filtered["train"] = filtered["train"].select(
            range(min(overfit_n, len(filtered["train"])))
        )

    if verbose:
        print(f"\nTokenizing with max_length={max_length} ...")

    def _tokenize(example):
        return tokenize_and_mask(example, tokenizer, max_length, prompt_variant)

    tokenized = filtered.map(
        _tokenize,
        remove_columns=filtered["train"].column_names,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch")

    if verbose:
        sample = tokenized["train"][0]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        masked_count = (labels == -100).sum().item()
        total = len(labels)
        print("\nTokenization check (first training example):")
        print(f"  input_ids length    : {total}")
        print(f"  masked tokens (-100): {masked_count} / {total}")
        print(f"  Cypher tokens (loss): {total - masked_count}")

        cypher_ids = [
            tok_id
            for tok_id, label in zip(input_ids.tolist(), labels.tolist(), strict=False)
            if label != -100
        ]
        decoded_cypher = tokenizer.decode(cypher_ids, skip_special_tokens=False)
        print(f"  Decoded assistant turn:\n    {decoded_cypher[:300]}")

    return tokenized, tokenizer
