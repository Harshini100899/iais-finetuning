"""Unit tests for text2cypher configuration constants."""

from text2cypher.config import CYSPIDER_PREFIX, DATASET_NAME, MAX_LENGTH, MODEL_NAME


def test_constants():
    assert MODEL_NAME == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert DATASET_NAME == "RomanTeucher/text2cypher-curated"
    assert CYSPIDER_PREFIX == "cyspider"
    assert MAX_LENGTH == 1024
