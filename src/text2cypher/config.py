"""Central configuration constants for the text2cypher package."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the repo root unless it is already absolute.

    Output artifacts (checkpoints, results, mlflow.db) anchor to the repo root so
    they land in the same place regardless of the launch directory, without
    mutating the process working directory.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_tracking_uri(uri: str) -> str:
    """Anchor a relative ``sqlite:///`` MLflow URI to the repo root."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    db_path = Path(uri[len(prefix) :])
    if db_path.is_absolute():
        return uri
    return f"{prefix}{(REPO_ROOT / db_path).as_posix()}"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML experiment config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_NAME = "RomanTeucher/text2cypher-curated"

SYSTEM_PROMPT = (
    "You are an expert in Neo4j and the Cypher query language. "
    "Given a graph schema and a natural-language question, "
    "generate a valid Cypher query that answers the question. "
    "Output ONLY the Cypher query, nothing else."
)

SYSTEM_PROMPT_FEWSHOT = (
    "You are a Neo4j Cypher query expert.\n"
    "Convert the natural language question into a Cypher query using ONLY\n"
    "the nodes, relationships, and properties in the provided schema.\n\n"
    "Rules:\n"
    "- Use single-letter lowercase aliases matching the label: (m:Movie), (p:Person)\n"
    "- Use WHERE for numeric/boolean conditions, inline {} for string equality\n"
    "- String values must use single quotes: {name: 'Alice'}\n"
    "- Always end with RETURN\n"
    "- Never invent properties or labels not in the schema\n\n"
    "Output ONLY the Cypher query, nothing else."
)

# Drawn from the train split of RomanTeucher/text2cypher-curated (never val/test
# — that would leak held-out examples into the prompt). instance_id_18568,
# instance_id_27874, instance_id_26221, instance_id_39966; schemas hand-compacted
# for brevity. Covers filter+DISTINCT, single-hop aggregation+ORDER BY, multi-hop
# traversal, and COLLECT+ORDER BY+LIMIT.
FEW_SHOT_EXAMPLES = [
    {
        "schema": "Article {article_id, title}",
        "question": "Retrieve distinct values of the title from Article "
        "where article_id is not 1010!",
        "cypher": "MATCH (n:Article) WHERE n.article_id <> '1010' "
        "RETURN DISTINCT n.title AS title",
    },
    {
        "schema": "Movie {title, votes, tagline, released}, Person {born, name}, "
        "(:Person)-[:ACTED_IN]->(:Movie)",
        "question": "Which actors played in the most movies?",
        "cypher": "MATCH (p:Person)-[a:ACTED_IN]->(m:Movie) "
        "RETURN p.name, COUNT(m) AS movies_count ORDER BY movies_count DESC",
    },
    {
        "schema": "Question {title}, Tag {name}, User {display_name}, "
        "(:User)-[:ASKED]->(:Question), (:Question)-[:TAGGED]->(:Tag)",
        "question": "Which users have asked questions tagged with 'react-apollo'?",
        "cypher": "MATCH (u:User)-[:ASKED]->(q:Question)-[:TAGGED]->"
        "(t:Tag {name: 'react-apollo'}) RETURN u",
    },
    {
        "schema": "Movie {title, votes, tagline, released}, Person {born, name}, "
        "(:Person)-[:DIRECTED]->(:Movie)",
        "question": "Who are the first 3 oldest directors and the movies "
        "they have directed?",
        "cypher": "MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN p.name AS Director, "
        "p.born AS BirthYear, collect(m.title) AS Movies ORDER BY p.born ASC LIMIT 3",
    },
]


def get_system_prompt(prompt_variant: str = "default") -> str:
    """Resolve the system prompt text for a given prompt_variant config value."""
    if prompt_variant == "fewshot":
        examples_text = "\n\n".join(
            f"Schema: {ex['schema']}\n"
            f"Question: {ex['question']}\n"
            f"Cypher: {ex['cypher']}"
            for ex in FEW_SHOT_EXAMPLES
        )
        return SYSTEM_PROMPT_FEWSHOT + "\n\nExamples:\n" + examples_text
    return SYSTEM_PROMPT


CYSPIDER_PREFIX = "cyspider"
MAX_LENGTH = 1024
