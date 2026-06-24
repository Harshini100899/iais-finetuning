"""Isolated metric implementations for Text2Cypher evaluation."""

from __future__ import annotations

import re
from typing import Any


def _normalize(query: str) -> str:
    """Lowercase and collapse all internal whitespace to a single space."""
    return " ".join(query.lower().split())


# Variable declared right after an opening ( or [, e.g. (n:Movie), [r:ACTED_IN], (m).
_RE_VAR_DECL = re.compile(r"[\(\[]\s*([a-zA-Z_]\w*)\s*(?=[:\)\]\{])")


def normalize_aliases(query: str) -> str:
    """
    Rename node/relationship variables to canonical placeholders (var0, var1, ...)
    in order of first appearance, so two queries that differ only in alias choice
    (e.g. `(n:Movie)` vs `(m:Movie)`) compare equal.
    """
    seen: dict[str, str] = {}
    for match in _RE_VAR_DECL.finditer(query):
        var = match.group(1)
        if var not in seen:
            seen[var] = f"var{len(seen)}"

    normalized = query
    for var, placeholder in seen.items():
        normalized = re.sub(rf"\b{re.escape(var)}\b", placeholder, normalized)
    return normalized


def exact_match_strict(gold: str, pred: str) -> bool:
    """
    Return True iff gold and pred are identical after lowercase + whitespace
    normalization only (no alias normalization). Kept for comparison against
    `normalized_exact_match` to quantify how much alias choice was masking
    correct predictions.
    """
    return _normalize(gold) == _normalize(pred)


def normalized_exact_match(gold: str, pred: str) -> bool:
    """
    Return True iff gold and pred are identical after normalization.

    Normalization: lowercase + collapse whitespace + canonicalize variable
    aliases. A prediction that uses different (but consistent) variable names
    than gold — e.g. `(m:Movie)` instead of `(n:Movie)` — still counts as a
    match, since the queries are semantically identical.
    Does NOT strip punctuation — a missing semicolon counts as a difference.
    """
    return normalize_aliases(_normalize(gold)) == normalize_aliases(_normalize(pred))


# Regex patterns for lightweight structural parsing

# Node labels: (n:Label) or (:Label)  — capture the label part
_RE_NODE_LABELS = re.compile(r"\([\w\s]*:(\w+)", re.IGNORECASE)

# Relationship types: -[:REL_TYPE]->  or <-[:REL_TYPE]-
_RE_REL_TYPES = re.compile(r"\[:(\w+)\]", re.IGNORECASE)

# Property keys (inside curly braces): {key: value, key2: value2}
# We extract keys only (not values) — value correctness is not captured here.
_RE_PROP_KEYS = re.compile(r"\{([^}]+)\}", re.IGNORECASE)
_RE_PROP_KEY_NAMES = re.compile(r"(\w+)\s*:", re.IGNORECASE)

# WHERE predicates: simple key-operator-value tokens, e.g. n.age > 30
# We extract as normalized "token" strings like "n.age>30"
_RE_WHERE_BLOCK = re.compile(
    r"\bWHERE\b(.*?)(?:\bRETURN\b|\bWITH\b|\bORDER\b|\bLIMIT\b|\bSKIP\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_RE_WHERE_PREDICATES = re.compile(
    r"[\w.]+\s*(?:=|<>|!=|<=|>=|<|>|CONTAINS|STARTS WITH|ENDS WITH)\s*[\w.'\"]+",
    re.IGNORECASE,
)

# RETURN items: everything after RETURN up to ORDER/LIMIT/SKIP/end
_RE_RETURN_BLOCK = re.compile(
    r"\bRETURN\b(.*?)(?:\bORDER\b|\bLIMIT\b|\bSKIP\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_components(query: str) -> dict[str, set[str]]:
    """
    Extract structural components from a Cypher query string.

    Returns a dict with keys:
        node_labels, rel_types, prop_keys, where_predicates, return_items
    Each value is a set of lowercase strings.
    """
    q = query.strip()

    node_labels: set[str] = {m.lower() for m in _RE_NODE_LABELS.findall(q)}
    rel_types: set[str] = {m.lower() for m in _RE_REL_TYPES.findall(q)}

    prop_keys: set[str] = set()
    for block in _RE_PROP_KEYS.findall(q):
        for key in _RE_PROP_KEY_NAMES.findall(block):
            prop_keys.add(key.lower())

    where_preds: set[str] = set()
    where_match = _RE_WHERE_BLOCK.search(q)
    if where_match:
        where_block = where_match.group(1)
        for pred in _RE_WHERE_PREDICATES.findall(where_block):
            where_preds.add(re.sub(r"\s+", "", pred).lower())

    return_items: set[str] = set()
    return_match = _RE_RETURN_BLOCK.search(q)
    if return_match:
        for item in return_match.group(1).split(","):
            tok = item.strip().lower()
            if tok:
                return_items.add(tok)

    return {
        "node_labels": node_labels,
        "rel_types": rel_types,
        "prop_keys": prop_keys,
        "where_predicates": where_preds,
        "return_items": return_items,
    }


def _set_prf(gold_set: set[str], pred_set: set[str]) -> tuple[float, float, float]:
    """Micro precision, recall, F1 for two sets of string tokens."""
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return round(precision, 4), round(recall, 4), round(f1, 4)


def component_f1(gold: str, pred: str) -> dict[str, Any]:
    """
    Compute component-level precision, recall, F1 between gold and predicted Cypher.

    Each component (node_labels, rel_types, prop_keys, where_predicates,
    return_items) is scored as set overlap; the aggregate is the macro-average.

    Returns a dict with keys:
        precision, recall, f1,
        per_component (sub-dict with the same keys per component)
    """
    gold_c = _extract_components(gold)
    pred_c = _extract_components(pred)

    per_component: dict[str, dict[str, float]] = {}
    all_p, all_r, all_f = [], [], []

    for key in gold_c:
        p, r, f = _set_prf(gold_c[key], pred_c[key])
        per_component[key] = {"precision": p, "recall": r, "f1": f}
        all_p.append(p)
        all_r.append(r)
        all_f.append(f)

    return {
        "precision": round(sum(all_p) / len(all_p), 4),
        "recall": round(sum(all_r) / len(all_r), 4),
        "f1": round(sum(all_f) / len(all_f), 4),
        "per_component": per_component,
    }


def syntactic_validity(query: str) -> bool:
    """
    Lightweight structural plausibility check for a Cypher query.

    Checks performed (all must pass):
    1. Non-empty after stripping.
    2. Balanced parentheses.
    3. Balanced square brackets.
    4. Balanced curly braces.
    5. Contains at least one of: MATCH, CREATE, MERGE, CALL, UNWIND.
    6. Contains RETURN (almost every valid read query must have one;
       write-only queries without RETURN are edge cases not common in this dataset).
    """
    q = query.strip()
    if not q:
        return False

    # Balance checks
    if q.count("(") != q.count(")"):
        return False
    if q.count("[") != q.count("]"):
        return False
    if q.count("{") != q.count("}"):
        return False

    upper = q.upper()

    # Must start with or contain an entry clause
    entry_clauses = ("MATCH", "CREATE", "MERGE", "CALL", "UNWIND", "WITH")
    if not any(clause in upper for clause in entry_clauses):
        return False

    # Must have a RETURN (covers read queries which dominate this dataset)
    if "RETURN" not in upper:
        return False

    return True


# Properties inside curly braces, e.g. "Movie {title, year}" or "{name: 'Alice'}".
_RE_SCHEMA_PROP_BLOCK = re.compile(r"\{([^}]+)\}")
# Properties in the verbose markdown schema format, e.g. "- `title`: STRING".
_RE_SCHEMA_PROP_BACKTICK = re.compile(r"`([^`:]+)`")
# Property access in a Cypher query, e.g. "m.title" — requires an identifier
# (not a bare number, so decimal literals like "2000.5" aren't mistaken for one).
_RE_PROPERTY_ACCESS = re.compile(r"[a-zA-Z_]\w*\.(\w+)")


def _extract_schema_properties(schema: str) -> set[str]:
    """Extract the set of property names declared in a schema string."""
    props: set[str] = set()
    for block in _RE_SCHEMA_PROP_BLOCK.findall(schema):
        for p in block.split(","):
            name = p.split(":")[0].strip().strip("'\"")
            if name:
                props.add(name.lower())
    for m in _RE_SCHEMA_PROP_BACKTICK.finditer(schema):
        props.add(m.group(1).strip().lower())
    return props


def _extract_pred_properties(pred: str) -> set[str]:
    """
    Extract property names referenced in a predicted Cypher query, covering both:
      - dot access:  m.title  -> "title"
      - inline maps: (p:Person {name: 'Alice'})  -> "name"
    Inline-map keys matter because the prompt teaches the model to use that style,
    so hallucinations often show up there rather than in dot access.
    """
    props = {m.lower() for m in _RE_PROPERTY_ACCESS.findall(pred)}
    for block in _RE_PROP_KEYS.findall(pred):
        for key in _RE_PROP_KEY_NAMES.findall(block):
            props.add(key.lower())
    return props


def schema_grounding(pred: str, schema: str) -> tuple[int, int]:
    """
    Return (num_grounded, num_predicted) property references for one prediction:
      - num_predicted: how many distinct properties the query references
      - num_grounded : how many of those actually exist in the schema

    Counts (not a ratio) so they can be summed into an honest micro-average over
    the test set — a prediction that references no properties contributes (0, 0)
    and therefore can't inflate the aggregate.
    """
    schema_props = _extract_schema_properties(schema)
    pred_props = _extract_pred_properties(pred)
    grounded = len(pred_props & schema_props)
    return grounded, len(pred_props)


def schema_grounding_score(pred: str, schema: str) -> float | None:
    """
    Per-sample grounding ratio (grounded / predicted), or None when the query
    references no properties at all — None means "not applicable" rather than a
    misleading perfect score. For aggregate reporting, micro-average the counts
    from ``schema_grounding`` instead of averaging these ratios.
    """
    grounded, total = schema_grounding(pred, schema)
    if total == 0:
        return None
    return grounded / total
