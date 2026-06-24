"""Unit tests for exact match, component F1, and syntactic validity."""

from text2cypher.metrics import component_f1, normalized_exact_match, syntactic_validity


class TestNormalizedExactMatch:
    def test_identical_strings(self):
        assert normalized_exact_match("MATCH (n) RETURN n", "MATCH (n) RETURN n")

    def test_case_difference(self):
        assert normalized_exact_match("match (n) return n", "MATCH (N) RETURN N")

    def test_whitespace_difference(self):
        assert normalized_exact_match("MATCH  (n) \n RETURN n", "MATCH (n) RETURN n")

    def test_case_and_whitespace_combined(self):
        assert normalized_exact_match("match   (n) \n return n", "MATCH (N) RETURN N")

    def test_alias_only_difference_matches(self):
        # Differing only by variable name is treated as a match (alias-normalized).
        assert normalized_exact_match("MATCH (n) RETURN n", "MATCH (m) RETURN m")

    def test_genuinely_different_queries(self):
        assert not normalized_exact_match(
            "MATCH (n:Movie) RETURN n.title", "MATCH (n:Person) RETURN n.name"
        )

    def test_different_predicate(self):
        assert not normalized_exact_match(
            "MATCH (n) WHERE n.age > 30 RETURN n", "MATCH (n) WHERE n.age > 40 RETURN n"
        )

    def test_empty_vs_nonempty(self):
        assert not normalized_exact_match("", "MATCH (n) RETURN n")

    def test_both_empty(self):
        assert normalized_exact_match("", "")


class TestComponentF1:
    def test_perfect_match_gives_f1_1(self):
        query = (
            "MATCH (p:Person {name: 'Alice'})-[:ACTED_IN]->(m:Movie) "
            "WHERE m.year > 2000 RETURN m.title"
        )
        res = component_f1(query, query)
        assert res["f1"] == 1.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0

    def test_completely_wrong_gives_low_f1(self):
        res = component_f1(
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN m.title",
            "CREATE (u:User {name: 'Bob'})",
        )
        assert res["f1"] == 0.0

    def test_node_labels_recall(self):
        res = component_f1(
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN m",
            "MATCH (p:Person) RETURN p",
        )
        assert res["per_component"]["node_labels"]["recall"] == 0.5


class TestSyntacticValidity:
    def test_valid_simple_query(self):
        assert syntactic_validity("MATCH (n) RETURN n")

    def test_valid_complex_query(self):
        assert syntactic_validity(
            "MATCH (a:Person)-[:FRIEND]->(b:Person) WHERE a.age > b.age RETURN a.name"
        )

    def test_invalid_unbalanced_parentheses(self):
        assert not syntactic_validity("MATCH (n RETURN n")

    def test_invalid_missing_return(self):
        assert not syntactic_validity("MATCH (n)")

    def test_invalid_no_entry_clause(self):
        assert not syntactic_validity("RETURN n")

    def test_invalid_empty_string(self):
        assert not syntactic_validity("")
