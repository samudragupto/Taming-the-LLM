from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.semantic import SemanticEvaluator, SemanticScore

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def evaluator() -> SemanticEvaluator:
    return SemanticEvaluator.get_instance("all-MiniLM-L6-v2")


@pytest.mark.semantic
@pytest.mark.slow
class TestSemanticEvaluator:
    def test_identical_texts_score_one(self, evaluator: SemanticEvaluator) -> None:
        text = "Python is a popular programming language."
        score = evaluator.similarity(text, text)
        assert score > 0.99

    def test_similar_texts_high_score(self, evaluator: SemanticEvaluator) -> None:
        a = "The cat sat on the mat."
        b = "A cat was sitting on a mat."
        score = evaluator.similarity(a, b)
        assert score > 0.6

    def test_unrelated_texts_low_score(self, evaluator: SemanticEvaluator) -> None:
        a = "Quantum physics describes subatomic particles."
        b = "I made pancakes for breakfast this morning."
        score = evaluator.similarity(a, b)
        assert score < 0.3

    def test_evaluate_returns_semantic_score(self, evaluator: SemanticEvaluator) -> None:
        result = evaluator.evaluate(
            candidate="Python is great for scripting.",
            reference="Python is excellent for writing scripts.",
            threshold=0.5,
        )
        assert isinstance(result, SemanticScore)
        assert result.passed is True
        assert 0.0 <= result.similarity <= 1.0

    def test_evaluate_fails_below_threshold(self, evaluator: SemanticEvaluator) -> None:
        result = evaluator.evaluate(
            candidate="The stock market crashed today.",
            reference="I enjoy hiking in the mountains.",
            threshold=0.8,
        )
        assert result.passed is False

    def test_evaluate_best_match(self, evaluator: SemanticEvaluator) -> None:
        references = [
            "JavaScript is used for web development.",
            "Python is a versatile programming language.",
            "Cooking requires patience and practice.",
        ]
        result = evaluator.evaluate_best_match(
            candidate="Python is great for many types of software projects.",
            references=references,
            threshold=0.5,
        )
        assert result.passed is True
        assert "Python" in result.reference

    def test_evaluate_best_match_empty_references(self, evaluator: SemanticEvaluator) -> None:
        result = evaluator.evaluate_best_match(
            candidate="Some text.",
            references=[],
            threshold=0.5,
        )
        assert result.passed is False
        assert result.similarity == 0.0

    def test_batch_embedding_shape(self, evaluator: SemanticEvaluator) -> None:
        texts = ["Hello world", "Foo bar", "Testing embeddings"]
        embeddings = evaluator.embed_batch(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0


@pytest.mark.semantic
@pytest.mark.slow
class TestSemanticFromFixtures:
    def test_evaluation_cases(self, evaluator: SemanticEvaluator) -> None:
        with open(FIXTURES_DIR / "evaluation_cases.json") as f:
            data = json.load(f)

        for case in data["semantic_similarity_cases"]:
            result = evaluator.evaluate(
                candidate=case["candidate"],
                reference=case["reference"],
                threshold=case["min_threshold"],
            )
            if case["expected_similar"]:
                assert result.passed, (
                    f"Case '{case['name']}' expected similar but got "
                    f"similarity={result.similarity:.4f} (threshold={case['min_threshold']})"
                )
            else:
                assert not result.passed, (
                    f"Case '{case['name']}' expected dissimilar but got "
                    f"similarity={result.similarity:.4f} (threshold={case['min_threshold']})"
                )


@pytest.mark.semantic
@pytest.mark.slow
class TestSemanticPlugin:
    def test_assert_similar_passes(self, semantic: Any) -> None:
        semantic.assert_similar(
            candidate="The dog is running in the park.",
            reference="A dog runs through the park.",
            threshold=0.5,
        )

    def test_assert_similar_fails(self, semantic: Any) -> None:
        with pytest.raises(AssertionError, match="Semantic Assertion Failed"):
            semantic.assert_similar(
                candidate="Quantum entanglement is a physical phenomenon.",
                reference="I ate a sandwich for lunch.",
                threshold=0.8,
            )

    def test_assert_not_similar(self, semantic: Any) -> None:
        semantic.assert_not_similar(
            candidate="The sky is blue.",
            reference="Database normalization reduces redundancy.",
            threshold=0.8,
        )