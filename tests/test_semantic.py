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


@pytest.mark.semantic
@pytest.mark.slow
class TestSemanticEvaluatorEdgeCases:
    def test_embed_single_word(self, evaluator: SemanticEvaluator) -> None:
        embedding = evaluator.embed("word")
        assert embedding.shape[0] > 0

    def test_embed_empty_string(self, evaluator: SemanticEvaluator) -> None:
        embedding = evaluator.embed("")
        assert embedding.shape[0] > 0

    def test_embed_batch_single_item(self, evaluator: SemanticEvaluator) -> None:
        embeddings = evaluator.embed_batch(["single"])
        assert embeddings.shape[0] == 1

    def test_similarity_boundary_values(self, evaluator: SemanticEvaluator) -> None:
        score = evaluator.similarity("test", "test")
        assert 0.0 <= score <= 1.0

    def test_evaluate_exact_threshold(self, evaluator: SemanticEvaluator) -> None:
        result = evaluator.evaluate("The cat sat.", "A cat was sitting.", threshold=0.6)
        assert isinstance(result.similarity, float)
        assert result.threshold == 0.6

    def test_evaluate_best_match_first_reference_best(self, evaluator: SemanticEvaluator) -> None:
        references = [
            "Python is a programming language.",
            "The weather is sunny.",
            "Cooking is an art.",
        ]
        result = evaluator.evaluate_best_match(
            "Python is used for coding.", references, threshold=0.5
        )
        assert "Python" in result.reference

    def test_get_instance_caching(self) -> None:
        instance1 = SemanticEvaluator.get_instance("all-MiniLM-L6-v2")
        instance2 = SemanticEvaluator.get_instance("all-MiniLM-L6-v2")
        assert instance1 is instance2

    def test_evaluate_best_match_no_match_below_threshold(
        self, evaluator: SemanticEvaluator
    ) -> None:
        references = ["Quantum physics", "Stock market trends"]
        result = evaluator.evaluate_best_match("Cooking pasta", references, threshold=0.9)
        assert result.passed is False


[
    {
        "resource": "/c:/Users/user/Desktop/taming-the-llm/src/evaluation/semantic.py",
        "owner": "Pylance10",
        "code": {
            "value": "reportUndefinedVariable",
            "target": {
                "$mid": 1,
                "path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
                "scheme": "https",
                "authority": "github.com",
            },
        },
        "severity": 4,
        "message": '"pytest" is not defined',
        "source": "Pylance",
        "startLineNumber": 95,
        "startColumn": 2,
        "endLineNumber": 95,
        "endColumn": 8,
        "modelVersionId": 15,
        "origin": "extHost1",
    },
    {
        "resource": "/c:/Users/user/Desktop/taming-the-llm/src/evaluation/semantic.py",
        "owner": "Pylance10",
        "code": {
            "value": "reportUndefinedVariable",
            "target": {
                "$mid": 1,
                "path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
                "scheme": "https",
                "authority": "github.com",
            },
        },
        "severity": 4,
        "message": '"pytest" is not defined',
        "source": "Pylance",
        "startLineNumber": 96,
        "startColumn": 2,
        "endLineNumber": 96,
        "endColumn": 8,
        "modelVersionId": 15,
        "origin": "extHost1",
    },
]


@pytest.mark.semantic
@pytest.mark.slow
class TestSemanticEvaluatorInternals:
    def test_model_initialization(self) -> None:
        evaluator = SemanticEvaluator(model_name="all-MiniLM-L6-v2")
        assert evaluator._model is not None
        assert evaluator._model_name == "all-MiniLM-L6-v2"

    def test_embed_returns_normalized_vector(self, evaluator: SemanticEvaluator) -> None:
        embedding = evaluator.embed("test")
        assert embedding.min() >= -1.0
        assert embedding.max() <= 1.0

    def test_embed_batch_consistency(self, evaluator: SemanticEvaluator) -> None:
        import numpy as np

        texts = ["hello", "world"]
        batch_embeddings = evaluator.embed_batch(texts)
        individual_embedding_0 = evaluator.embed(texts[0])
        individual_embedding_1 = evaluator.embed(texts[1])

        assert np.allclose(batch_embeddings[0], individual_embedding_0, atol=1e-5)
        assert np.allclose(batch_embeddings[1], individual_embedding_1, atol=1e-5)

    def test_similarity_symmetry(self, evaluator: SemanticEvaluator) -> None:
        score_ab = evaluator.similarity("hello world", "world hello")
        score_ba = evaluator.similarity("world hello", "hello world")
        assert abs(score_ab - score_ba) < 0.01

    def test_evaluate_all_fields_populated(self, evaluator: SemanticEvaluator) -> None:
        result = evaluator.evaluate("test", "test", threshold=0.5)
        assert result.similarity is not None
        assert result.passed is not None
        assert result.threshold == 0.5
        assert result.candidate == "test"
        assert result.reference == "test"

    def test_evaluate_best_match_single_reference(self, evaluator: SemanticEvaluator) -> None:
        result = evaluator.evaluate_best_match(
            "Python is great", ["Python is great"], threshold=0.8
        )
        assert result.passed is True
        assert result.reference == "Python is great"
