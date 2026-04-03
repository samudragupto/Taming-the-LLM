from __future__ import annotations

import pytest

from src.evaluation.semantic import SemanticEvaluator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--semantic-threshold",
        action="store",
        default="0.75",
        help="Default similarity threshold for semantic assertions (0.0-1.0)",
    )
    parser.addoption(
        "--semantic-model",
        action="store",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name for semantic evaluation",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "semantic: mark test as a semantic similarity test")


class SemanticAssertionError(AssertionError):
    def __init__(self, candidate: str, reference: str, similarity: float, threshold: float):
        self.candidate = candidate
        self.reference = reference
        self.similarity = similarity
        self.threshold = threshold
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"\n--- Semantic Assertion Failed ---\n"
            f"Similarity: {self.similarity:.4f} (threshold: {self.threshold})\n"
            f"Candidate:  {self.candidate[:200]}{'...' if len(self.candidate) > 200 else ''}\n"
            f"Reference:  {self.reference[:200]}{'...' if len(self.reference) > 200 else ''}\n"
            f"---------------------------------"
        )


class SemanticAssert:
    """Provides assertion methods that compare meaning rather than exact strings."""

    def __init__(self, evaluator: SemanticEvaluator, default_threshold: float) -> None:
        self._evaluator = evaluator
        self._default_threshold = default_threshold

    def assert_similar(
        self,
        candidate: str,
        reference: str,
        threshold: float | None = None,
    ) -> None:
        t = threshold if threshold is not None else self._default_threshold
        result = self._evaluator.evaluate(candidate, reference, threshold=t)
        if not result.passed:
            raise SemanticAssertionError(candidate, reference, result.similarity, t)

    def assert_similar_to_any(
        self,
        candidate: str,
        references: list[str],
        threshold: float | None = None,
    ) -> None:
        t = threshold if threshold is not None else self._default_threshold
        result = self._evaluator.evaluate_best_match(candidate, references, threshold=t)
        if not result.passed:
            raise SemanticAssertionError(candidate, result.reference, result.similarity, t)

    def assert_not_similar(
        self,
        candidate: str,
        reference: str,
        threshold: float | None = None,
    ) -> None:
        t = threshold if threshold is not None else self._default_threshold
        result = self._evaluator.evaluate(candidate, reference, threshold=t)
        if result.passed:
            raise AssertionError(
                f"\n--- Semantic Assertion Failed (expected dissimilarity) ---\n"
                f"Similarity: {result.similarity:.4f} (threshold: {t})\n"
                f"Texts are too similar when they should not be.\n"
            )

    def similarity(self, text_a: str, text_b: str) -> float:
        return self._evaluator.similarity(text_a, text_b)


@pytest.fixture(scope="session")
def semantic_evaluator(request: pytest.FixtureRequest) -> SemanticEvaluator:
    model_name = request.config.getoption("--semantic-model")
    return SemanticEvaluator.get_instance(model_name)


@pytest.fixture(scope="session")
def semantic(
    request: pytest.FixtureRequest, semantic_evaluator: SemanticEvaluator
) -> SemanticAssert:
    threshold = float(request.config.getoption("--semantic-threshold"))
    return SemanticAssert(semantic_evaluator, default_threshold=threshold)


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    if call.excinfo is not None and isinstance(call.excinfo.value, SemanticAssertionError):
        err = call.excinfo.value
        if hasattr(item, "user_properties"):
            item.user_properties.append(("semantic_similarity", err.similarity))
            item.user_properties.append(("semantic_threshold", err.threshold))
