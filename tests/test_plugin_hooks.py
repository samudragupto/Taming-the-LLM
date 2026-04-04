"""Test pytest plugin integration and hooks."""

from __future__ import annotations

from pytest_semantic.plugin import SemanticAssertionError


def test_semantic_assertion_error_message_formatting() -> None:
    error = SemanticAssertionError(
        candidate="This is a test output",
        reference="This is the expected output",
        similarity=0.65,
        threshold=0.75,
    )
    message = str(error)
    assert "0.6500" in message
    assert "0.75" in message
    assert "Semantic Assertion Failed" in message


def test_semantic_assertion_error_truncates_long_text() -> None:
    long_candidate = "a" * 300
    error = SemanticAssertionError(
        candidate=long_candidate,
        reference="short",
        similarity=0.5,
        threshold=0.8,
    )
    message = str(error)
    assert "..." in message
    assert len(message) < len(long_candidate) + 200


def test_semantic_assertion_error_attributes() -> None:
    error = SemanticAssertionError(
        candidate="candidate text",
        reference="reference text",
        similarity=0.55,
        threshold=0.70,
    )
    assert error.candidate == "candidate text"
    assert error.reference == "reference text"
    assert error.similarity == 0.55
    assert error.threshold == 0.70
