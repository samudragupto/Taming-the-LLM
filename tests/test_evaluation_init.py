"""Test that evaluation module exports are accessible."""

from __future__ import annotations


def test_evaluation_module_imports() -> None:
    from src.evaluation import GuardrailChecker, PropertyChecker, SemanticEvaluator

    assert GuardrailChecker is not None
    assert PropertyChecker is not None
    assert SemanticEvaluator is not None


def test_guardrail_checker_importable() -> None:
    from src.evaluation import GuardrailChecker

    result = GuardrailChecker.check_harmful_content("Clean text.")
    assert result.passed is True


def test_property_checker_importable() -> None:
    from src.evaluation import PropertyChecker

    result = PropertyChecker.check_not_empty("Some content")
    assert result.passed is True


def test_semantic_evaluator_importable() -> None:
    from src.evaluation import SemanticEvaluator

    evaluator = SemanticEvaluator.get_instance()
    assert evaluator is not None


def test_all_exports_accessible_via_star_import() -> None:
    """Test __all__ exports if defined."""
    import src.evaluation as eval_module

    assert hasattr(eval_module, "SemanticEvaluator")
    assert hasattr(eval_module, "PropertyChecker")
    assert hasattr(eval_module, "GuardrailChecker")
