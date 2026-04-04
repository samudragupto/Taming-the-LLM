"""Test that evaluation module exports are accessible."""

from __future__ import annotations


def test_evaluation_module_all_imports() -> None:
    """Test that all three classes can be imported from the evaluation module."""
    import src.evaluation

    assert hasattr(src.evaluation, "GuardrailChecker")
    assert hasattr(src.evaluation, "PropertyChecker")
    assert hasattr(src.evaluation, "SemanticEvaluator")


def test_guardrail_checker_direct_import() -> None:
    """Test GuardrailChecker can be imported directly."""
    from src.evaluation import GuardrailChecker

    result = GuardrailChecker.check_harmful_content("Clean text.")
    assert result.passed is True
    assert result.severity == "info"


def test_property_checker_direct_import() -> None:
    """Test PropertyChecker can be imported directly."""
    from src.evaluation import PropertyChecker

    result = PropertyChecker.check_not_empty("Some content")
    assert result.passed is True


def test_semantic_evaluator_direct_import() -> None:
    """Test SemanticEvaluator can be imported directly."""
    from src.evaluation import SemanticEvaluator

    evaluator = SemanticEvaluator.get_instance()
    assert evaluator is not None
    assert evaluator._model_name == "all-MiniLM-L6-v2"


def test_all_classes_instantiable_from_module() -> None:
    """Test that classes imported from the module work correctly."""
    from src.evaluation import GuardrailChecker, PropertyChecker, SemanticEvaluator

    # GuardrailChecker is a class with static methods
    assert callable(GuardrailChecker.check_harmful_content)

    # PropertyChecker is a class with static methods
    assert callable(PropertyChecker.check_not_empty)

    # SemanticEvaluator is instantiable
    evaluator = SemanticEvaluator.get_instance()
    assert evaluator is not None


def test_evaluation_module_structure() -> None:
    """Test the module has the expected structure."""
    import src.evaluation

    # Check module-level exports
    module_attrs = dir(src.evaluation)
    assert "GuardrailChecker" in module_attrs
    assert "PropertyChecker" in module_attrs
    assert "SemanticEvaluator" in module_attrs
