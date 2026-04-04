from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.properties import PropertyChecker

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestPropertyCheckerBasic:
    def test_max_length_pass(self) -> None:
        result = PropertyChecker.check_max_length("short text", max_chars=100)
        assert result.passed is True

    def test_max_length_fail(self) -> None:
        result = PropertyChecker.check_max_length("a" * 200, max_chars=100)
        assert result.passed is False

    def test_max_sentences_pass(self) -> None:
        result = PropertyChecker.check_max_sentences(
            "First sentence. Second sentence. Third sentence.", max_sentences=3
        )
        assert result.passed is True

    def test_max_sentences_fail(self) -> None:
        text = "One. Two. Three. Four. Five."
        result = PropertyChecker.check_max_sentences(text, max_sentences=3)
        assert result.passed is False

    def test_no_pii_clean(self) -> None:
        result = PropertyChecker.check_no_pii("This is a normal text without personal data.")
        assert result.passed is True

    def test_no_pii_email(self) -> None:
        result = PropertyChecker.check_no_pii("Contact me at john@example.com for details.")
        assert result.passed is False
        assert "email" in result.message

    def test_no_pii_phone(self) -> None:
        result = PropertyChecker.check_no_pii("Call me at 555-123-4567.")
        assert result.passed is False
        assert "phone" in result.message

    def test_no_pii_ssn(self) -> None:
        result = PropertyChecker.check_no_pii("SSN: 123-45-6789")
        assert result.passed is False
        assert "SSN" in result.message

    def test_valid_json_pass(self) -> None:
        result = PropertyChecker.check_valid_json('[{"key": "value"}]')
        assert result.passed is True

    def test_valid_json_fail(self) -> None:
        result = PropertyChecker.check_valid_json("not json at all")
        assert result.passed is False

    def test_json_schema_array(self) -> None:
        text = '[{"entity": "Guido", "type": "PERSON"}]'
        result = PropertyChecker.check_json_schema(text, required_keys=["entity", "type"])
        assert result.passed is True

    def test_json_schema_missing_key(self) -> None:
        text = '[{"entity": "Guido"}]'
        result = PropertyChecker.check_json_schema(text, required_keys=["entity", "type"])
        assert result.passed is False

    def test_value_in_set_pass(self) -> None:
        result = PropertyChecker.check_value_in_set("positive", {"positive", "negative", "neutral"})
        assert result.passed is True

    def test_value_in_set_fail(self) -> None:
        result = PropertyChecker.check_value_in_set("maybe", {"positive", "negative", "neutral"})
        assert result.passed is False

    def test_not_empty_pass(self) -> None:
        result = PropertyChecker.check_not_empty("some content")
        assert result.passed is True

    def test_not_empty_fail(self) -> None:
        result = PropertyChecker.check_not_empty("   ")
        assert result.passed is False

    def test_no_refusal_clean(self) -> None:
        result = PropertyChecker.check_no_refusal("Here is the summary of the document.")
        assert result.passed is True

    def test_no_refusal_detected(self) -> None:
        result = PropertyChecker.check_no_refusal(
            "I apologize, but as an AI language model, I cannot help with that."
        )
        assert result.passed is False


class TestPropertyReport:
    def test_all_passed(self) -> None:
        checker = PropertyChecker()
        report = checker.run_checks(
            [
                PropertyChecker.check_not_empty("hello"),
                PropertyChecker.check_max_length("hello", 100),
            ]
        )
        assert report.all_passed is True
        assert len(report.failures) == 0

    def test_mixed_results(self) -> None:
        checker = PropertyChecker()
        report = checker.run_checks(
            [
                PropertyChecker.check_not_empty("hello"),
                PropertyChecker.check_max_length("a" * 200, 100),
            ]
        )
        assert report.all_passed is False
        assert len(report.failures) == 1

    def test_summary_output(self) -> None:
        checker = PropertyChecker()
        report = checker.run_checks(
            [
                PropertyChecker.check_not_empty(""),
                PropertyChecker.check_max_length("ok", 100),
            ]
        )
        summary = report.summary()
        assert "1/2 passed" in summary
        assert "FAIL" in summary


class TestPropertyFromFixtures:
    def test_fixture_cases(self) -> None:
        with open(FIXTURES_DIR / "evaluation_cases.json") as f:
            data = json.load(f)

        for case in data["property_test_cases"]:
            output = case["output"]
            for check_name in case["checks"]:
                if check_name == "value_in_set":
                    result = PropertyChecker.check_value_in_set(output, set(case["allowed_values"]))
                    assert result.passed, f"Case '{case['name']}' failed {check_name}"
                elif check_name == "valid_json":
                    result = PropertyChecker.check_valid_json(output)
                    assert result.passed, f"Case '{case['name']}' failed {check_name}"
                elif check_name == "json_schema":
                    result = PropertyChecker.check_json_schema(
                        output, required_keys=case["required_keys"]
                    )
                    assert result.passed, f"Case '{case['name']}' failed {check_name}"
                elif check_name == "max_sentences":
                    result = PropertyChecker.check_max_sentences(
                        output, max_sentences=case["max_sentences"]
                    )
                    assert result.passed, f"Case '{case['name']}' failed {check_name}"
                elif check_name == "not_empty":
                    result = PropertyChecker.check_not_empty(output)
                    assert result.passed, f"Case '{case['name']}' failed {check_name}"


@pytest.mark.property
class TestPropertyHypothesis:
    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_not_empty_consistency(self, text: str) -> None:
        result = PropertyChecker.check_not_empty(text)
        assert result.passed == (len(text.strip()) > 0)

    @given(text=st.text(min_size=0, max_size=1000))
    @settings(max_examples=100)
    def test_max_length_consistency(self, text: str) -> None:
        max_chars = 200
        result = PropertyChecker.check_max_length(text, max_chars)
        assert result.passed == (len(text) <= max_chars)

    @given(
        value=st.sampled_from(["positive", "negative", "neutral", "unknown", "mixed", "happy"]),
    )
    def test_value_in_set_deterministic(self, value: str) -> None:
        allowed = {"positive", "negative", "neutral"}
        result = PropertyChecker.check_value_in_set(value, allowed)
        assert result.passed == (value.strip().lower() in allowed)

    @given(
        text=st.text(
            min_size=1,
            max_size=500,
            alphabet=st.characters(categories=("L", "N", "P", "Z")),
        )
    )
    @settings(max_examples=50)
    def test_no_pii_no_false_positive_on_plain_text(self, text: str) -> None:
        if "@" not in text and not any(c.isdigit() for c in text):
            result = PropertyChecker.check_no_pii(text)
            assert result.passed is True


class TestPropertyCheckerEdgeCases:
    def test_max_length_exact_boundary(self) -> None:
        result = PropertyChecker.check_max_length("a" * 100, max_chars=100)
        assert result.passed is True

    def test_max_sentences_with_exclamation_marks(self) -> None:
        result = PropertyChecker.check_max_sentences("One! Two! Three!", max_sentences=3)
        assert result.passed is True

    def test_max_sentences_with_question_marks(self) -> None:
        result = PropertyChecker.check_max_sentences(
            "First? Second? Third? Fourth?", max_sentences=3
        )
        assert result.passed is False

    def test_max_sentences_empty_text(self) -> None:
        result = PropertyChecker.check_max_sentences("", max_sentences=3)
        assert result.passed is True

    def test_no_pii_multiple_patterns(self) -> None:
        result = PropertyChecker.check_no_pii(
            "Contact john@example.com or call 555-123-4567. SSN: 123-45-6789"
        )
        assert result.passed is False
        assert "email" in result.message
        assert "phone" in result.message or "SSN" in result.message

    def test_valid_json_empty_object(self) -> None:
        result = PropertyChecker.check_valid_json("{}")
        assert result.passed is True

    def test_valid_json_empty_array(self) -> None:
        result = PropertyChecker.check_valid_json("[]")
        assert result.passed is True

    def test_json_schema_dict_with_all_keys(self) -> None:
        result = PropertyChecker.check_json_schema(
            '{"entity": "test", "type": "PERSON", "extra": "ignored"}',
            required_keys=["entity", "type"],
        )
        assert result.passed is True

    def test_json_schema_invalid_json(self) -> None:
        result = PropertyChecker.check_json_schema("not json", required_keys=["key"])
        assert result.passed is False
        assert "invalid JSON" in result.message

    def test_json_schema_non_dict_non_list(self) -> None:
        result = PropertyChecker.check_json_schema('"just a string"', required_keys=["key"])
        assert result.passed is False

    def test_json_schema_array_with_non_object(self) -> None:
        result = PropertyChecker.check_json_schema("[1, 2, 3]", required_keys=["key"])
        assert result.passed is False

    def test_value_in_set_case_normalization(self) -> None:
        result = PropertyChecker.check_value_in_set("  POSITIVE  ", {"positive", "negative"})
        assert result.passed is True

    def test_not_empty_with_whitespace_only(self) -> None:
        result = PropertyChecker.check_not_empty("\t\n  \r")
        assert result.passed is False

    def test_no_refusal_multiple_patterns(self) -> None:
        result = PropertyChecker.check_no_refusal(
            "I cannot help with that. As an AI, I'm unable to assist."
        )
        assert result.passed is False
        assert len(result.message.split(",")) >= 2

    def test_no_refusal_edge_case_mixed_case(self) -> None:
        result = PropertyChecker.check_no_refusal("I CaNnOt assist with this request.")
        assert result.passed is False

    def test_run_checks_all_pass(self) -> None:
        checker = PropertyChecker()
        results = [
            PropertyChecker.check_not_empty("text"),
            PropertyChecker.check_max_length("short", 100),
            PropertyChecker.check_no_pii("clean text"),
        ]
        report = checker.run_checks(results)
        assert report.all_passed is True
        assert len(report.failures) == 0

    def test_run_checks_summary_contains_failure_details(self) -> None:
        checker = PropertyChecker()
        results = [PropertyChecker.check_not_empty("")]
        report = checker.run_checks(results)
        summary = report.summary()
        assert "0/1 passed" in summary
        assert "FAIL" in summary
