from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PropertyResult:
    name: str
    passed: bool
    message: str


@dataclass
class PropertyReport:
    results: list[PropertyResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[PropertyResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        lines = [f"Property Check: {passed}/{total} passed"]
        for failure in self.failures:
            lines.append(f"  FAIL: {failure.name} - {failure.message}")
        return "\n".join(lines)


class PropertyChecker:
    """Validates structural and behavioral properties of LLM output
    without relying on exact string matching."""

    @staticmethod
    def check_max_length(text: str, max_chars: int) -> PropertyResult:
        length = len(text)
        passed = length <= max_chars
        return PropertyResult(
            name="max_length",
            passed=passed,
            message=f"Length {length} <= {max_chars}"
            if passed
            else f"Length {length} > {max_chars}",
        )

    @staticmethod
    def check_max_sentences(text: str, max_sentences: int) -> PropertyResult:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        count = len(sentences)
        passed = count <= max_sentences
        return PropertyResult(
            name="max_sentences",
            passed=passed,
            message=(
                f"Sentence count {count} <= {max_sentences}"
                if passed
                else f"Sentence count {count} > {max_sentences}"
            ),
        )

    @staticmethod
    def check_no_pii(text: str) -> PropertyResult:
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"

        findings = []
        if re.search(email_pattern, text):
            findings.append("email address")
        if re.search(phone_pattern, text):
            findings.append("phone number")
        if re.search(ssn_pattern, text):
            findings.append("SSN-like pattern")

        passed = len(findings) == 0
        return PropertyResult(
            name="no_pii",
            passed=passed,
            message="No PII detected" if passed else f"PII detected: {', '.join(findings)}",
        )

    @staticmethod
    def check_valid_json(text: str) -> PropertyResult:
        try:
            json.loads(text)
            return PropertyResult(name="valid_json", passed=True, message="Valid JSON")
        except json.JSONDecodeError as exc:
            return PropertyResult(name="valid_json", passed=False, message=f"Invalid JSON: {exc}")

    @staticmethod
    def check_json_schema(text: str, required_keys: list[str]) -> PropertyResult:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return PropertyResult(
                name="json_schema", passed=False, message="Cannot validate schema: invalid JSON"
            )

        if isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    return PropertyResult(
                        name="json_schema",
                        passed=False,
                        message=f"Array element {i} is not an object",
                    )
                missing = [k for k in required_keys if k not in item]
                if missing:
                    return PropertyResult(
                        name="json_schema",
                        passed=False,
                        message=f"Element {i} missing keys: {missing}",
                    )
            return PropertyResult(name="json_schema", passed=True, message="Schema valid")

        if isinstance(data, dict):
            missing = [k for k in required_keys if k not in data]
            if missing:
                return PropertyResult(
                    name="json_schema", passed=False, message=f"Missing keys: {missing}"
                )
            return PropertyResult(name="json_schema", passed=True, message="Schema valid")

        return PropertyResult(
            name="json_schema", passed=False, message=f"Unexpected JSON type: {type(data).__name__}"
        )

    @staticmethod
    def check_value_in_set(value: str, allowed: set[str]) -> PropertyResult:
        normalized = value.strip().lower()
        passed = normalized in allowed
        return PropertyResult(
            name="value_in_set",
            passed=passed,
            message=(
                f"'{normalized}' is in allowed set"
                if passed
                else f"'{normalized}' not in {allowed}"
            ),
        )

    @staticmethod
    def check_not_empty(text: str) -> PropertyResult:
        passed = len(text.strip()) > 0
        return PropertyResult(
            name="not_empty",
            passed=passed,
            message="Output is non-empty" if passed else "Output is empty",
        )

    @staticmethod
    def check_no_refusal(text: str) -> PropertyResult:
        refusal_phrases = [
            "i cannot",
            "i can't",
            "i am unable",
            "i'm unable",
            "as an ai",
            "as a language model",
            "i apologize",
            "sorry, but i",
        ]
        lower_text = text.lower()
        found = [phrase for phrase in refusal_phrases if phrase in lower_text]
        passed = len(found) == 0
        return PropertyResult(
            name="no_refusal",
            passed=passed,
            message="No refusal detected" if passed else f"Refusal phrases found: {found}",
        )

    def run_checks(self, results: list[PropertyResult]) -> PropertyReport:
        return PropertyReport(results=results)
