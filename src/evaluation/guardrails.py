from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    name: str
    passed: bool
    message: str
    severity: str  # "critical", "warning", "info"


class GuardrailChecker:
    """Production guardrails that should block or flag dangerous LLM outputs."""

    HARMFUL_PATTERNS = [
        r"(?i)\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)",
        r"(?i)\b(kill|murder|assassinate)\s+(yourself|someone|people)",
        r"(?i)\b(hack|crack|break)\s+(into\s+)?(a\s+)?(bank|system|account)",
    ]

    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)you\s+are\s+now\s+",
        r"(?i)disregard\s+(all\s+)?(prior|previous|above)",
        r"(?i)new\s+instructions?:",
        r"(?i)system\s*prompt\s*:",
    ]

    @classmethod
    def check_harmful_content(cls, text: str) -> GuardrailResult:
        for pattern in cls.HARMFUL_PATTERNS:
            if re.search(pattern, text):
                return GuardrailResult(
                    name="harmful_content",
                    passed=False,
                    message=f"Harmful content pattern detected: {pattern}",
                    severity="critical",
                )
        return GuardrailResult(
            name="harmful_content",
            passed=True,
            message="No harmful content detected",
            severity="info",
        )

    @classmethod
    def check_prompt_injection(cls, text: str) -> GuardrailResult:
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text):
                return GuardrailResult(
                    name="prompt_injection",
                    passed=False,
                    message=f"Potential prompt injection detected: {pattern}",
                    severity="critical",
                )
        return GuardrailResult(
            name="prompt_injection",
            passed=True,
            message="No prompt injection detected",
            severity="info",
        )

    @staticmethod
    def check_output_length(text: str, max_tokens_estimate: int = 2000) -> GuardrailResult:
        estimated_tokens = len(text.split())
        passed = estimated_tokens <= max_tokens_estimate
        return GuardrailResult(
            name="output_length",
            passed=passed,
            message=(
                f"Estimated {estimated_tokens} tokens (limit: {max_tokens_estimate})"
            ),
            severity="warning" if not passed else "info",
        )

    @staticmethod
    def check_language_consistency(text: str, expected_script: str = "latin") -> GuardrailResult:
        if expected_script == "latin":
            non_latin = sum(
                1 for c in text if ord(c) > 127 and not c.isspace() and c not in ".,;:!?'-\""
            )
            ratio = non_latin / max(len(text), 1)
            passed = ratio < 0.3
            return GuardrailResult(
                name="language_consistency",
                passed=passed,
                message=f"Non-latin character ratio: {ratio:.2%}",
                severity="warning" if not passed else "info",
            )
        return GuardrailResult(
            name="language_consistency",
            passed=True,
            message="Script check not implemented for: " + expected_script,
            severity="info",
        )

    @classmethod
    def run_all(cls, text: str) -> list[GuardrailResult]:
        return [
            cls.check_harmful_content(text),
            cls.check_prompt_injection(text),
            cls.check_output_length(text),
            cls.check_language_consistency(text),
        ]

    @classmethod
    def any_critical_failure(cls, text: str) -> bool:
        results = cls.run_all(text)
        return any(not r.passed and r.severity == "critical" for r in results)