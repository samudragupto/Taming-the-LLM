from __future__ import annotations

from typing import Any

import pytest

from src.agent import Agent, AgentError, Entity
from src.client import LLMClient


class TestAgentSummarize:
    def test_returns_non_empty_summary(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("summarize", "good_response")
        result = mock_agent.summarize("Some long text about Python programming.")
        assert len(result) > 0

    def test_summary_content_is_string(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("summarize", "good_response")
        result = mock_agent.summarize("Some long text about Python programming.")
        assert isinstance(result, str)


class TestAgentClassifySentiment:
    def test_positive_sentiment(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("classify_sentiment", "positive_response")
        result = mock_agent.classify_sentiment("I love this product!")
        assert result == "positive"

    def test_negative_sentiment(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("classify_sentiment", "negative_response")
        result = mock_agent.classify_sentiment("This is terrible.")
        assert result == "negative"

    def test_invalid_sentiment_raises(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("classify_sentiment", "invalid_response")
        with pytest.raises(AgentError, match="Unexpected sentiment value"):
            mock_agent.classify_sentiment("I feel okay about it.")


class TestAgentExtractEntities:
    def test_returns_entity_list(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("extract_entities", "good_response")
        result = mock_agent.extract_entities("Guido van Rossum created Python in the Netherlands.")
        assert isinstance(result, list)
        assert all(isinstance(e, Entity) for e in result)

    def test_entity_types_are_valid(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("extract_entities", "good_response")
        result = mock_agent.extract_entities("Guido van Rossum created Python in the Netherlands.")
        valid_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "OTHER"}
        for entity in result:
            assert entity.type in valid_types

    def test_malformed_json_raises(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("extract_entities", "malformed_json_response")
        with pytest.raises(AgentError, match="Failed to parse entity JSON"):
            mock_agent.extract_entities("Some text with entities.")


class TestAgentAnswerQuestion:
    def test_returns_answer(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("answer_question", "good_response")
        result = mock_agent.answer_question(
            context="Python was created by Guido van Rossum in 1991.",
            question="Who created Python?",
        )
        assert len(result) > 0

    def test_returns_unknown_when_no_context(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
    ) -> None:
        configure_mock_completion("answer_question", "unknown_response")
        result = mock_agent.answer_question(
            context="The weather is sunny.",
            question="What is the speed of light?",
        )
        assert "don't know" in result.lower()