from __future__ import annotations

from typing import Any

import pytest

from src.agent import Agent, AgentError, Entity


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


class TestAgentEdgeCases:
    def test_extract_entities_with_invalid_type(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
        mock_responses: dict[str, Any],
    ) -> None:
        """Test entity extraction with invalid type gets normalized to OTHER."""
        import json
        from unittest.mock import MagicMock

        invalid_response = mock_responses["extract_entities"]["good_response"].copy()
        entities_with_invalid = [
            {"entity": "Test", "type": "INVALID_TYPE"},
            {"entity": "Test2", "type": "PERSON"},
        ]
        invalid_response["choices"][0]["message"]["content"] = json.dumps(entities_with_invalid)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = invalid_response
        mock_response.raise_for_status = MagicMock()
        mock_agent._client._http.post.return_value = mock_response

        result = mock_agent.extract_entities("Some text")
        assert any(e.type == "OTHER" for e in result)

    def test_extract_entities_with_markdown_json(
        self,
        mock_agent: Agent,
        configure_mock_completion: Any,
        mock_responses: dict[str, Any],
    ) -> None:
        """Test entity extraction strips markdown code blocks."""
        import json
        from unittest.mock import MagicMock

        base_response = mock_responses["extract_entities"]["good_response"].copy()
        entities = [{"entity": "Test", "type": "PERSON"}]
        base_response["choices"][0]["message"]["content"] = f"```json\n{json.dumps(entities)}\n```"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = base_response
        mock_response.raise_for_status = MagicMock()
        mock_agent._client._http.post.return_value = mock_response

        result = mock_agent.extract_entities("Some text")
        assert len(result) > 0
        assert result[0].entity == "Test"
