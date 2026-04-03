from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.agent import Agent
from src.client import LLMClient, LLMClientError
from src.evaluation.guardrails import GuardrailChecker
from src.evaluation.properties import PropertyChecker


@pytest.mark.integration
class TestLLMClientWithMockedHTTP:
    def test_successful_completion(
        self,
        respx_mock_openai: respx.MockRouter,
        llm_client_with_respx: LLMClient,
        mock_responses: dict[str, Any],
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_responses["summarize"]["good_response"])
        )
        response = llm_client_with_respx.complete(
            system="You are a summarizer.",
            user="Summarize this text.",
        )
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "gpt-4o-mini"
        assert response.prompt_tokens > 0
        assert response.latency_ms >= 0

    def test_retry_on_server_error(
        self,
        respx_mock_openai: respx.MockRouter,
        llm_client_with_respx: LLMClient,
        mock_responses: dict[str, Any],
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(500, json={"error": "internal server error"}),
                httpx.Response(200, json=mock_responses["summarize"]["good_response"]),
            ]
        )
        response = llm_client_with_respx.complete(
            system="You are a summarizer.",
            user="Summarize this.",
        )
        assert response.content is not None

    def test_all_retries_exhausted(
        self,
        respx_mock_openai: respx.MockRouter,
        llm_client_with_respx: LLMClient,
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "server down"})
        )
        with pytest.raises(LLMClientError, match="All 2 attempts failed"):
            llm_client_with_respx.complete(
                system="System prompt.",
                user="User message.",
            )


@pytest.mark.integration
class TestEndToEndAgentPipeline:
    def test_summarize_pipeline(
        self,
        respx_mock_openai: respx.MockRouter,
        mock_responses: dict[str, Any],
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_responses["summarize"]["good_response"])
        )
        client = LLMClient(api_key="fake", base_url="https://api.openai.com/v1")
        agent = Agent(client=client)

        summary = agent.summarize(mock_responses["summarize"]["input_text"], max_sentences=3)

        assert PropertyChecker.check_not_empty(summary).passed
        assert PropertyChecker.check_max_sentences(summary, max_sentences=5).passed
        assert PropertyChecker.check_no_pii(summary).passed
        assert not GuardrailChecker.any_critical_failure(summary)

    def test_sentiment_pipeline(
        self,
        respx_mock_openai: respx.MockRouter,
        mock_responses: dict[str, Any],
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(
                200, json=mock_responses["classify_sentiment"]["positive_response"]
            )
        )
        client = LLMClient(api_key="fake", base_url="https://api.openai.com/v1")
        agent = Agent(client=client)

        sentiment = agent.classify_sentiment("I love this product so much!")

        assert PropertyChecker.check_value_in_set(
            sentiment, {"positive", "negative", "neutral"}
        ).passed
        assert PropertyChecker.check_not_empty(sentiment).passed
        assert not GuardrailChecker.any_critical_failure(sentiment)

    def test_entity_extraction_pipeline(
        self,
        respx_mock_openai: respx.MockRouter,
        mock_responses: dict[str, Any],
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(
                200, json=mock_responses["extract_entities"]["good_response"]
            )
        )
        client = LLMClient(api_key="fake", base_url="https://api.openai.com/v1")
        agent = Agent(client=client)

        entities = agent.extract_entities(
            "Guido van Rossum created Python at Google in the Netherlands."
        )

        assert len(entities) > 0
        valid_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "OTHER"}
        for entity in entities:
            assert entity.type in valid_types
            assert len(entity.entity) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestSemanticIntegration:
    def test_summary_semantic_quality(
        self,
        respx_mock_openai: respx.MockRouter,
        mock_responses: dict[str, Any],
        semantic: Any,
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_responses["summarize"]["good_response"])
        )
        client = LLMClient(api_key="fake", base_url="https://api.openai.com/v1")
        agent = Agent(client=client)

        summary = agent.summarize(mock_responses["summarize"]["input_text"])

        semantic.assert_similar(
            candidate=summary,
            reference="Python is a popular and versatile programming language used for web development, data science, and automation.",
            threshold=0.5,
        )

    def test_answer_semantic_quality(
        self,
        respx_mock_openai: respx.MockRouter,
        mock_responses: dict[str, Any],
        semantic: Any,
    ) -> None:
        respx_mock_openai.post("/chat/completions").mock(
            return_value=httpx.Response(
                200, json=mock_responses["answer_question"]["good_response"]
            )
        )
        client = LLMClient(api_key="fake", base_url="https://api.openai.com/v1")
        agent = Agent(client=client)

        answer = agent.answer_question(
            context="Python was created by Guido van Rossum and first released in 1991.",
            question="Who created Python?",
        )

        semantic.assert_similar(
            candidate=answer,
            reference="Guido van Rossum created Python.",
            threshold=0.5,
        )

        semantic.assert_not_similar(
            candidate=answer,
            reference="Java was created by James Gosling at Sun Microsystems.",
            threshold=0.85,
        )