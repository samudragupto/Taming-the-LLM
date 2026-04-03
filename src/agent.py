from __future__ import annotations

import json
from dataclasses import dataclass

import structlog

from src.client import LLMClient, LLMResponse
from src.prompts import ANSWER_QUESTION, CLASSIFY_SENTIMENT, EXTRACT_ENTITIES, SUMMARIZE

logger = structlog.get_logger()


@dataclass(frozen=True)
class Entity:
    entity: str
    type: str


class AgentError(Exception):
    pass


class Agent:
    """High-level AI agent that wraps structured tasks around the LLM client."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        user_msg = SUMMARIZE.render_user(text=text, max_sentences=str(max_sentences))
        response = self._client.complete(system=SUMMARIZE.system, user=user_msg)
        self._log_response("summarize", response)
        return response.content.strip()

    def classify_sentiment(self, text: str) -> str:
        user_msg = CLASSIFY_SENTIMENT.render_user(text=text)
        response = self._client.complete(
            system=CLASSIFY_SENTIMENT.system,
            user=user_msg,
            temperature=0.0,
        )
        self._log_response("classify_sentiment", response)
        sentiment = response.content.strip().lower()
        if sentiment not in ("positive", "negative", "neutral"):
            raise AgentError(
                f"Unexpected sentiment value: '{sentiment}'. "
                "Expected one of: positive, negative, neutral."
            )
        return sentiment

    def extract_entities(self, text: str) -> list[Entity]:
        user_msg = EXTRACT_ENTITIES.render_user(text=text)
        response = self._client.complete(system=EXTRACT_ENTITIES.system, user=user_msg)
        self._log_response("extract_entities", response)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            raw_entities = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AgentError(f"Failed to parse entity JSON: {content}") from exc

        valid_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "OTHER"}
        entities = []
        for item in raw_entities:
            entity_type = item.get("type", "OTHER").upper()
            if entity_type not in valid_types:
                entity_type = "OTHER"
            entities.append(Entity(entity=item["entity"], type=entity_type))
        return entities

    def answer_question(self, context: str, question: str) -> str:
        user_msg = ANSWER_QUESTION.render_user(context=context, question=question)
        response = self._client.complete(system=ANSWER_QUESTION.system, user=user_msg)
        self._log_response("answer_question", response)
        return response.content.strip()

    @staticmethod
    def _log_response(task: str, response: LLMResponse) -> None:
        logger.info(
            "agent_task_completed",
            task=task,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
