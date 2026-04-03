from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from src.agent import Agent
from src.client import LLMClient


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def mock_responses() -> dict[str, Any]:
    with open(FIXTURES_DIR / "mock_responses.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def evaluation_cases() -> dict[str, Any]:
    with open(FIXTURES_DIR / "evaluation_cases.json") as f:
        return json.load(f)


@pytest.fixture
def mock_llm_client() -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client._model = "gpt-4o-mini"
    client._temperature = 0.0
    client._max_retries = 1
    client._http = MagicMock()
    return client


@pytest.fixture
def mock_agent(mock_llm_client: LLMClient) -> Agent:
    return Agent(client=mock_llm_client)


def _build_httpx_response(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=data)


@pytest.fixture
def respx_mock_openai() -> Generator[respx.MockRouter, None, None]:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        yield router


@pytest.fixture
def llm_client_with_respx() -> LLMClient:
    return LLMClient(
        api_key="test-key-not-real",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        timeout=5.0,
        max_retries=2,
    )


@pytest.fixture
def configure_mock_completion(
    mock_llm_client: LLMClient, mock_responses: dict[str, Any]
) -> Any:
    def _configure(task: str, response_key: str) -> None:
        response_data = mock_responses[task][response_key]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_llm_client._http.post.return_value = mock_response  # type: ignore[union-attr]

    return _configure