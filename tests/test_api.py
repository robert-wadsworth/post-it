from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from api import app


def make_agent_result(text: str = "Here's your post!", image_url: str | None = None) -> dict:
    return {
        "messages": [AIMessage(content=text)],
        "image_url": image_url,
        "llm_calls": 3,
    }


async def test_generate_returns_200_with_valid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "test-token")

    with patch("api.agent.invoke", return_value=make_agent_result(image_url="https://example.com/image.png")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={"prompt": "Write about AI"},
                headers={"Authorization": "Bearer test-token"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Here's your post!"
    assert data["image_url"] == "https://example.com/image.png"
    assert data["llm_calls"] == 3


async def test_generate_returns_401_with_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "real-token")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/generate",
            json={"prompt": "Write about AI"},
            headers={"Authorization": "Bearer wrong-token"},
        )

    assert response.status_code == 401


async def test_generate_returns_401_with_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "real-token")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/generate", json={"prompt": "Write about AI"})

    assert response.status_code == 401
