from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from api import app


def make_agent_result(
    text: str = "Here's your post!", image_url: str | None = None
) -> dict:
    return {
        "messages": [AIMessage(content=text)],
        "image_url": image_url,
        "llm_calls": 3,
    }


async def test_generate_returns_200() -> None:
    with patch(
        "api.agent.invoke",
        return_value=make_agent_result(image_url="https://example.com/image.png"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/generate", json={"prompt": "Write about AI"})

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Here's your post!"
    assert data["image_url"] == "https://example.com/image.png"
    assert data["llm_calls"] == 3
    assert "request_id" in data


async def test_generate_rejects_empty_prompt() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/generate", json={"prompt": ""})

    assert response.status_code == 422


async def test_generate_returns_500_on_agent_failure() -> None:
    with patch("api.agent.invoke", side_effect=Exception("something went wrong")):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/generate", json={"prompt": "Write about AI"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to generate post"
