from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from nodes.content_nodes import (
    draft_text_node,
    generate_image_prompt_node,
    generate_image_with_openai,
    review_draft_node,
)
from schemas import ReviewDecision


def make_state(
    messages: list | None = None, llm_calls: int = 0, revision_count: int = 0
) -> dict:
    return {
        "messages": messages or [],
        "llm_calls": llm_calls,
        "approved": False,
        "image_url": None,
        "draft_text": "",
        "image_prompt": "",
        "review_feedback": "",
        "revision_count": revision_count,
    }


@patch("nodes.content_nodes.model")
def test_draft_text_node(mock_model: MagicMock) -> None:
    mock_response = AIMessage(content="Draft post content")
    mock_model.invoke.return_value = mock_response

    result = draft_text_node(
        make_state(messages=[HumanMessage(content="Write about AI")])
    )

    assert result["messages"] == [mock_response]
    assert result["llm_calls"] == 1
    assert result["draft_text"] == "Draft post content"


@patch("nodes.content_nodes.model")
def test_review_draft_node_approved(mock_model: MagicMock) -> None:
    mock_model.with_structured_output.return_value.invoke.return_value = ReviewDecision(
        feedback="Looks great!", approved=True
    )

    result = review_draft_node(
        make_state(messages=[HumanMessage(content="Draft post")])
    )

    assert result["approved"] is True
    assert result["messages"][0].content == "Looks great!"
    assert result["llm_calls"] == 1
    assert result["review_feedback"] == "Looks great!"
    assert result["revision_count"] == 1


@patch("nodes.content_nodes.model")
def test_review_draft_node_not_approved(mock_model: MagicMock) -> None:
    mock_model.with_structured_output.return_value.invoke.return_value = ReviewDecision(
        feedback="Needs more detail.", approved=False
    )

    result = review_draft_node(
        make_state(messages=[HumanMessage(content="Draft post")])
    )

    assert result["approved"] is False
    assert result["messages"][0].content == "Needs more detail."
    assert result["review_feedback"] == "Needs more detail."
    assert result["revision_count"] == 1


@patch("nodes.content_nodes.model")
def test_generate_image_prompt_node(mock_model: MagicMock) -> None:
    mock_response = AIMessage(
        content="A vivid digital painting of robots collaborating"
    )
    mock_model.invoke.return_value = mock_response

    result = generate_image_prompt_node(
        make_state(messages=[HumanMessage(content="Post about AI")])
    )

    assert result["messages"] == [mock_response]
    assert result["llm_calls"] == 1
    assert result["image_prompt"] == "A vivid digital painting of robots collaborating"


@patch("nodes.content_nodes.client")
def test_generate_image_with_openai(mock_client: MagicMock) -> None:
    mock_client.images.generate.return_value.data = [
        MagicMock(url="https://example.com/image.png")
    ]

    result = generate_image_with_openai(
        make_state(messages=[HumanMessage(content="A vivid digital painting")])
    )

    assert result["image_url"] == "https://example.com/image.png"


def test_generate_image_raises_on_empty_messages() -> None:
    with pytest.raises(ValueError, match="No messages found"):
        generate_image_with_openai(make_state(messages=[]))


@patch("nodes.content_nodes.client")
def test_generate_image_raises_on_empty_data(mock_client: MagicMock) -> None:
    mock_client.images.generate.return_value.data = []

    with pytest.raises(ValueError, match="no data"):
        generate_image_with_openai(make_state(messages=[HumanMessage(content="A prompt")]))


@patch("nodes.content_nodes.client")
def test_generate_image_raises_on_provider_error(mock_client: MagicMock) -> None:
    mock_client.images.generate.side_effect = RuntimeError("provider error")

    with pytest.raises(RuntimeError, match="provider error"):
        generate_image_with_openai(make_state(messages=[HumanMessage(content="A prompt")]))
