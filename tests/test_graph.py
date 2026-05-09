from collections import deque
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from main import agent
from schemas import ReviewDecision
from settings import settings


def _stub_image_client(mock_client: MagicMock) -> None:
    mock_client.images.generate.return_value.data = [MagicMock(url="https://example.com/img.png")]


def sequential(*responses):
    """
    Returns a side_effect callable that yields responses in order.
    Raises IndexError (not StopIteration) when exhausted, so it does not
    trigger PEP 479 RuntimeError inside LangGraph's generator-based runner.
    """
    queue: deque = deque(responses)

    def _call(*args, **kwargs):
        return queue.popleft()

    return _call


@patch("nodes.content_nodes.client")
@patch("nodes.content_nodes.model")
def test_graph_reject_once_then_approve(mock_model: MagicMock, mock_client: MagicMock) -> None:
    """
    Graph flow (DRAFT_TEXT → REVIEW_DRAFT → conditional):
      1. DRAFT_TEXT: model.invoke → "First draft"
      2. REVIEW_DRAFT: review → rejected, revision_count=1
      3. DRAFT_TEXT: model.invoke → "Revised draft"
      4. REVIEW_DRAFT: review → approved, revision_count=2
      5. GENERATE_IMAGE_PROMPT: model.invoke → "Image prompt text"
      6. GENERATE_IMAGE: client.images.generate
    """
    _stub_image_client(mock_client)
    mock_model.invoke.side_effect = sequential(
        AIMessage(content="First draft"),
        AIMessage(content="Revised draft"),
        AIMessage(content="Image prompt text"),
    )
    mock_model.with_structured_output.return_value.invoke.side_effect = sequential(
        ReviewDecision(feedback="Needs work.", approved=False),
        ReviewDecision(feedback="Looks good.", approved=True),
    )

    result = agent.invoke({"messages": [HumanMessage(content="Write about AI")]})

    assert result["approved"] is True
    assert result["revision_count"] == 2
    assert result["draft_text"] == "Revised draft"
    assert result["review_feedback"] == "Looks good."
    assert result["image_prompt"] == "Image prompt text"
    assert result["image_url"] == "https://example.com/img.png"


@patch("nodes.content_nodes.client")
@patch("nodes.content_nodes.model")
def test_graph_escape_hatch_fires_at_max_revisions(mock_model: MagicMock, mock_client: MagicMock) -> None:
    """All reviews reject; graph escapes to image generation after max_revisions."""
    _stub_image_client(mock_client)
    mock_model.invoke.return_value = AIMessage(content="Draft content")
    mock_model.with_structured_output.return_value.invoke.return_value = ReviewDecision(
        feedback="Still not good.", approved=False
    )

    result = agent.invoke({"messages": [HumanMessage(content="Write about AI")]})

    assert result["approved"] is False
    assert result["revision_count"] >= settings.max_revisions
    assert result["image_url"] == "https://example.com/img.png"


@patch("nodes.content_nodes.client")
@patch("nodes.content_nodes.model")
def test_graph_draft_text_is_explicit_not_scanned(mock_model: MagicMock, mock_client: MagicMock) -> None:
    """draft_text is written by the node directly — not derived from message history."""
    _stub_image_client(mock_client)
    mock_model.invoke.side_effect = sequential(
        AIMessage(content="The actual post text"),
        AIMessage(content="Image prompt text"),
    )
    mock_model.with_structured_output.return_value.invoke.return_value = ReviewDecision(
        feedback="Approved.", approved=True
    )

    result = agent.invoke({"messages": [HumanMessage(content="Write about AI")]})

    assert result["draft_text"] == "The actual post text"
    assert result["image_prompt"] == "Image prompt text"
