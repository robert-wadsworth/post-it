from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from nodes.prompts import (
    DRAFT_TEXT_SYSTEM_PROMPT,
    GENERATE_IMAGE_PROMPT_SYSTEM_PROMPT,
    REVIEW_DRAFT_SYSTEM_PROMPT,
)
from schemas import ReviewDecision
from settings import settings
from state import MessageState

client = OpenAI()

model = ChatOpenAI(model=settings.model_name, temperature=0.7)


def draft_text_node(state: MessageState) -> MessageState:
    """Create an initial draft for a social media post"""

    messages = [SystemMessage(content=DRAFT_TEXT_SYSTEM_PROMPT)] + state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def review_draft_node(state: MessageState) -> MessageState:
    """Review the draft for accuracy, clarity, and best practices"""

    messages = [SystemMessage(content=REVIEW_DRAFT_SYSTEM_PROMPT)] + state["messages"]

    decision: ReviewDecision = model.with_structured_output(ReviewDecision).invoke(
        messages
    )

    return {
        "messages": [AIMessage(content=decision.feedback)],
        "approved": decision.approved,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def generate_image_prompt_node(state: MessageState) -> MessageState:
    """Generate a prompt for an image to accompany the post"""

    messages = [SystemMessage(content=GENERATE_IMAGE_PROMPT_SYSTEM_PROMPT)] + state[
        "messages"
    ]
    response = model.invoke(messages)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def generate_image_with_openai(state: MessageState) -> MessageState:
    """Generate an image based on the prompt from the last message"""

    # Extract the prompt from the last message
    if not state.get("messages"):
        raise ValueError("No messages found in state to extract image prompt")

    last_message = state["messages"][-1]
    prompt = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    image_response = client.images.generate(
        model=settings.image_model,
        prompt=prompt,
        n=1,
        size=settings.image_size,
    )

    if not image_response.data:
        raise ValueError("Image generation returned no data")

    return {
        "image_url": image_response.data[0].url,
        "llm_calls": state.get("llm_calls", 0),
    }
