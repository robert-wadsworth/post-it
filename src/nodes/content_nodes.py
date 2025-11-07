import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from nodes.constants import (
    DRAFT_TEXT,
    GENERATE_IMAGE,
    GENERATE_IMAGE_PROMPT,
    REVIEW_DRAFT,
)
from state import MessageState

client = OpenAI()


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def draft_text_node(state: MessageState) -> MessageState:
    """Create an initial draft for a social media post"""

    messages = [
        SystemMessage(
            content="""
            You are a helpful assistant that creates social media posts.
            You will be given a topic and you will need to create a draft of a social media post.
            You will need to return the draft of the post.

            If the user provides feedback, revise the most recent draft of the post to address the feedback.
            """
        )
    ] + state["messages"]

    response = model.invoke(messages)

    return {
        "node_name": DRAFT_TEXT,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def review_draft_node(state: MessageState) -> MessageState:
    """Review the draft for accuracy, clarity, and best practices"""

    messages = [
        SystemMessage(
            content="""
            You are a helpful assistant that reviews social media posts for accuracy and clarity and best practices.
            You will be given a draft of a social media post and you will need to review it for accuracy, clarity, and best practices.
            You will need to return a list of suggestions for the post.
            Do not provide a revised version of the post, only suggest changes.
            """
        )
    ] + state["messages"]

    response = model.invoke(messages)

    return {
        "node_name": REVIEW_DRAFT,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def generate_image_prompt_node(state: MessageState) -> MessageState:
    """Generate a prompt for an image to accompany the post"""

    messages = [
        SystemMessage(
            content="""
            You are an expert Prompt Engineer that generates prompts for images to accompany social media posts.
            Given the post, generate a prompt for an image that would accompany the post.
            You will need to return the prompt.
            Do not provide a revised version of the post, only suggest changes.
            """
        )
    ] + state["messages"]
    response = model.invoke(messages)

    return {
        "node_name": GENERATE_IMAGE_PROMPT,
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

    # Generate the image using OpenAI
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
    )

    # Create a message with the image URL
    image_url = image_response.data[0].url
    image_message = HumanMessage(content=f"Image generated: {image_url}")

    return {
        "node_name": GENERATE_IMAGE,
        "messages": [image_message],
        "llm_calls": state.get("llm_calls", 0),
    }
