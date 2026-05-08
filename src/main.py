from dotenv import load_dotenv

# Load environment variables BEFORE any other imports
load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from nodes.constants import (  # noqa: E402
    DRAFT_TEXT,
    GENERATE_IMAGE,
    GENERATE_IMAGE_PROMPT,
    REVIEW_DRAFT,
)
from nodes.content_nodes import (  # noqa: E402
    draft_text_node,
    generate_image_prompt_node,
    generate_image_with_openai,
    review_draft_node,
)
from state import MessageState  # noqa: E402


def should_continue(state: MessageState) -> str:
    """Determine if the agent should continue"""
    if state.get("llm_calls", 0) > 3:
        return GENERATE_IMAGE_PROMPT
    return REVIEW_DRAFT


# build the agent
agent_builder = StateGraph(state_schema=MessageState)

# add nodes
agent_builder.add_node(DRAFT_TEXT, draft_text_node)
agent_builder.add_node(REVIEW_DRAFT, review_draft_node)
agent_builder.add_node(GENERATE_IMAGE_PROMPT, generate_image_prompt_node)
agent_builder.add_node(GENERATE_IMAGE, generate_image_with_openai)
agent_builder.set_entry_point(DRAFT_TEXT)

# add edges
agent_builder.add_conditional_edges(
    DRAFT_TEXT,
    should_continue,
    path_map={GENERATE_IMAGE_PROMPT: GENERATE_IMAGE_PROMPT, REVIEW_DRAFT: REVIEW_DRAFT},
)
agent_builder.add_edge(REVIEW_DRAFT, DRAFT_TEXT)
agent_builder.add_edge(GENERATE_IMAGE_PROMPT, GENERATE_IMAGE)
agent_builder.add_edge(GENERATE_IMAGE, END)

agent = agent_builder.compile()

if __name__ == "__main__":
    query = """
    Write a post about artificial intelligence and the most common use cases for agentic workflows."
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    result = agent.invoke(inputs)
    print(result)
