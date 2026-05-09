import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    approved: bool
    image_url: Optional[str]
    draft_text: str
    image_prompt: str
    review_feedback: str
    revision_count: int
