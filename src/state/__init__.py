import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage


class MessageState(TypedDict):
    node_name: str
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    approved: bool
    image_url: Optional[str]
