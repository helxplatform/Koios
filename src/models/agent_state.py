import operator
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from pydantic import Field
from langchain_core.messages import AIMessage


class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    input: str
    # The 'next' field indicates where to route to next
    next: str
    output: Optional[AIMessage]
    chat_history: List = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    user_intent: Optional[Dict[str, Any]] = Field(default_factory=dict)
