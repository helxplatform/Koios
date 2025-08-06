from pydantic import Field, BaseModel
from typing import List, Tuple


class Question(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})

class SimpleQuery(BaseModel):
    query: str