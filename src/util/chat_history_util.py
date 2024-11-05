from bs4 import BeautifulSoup
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Tuple


def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    """Formats chat history as AI and Human messages"""
    buffer = []
    for human, ai in chat_history:
        soup = BeautifulSoup(human, features="html.parser")
        buffer.append(HumanMessage(content=soup.get_text()))
        soup = BeautifulSoup(ai, features="html.parser")
        buffer.append(AIMessage(content=soup.get_text()))
    return buffer

