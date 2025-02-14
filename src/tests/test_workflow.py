import os
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from agents.intent_agent_graph import graph 
import langchain_openai

def test_intent_processing():
    """
    Tests the intent detection and KG/QV lookup process.
    """
    test_state = {
        "chat_history": [
            {"role": "user", "content": "Tell me more about Ibuprofen."}
        ],
        "input": [
            HumanMessage(content="Tell me more about Ibuprofen.")
        ]
    }

    thread_config = {"configurable": {"thread_id": "1"}}

    for state in graph.stream(test_state, config=thread_config):
        print(state)

if __name__ == "__main__":
    test_intent_processing()