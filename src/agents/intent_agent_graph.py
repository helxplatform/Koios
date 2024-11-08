import functools
import operator
import json
from typing import Sequence, TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from util.llm_helper import LLMFactory
from langgraph.graph import END, StateGraph, START
from agents.utils import *
from agents.supervisor import SupervisorAgent
from langgraph.checkpoint.memory import MemorySaver
from chains.question_lookup_chain import QuestionLookupChain
from chains.kg_chain import KGChain
import config as app_config


class AgentState(TypedDict):
    # The agent state is the input to each node in the graph
    # Our state Schema (https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)
    # The annotation tells the graph that new messages will always
    # be added to the current states
    input: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str
    chat_history: List = []


def analyze_intent(query: str) -> List[int]:
    # Helper function to analyze user intent
    intent_cat_query = (
        f"Please analyze the potential intent of the following query and identify it as one or more of the given categories: "
        f"'{query}'. Categories: 1. Factual Queries, 2. Explanatory Inquiries, 3. Troubleshooting Assistance, "
        f"4. Decision Support, 5. Learning Support, 6. Personal Advice, 7. Data Processing, 8. Research Questions, 9. Not Research Related. "
        "Respond only with the category numbers."
    )
    # Run the LLM to analyze the query
    llm = LLMFactory(config=app_config)
    response = llm([HumanMessage(content=intent_cat_query)])
    # Convert the response to a list of integers
    return list(map(int, response.content.split(",")))


def log_query_intent(query: str, intents: List[int], filename: str = "query_intents.json"):
    # Helper function to log queries and intents to a JSON file
    try:
        # Load existing data
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, initialize an empty list
        data = []

    # Append new query and intents
    data.append({
        "query": query,
        "intents": intents
    })
    
    # Write updated data to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Define the intent node
def intent_node(state: AgentState) -> AgentState:
    # Get the current user input
    query = state['input'][-1].content
    # Analyze the intent of the query
    intents = analyze_intent(query)
    # Save the intent categories to the state (you can log or process it further)
    state['intents'] = intents
    
    # Log the query and intents to a JSON file
    log_query_intent(query, intents)
    
    # After identifying the intent, route to the appropriate agent or supervisor
    state['next'] = "supervisor"  # or another agent based on intent
    return state


# Create our agents (KG lookup and QV lookup)
kg_lookup_agent_node = functools.partial(agent_node_dict, agent=KGChain(app_config).as_generative_chain(), name="KG_lookup_agent")
qv_lookup_agent_node = functools.partial(agent_node_dict, agent=QuestionLookupChain(app_config).as_generative_chain(), name="QV_lookup_agent")
supervisor_agent = SupervisorAgent(app_config)
members = supervisor_agent.members

# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)

# Add the intent detection node
workflow.add_node("intent", intent_node)
# Add the nodes to the workflow
workflow.add_node("KG_lookup", kg_lookup_agent_node)
workflow.add_node("QV_lookup", qv_lookup_agent_node)
workflow.add_node("supervisor", supervisor_agent.as_generative_chain())


# Define how members are laid out (researcher, comedian, etc.)
for member in members:
    workflow.add_edge(member, END)  # Ends after running the agent unless routed to the supervisor

# The supervisor populates the "next" field in the graph state which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
# Connect supervisor node with all the members
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Add the entry point, making the supervisor the one that accepts user input
workflow.add_edge(START, "intent")  # Intent analysis is the first step, then routes to supervisor
workflow.add_edge("intent", "supervisor")

# Set up memory
memory = MemorySaver()

# Compile the graph with memory
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    # Test code to run
    graph.get_graph().print_ascii()
    thread_config = {"configurable": {"thread_id": "1"}}
    
    for s in graph.stream(
            {
                # Mimicking previous interactions.
                "chat_history": [
                    ("jokes around Heart", "the heart is melting"),
                ],
                # Current question.
                "input": [
                    HumanMessage(content="explain that more"),
                ]
            }, config=thread_config
    ):
        if "__end__" not in s:
            print(s)
            state = graph.get_state(thread_config)
            print(state)
            # Prints the detected intents


# the JSON storage file and intents should look like this 

# [
#     {
#         "query": "explain that more",
#         "intents": [2, 5]
#     },
#     {
#         "query": "how to troubleshoot this error?",
#         "intents": [3]
#     }
# ]