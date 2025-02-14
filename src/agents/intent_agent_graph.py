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
from typing import Sequence, TypedDict, Annotated, Any

class AgentState(TypedDict):
    """
    The state schema for the agent workflow. This keeps track of:
    - Input messages
    - The next agent to call
    - Chat history
    - Extra metadata (like user preferences)
    """
    input: Annotated[Sequence[BaseMessage], operator.add]  # Sequence of messages
    next: str  # Next node in workflow
    chat_history: list[BaseMessage]  # Stores user interactions
    extra: dict[str, Any]  # Stores extracted preferences and metadata


def extract_user_preferences_node(state: AgentState) -> AgentState:
    """
    Uses LLM to extract user preferences based on chat history.
    Stores the extracted preferences in the `extra` field of `AgentState`.
    """
    chat_history = state.get("chat_history", [])
    preference_query = (
        "Analyze the user's past messages and infer preferences for response formatting, "
        "content restrictions, and preferred response structure. "
        "Respond in JSON format with keys like 'blocked_terms', 'response_format', etc."
    )

    llm = LLMFactory(config=app_config)
    response = llm.invoke([HumanMessage(content=preference_query + "\n\nChat History:\n" + str(chat_history))])

    try:
        extracted_preferences = json.loads(response.content)
    except json.JSONDecodeError:
        extracted_preferences = {"blocked_terms": [], "response_format": "list"}  # Default preferences

    # Store preferences in state
    state["extra"]["user_preferences"] = extracted_preferences
    state["next"] = "supervisor"  # Route to the supervisor next

    return state

def analyze_intent_and_scope(query: str) -> dict:
    """
    Analyzes the user intent and determines if they are interested in a single node or multiple nodes.
    """
    intent_cat_query = (
        f"Please analyze the intent of the following query and classify it into one or more of the given categories: "
        f"'{query}'. Categories: 1. Factual Queries, 2. Explanatory Inquiries, 3. Troubleshooting Assistance, "
        f"4. Decision Support, 5. Learning Support, 6. Personal Advice, 7. Data Processing, 8. Research Questions, 9. Not Research Related. "
        "Respond only with the category numbers."
    )

    scope_query = (
        f"Determine if the following query refers to a single entity or multiple entities: '{query}'. "
        "Respond with 'single' or 'multiple'."
    )

    llm = LLMFactory(config=app_config)
    
    print("LLM type:", type(llm)) # debugging

    intent_response = llm.invoke([HumanMessage(content=intent_cat_query)])
    scope_response = llm.invoke([HumanMessage(content=scope_query)])


    # Parse the responses
    try:
        intents = list(map(int, intent_response.content.split(",")))
    except ValueError:
        intents = []

    scope = scope_response.content.strip().lower()  # Expecting 'single' or 'multiple'

    return {"intents": intents, "scope": scope}


def intent_node(state: AgentState) -> AgentState:
    # Get the current user input
    query = state['input'][-1].content

    # Analyze intent and scope
    analysis = analyze_intent_and_scope(query)
    intents = analysis["intents"]
    scope = analysis["scope"]  # "single" or "multiple"

    # Save the analysis to the state
    state['intents'] = intents
    state['scope'] = scope

    # Log the query, intents, and scope to a JSON file
    log_query_intent(query, intents, scope)

    # Route to the appropriate agent or supervisor
    state['next'] = "supervisor"
    return state


def log_query_intent(query: str, intents: List[int], scope: str, filename: str = "query_intents.json"):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append({"query": query, "intents": intents, "scope": scope})

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
   
# Define the intent node
def intent_node(state: AgentState) -> AgentState:
    # Get the current user input
    query = state['input'][-1].content
    # Analyze the intent of the query
    intents, scope = analyze_intent_and_scope(query)
    # Saving the intents for now 
    state['intents'] = intents

    state['scope'] = scope 

    # Log the query and intents to a JSON file
    log_query_intent(query, intents, scope)
    
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

# Add the intent detection node (if intent detection is used)
workflow.add_node("intent", intent_node)

# Add the user preference extraction node BEFORE the supervisor
workflow.add_node("extract_user_preferences", extract_user_preferences_node)

# Add the main agents (KG lookup and QV lookup)
workflow.add_node("KG_lookup", kg_lookup_agent_node)
workflow.add_node("QV_lookup", qv_lookup_agent_node)

# Add the supervisor (which now considers extracted preferences)
workflow.add_node("supervisor", supervisor_agent.as_generative_chain())

# Connect nodes in order
workflow.add_edge(START, "intent")  # Detecting intent
workflow.add_edge("intent", "extract_user_preferences")  # Extracting user preferences
workflow.add_edge("extract_user_preferences", "supervisor")  # Passing preferences to supervisor

# Define how members are laid out (KG_lookup, QV_lookup)
for member in members:
    workflow.add_edge(member, "supervisor")  # After lookup, return to supervisor

# Define conditional routing logic
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
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