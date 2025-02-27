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
    input: Annotated[Sequence[BaseMessage], operator.add]
    # Stores a list of the next agents to process the request
    next: Annotated[List[str], operator.add] 
    # stores previous user interactions 
    chat_history: list[BaseMessage]
    # used to store extracted user preferences (can be more than one)
    extra: dict[str, Any]


def extract_user_preferences_node(state: AgentState) -> AgentState:
    chat_history = state.get("chat_history", [])
    preference_query = (
        "Analyze the user's past messages and infer preferences for response formatting, "
        "content restrictions, and preferred response structure. "
        "Respond in JSON format with keys like 'blocked_terms', 'response_format', etc."
    )

    llm = LLMFactory(config=app_config)
    response = llm.invoke([HumanMessage(content=preference_query + "\n\nChat History:\n" + str(chat_history))])

    try:
        # extract user preference from the response 
        extracted_preferences = json.loads(response.content)
    except json.JSONDecodeError:
        # typically in LLMs, for safety reasons, we can send in a blocked_term
        extracted_preferences = {"blocked_terms": [], "response_format": "list"}  

    if not isinstance(state.get("next"), list):
        state["next"] = []

    # supervisor is always added next so it continues routing
    if "supervisor" not in state["next"]:
        state["next"].append("supervisor")

    
    print(f"[DEBUG] After fixing `next` in `extract_user_preferences_node`: {state['next']} (Type: {type(state['next'])})")

    # Store extracted preferences to next
    state.setdefault("extra", {})
    state["extra"]["user_preferences"] = extracted_preferences

    return state



def analyze_intent_and_scope(query: str) -> dict:
    """
    Analyzes the user intent and determines what kind of question they're interested in
    """
    # what type of question is a user asking?
    intent_cat_query = (
        f"Please analyze the intent of the following query and classify it into one or more of the given categories: "
        f"'{query}'. Categories: 1. Factual Queries, 2. Explanatory Inquiries, 3. Troubleshooting Assistance, "
        f"4. Decision Support, 5. Learning Support, 6. Personal Advice, 7. Data Processing, 8. Research Questions, 9. Not Research Related. "
        "Respond only with the category numbers."
    )
    # what is the scope of their query? a single or multiple node response
    scope_query = (
        f"Determine if the following query refers to a single entity or multiple entities: '{query}'. "
        "Respond with 'single' or 'multiple'."
    )

    llm = LLMFactory(config=app_config)

    intent_response = llm.invoke([HumanMessage(content=intent_cat_query)])
    scope_response = llm.invoke([HumanMessage(content=scope_query)])

    # Parse the responses
    try:
        intents = list(map(int, intent_response.content.split(",")))
    except ValueError:
        intents = []

    # Expecting 'single' or 'multiple'
    scope = scope_response.content.strip().lower()  

    return {"intents": intents, "scope": scope}

def intent_node(state: AgentState) -> AgentState:
    # logs the detected intent and scope and ensures there are no duplicate
    # agents in next
    query = state["input"][-1].content
    analysis = analyze_intent_and_scope(query)
    intents = analysis["intents"]
    scope = analysis["scope"]

    state["intents"] = intents
    state["scope"] = scope

    log_query_intent(query, intents, scope)

    if not isinstance(state.get("next"), list):
        state["next"] = []

    state["next"] = list(set(state["next"]))

    print(f"[DEBUG] After fixing `next` in `intent_node`: {state['next']} (Type: {type(state['next'])})")

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
conditional_map["supervisor"] = "supervisor"  
conditional_map["FINISH"] = END

# Ensure supervisor is added as a valid node
if "supervisor" not in members:
    members["supervisor"] = "Main agent that decides the next step based on user preferences"


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