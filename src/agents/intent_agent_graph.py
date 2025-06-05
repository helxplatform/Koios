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
from chains import QuestionLookupChain, KGChain
import config as app_config
from typing import Sequence, TypedDict, Annotated, Any
import config


class AgentState(TypedDict):
    input: Annotated[Sequence[BaseMessage], operator.add]
    # Stores a list of the next agents to process the request
    next: List[str]
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

    state.setdefault("extra", {})  

    state["extra"]["intents"] = intents
    state["extra"]["scope"] = scope

    return state


    return state



    

# Create our agents (KG lookup and QV lookup)
kg_lookup_agent_node = functools.partial(agent_node_dict, agent=KGChain(app_config).as_generative_chain(), name="KG_lookup_agent")
qv_lookup_agent_node = functools.partial(agent_node_dict, agent=QuestionLookupChain(app_config).as_generative_chain(), name="QV_lookup_agent")
supervisor_agent = SupervisorAgent(app_config)
members = supervisor_agent.members

# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)

# Add the intent and preference nodes
workflow.add_node("intent", intent_node)
workflow.add_node("extract_user_preferences", extract_user_preferences_node)

# Add KG and QV nodes
workflow.add_node("KG_lookup", kg_lookup_agent_node)
workflow.add_node("QV_lookup", qv_lookup_agent_node)

# Add the supervisor node
workflow.add_node("supervisor", supervisor_agent.as_generative_chain())

# Routing sequence
workflow.add_edge(START, "intent")
workflow.add_edge("intent", "extract_user_preferences")
workflow.add_edge("extract_user_preferences", "supervisor")

# After each lookup, return to supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# Supervisor decides where to go next or ends
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Set up memory
memory = MemorySaver()

# Compile the graph with memory
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    # Test code to run
    graph.get_graph().print_ascii()
    from langfuse.callback import CallbackHandler
    langfuse_callback = CallbackHandler(
        host=config.LANGFUSE_HOST,
        secret_key=config.LANGFUSE_SECRET_KEY,
        public_key=config.LANGFUSE_PUBLIC_KEY
    )
    thread_config = {"configurable": {"thread_id": "1"}, "callbacks": [langfuse_callback]}

    for s in graph.stream(
            {
                "chat_history": [
                ],
                # Current question.
                "input": [
                    HumanMessage(content="What variables and studies are around sickle cell?"),
                ]
            }, config=thread_config
    ):
        if "__end__" not in s:
            print(s)
            state = graph.get_state(thread_config)
            print(state)
       