import functools
import operator

from typing import Sequence, TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, StateGraph, START

import config
from agents.utils import *
from agents.supervisor import SupervisorAgent
from langgraph.checkpoint.memory import MemorySaver
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
import config as app_config
from pydantic import BaseModel, Field

from agents.intent_agent_graph import intent_node, extract_user_preferences_node 

# The agent state is the input to each node in the graph
# Our state Schema (https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    input: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: Annotated[List[str], operator.add]
    chat_history: List[BaseMessage]
    extra: Dict[str, Any]


# Create our agents (KG lookup and QV lookup)
kg_lookup_agent_node = functools.partial(agent_node_dict, agent=KGChain(app_config).as_generative_chain(), name="KG_lookup_agent")
qv_lookup_agent_node = functools.partial(agent_node_dict, agent=QuestionLookupChain(app_config).as_generative_chain(), name="QV_lookup_agent")
supervisor_agent = SupervisorAgent(app_config)
members = supervisor_agent.members

# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)

workflow.add_node("intent", intent_node)
workflow.add_node("extract_user_preferences", extract_user_preferences_node)
workflow.add_node("KG_lookup", kg_lookup_agent_node)
workflow.add_node("QV_lookup", qv_lookup_agent_node)
workflow.add_node("supervisor", supervisor_agent.as_generative_chain())

workflow.add_edge(START, "intent")
workflow.add_edge("intent", "extract_user_preferences")
workflow.add_edge("extract_user_preferences", "supervisor")


# Define how members are laid out (researcher, comedian, etc.)
for member in members:
    workflow.add_edge(member, "supervisor")  # Ends after running the agent unless routed to the supervisor

# The supervisor populates the "next" field in the graph state which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

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
    thread_config = {"configurable": {"thread_id": "1"}, "callbacks":[langfuse_callback]}

    for s in graph.stream(
            {
                # Mimicking previous interactions.
                "chat_history": [
                    # ("wHr ", "the heart is melting"),
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
            # print(state['intents'])  # Prints the detected intents

