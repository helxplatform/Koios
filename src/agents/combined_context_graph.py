import functools
import operator

from typing import TypedDict, Annotated, Optional
from pydantic import Field
from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START


from langfuse.callback import CallbackHandler
from langchain.callbacks.manager import CallbackManager
import config
from agents.utils import *
from langgraph.checkpoint.memory import MemorySaver
from chains.qvkg_chain import QVKGChain
import config as app_config
from guardrails.input_guard import InputGuard


# The agent state is the input to each node in the graph
# Our state Schema (https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    input: str
    # The 'next' field indicates where to route to next
    next: str
    output: Optional[AIMessage]
    chat_history: List = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


# Create our agents (KG lookup and QV lookup)
qv_kg_lookup_agent_node = functools.partial(agent_node_dict, agent=QVKGChain(app_config).as_generative_chain(),
                                         name="lookup_agent")

guardrails_instance = InputGuard(app_config)


def guardrails_node(state: AgentState) -> AgentState:
    # Process through guardrails
    result = guardrails_instance.invoke({"input": state["input"]}) #[-1].content})

    if "I'm sorry, I can't respond to that." in result.get("output", ""):
        state["next"] = "FINISH"
        state["output"] = AIMessage(content=result.get("output", "I'm sorry, I can't respond to that."))
        return state

    state["next"] = "lookup_agent"

    return state


# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)


workflow.add_node("guardrails", guardrails_node)
workflow.add_node("lookup_agent", qv_kg_lookup_agent_node)

workflow.add_edge(START, "guardrails")
workflow.add_conditional_edges("guardrails", lambda x: x["next"],
                               {"lookup_agent": "lookup_agent", "FINISH": END})
workflow.add_edge("lookup_agent", END)

langfuse_callback = CallbackHandler(
    host=config.LANGFUSE_HOST,
    secret_key=config.LANGFUSE_SECRET_KEY,
    public_key=config.LANGFUSE_PUBLIC_KEY
)
callback_manager = CallbackManager([langfuse_callback])
# Compile the graph with memory
graph = workflow.compile(checkpointer=MemorySaver()).with_config(callbacks=callback_manager)

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
                # Mimicking previous interactions.
                "chat_history": [
                    # ("wHr ", "the heart is melting"),
                ],
                # Current question.
                "input": "What variables and studies are around sickle cell?",

            }, config=thread_config
    ):
        if "__end__" not in s:
            print(s)
            state = graph.get_state(thread_config)
            print(state)
            # print(state['intents'])  # Prints the detected intents

