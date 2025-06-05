import functools
from langgraph.graph import END, StateGraph, START
from langfuse.callback import CallbackHandler
from langchain.callbacks.manager import CallbackManager
import config
from agents.utils import *
from models.agent_state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from chains import QVKGChain, UserIntentChain
import config as app_config


# The agent state is the input to each node in the graph
# Our state Schema (https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)


# Create our agents (KG lookup and QV lookup)
qv_kg_lookup_agent_node = functools.partial(agent_node_dict, agent=QVKGChain(app_config).as_generative_chain(),
                                         name="lookup_agent")
intent_agent_node = functools.partial(agent_node_dict, agent=UserIntentChain(app_config).as_generative_chain(),
                                      name="intent_agent")


# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)


workflow.add_node("guardrails", guardrails_node)
workflow.add_node("lookup_agent", qv_kg_lookup_agent_node)
workflow.add_node("intent_agent", intent_agent_node)

workflow.add_edge(START, "guardrails")
workflow.add_conditional_edges("guardrails", lambda x: x["next"],
                               {"continue": "intent_agent", "FINISH": END})
workflow.add_edge("intent_agent", "lookup_agent")
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

