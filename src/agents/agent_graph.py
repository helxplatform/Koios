import functools
import operator
from typing import Sequence, TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START
from agents.utils import *
from agents.supervisor import llm, supervisor_chain
from langgraph.checkpoint.memory import MemorySaver
from chain import init_chain
from agents.supervisor import members


# The agent state is the input to each node in the graph
# Our state Schema (https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    input: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str
    chat_history: List = []



# here lets create some nodes to add to the workflow.

# we will grab our existing chain (the vector lookup) and fix it to the agent_node(state, agent, name) function using partial function
research_node = functools.partial(agent_node, agent=init_chain(), name="researcher")


# we also create a new llm chain that has the system prompt here, and an llm to create a new chain.
comedian_agent = create_agent(
    llm,
    "You are a comedian",
)
# same thing convert this to a node , but wrapping it into the agent node function.
comedian_node = functools.partial(agent_node, agent=comedian_agent, name="comedian")

# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)

# add the nodes
workflow.add_node("researcher", research_node)
workflow.add_node("comedian", comedian_node)
workflow.add_node("supervisor", supervisor_chain)

# Define how members are layed out. (note members is just a list  ['researcher', 'comedian']
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    # workflow.add_edge(member, "supervisor")
    # uncomment the line above to pass messages back to the supervisor.
    # or else the following line just make the graph end after an agent runs.
    workflow.add_edge(member, END)

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
# connect supervisor node with all the members
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint , making the supervisor the one that accepts user input.
workflow.add_edge(START, "supervisor")

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
                # this is mimicing prevous interactions.
                "chat_history": [(
                        ("jokes  around Heart"),
                        ("the heart is melting"),)
                ],
                # current question.
                "input": [
                    HumanMessage(content="explain that more"),
                ]
            }, config=thread_config
    ):
        if "__end__" not in s:
            print(s)
            state = graph.get_state(thread_config)
            print(state.next)
