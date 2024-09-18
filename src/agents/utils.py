# Helper package to create "agents" or more like nodes,
# By definition an agent is llm prompt with set of tools for llm to utilize in performing a task.
# Here we don't have tools, hence the quotes around agents.
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


# basic chain with a prompt and a llm with output parsed to a string.
def create_agent(llm: ChatOpenAI, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="input"),
            # MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return prompt | llm | StrOutputParser()

# Nodes in lang graph are just function calls that just accept the Langraph State.
# the langraph state is where agents (chains) and other nodes more generally write to, so it's accessible by others.
# more info (https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph)
# here we are passing other information such as agent (more of a chain in our case) and name parameters. These are going
# to have to be static. When we are constracting the langgraph Graph we will use something like
# node_1 = functools.partial(agent_node, my_intialized_chain, "my-chain")
# so node_1 becomes a new function with just one parameter node_1(state) , that is going to be a node in the langraph.
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"input": [AIMessage(content=result, name=name)]}
