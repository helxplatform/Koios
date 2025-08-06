# Helper package to create "agents" or more like nodes,
# By definition an agent is llm prompt with set of tools for llm to utilize in performing a task.
# Here we don't have tools, hence the quotes around agents.
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langfuse.decorators import observe, langfuse_context
from guardrails.input_guard import InputGuard
import config as app_config

import logging_util
from models.agent_state import AgentState

logger = logging_util.logger


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


@observe()
def agent_node_dict(state: AgentState, agent, name):
    trace_id = langfuse_context.get_current_trace_id()

    chat_history = state.chat_history
    # this is the input for the next agent
    input_data = {
        "input": state.input,
        "chat_history": chat_history,
        "extra": state.extra,
        "user_intent": state.user_intent,
        "return_prompt": state.return_prompt
    }
    result = agent.invoke(input_data)
    extra = result.get('extra', {})
    extra.update({"trace_id": trace_id})
    # this output is what the next agent will see.
    if state.return_prompt:
        extra.update({"context": result.get("prompt", "")})
    output = {
        "output": AIMessage(content=result.get('output', ''), name=name),
        "next": result.get('next', ""),
        "input": state.input,
        "extra": extra,
        "user_intent": result.get("user_intent", {})
    }
    return output


def guardrails_node(state: AgentState):
    # Process through guardrails
    guardrails_instance = InputGuard(app_config)
    result = guardrails_instance.invoke({"input": state.input}) #[-1].content})
    if "I'm sorry, I can't respond to that." in result.get("output", ""):
        state.next = "FINISH"
        state.output = AIMessage(content="I'm sorry, but I can't answer that question. I’m a assistant "
                                            "designed exclusively to support inquiries related to  NHLBI - "
                                            "BioData Catalyst research studies. Please redirect your question "
                                            "to focus on topics related to NHLBI-supported medical research studies.")
        return state

    state.next = "continue"

    return state
